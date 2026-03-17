import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# TRITON KERNELS
# ---------------------------------------------------------------------------

@triton.jit
def rms_norm_kernel(
    x_ptr, y_ptr,
    stride_xb, stride_xt, stride_xd,
    stride_yb, stride_yt, stride_yd,
    D: tl.constexpr, eps: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    b_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    
    x_ptr += b_idx * stride_xb + t_idx * stride_xt
    y_ptr += b_idx * stride_yb + t_idx * stride_yt

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    x = tl.load(x_ptr + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / D
    y = (x * tl.math.rsqrt(variance + eps)).to(tl.bfloat16)
    
    tl.store(y_ptr + offs_d * stride_yd, y, mask=mask)

@triton.jit
def fused_qkvg_norm_rope_kernel(
    x_ptr, wq_ptr, wk_ptr, wv_ptr, wg_ptr,
    factor1_ptr, factor2_ptr, 
    q_out_ptr, k_out_ptr, v_out_ptr, g_out_ptr,
    M, T, H, D: tl.constexpr, eps: tl.constexpr,
    
    stride_xb, stride_xt, stride_xd,
    stride_wq_out, stride_wq_in,
    stride_wk_out, stride_wk_in,
    stride_wv_out, stride_wv_in,
    stride_wg_out, stride_wg_in,
    
    stride_f1_t, stride_f1_d,
    stride_f2_t, stride_f2_d,
    
    stride_qb, stride_qh, stride_qt, stride_qn,
    stride_kb, stride_kh, stride_kt, stride_kn,
    stride_vb, stride_vh, stride_vt, stride_vn,
    stride_gb, stride_gh, stride_gt, stride_gn, # Now fully 4D (B, H, T, N)
    
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    N: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * H
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_h = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    # Native 3D decomposition
    b_idx = offs_m // T
    t_idx = offs_m % T
    
    offs_n = pid_h * N + tl.arange(0, N)
    offs_N = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_q = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        # Native 3D pointer tracking
        x_ptrs = x_ptr + (b_idx[:, None] * stride_xb + t_idx[:, None] * stride_xt + (k + offs_k)[None, :] * stride_xd)
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

        wq = tl.load(wq_ptr + (offs_n[None, :] * stride_wq_out + (k + offs_k)[:, None] * stride_wq_in))
        wk = tl.load(wk_ptr + (offs_n[None, :] * stride_wk_out + (k + offs_k)[:, None] * stride_wk_in))
        wv = tl.load(wv_ptr + (offs_n[None, :] * stride_wv_out + (k + offs_k)[:, None] * stride_wv_in))
        wg = tl.load(wg_ptr + (offs_n[None, :] * stride_wg_out + (k + offs_k)[:, None] * stride_wg_in))

        acc_q += tl.dot(x, wq)
        acc_k += tl.dot(x, wk)
        acc_v += tl.dot(x, wv)
        acc_g += tl.dot(x, wg)

    q_var = tl.sum(acc_q * acc_q, axis=1) / N
    q_normed = (acc_q * tl.math.rsqrt(q_var + eps)[:, None]).to(tl.bfloat16)

    k_var = tl.sum(acc_k * acc_k, axis=1) / N
    k_normed = (acc_k * tl.math.rsqrt(k_var + eps)[:, None]).to(tl.bfloat16)
    
    v_out = acc_v.to(tl.bfloat16)
    
    f1 = tl.load(factor1_ptr + t_idx[:, None] * stride_f1_t + offs_N[None, :] * stride_f1_d, mask=mask_m[:, None], other=0.0)
    f2 = tl.load(factor2_ptr + t_idx[:, None] * stride_f2_t + offs_N[None, :] * stride_f2_d, mask=mask_m[:, None], other=0.0)

    # -----------------------------------------------------------
    # NATIVE OUTPUT MAPPING (Zero Views)
    # -----------------------------------------------------------
    q_out_ptrs = q_out_ptr + (b_idx[:, None] * stride_qb + pid_h * stride_qh + t_idx[:, None] * stride_qt + offs_N[None, :] * stride_qn)
    k_out_ptrs = k_out_ptr + (b_idx[:, None] * stride_kb + pid_h * stride_kh + t_idx[:, None] * stride_kt + offs_N[None, :] * stride_kn)
    v_out_ptrs = v_out_ptr + (b_idx[:, None] * stride_vb + pid_h * stride_vh + t_idx[:, None] * stride_vt + offs_N[None, :] * stride_vn)
    
    # G now aligns perfectly to (B, H, T, N) target
    g_out_ptrs = g_out_ptr + (b_idx[:, None] * stride_gb + pid_h * stride_gh + t_idx[:, None] * stride_gt + offs_N[None, :] * stride_gn)

    g_sig = tl.sigmoid(acc_g).to(tl.bfloat16)
    tl.store(v_out_ptrs, v_out, mask=mask_m[:, None])
    tl.store(g_out_ptrs, g_sig, mask=mask_m[:, None])

    tl.store(q_out_ptrs, q_normed, mask=mask_m[:, None])
    tl.store(k_out_ptrs, k_normed, mask=mask_m[:, None])
    
    tl.debug_barrier()

    offs_N_flip = offs_N ^ 1
    
    q_flip_ptrs = q_out_ptr + (b_idx[:, None] * stride_qb + pid_h * stride_qh + t_idx[:, None] * stride_qt + offs_N_flip[None, :] * stride_qn)
    k_flip_ptrs = k_out_ptr + (b_idx[:, None] * stride_kb + pid_h * stride_kh + t_idx[:, None] * stride_kt + offs_N_flip[None, :] * stride_kn)

    q_flip = tl.load(q_flip_ptrs, mask=mask_m[:, None])
    k_flip = tl.load(k_flip_ptrs, mask=mask_m[:, None])

    q_rope = q_normed * f1 + q_flip * f2
    k_rope = k_normed * f1 + k_flip * f2

    tl.store(q_out_ptrs, q_rope, mask=mask_m[:, None])
    tl.store(k_out_ptrs, k_rope, mask=mask_m[:, None])

@triton.jit
def gate_repack_kernel(
    attn_ptr, g_ptr, out_ptr,
    stride_ab, stride_ah, stride_at, stride_an,
    stride_gb, stride_gh, stride_gt, stride_gn,
    stride_ob, stride_ot, stride_on,
    T,
    BLOCK_T: tl.constexpr, N: tl.constexpr
):
    b_idx = tl.program_id(0)
    t_pid = tl.program_id(1)
    h_idx = tl.program_id(2)

    offs_t = t_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    offs_n = tl.arange(0, N) 

    # Read from (B, H, T, N) native SDPA output layout
    attn_ptrs = attn_ptr + (b_idx * stride_ab + h_idx * stride_ah + offs_t[:, None] * stride_at + offs_n[None, :] * stride_an)
    attn = tl.load(attn_ptrs, mask=mask_t[:, None])

    # Read from (B, H, T, N) matching native gate layout
    g_ptrs = g_ptr + (b_idx * stride_gb + h_idx * stride_gh + offs_t[:, None] * stride_gt + offs_n[None, :] * stride_gn)
    g = tl.load(g_ptrs, mask=mask_t[:, None])

    out = attn * g

    # Write directly to final (B, T, H*N) memory block
    g_n_idx = h_idx * N + offs_n
    out_ptrs = out_ptr + (b_idx * stride_ob + offs_t[:, None] * stride_ot + g_n_idx[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=mask_t[:, None])

import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel_recompute(
    x_ptr, y_ptr, inv_rms_ptr,  # <-- Added inv_rms_ptr
    stride_xb, stride_xt, stride_xd,
    stride_yb, stride_yt, stride_yd,
    stride_ir_b, stride_ir_t,   # <-- Added strides for inv_rms (B, T)
    D: tl.constexpr, eps: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    b_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    
    x_ptr += b_idx * stride_xb + t_idx * stride_xt
    y_ptr += b_idx * stride_yb + t_idx * stride_yt

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    x = tl.load(x_ptr + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / D
    inv_rms = tl.math.rsqrt(variance + eps)  # Calculate rsqrt once
    
    y = (x * inv_rms).to(tl.bfloat16)
    
    tl.store(y_ptr + offs_d * stride_yd, y, mask=mask)
    
    # Store the inverse RMS scalar for this (b, t) token
    inv_rms_ptr += b_idx * stride_ir_b + t_idx * stride_ir_t
    tl.store(inv_rms_ptr, inv_rms.to(tl.float32))

@triton.jit
def fused_qkvg_norm_rope_kernel_recompute(
    x_ptr, wq_ptr, wk_ptr, wv_ptr, wg_ptr,
    factor1_ptr, factor2_ptr, 
    q_out_ptr, k_out_ptr, v_out_ptr, g_out_ptr,
    q_inv_rms_ptr, k_inv_rms_ptr,  # <-- Added q and k inv_rms pointers
    M, T, H, D: tl.constexpr, eps: tl.constexpr,
    
    stride_xb, stride_xt, stride_xd,
    stride_wq_out, stride_wq_in,
    stride_wk_out, stride_wk_in,
    stride_wv_out, stride_wv_in,
    stride_wg_out, stride_wg_in,
    
    stride_f1_t, stride_f1_d,
    stride_f2_t, stride_f2_d,
    
    stride_qb, stride_qh, stride_qt, stride_qn,
    stride_kb, stride_kh, stride_kt, stride_kn,
    stride_vb, stride_vh, stride_vt, stride_vn,
    stride_gb, stride_gh, stride_gt, stride_gn, 
    
    stride_iq_b, stride_iq_h, stride_iq_t,  # <-- Strides for q_inv_rms (B, H, T)
    stride_ik_b, stride_ik_h, stride_ik_t,  # <-- Strides for k_inv_rms (B, H, T)
    
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    N: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * H
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_h = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    # Native 3D decomposition
    b_idx = offs_m // T
    t_idx = offs_m % T
    
    offs_n = pid_h * N + tl.arange(0, N)
    offs_N = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_q = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        x_ptrs = x_ptr + (b_idx[:, None] * stride_xb + t_idx[:, None] * stride_xt + (k + offs_k)[None, :] * stride_xd)
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

        wq = tl.load(wq_ptr + (offs_n[None, :] * stride_wq_out + (k + offs_k)[:, None] * stride_wq_in))
        wk = tl.load(wk_ptr + (offs_n[None, :] * stride_wk_out + (k + offs_k)[:, None] * stride_wk_in))
        wv = tl.load(wv_ptr + (offs_n[None, :] * stride_wv_out + (k + offs_k)[:, None] * stride_wv_in))
        wg = tl.load(wg_ptr + (offs_n[None, :] * stride_wg_out + (k + offs_k)[:, None] * stride_wg_in))

        acc_q += tl.dot(x, wq)
        acc_k += tl.dot(x, wk)
        acc_v += tl.dot(x, wv)
        acc_g += tl.dot(x, wg)

    # Calculate variables and rsqrt 
    q_var = tl.sum(acc_q * acc_q, axis=1) / N
    q_inv_rms = tl.math.rsqrt(q_var + eps)
    q_normed = (acc_q * q_inv_rms[:, None]).to(tl.bfloat16)

    k_var = tl.sum(acc_k * acc_k, axis=1) / N
    k_inv_rms = tl.math.rsqrt(k_var + eps)
    k_normed = (acc_k * k_inv_rms[:, None]).to(tl.bfloat16)
    
    v_out = acc_v.to(tl.bfloat16)
    
    f1 = tl.load(factor1_ptr + t_idx[:, None] * stride_f1_t + offs_N[None, :] * stride_f1_d, mask=mask_m[:, None], other=0.0)
    f2 = tl.load(factor2_ptr + t_idx[:, None] * stride_f2_t + offs_N[None, :] * stride_f2_d, mask=mask_m[:, None], other=0.0)

    # -----------------------------------------------------------
    # OUTPUT MAPPING
    # -----------------------------------------------------------
    q_out_ptrs = q_out_ptr + (b_idx[:, None] * stride_qb + pid_h * stride_qh + t_idx[:, None] * stride_qt + offs_N[None, :] * stride_qn)
    k_out_ptrs = k_out_ptr + (b_idx[:, None] * stride_kb + pid_h * stride_kh + t_idx[:, None] * stride_kt + offs_N[None, :] * stride_kn)
    v_out_ptrs = v_out_ptr + (b_idx[:, None] * stride_vb + pid_h * stride_vh + t_idx[:, None] * stride_vt + offs_N[None, :] * stride_vn)
    g_out_ptrs = g_out_ptr + (b_idx[:, None] * stride_gb + pid_h * stride_gh + t_idx[:, None] * stride_gt + offs_N[None, :] * stride_gn)

    # Inv RMS Pointers (Shape: B, H, T)
    q_inv_rms_ptrs = q_inv_rms_ptr + (b_idx * stride_iq_b + pid_h * stride_iq_h + t_idx * stride_iq_t)
    k_inv_rms_ptrs = k_inv_rms_ptr + (b_idx * stride_ik_b + pid_h * stride_ik_h + t_idx * stride_ik_t)

    g_sig = tl.sigmoid(acc_g).to(tl.bfloat16)
    tl.store(v_out_ptrs, v_out, mask=mask_m[:, None])
    tl.store(g_out_ptrs, g_sig, mask=mask_m[:, None])

    # Store the inverse RMS arrays
    tl.store(q_inv_rms_ptrs, q_inv_rms.to(tl.float32), mask=mask_m)
    tl.store(k_inv_rms_ptrs, k_inv_rms.to(tl.float32), mask=mask_m)

    tl.store(q_out_ptrs, q_normed, mask=mask_m[:, None])
    tl.store(k_out_ptrs, k_normed, mask=mask_m[:, None])
    
    tl.debug_barrier()

    offs_N_flip = offs_N ^ 1
    
    q_flip_ptrs = q_out_ptr + (b_idx[:, None] * stride_qb + pid_h * stride_qh + t_idx[:, None] * stride_qt + offs_N_flip[None, :] * stride_qn)
    k_flip_ptrs = k_out_ptr + (b_idx[:, None] * stride_kb + pid_h * stride_kh + t_idx[:, None] * stride_kt + offs_N_flip[None, :] * stride_kn)

    q_flip = tl.load(q_flip_ptrs, mask=mask_m[:, None])
    k_flip = tl.load(k_flip_ptrs, mask=mask_m[:, None])

    q_rope = q_normed * f1 + q_flip * f2
    k_rope = k_normed * f1 + k_flip * f2

    tl.store(q_out_ptrs, q_rope, mask=mask_m[:, None])
    tl.store(k_out_ptrs, k_rope, mask=mask_m[:, None])

@triton.jit
def gate_repack_kernel_bwd_recompute(
    attn_ptr, g_ptr, out_ptr,
    stride_ab, stride_ah, stride_at, stride_an,
    stride_gb, stride_gh, stride_gt, stride_gn,
    stride_om, stride_on,  # <-- Output strides are now just 2D: (M, N) where M = B*T
    T,
    BLOCK_T: tl.constexpr, N: tl.constexpr
):
    b_idx = tl.program_id(0)
    t_pid = tl.program_id(1)
    h_idx = tl.program_id(2)

    offs_t = t_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    offs_n = tl.arange(0, N) 

    # Read from (B, H, T, N) native SDPA output layout
    attn_ptrs = attn_ptr + (b_idx * stride_ab + h_idx * stride_ah + offs_t[:, None] * stride_at + offs_n[None, :] * stride_an)
    attn = tl.load(attn_ptrs, mask=mask_t[:, None])

    # Read from (B, H, T, N) matching native gate layout
    g_ptrs = g_ptr + (b_idx * stride_gb + h_idx * stride_gh + offs_t[:, None] * stride_gt + offs_n[None, :] * stride_gn)
    g = tl.load(g_ptrs, mask=mask_t[:, None])

    out = attn * g

    # Calculate flat row index for M = B*T
    m_idx = b_idx * T + offs_t
    g_n_idx = h_idx * N + offs_n
    
    # Write directly to final 2D (B*T, H*N) memory block
    out_ptrs = out_ptr + (m_idx[:, None] * stride_om + g_n_idx[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=mask_t[:, None])
    
# ---------------------------------------------------------------------------
# AUTOGRAD FUNCTION
# ---------------------------------------------------------------------------

class FusedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, g_weight, o_weight, factor1, factor2, H, N, eps):
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # 1. RMS Norm (Pre-projection)
        x_normed = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        rms_norm_kernel[(B, T)](
            x, x_normed,
            x.stride(0), x.stride(1), x.stride(2),
            x_normed.stride(0), x_normed.stride(1), x_normed.stride(2),
            D=D, eps=eps, BLOCK_D=BLOCK_D, num_warps=4
        )

        # 2. Fused QKVG Projection + Norm + RoPE
        q_out = torch.empty((B, H, T, N), device=device, dtype=dtype)
        k_out = torch.empty((B, H, T, N), device=device, dtype=dtype)
        v_out = torch.empty((B, H, T, N), device=device, dtype=dtype)
        g_out = torch.empty((B, H, T, N), device=device, dtype=dtype)

        BLOCK_M = 32
        BLOCK_K = 32
        grid = (triton.cdiv(B * T, BLOCK_M) * H, )

        fused_qkvg_norm_rope_kernel[grid](
            x_normed, 
            q_weight, k_weight, v_weight, g_weight,
            factor1, factor2, 
            q_out, k_out, v_out, g_out,
            M=B * T, T=T, H=H, D=D, eps=eps,
            
            stride_xb=x_normed.stride(0), stride_xt=x_normed.stride(1), stride_xd=x_normed.stride(2),
            stride_wq_out=q_weight.stride(0), stride_wq_in=q_weight.stride(1),
            stride_wk_out=k_weight.stride(0), stride_wk_in=k_weight.stride(1),
            stride_wv_out=v_weight.stride(0), stride_wv_in=v_weight.stride(1),
            stride_wg_out=g_weight.stride(0), stride_wg_in=g_weight.stride(1),
            
            stride_f1_t=factor1.stride(0), stride_f1_d=factor1.stride(1),
            stride_f2_t=factor2.stride(0), stride_f2_d=factor2.stride(1),
            
            stride_qb=q_out.stride(0), stride_qh=q_out.stride(1), stride_qt=q_out.stride(2), stride_qn=q_out.stride(3),
            stride_kb=k_out.stride(0), stride_kh=k_out.stride(1), stride_kt=k_out.stride(2), stride_kn=k_out.stride(3),
            stride_vb=v_out.stride(0), stride_vh=v_out.stride(1), stride_vt=v_out.stride(2), stride_vn=v_out.stride(3),
            stride_gb=g_out.stride(0), stride_gh=g_out.stride(1), stride_gt=g_out.stride(2), stride_gn=g_out.stride(3),
            
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, N=N, GROUP_SIZE_M=32,
            num_warps=4, num_stages=2
        )

        # 3. Scaled Dot Product Attention (Switched to ATen for LSE + RNG state)
        scale = 1.0 / math.sqrt(N)
        
        (
            attn_out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask
        ) = torch.ops.aten._scaled_dot_product_flash_attention(
            q_out, k_out, v_out,
            dropout_p=0.0,
            is_causal=True,
            return_debug_mask=False,
            scale=scale
        )

        # 4. Gating and Repackaging
        final_out_repacked = torch.empty((B, T, H * N), device=device, dtype=dtype)
        BLOCK_T_REPACK = 64
        grid_gate = (B, triton.cdiv(T, BLOCK_T_REPACK), H)

        gate_repack_kernel[grid_gate](
            attn_out, g_out, final_out_repacked,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            g_out.stride(0), g_out.stride(1), g_out.stride(2), g_out.stride(3),
            final_out_repacked.stride(0), final_out_repacked.stride(1), final_out_repacked.stride(2),
            T=T, BLOCK_T=BLOCK_T_REPACK, N=N, num_warps=4
        )

        # 5. Output Projection
        output = F.linear(final_out_repacked, o_weight)

        # In FusedAttentionFunction.forward:
        ctx.save_for_backward(
            x, q_weight, k_weight, v_weight, g_weight, o_weight, 
            factor1, factor2, attn_out, logsumexp
        )
        
        # Save integers and non-tensor metadata directly to the context object
        ctx.cum_seq_q = cum_seq_q
        ctx.cum_seq_k = cum_seq_k
        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.H = H
        ctx.N = N
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x, q_weight, k_weight, v_weight, g_weight, o_weight, 
         factor1, factor2, attn_out, logsumexp) = ctx.saved_tensors
         
        cum_seq_q = ctx.cum_seq_q
        cum_seq_k = ctx.cum_seq_k
        max_q = ctx.max_q
        max_k = ctx.max_k
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        H, N, eps = ctx.H, ctx.N, ctx.eps
        
        B, T, D = x.shape
        scale = 1.0 / math.sqrt(N)

        def swap_even_odd(tensor):
            return tensor.view(*tensor.shape[:-1], N // 2, 2).flip(-1).view_as(tensor)

        # =========================================================================
        # RECOMPUTATION PHASE
        # =========================================================================
        
        # 1. Main RMS Norm Recomputation
        x_normed = torch.empty_like(x)
        x_inv_rms = torch.empty((B, T), device=x.device, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(D)
        
        rms_norm_kernel_recompute[(B, T)](
            x, x_normed, x_inv_rms,
            x.stride(0), x.stride(1), x.stride(2),
            x_normed.stride(0), x_normed.stride(1), x_normed.stride(2),
            x_inv_rms.stride(0), x_inv_rms.stride(1),
            D=D, eps=eps, BLOCK_D=BLOCK_D, num_warps=4
        )
        
        # 2. Fused QKVG Recomputation
        q_out = torch.empty((B, H, T, N), device=x.device, dtype=x.dtype)
        k_out = torch.empty((B, H, T, N), device=x.device, dtype=x.dtype)
        v_out = torch.empty((B, H, T, N), device=x.device, dtype=x.dtype)
        g_out = torch.empty((B, H, T, N), device=x.device, dtype=x.dtype)
        
        q_inv_rms = torch.empty((B, H, T), device=x.device, dtype=torch.float32)
        k_inv_rms = torch.empty((B, H, T), device=x.device, dtype=torch.float32)
        
        BLOCK_M = 32
        BLOCK_K = 32
        grid = (triton.cdiv(B * T, BLOCK_M) * H, )

        fused_qkvg_norm_rope_kernel_recompute[grid](
            x_normed, 
            q_weight, k_weight, v_weight, g_weight,
            factor1, factor2, 
            q_out, k_out, v_out, g_out,
            q_inv_rms, k_inv_rms,
            M=B * T, T=T, H=H, D=D, eps=eps,
            
            stride_xb=x_normed.stride(0), stride_xt=x_normed.stride(1), stride_xd=x_normed.stride(2),
            stride_wq_out=q_weight.stride(0), stride_wq_in=q_weight.stride(1),
            stride_wk_out=k_weight.stride(0), stride_wk_in=k_weight.stride(1),
            stride_wv_out=v_weight.stride(0), stride_wv_in=v_weight.stride(1),
            stride_wg_out=g_weight.stride(0), stride_wg_in=g_weight.stride(1),
            
            stride_f1_t=factor1.stride(0), stride_f1_d=factor1.stride(1),
            stride_f2_t=factor2.stride(0), stride_f2_d=factor2.stride(1),
            
            stride_qb=q_out.stride(0), stride_qh=q_out.stride(1), stride_qt=q_out.stride(2), stride_qn=q_out.stride(3),
            stride_kb=k_out.stride(0), stride_kh=k_out.stride(1), stride_kt=k_out.stride(2), stride_kn=k_out.stride(3),
            stride_vb=v_out.stride(0), stride_vh=v_out.stride(1), stride_vt=v_out.stride(2), stride_vn=v_out.stride(3),
            stride_gb=g_out.stride(0), stride_gh=g_out.stride(1), stride_gt=g_out.stride(2), stride_gn=g_out.stride(3),
            
            stride_iq_b=q_inv_rms.stride(0), stride_iq_h=q_inv_rms.stride(1), stride_iq_t=q_inv_rms.stride(2),
            stride_ik_b=k_inv_rms.stride(0), stride_ik_h=k_inv_rms.stride(1), stride_ik_t=k_inv_rms.stride(2),
            
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, N=N, GROUP_SIZE_M=32,
            num_warps=4, num_stages=2
        )

        # 3. Recompute final_out_repacked directly into a flattened 2D tensor
        final_out_repacked_flat = torch.empty((B * T, H * N), device=x.device, dtype=x.dtype)
        BLOCK_T_REPACK = 64
        grid_gate = (B, triton.cdiv(T, BLOCK_T_REPACK), H)

        gate_repack_kernel_bwd_recompute[grid_gate](
            attn_out, g_out, final_out_repacked_flat,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            g_out.stride(0), g_out.stride(1), g_out.stride(2), g_out.stride(3),
            final_out_repacked_flat.stride(0), final_out_repacked_flat.stride(1),
            T=T, BLOCK_T=BLOCK_T_REPACK, N=N, num_warps=4
        )

        # =========================================================================
        # GRADIENT COMPUTATION PHASE
        # =========================================================================

        # --- 5. Output Projection Backward ---
        grad_output_flat = grad_output.reshape(-1, H * N)
        grad_final_out_repacked_flat = grad_output_flat.matmul(o_weight) 
        grad_o_weight = grad_output_flat.t().matmul(final_out_repacked_flat) 

        # --- 4. Gating and Repackaging Backward ---
        grad_final_out_repacked_view = grad_final_out_repacked_flat.view(B, T, H, N).transpose(1, 2) 
        grad_attn_out = grad_final_out_repacked_view * g_out
        grad_g_out = grad_final_out_repacked_view * attn_out

        # --- 3. Scaled Dot Product Attention Backward ---
        grad_q_out, grad_k_out, grad_v_out = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_attn_out.contiguous(),
            q_out.contiguous(),
            k_out.contiguous(),
            v_out.contiguous(),
            attn_out.contiguous(),
            logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
            dropout_p=0.0, is_causal=True,
            philox_seed=philox_seed, philox_offset=philox_offset, scale=scale
        )

        # --- 2. Fused QKVG + Norm + RoPE Backward ---
        
        # a. Gating reverse
        grad_acc_g = grad_g_out * g_out * (1.0 - g_out) 

        # b. Head RMS Norm reverse exploiting RoPE Orthogonality
        # Because dot products are invariant to rotation, we can compute the RMS norm
        # gradient in the rotated space!
        def rms_norm_backward_rotated(grad_out, out, inv_rms, dim_size):
            inv_rms = inv_rms.unsqueeze(-1).to(grad_out.dtype)
            dot_term = (grad_out * out).sum(dim=-1, keepdim=True)
            return inv_rms * (grad_out - (out / dim_size) * dot_term)

        grad_acc_q_rotated = rms_norm_backward_rotated(grad_q_out, q_out, q_inv_rms, N)
        grad_acc_k_rotated = rms_norm_backward_rotated(grad_k_out, k_out, k_inv_rms, N)

        # c. Apply Inverse RoPE to the rotated RMS gradients
        f1_sliced = factor1[:T, :].view(1, 1, T, N)
        f2_sliced = factor2[:T, :].view(1, 1, T, N)

        grad_acc_q = grad_acc_q_rotated * f1_sliced + swap_even_odd(grad_acc_q_rotated * f2_sliced)
        grad_acc_k = grad_acc_k_rotated * f1_sliced + swap_even_odd(grad_acc_k_rotated * f2_sliced)
        grad_acc_v = grad_v_out

        # d. Gradients w.r.t RoPE factors (Mathematically rewritten to avoid q_normed/k_normed)
        # Using algebraic expansions of the swap function, we can compute grad_factor 
        # using only q_out and grad_q_out.
        dot_grad_q = (grad_q_out * q_out).sum(dim=(0, 1))
        dot_grad_k = (grad_k_out * k_out).sum(dim=(0, 1))
        dot_grad_q_swap = (grad_q_out * swap_even_odd(q_out)).sum(dim=(0, 1))
        dot_grad_k_swap = (grad_k_out * swap_even_odd(k_out)).sum(dim=(0, 1))
        
        f1_2d = factor1[:T, :]
        f2_2d = factor2[:T, :]

        grad_f1_active = (dot_grad_q + dot_grad_k) * f1_2d - (dot_grad_q_swap + dot_grad_k_swap) * f2_2d
        grad_f2_active = (dot_grad_q_swap + dot_grad_k_swap) * f1_2d + (dot_grad_q + dot_grad_k) * f2_2d

        grad_factor1 = torch.zeros_like(factor1)
        grad_factor2 = torch.zeros_like(factor2)
        grad_factor1[:T, :] = grad_f1_active
        grad_factor2[:T, :] = grad_f2_active

        # e. Gradients w.r.t Q, K, V, G weights
        grad_acc_q_flat = grad_acc_q.transpose(1, 2).reshape(-1, H*N)
        grad_acc_k_flat = grad_acc_k.transpose(1, 2).reshape(-1, H*N)
        grad_acc_v_flat = grad_acc_v.transpose(1, 2).reshape(-1, H*N)
        grad_acc_g_flat = grad_acc_g.transpose(1, 2).reshape(-1, H*N)

        x_normed_flat = x_normed.view(-1, D)

        grad_q_weight = grad_acc_q_flat.t().matmul(x_normed_flat)
        grad_k_weight = grad_acc_k_flat.t().matmul(x_normed_flat)
        grad_v_weight = grad_acc_v_flat.t().matmul(x_normed_flat)
        grad_g_weight = grad_acc_g_flat.t().matmul(x_normed_flat)

        # f. Pass gradients down to x_normed
        grad_x_normed_flat = (
            grad_acc_q_flat.matmul(q_weight) +
            grad_acc_k_flat.matmul(k_weight) +
            grad_acc_v_flat.matmul(v_weight) +
            grad_acc_g_flat.matmul(g_weight)
        )
        grad_x_normed = grad_x_normed_flat.view(B, T, D)

        # --- 1. Main RMS Norm Backward ---
        def rms_norm_backward_new(grad_normed, normed_output, inv_rms, dim_size):
            inv_rms = inv_rms.unsqueeze(-1).to(grad_normed.dtype)
            dot_term = (grad_normed * normed_output).sum(dim=-1, keepdim=True)
            return inv_rms * (grad_normed - (normed_output / dim_size) * dot_term)

        grad_x = rms_norm_backward_new(grad_x_normed, x_normed, x_inv_rms, D)

        return (
            grad_x, grad_q_weight, grad_k_weight, grad_v_weight, grad_g_weight, grad_o_weight, 
            grad_factor1, grad_factor2, 
            None, None, None # For H, N, eps
        )

# ---------------------------------------------------------------------------
# UPDATED FORWARD WRAPPER
# ---------------------------------------------------------------------------

def fused_attn(x, q_proj, k_proj, v_proj, g_proj, o_proj, yarn_module, H, N, eps=1e-6):
    """
    Wrapper to call the autograd function.
    """
    return FusedAttentionFunction.apply(
        x, 
        q_proj.weight, 
        k_proj.weight, 
        v_proj.weight, 
        g_proj.weight, 
        o_proj.weight,
        yarn_module.factor1, 
        yarn_module.factor2,
        H, 
        N, 
        eps
    )
