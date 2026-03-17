Results on pretraining 50M model with bsz=262k tokens: -1GB of memory usage.

The commented-out code is equivalent to `fused_attn`

```python
class CausalSoftmaxAttention(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int, 
        head_dim: int, 
    ):
        super().__init__()
        
        self.head_dim = head_dim
        self.num_heads = num_heads

        N = self.head_dim
        H = self.num_heads
        C = input_dim

        with torch.no_grad():
            init_bounds = 0.5 / (C ** 0.5)
    
            self.q_proj = nn.Linear(C, H*N, bias=False)
            self.k_proj = nn.Linear(C, H*N, bias=False)
            self.v_proj = nn.Linear(C, H*N, bias=False)
            self.g_proj = nn.Linear(C, H*N, bias=False)
            self.o_proj = nn.Linear(H*N, C, bias=False)
    
            self.q_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.k_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.v_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.g_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.o_proj.weight.data.zero_()

    def forward(self, x, yarn):
        B, T, C = x.size()
        N = self.head_dim
        H = self.num_heads

        # def forward1(x):
        #     x = norm(x)
            
        #     q = self.q_proj(x).view(B, T, H, N)
        #     k = self.k_proj(x).view(B, T, H, N)
        #     v = self.v_proj(x).view(B, T, H, N)
        #     g = self.g_proj(x)

        #     q, k = yarn.rotary(norm(q)), yarn.rotary(norm(k))
            
        #     return (q, k, v, g)

        # (q, k, v, g) = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)

        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        #     x = F.scaled_dot_product_attention(
        #         q.transpose(1, 2), 
        #         k.transpose(1, 2), 
        #         v.transpose(1, 2), 
        #         is_causal=True, 
        #     ).transpose(1, 2).contiguous().view(B, T, H*N)

        # x = self.o_proj(x * torch.sigmoid(g))

        x = fused_attn(x, self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj, yarn, H, N)
        
        return x
```
