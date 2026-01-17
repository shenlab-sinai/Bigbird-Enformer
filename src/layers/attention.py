import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce

def local_block_sparse_attention(q, k, v, block_size):
    # q,k,v: [B,H,N,Dh] for non global tokens only
    B, H, N, Dh = q.shape
    assert N % block_size == 0, "Non-global length must be divisible by block_size"
    nb = N // block_size

    q_blocks = q.view(B, H, nb, block_size, Dh)
    k_blocks = k.view(B, H, nb, block_size, Dh)
    v_blocks = v.view(B, H, nb, block_size, Dh)

    k_left  = torch.roll(k_blocks,  1, dims=2)
    k_right = torch.roll(k_blocks, -1, dims=2)
    k_window = torch.cat([k_left, k_blocks, k_right], dim=3)  # [B,H,nb,3B,Dh]

    v_left  = torch.roll(v_blocks,  1, dims=2)
    v_right = torch.roll(v_blocks, -1, dims=2)
    v_window = torch.cat([v_left, v_blocks, v_right], dim=3)

    scale = 1.0 / math.sqrt(Dh)
    scores = torch.matmul(q_blocks, k_window.transpose(-1, -2)) * scale  # [B,H,nb,B,3B]
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_window)  # [B,H,nb,B,Dh]
    return out.view(B, H, N, Dh)

class BigBirdAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=128, dim_key=64, dim_value=64, num_global_tokens=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.inner_k = num_heads * dim_key
        self.inner_v = num_heads * dim_value
        self.num_global_tokens = num_global_tokens

        self.q_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_v, bias=False)
        self.out_proj = nn.Linear(self.inner_v, d_model)

    def forward(self, x, global_indices=None):
        # x: [B, G+N, D]
        B, T, _ = x.shape
        G = self.num_global_tokens
        assert T > G
        xg = x[:, :G, :]   # [B,G,D]
        xx = x[:, G:, :]   # [B,N,D]
        N = xx.size(1)

        # project combined to avoid extra matmuls?
        q = self.q_proj(x).view(B, T, self.num_heads, self.dim_key).transpose(1, 2)   # [B,H,T,dk]
        k = self.k_proj(x).view(B, T, self.num_heads, self.dim_key).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.dim_value).transpose(1, 2)

        qg, qx = q[:, :, :G, :], q[:, :, G:, :]   # [B,H,G,dk], [B,H,N,dk]
        kg, kx = k[:, :, :G, :], k[:, :, G:, :]
        vg, vx = v[:, :, :G, :], v[:, :, G:, :]

        # local sparse among x tokens
        out_local_x = local_block_sparse_attention(qx, kx, vx, self.block_size)  # [B,H,N,dv]

        # x attends to globals (read)
        scale = 1.0 / math.sqrt(self.dim_key)
        scores_xg = torch.matmul(qx, kg.transpose(-1, -2)) * scale    # [B,H,N,G]
        attn_xg = F.softmax(scores_xg, dim=-1)
        out_xg = torch.matmul(attn_xg, vg)                            # [B,H,N,dv]

        out_x = out_local_x + out_xg                                  # [B,H,N,dv]

        # globals attend to x (update)
        scores_gx = torch.matmul(qg, kx.transpose(-1, -2)) * scale    # [B,H,G,N]
        attn_gx = F.softmax(scores_gx, dim=-1)
        out_g = torch.matmul(attn_gx, vx)                             # [B,H,G,dv]

        # concat globals + x so globals persist to next layer
        out = torch.cat([out_g, out_x], dim=2)                        # [B,H,G+N,dv]
        out = out.transpose(1, 2).reshape(B, T, self.inner_v)         # [B,T,H*dv]
        return self.out_proj(out)

class FullAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.,
        **kwargs 
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        attn = dots.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)