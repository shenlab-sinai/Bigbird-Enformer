# src/layers/attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 
# local sparse helpers (2-block and 3-block)
# 

def _edge_mask_2(nb: int, B: int, device: torch.device, window_dir: str):
    """
    mask: [1,1,nb,1,2B], True means invalid positions
    window_dir:
      - "left" : [left, current]  -> first block's left half is padding
      - "right": [current, right] -> last  block's right half is padding
    """
    m = torch.zeros((nb, 2 * B), device=device, dtype=torch.bool)
    if window_dir == "left":
        m[0, :B] = True
    elif window_dir == "right":
        m[-1, B:] = True
    else:
        raise ValueError(f"window_dir must be 'left' or 'right', got {window_dir}")
    return m.view(1, 1, nb, 1, 2 * B)


def _edge_mask_3(nb: int, B: int, device: torch.device):
    """
    mask: [1,1,nb,1,3B]
    window = [left, current, right]
      - first block: left part invalid
      - last  block: right part invalid
    """
    m = torch.zeros((nb, 3 * B), device=device, dtype=torch.bool)
    m[0, :B] = True
    m[-1, 2 * B:] = True
    return m.view(1, 1, nb, 1, 3 * B)


def local_sparse_2block(
    q, k, v,
    block_size: int,
    window_dir: str,                 # "left" or "right"
    attn_dropout_p: float = 0.0,
    training: bool = True,
):
    """
    2-block local sparse:
      window_dir="left"  -> [left, current]
      window_dir="right" -> [current, right]
    q,k: [B,H,N,Dk], v: [B,H,N,Dv]
    """
    b, h, n, dk = q.shape
    assert n % block_size == 0, f"N={n} must be divisible by block_size={block_size}"
    nb = n // block_size
    B = block_size
    dv = v.shape[-1]

    q_blocks = q.reshape(b, h, nb, B, dk)
    k_blocks = k.reshape(b, h, nb, B, dk)
    v_blocks = v.reshape(b, h, nb, B, dv)

    # pad with zeros at both ends (no wrap)
    z_k = torch.zeros_like(k_blocks[:, :, :1])
    z_v = torch.zeros_like(v_blocks[:, :, :1])
    k_pad = torch.cat([z_k, k_blocks, z_k], dim=2)  # [b,h,nb+2,B,dk]
    v_pad = torch.cat([z_v, v_blocks, z_v], dim=2)  # [b,h,nb+2,B,dv]

    if window_dir == "left":
        k_a = k_pad[:, :, 0:nb]
        k_b = k_pad[:, :, 1:nb+1]
        v_a = v_pad[:, :, 0:nb]
        v_b = v_pad[:, :, 1:nb+1]
    elif window_dir == "right":
        k_a = k_pad[:, :, 1:nb+1]
        k_b = k_pad[:, :, 2:nb+2]
        v_a = v_pad[:, :, 1:nb+1]
        v_b = v_pad[:, :, 2:nb+2]
    else:
        raise ValueError(f"window_dir must be 'left' or 'right', got {window_dir}")

    k_window = torch.cat([k_a, k_b], dim=3)  # [b,h,nb,2B,dk]
    v_window = torch.cat([v_a, v_b], dim=3)  # [b,h,nb,2B,dv]

    scale = 1.0 / math.sqrt(dk)
    scores = torch.matmul(q_blocks.float(), k_window.float().transpose(-1, -2)) * scale  # [b,h,nb,B,2B]

    # mask padded half-window at edges
    scores = scores.masked_fill(_edge_mask_2(nb, B, q.device, window_dir), float("-inf"))

    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    if attn_dropout_p > 0.0:
        attn = F.dropout(attn, p=attn_dropout_p, training=training)

    out = torch.matmul(attn, v_window)  # [b,h,nb,B,dv]
    return out.reshape(b, h, n, dv)

def local_sparse_3block(q, k, v, block_size, attn_dropout_p=0.0, training=True):
    """
    Zero-copy implementation using as_strided.
    """
    b, h, n, dk = q.shape
    dv = v.shape[-1]
    nb = n // block_size
    B = block_size

    # 1. View as blocks [b, h, nb, B, d]
    q_blocks = q.view(b, h, nb, B, dk)
    k_blocks = k.view(b, h, nb, B, dk)
    v_blocks = v.view(b, h, nb, B, dv)

    # 2. Pad K and V for the windows (allocates small memory)
    # We need 1 block padding on left and right
    z_k = torch.zeros(b, h, 1, B, dk, device=k.device, dtype=k.dtype)
    z_v = torch.zeros(b, h, 1, B, dv, device=v.device, dtype=v.dtype)
    
    # We must concatenate ONCE to create the padded canvas
    k_pad = torch.cat([z_k, k_blocks, z_k], dim=2) 
    v_pad = torch.cat([z_v, v_blocks, z_v], dim=2)

    s_b, s_h, s_nb, s_B, s_d = k_pad.stride()

    k_unfolded = k_pad.unfold(dimension=2, size=3, step=1) # [b, h, nb, B, dk, 3]
    v_unfolded = v_pad.unfold(dimension=2, size=3, step=1)
    
    k_window = k_unfolded.permute(0, 1, 2, 5, 3, 4).reshape(b, h, nb, 3*B, dk)
    v_window = v_unfolded.permute(0, 1, 2, 5, 3, 4).reshape(b, h, nb, 3*B, dv)

    scale = 1.0 / math.sqrt(dk)
    scores = torch.matmul(q_blocks, k_window.transpose(-1, -2)) * scale
    
    scores = scores.masked_fill(_edge_mask_3(nb, B, q.device), float("-inf"))
    
    attn = torch.softmax(scores, dim=-1)
    if attn_dropout_p > 0.0:
        attn = F.dropout(attn, p=attn_dropout_p, training=training)
        
    out = torch.matmul(attn, v_window)
    return out.reshape(b, h, n, dv)


class HierarchicalAttention(nn.Module):
    """
    Layout: [g0, chunk0..., g1, chunk1..., ..., g(C-1), chunk(C-1)...]
    Optimized to avoid OOM by removing redundant tensor concatenations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_chunks: int,
        block_size: int = 128,
        dim_key: int = 64,
        dim_value: int = 64,
        attn_dropout: float = 0.0,
        local_window_blocks: int = 3,
        layer_idx: int = 0,
    ):
        super().__init__()
        assert num_chunks >= 1
        assert local_window_blocks in (2, 3)

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_chunks = int(num_chunks)
        self.block_size = block_size
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.inner_k = num_heads * dim_key
        self.inner_v = num_heads * dim_value
        self.attn_dropout_p = float(attn_dropout)
        self.local_window_blocks = int(local_window_blocks)
        self.layer_idx = int(layer_idx)

        # For sparse_2 alternate direction
        if self.local_window_blocks == 2:
            self.window_dir = "left" if (self.layer_idx % 2 == 0) else "right"
        else:
            self.window_dir = None

        self.q_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_v, bias=False)
        self.out_proj = nn.Linear(self.inner_v, d_model)

    def _softmax_fp32(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        attn = torch.softmax(scores, dim=dim, dtype=torch.float32)
        return attn.to(scores.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads
        dk = self.dim_key
        dv = self.dim_value
        C = self.num_chunks

        assert T > C
        N = T - C
        Nchunk = N // C
        
        scale = 1.0 / math.sqrt(dk)

        q = self.q_proj(x).view(B, T, H, dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dv).transpose(1, 2)

        # --- 1. Slice Interleaved Layout ---
        stride = Nchunk + 1
        g_idx = torch.arange(0, C * stride, stride, device=x.device)
        
        # Globals: [B,H,C,1,dk]
        gq = q.index_select(dim=2, index=g_idx).unsqueeze(3)
        gk = k.index_select(dim=2, index=g_idx).unsqueeze(3)
        gv = v.index_select(dim=2, index=g_idx).unsqueeze(3)

        # Locals: [B,H,C*Nchunk,dk]
        base = g_idx + 1
        offsets = torch.arange(0, Nchunk, device=x.device)
        local_idx = (base[:, None] + offsets[None, :]).reshape(-1)
        
        qx = q.index_select(dim=2, index=local_idx)
        kx = k.index_select(dim=2, index=local_idx)
        vx = v.index_select(dim=2, index=local_idx)

        # Reshape locals to chunked form: [B,H,C,Nchunk,dk]
        qx = qx.view(B, H, C, Nchunk, dk)
        kx = kx.view(B, H, C, Nchunk, dk)
        vx = vx.view(B, H, C, Nchunk, dv)

        # --- 2. Local Sparse Attention (Optimized) ---
        # Flatten chunks into batch -> [B*C, H, Nchunk, dk]
        q_flat = qx.permute(0, 2, 1, 3, 4).reshape(B * C, H, Nchunk, dk)
        k_flat = kx.permute(0, 2, 1, 3, 4).reshape(B * C, H, Nchunk, dk)
        v_flat = vx.permute(0, 2, 1, 3, 4).reshape(B * C, H, Nchunk, dv)

        if self.local_window_blocks == 3:
            # Use the new zero-copy optimized function
            out_local_flat = local_sparse_3block(
                q_flat, k_flat, v_flat,
                block_size=self.block_size,
                attn_dropout_p=self.attn_dropout_p,
                training=self.training,
            )
        else:
            # Keep existing 2-block or write a similar optimized version
            out_local_flat = local_sparse_2block(
                q_flat, k_flat, v_flat,
                block_size=self.block_size,
                window_dir=self.window_dir,
                attn_dropout_p=self.attn_dropout_p,
                training=self.training,
            )

        out_local = out_local_flat.view(B, C, H, Nchunk, dv).permute(0, 2, 1, 3, 4)

        # --- 3. Global Mixing (Optimized Split Calculation) ---
        
        # Locals attend to their own global
        # Simple addition to local output
        out_xg = gv.expand(-1, -1, -1, Nchunk, -1)
        if self.attn_dropout_p > 0.0:
            out_xg = F.dropout(out_xg, p=self.attn_dropout_p, training=self.training)
        out_chunks = out_local + out_xg

        # Globals attend to [All Globals] + [Own Chunk Locals]
        # We split this calculation to avoid concatenating huge K/V tensors
        
        # 3a. Global <-> Global scores [B,H,C,1,C]
        # We need all globals (gk_all) against each specific global
        gk_all = gk.squeeze(3) # [B,H,C,dk]
        gv_all = gv.squeeze(3)
        # Replicate for broadcast: [B,H,C,C,dk]
        gk_rep = gk_all.unsqueeze(2).expand(-1, -1, C, -1, -1)
        gv_rep = gv_all.unsqueeze(2).expand(-1, -1, C, -1, -1)
        
        scores_g_g = torch.matmul(gq, gk_rep.transpose(-1, -2)) * scale
        
        # 3b. Global <-> Chunk Local scores [B,H,C,1,Nchunk]
        # kx is [B,H,C,Nchunk,dk]
        scores_g_l = torch.matmul(gq, kx.transpose(-1, -2)) * scale
        
        # 3c. Concat SCORES only (Memory Safe)
        # Result: [B,H,C,1, C + Nchunk]
        scores_combined = torch.cat([scores_g_g, scores_g_l], dim=-1)
        
        attn_combined = self._softmax_fp32(scores_combined, dim=-1)
        if self.attn_dropout_p > 0.0:
            attn_combined = F.dropout(attn_combined, p=self.attn_dropout_p, training=self.training)
            
        # 3d. Weighted Sum
        # Split weights back
        attn_g_g = attn_combined[..., :C]      # [B,H,C,1,C]
        attn_g_l = attn_combined[..., C:]      # [B,H,C,1,Nchunk]
        
        out_g_g = torch.matmul(attn_g_g, gv_rep) # [B,H,C,1,dv]
        out_g_l = torch.matmul(attn_g_l, vx)     # [B,H,C,1,dv]
        
        out_g = out_g_g + out_g_l

        # --- 4. Final Reassembly ---
        out_interleaved = torch.cat([out_g, out_chunks], dim=3)
        out = out_interleaved.reshape(B, H, C * (1 + Nchunk), dv)
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_v)
        
        return self.out_proj(out)

class BigBirdAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 128,
        dim_key: int = 64,
        dim_value: int = 64,
        num_global_tokens: int = 2,
        attn_dropout: float = 0.0,

        # new knobs (safe defaults)
        local_window_blocks: int = 3,   # 2 => sparse_2, 3 => sparse_3
        layer_idx: int = 0,             # used only when local_window_blocks=2
    ):
        super().__init__()
        assert local_window_blocks in (2, 3), f"local_window_blocks must be 2 or 3, got {local_window_blocks}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.inner_k = num_heads * dim_key
        self.inner_v = num_heads * dim_value
        self.num_global_tokens = int(num_global_tokens)
        self.attn_dropout_p = float(attn_dropout)

        self.local_window_blocks = int(local_window_blocks)
        self.layer_idx = int(layer_idx)

        # for sparse_2 alternate direction, always starting from LEFT
        if self.local_window_blocks == 2:
            self.window_dir = "left" if (self.layer_idx % 2 == 0) else "right"
        else:
            self.window_dir = None

        self.q_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_v, bias=False)
        self.out_proj = nn.Linear(self.inner_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads

        assert self.num_global_tokens == 2, (
            f"This attention assumes exactly 2 globals (front+end). Got {self.num_global_tokens}"
        )
        assert T >= 3, f"Need at least 2 globals + 1 local. Got T={T}"

        scale = 1.0 / math.sqrt(self.dim_key)

        q = self.q_proj(x).view(B, T, H, self.dim_key).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.dim_key).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.dim_value).transpose(1, 2)

        # 2 globals: front and end
        qg = torch.cat([q[:, :, 0:1], q[:, :, -1:]], dim=2)  # [B,H,2,Dk]
        kg = torch.cat([k[:, :, 0:1], k[:, :, -1:]], dim=2)  # [B,H,2,Dk]
        vg = torch.cat([v[:, :, 0:1], v[:, :, -1:]], dim=2)  # [B,H,2,Dv]

        # locals
        qx = q[:, :, 1:-1]
        kx = k[:, :, 1:-1]
        vx = v[:, :, 1:-1]

        N = T - 2
        assert N % self.block_size == 0, f"Local N must be divisible by block_size. N={N}, block_size={self.block_size}"

        # local sparse
        if self.local_window_blocks == 2:
            out_local = local_sparse_2block(
                qx, kx, vx,
                block_size=self.block_size,
                window_dir=self.window_dir,
                attn_dropout_p=self.attn_dropout_p,
                training=self.training,
            )
        else:
            out_local = local_sparse_3block(
                qx, kx, vx,
                block_size=self.block_size,
                attn_dropout_p=self.attn_dropout_p,
                training=self.training,
            )

        # locals attend to globals
        scores_xg = torch.matmul(qx, kg.transpose(-1, -2)) * scale  # [B,H,N,2]
        attn_xg = torch.softmax(scores_xg, dim=-1)
        if self.attn_dropout_p > 0.0:
            attn_xg = F.dropout(attn_xg, p=self.attn_dropout_p, training=self.training)
        out_xg = torch.matmul(attn_xg, vg)  # [B,H,N,Dv]

        out_x = out_local + out_xg

        # globals attend to (globals + locals) => includes global<->global
        k_for_g = torch.cat([kg, kx], dim=2)  # [B,H,2+N,Dk]
        v_for_g = torch.cat([vg, vx], dim=2)  # [B,H,2+N,Dv]

        scores_g = torch.matmul(qg, k_for_g.transpose(-1, -2)) * scale  # [B,H,2,2+N]
        attn_g = torch.softmax(scores_g, dim=-1)
        if self.attn_dropout_p > 0.0:
            attn_g = F.dropout(attn_g, p=self.attn_dropout_p, training=self.training)
        out_g = torch.matmul(attn_g, v_for_g)  # [B,H,2,Dv]

        out_front, out_end = out_g[:, :, 0:1], out_g[:, :, 1:2]
        out = torch.cat([out_front, out_x, out_end], dim=2)  # [B,H,T,Dv]

        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_v)
        return self.out_proj(out)



class FullAttention(nn.Module):
    """
    Standard O(N^2) attention using plain matrix multiplication.
    """
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.05, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        
    def forward(self, x):
        B, N, _ = x.shape
        h = self.heads
        
        q = self.to_q(x).view(B, N, h, self.dim_key).transpose(1, 2)
        k = self.to_k(x).view(B, N, h, self.dim_key).transpose(1, 2)
        v = self.to_v(x).view(B, N, h, self.dim_value).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.dim_key)

        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        attn = torch.softmax(scores, dim=-1)
        
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
            
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, h * self.dim_value)
        return self.to_out(out)