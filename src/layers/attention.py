# src/layers/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BigBirdAttention(nn.Module):
    """
    BigBird attention with ONE learnable global token prepended at position 0.
    Layout: [global | local_0 ... local_{N-1}]  T = N+1, N must be divisible by block_size.
    Used with SingleGlobalTokenInjector / SingleGlobalTokenRemover.
    """
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64,
                 block_size=128, dropout=0.0, **kwargs):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.dropout_p  = float(dropout)
        self.inner_k    = heads * dim_key
        self.inner_v    = heads * dim_value

        self.to_q   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_k   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_v   = nn.Linear(dim, self.inner_v, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv, BS = self.heads, self.dim_key, self.dim_value, self.block_size
        N  = T - 1
        nb = N // BS
        drop = self.dropout_p if self.training else 0.0
        assert N % BS == 0, f"Local length N={N} must be divisible by block_size={BS}"

        q = self.to_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.to_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, dv).transpose(1, 2)

        # Global token attends to all
        out_g = F.scaled_dot_product_attention(
            q[:, :, :1, :], k, v, dropout_p=drop, is_causal=False
        )

        kg, vg = k[:, :, :1, :], v[:, :, :1, :]
        kx, vx = k[:, :, 1:, :], v[:, :, 1:, :]

        qx_blk = q[:, :, 1:, :].view(B, H, nb, BS, dk)
        kx_blk = kx.view(B, H, nb, BS, dk)
        vx_blk = vx.view(B, H, nb, BS, dv)

        zk    = kx.new_zeros(B, H, 1, BS, dk)
        zv    = vx.new_zeros(B, H, 1, BS, dv)
        kx_pad = torch.cat([zk, kx_blk, zk], dim=2)
        vx_pad = torch.cat([zv, vx_blk, zv], dim=2)

        k_local = torch.cat([kx_pad[:, :, 0:nb], kx_pad[:, :, 1:nb+1], kx_pad[:, :, 2:nb+2]], dim=3)
        v_local = torch.cat([vx_pad[:, :, 0:nb], vx_pad[:, :, 1:nb+1], vx_pad[:, :, 2:nb+2]], dim=3)

        kg_exp = kg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        vg_exp = vg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        k_full = torch.cat([kg_exp, k_local], dim=3)
        v_full = torch.cat([vg_exp, v_local], dim=3)

        q_flat = qx_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,         dk)
        k_flat = k_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS, dk)
        v_flat = v_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS, dv)

        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False)
        out_x = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)

        out = torch.cat([out_g, out_x], dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_v)
        return self.to_out(out)


class BlockSparseAttention(nn.Module):
    """
    Block Sparse Attention with Global Tokens.
    block_size includes the global token (e.g. 128 local + 1 global = 129).
    Used with ChunkGlobalTokenInjector / ChunkGlobalTokenRemover.
    """
    def __init__(self, dim, heads=8, block_size=129, dim_key=64, dim_value=64,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.heads      = heads
        self.block_size = block_size
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.dropout_p  = float(dropout)
        self.inner_k    = heads * dim_key
        self.inner_v    = heads * dim_value

        self.to_q   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_k   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_v   = nn.Linear(dim, self.inner_v, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

    def forward(self, x):
        B, N, D = x.shape
        H, BLK  = self.heads, self.block_size
        assert N % BLK == 0, f"Sequence length {N} must be divisible by block_size {BLK}"
        num_blocks = N // BLK

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q_blk_view = q.view(B, num_blocks, BLK, H, self.dim_key)
        k_blk_view = k.view(B, num_blocks, BLK, H, self.dim_key)
        v_blk_view = v.view(B, num_blocks, BLK, H, self.dim_value)

        q_blk = q_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        k_blk = k_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        v_blk = v_blk_view.reshape(B * num_blocks, BLK, H, self.dim_value).transpose(1, 2)

        out_blk = F.scaled_dot_product_attention(
            q_blk, k_blk, v_blk,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
        )
        out_blk = out_blk.view(B, num_blocks, H, BLK, self.dim_value)

        q_g = q_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        k_g = k_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        v_g = v_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        out_g = F.scaled_dot_product_attention(
            q_g, k_g, v_g,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
        )

        out_g_reshaped = out_g.permute(0, 2, 1, 3).unsqueeze(3)
        global_part  = out_blk[:, :, :, 0:1, :] + out_g_reshaped
        local_part   = out_blk[:, :, :, 1:,  :]

        out_combined = torch.cat([global_part, local_part], dim=3)
        out_combined = out_combined.permute(0, 1, 3, 2, 4).reshape(B, N, self.inner_v)
        return self.to_out(out_combined)


class FullAttention(nn.Module):
    """Standard Full Attention via SDPA (FlashAttention eligible)."""
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.0, **kwargs):
        super().__init__()
        self.heads     = heads
        self.dim_key   = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)
        self.inner_v   = heads * dim_value

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(heads * dim_value, dim)

    def forward(self, x):
        B, N, _ = x.shape
        H = self.heads
        q = self.to_q(x).view(B, N, H, self.dim_key).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, self.dim_key).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, self.dim_value).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class BigBirdAttentionAblation(nn.Module):
    """
    BigBird local-window attention — NO global tokens.
    Ablation baseline: pure sliding-window over N=1536 tokens.
    Each 128bp block attends only to [prev | curr | next] blocks.
    Use with nn.Identity() injector/remover.
    """
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64,
                 block_size=128, dropout=0.0, **kwargs):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.dropout_p  = float(dropout)
        self.inner_v    = heads * dim_value

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(heads * dim_value, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv, BS = self.heads, self.dim_key, self.dim_value, self.block_size
        nb   = N // BS
        drop = self.dropout_p if self.training else 0.0
        assert N % BS == 0, f"N={N} must be divisible by block_size={BS}"

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        q_blk = q.view(B, H, nb, BS, dk)
        k_blk = k.view(B, H, nb, BS, dk)
        v_blk = v.view(B, H, nb, BS, dv)

        zk    = k.new_zeros(B, H, 1, BS, dk)
        zv    = v.new_zeros(B, H, 1, BS, dv)
        k_pad = torch.cat([zk, k_blk, zk], dim=2)
        v_pad = torch.cat([zv, v_blk, zv], dim=2)

        k_win = torch.cat([k_pad[:, :, 0:nb], k_pad[:, :, 1:nb+1], k_pad[:, :, 2:nb+2]], dim=3)
        v_win = torch.cat([v_pad[:, :, 0:nb], v_pad[:, :, 1:nb+1], v_pad[:, :, 2:nb+2]], dim=3)

        q_flat = q_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,     dk)
        k_flat = k_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dk)
        v_flat = v_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dv)

        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False)
        out = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class BigBirdCCREAttention(nn.Module):
    """
    BigBird attention with biologically grounded in-place cCRE global tokens.
    """

    def __init__(self, dim, heads=8, dim_key=64, dim_value=64,
                 block_size=128, dropout=0.0, max_seq_len=1536, **kwargs):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.dropout_p  = float(dropout)
        self.inner_v    = heads * dim_value

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(heads * dim_value, dim)

        # Pre-build local window mask [N, N] once; registered as a non-parameter buffer.
        # local_window_mask[i, j] = True iff j is in the 3-block window around block(i).
        N, BS = max_seq_len, block_size
        assert N % BS == 0, f"max_seq_len={N} must be divisible by block_size={BS}"
        nb = N // BS
        local_mask = torch.zeros(N, N, dtype=torch.bool)
        for bi in range(nb):
            q_s = bi * BS
            q_e = q_s + BS
            k_s = max(0, (bi - 1) * BS)
            k_e = min(N, (bi + 2) * BS)
            local_mask[q_s:q_e, k_s:k_e] = True
        self.register_buffer("local_window_mask", local_mask)   # [N, N]

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        drop = self.dropout_p if self.training else 0.0

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)   # [B, H, N, dk]
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        if is_global is None or not is_global.any():
            out = self._local_window_forward(q, k, v, B, N, H, dk, dv, drop)
        else:
            #   local_window_mask : [1, 1, N, N]  same for all samples
            #   is_global rows    : [B, 1, N, 1]  global query --> attend to all keys
            #   is_global cols    : [B, 1, 1, N]  all queries --> attend to global keys
            attn_mask = (
                self.local_window_mask.view(1, 1, N, N)
                | is_global.view(B, 1, N, 1)
                | is_global.view(B, 1, 1, N)
            )                                                  # [B, 1, N, N] bool
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
            )

        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)

    def _local_window_forward(self, q, k, v, B, N, H, dk, dv, drop):
        """Block-gather sliding window — no mask, FlashAttention eligible."""
        BS = self.block_size
        nb = N // BS

        q_blk = q.view(B, H, nb, BS, dk)
        k_blk = k.view(B, H, nb, BS, dk)
        v_blk = v.view(B, H, nb, BS, dv)

        zk = k.new_zeros(B, H, 1, BS, dk)
        zv = v.new_zeros(B, H, 1, BS, dv)
        k_pad = torch.cat([zk, k_blk, zk], dim=2)
        v_pad = torch.cat([zv, v_blk, zv], dim=2)

        k_win = torch.cat([k_pad[:, :, 0:nb], k_pad[:, :, 1:nb+1], k_pad[:, :, 2:nb+2]], dim=3)
        v_win = torch.cat([v_pad[:, :, 0:nb], v_pad[:, :, 1:nb+1], v_pad[:, :, 2:nb+2]], dim=3)

        q_flat = q_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,     dk)
        k_flat = k_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dk)
        v_flat = v_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dv)

        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False)
        return out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)