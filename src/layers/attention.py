# src/layers/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigBirdAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        block_size: int = 128,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.dropout_p  = float(dropout)

        self.inner_k = heads * dim_key
        self.inner_v = heads * dim_value

        self.to_q   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_k   = nn.Linear(dim, self.inner_k, bias=False)
        self.to_v   = nn.Linear(dim, self.inner_v, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H  = self.heads
        dk = self.dim_key
        dv = self.dim_value
        BS = self.block_size
        N  = T - 1          # number of local tokens
        nb = N // BS        # number of local blocks
        drop = self.dropout_p if self.training else 0.0

        assert N % BS == 0, (
            f"Local length N={N} (= seq_len-1) must be divisible by block_size={BS}"
        )

        q = self.to_q(x).view(B, T, H, dk).transpose(1, 2) 
        k = self.to_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, dv).transpose(1, 2)

        kg = k[:, :, :1, :]   
        vg = v[:, :, :1, :]  
        kx = k[:, :, 1:, :]  
        vx = v[:, :, 1:, :]  

        # Global token
        out_g = F.scaled_dot_product_attention(
            q[:, :, :1, :], k, v, dropout_p=drop, is_causal=False
        ) 

        # Local blocks
        qx_blk = q[:, :, 1:, :].view(B, H, nb, BS, dk)
        kx_blk = kx.view(B, H, nb, BS, dk)
        vx_blk = vx.view(B, H, nb, BS, dv)

        zk = kx.new_zeros(B, H, 1, BS, dk)
        zv = vx.new_zeros(B, H, 1, BS, dv)
        kx_pad = torch.cat([zk, kx_blk, zk], dim=2)  
        vx_pad = torch.cat([zv, vx_blk, zv], dim=2)

        k_local = torch.cat(
            [kx_pad[:, :, 0:nb], kx_pad[:, :, 1:nb+1], kx_pad[:, :, 2:nb+2]], dim=3
        )
        v_local = torch.cat(
            [vx_pad[:, :, 0:nb], vx_pad[:, :, 1:nb+1], vx_pad[:, :, 2:nb+2]], dim=3
        )

        # Prepend global token to every block
        kg_exp = kg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        vg_exp = vg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        k_full = torch.cat([kg_exp, k_local], dim=3)
        v_full = torch.cat([vg_exp, v_local], dim=3)

        # Flatten (B, nb)
        q_flat = qx_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,          dk)
        k_flat = k_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS,  dk)
        v_flat = v_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS,  dv)

        # FlashAttention
        out_flat = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False
        ) 

        out_x = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)

        # Reassemble
        out = torch.cat([out_g, out_x], dim=2)           
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_v)
        return self.to_out(out)


class BlockSparseAttention(nn.Module):
    """
    Block Sparse Attention with Global Tokens.

    Structure: [Global, Local_1, Local_2, ..., Local_N] per block.

    Input: [Batch, SeqLen, Dim]
           SeqLen must be divisible by block_size.
           (block_size should include the global token, e.g. 128 local + 1 global = 129).
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        block_size: int = 129,
        dim_key: int = 64,
        dim_value: int = 64,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.block_size = block_size
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)

        self.inner_k = heads * dim_key
        self.inner_v = heads * dim_value

        self.to_q = nn.Linear(dim, self.inner_k, bias=False)
        self.to_k = nn.Linear(dim, self.inner_k, bias=False)
        self.to_v = nn.Linear(dim, self.inner_v, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads
        BLK = self.block_size

        assert N % BLK == 0, f"Sequence length {N} must be divisible by block_size {BLK}"
        num_blocks = N // BLK

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q_blk_view = q.view(B, num_blocks, BLK, H, self.dim_key)
        k_blk_view = k.view(B, num_blocks, BLK, H, self.dim_key)
        v_blk_view = v.view(B, num_blocks, BLK, H, self.dim_value)

        # Intra-block attention
        q_blk = q_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        k_blk = k_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        v_blk = v_blk_view.reshape(B * num_blocks, BLK, H, self.dim_value).transpose(1, 2)

        out_blk = F.scaled_dot_product_attention(
            q_blk, k_blk, v_blk,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        ) 

        out_blk = out_blk.view(B, num_blocks, H, BLK, self.dim_value)

        # Global-Global attention 
        q_g = q_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        k_g = k_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        v_g = v_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)

        out_g = F.scaled_dot_product_attention(
            q_g, k_g, v_g,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        # Combine 
        out_g_reshaped = out_g.permute(0, 2, 1, 3).unsqueeze(3)
        global_part = out_blk[:, :, :, 0:1, :] + out_g_reshaped
        local_part  = out_blk[:, :, :, 1:,  :]

        out_combined = torch.cat([global_part, local_part], dim=3)
        out_combined = out_combined.permute(0, 1, 3, 2, 4).reshape(B, N, self.inner_v)
        return self.to_out(out_combined)


class FullAttention(nn.Module):
    """
    Standard Full Attention (FlashAttention via SDPA).
    """
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.0, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)

        self.inner_k = heads * dim_key
        self.inner_v = heads * dim_value

        self.to_q = nn.Linear(dim, self.inner_k, bias=False)
        self.to_k = nn.Linear(dim, self.inner_k, bias=False)
        self.to_v = nn.Linear(dim, self.inner_v, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

    def forward(self, x):
        B, N, _ = x.shape
        H = self.heads

        q = self.to_q(x).view(B, N, H, self.dim_key).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, self.dim_key).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, self.dim_value).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)