import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Relative positional encoding helpers (ported from original Enformer)

def get_positional_features_exponential(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    min_half_life: float = 3.0,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Exponential decay basis functions with log-spaced half-lives."""
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]          # [1, features]
    positions = positions.abs()[..., None]     # [2N-1, 1]
    return torch.exp(-math.log(2.0) / half_life * positions).to(dtype)


def get_positional_features_central_mask(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Binary central-mask basis: 1 if |distance| ≤ 2^i, else 0."""
    center_widths = 2 ** torch.arange(
        1, features + 1, device=positions.device
    ).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)


def gamma_pdf(
    x: torch.Tensor,
    concentration: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    """Gamma probability density function (unnormalized log-space)."""
    log_unnorm = torch.xlogy(concentration - 1.0, x) - rate * x
    log_norm   = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnorm - log_norm)


def get_positional_features_gamma(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    stddev: float = None,
    start_mean: float = None,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Gamma-PDF basis functions with linearly spaced means."""
    if stddev is None:
        stddev = seq_len / (2 * features)
    if start_mean is None:
        start_mean = seq_len / features

    mean = torch.linspace(
        start_mean, seq_len, features, device=positions.device
    )
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probs = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probs = probs + eps
    return probs / torch.amax(probs, dim=-1, keepdim=True)


def get_positional_embed(
    seq_len: int,
    feature_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Build [2*seq_len - 1, feature_size] relative positional embeddings.

    Uses three basis function families (exponential, central_mask, gamma),
    each providing both a symmetric f(|d|) and an asymmetric sign(d)*f(|d|)
    variant.  The six groups are concatenated to fill `feature_size`.

    Matches the original Enformer paper (Extended Data Fig. 6) but always
    uses the PyTorch gamma implementation (no TF precomputed gammas).
    """
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]
    num_components = len(feature_functions) * 2   # symmetric + asymmetric

    if feature_size % num_components != 0:
        raise ValueError(
            f"feature_size={feature_size} must be divisible by "
            f"num_components={num_components} (3 basis families × 2)"
        )
    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(
            fn(distances, num_basis_per_class, seq_len, dtype=dtype)
        )
    embeddings = torch.cat(embeddings, dim=-1)          # [2N-1, feat/2]

    # Concatenate symmetric + asymmetric
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings.to(dtype)   # [2N-1, feature_size]


def relative_shift(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, H, N, 2N-1] relative-key logits into [B, H, N, N] with
    correct j-i distance indexing.  (Transformer-XL / Enformer convention.)
    """
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)         # [B, H, N, 2N]
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)                # [B, H, 2N, N]
    x = x[:, :, 1:, :]                          # [B, H, 2N-1, N]
    x = x.reshape(-1, h, t1, t2 - 1)            # [B, H, N, 2N-1]
    return x[..., :((t2 + 1) // 2)]             # [B, H, N, N]


class BigBirdCCREAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        block_size: int = 128,
        dropout: float = 0.0,
        pos_dropout: float = 0.01,
        num_rel_pos_features: int = None,
        max_seq_len: int = 1536,
        **kwargs,
    ):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.dropout_p  = float(dropout)
        self.inner_v    = heads * dim_value
        self.scale      = dim_key ** -0.5

        #
        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        if num_rel_pos_features is None:
            num_rel_pos_features = dim // heads
        self.num_rel_pos_features = num_rel_pos_features

        
        self.to_rel_k = nn.Linear(
            num_rel_pos_features, dim_key * heads, bias=False
        )
        
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, heads, 1, dim_key)
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, heads, 1, dim_key)
        )
 
        self.pos_dropout  = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)
  
        N = max_seq_len
        pos_embed = get_positional_embed(
            N, num_rel_pos_features, torch.device("cpu"), dtype=torch.float
        )
        self.register_buffer("pos_embed", pos_embed) 
        
        BS = block_size
        assert N % BS == 0, f"max_seq_len={N} must be divisible by block_size={BS}"
        nb = N // BS
        local_mask = torch.zeros(N, N, dtype=torch.bool)
        for bi in range(nb):
            q_s = bi * BS
            q_e = q_s + BS
            k_s = max(0, (bi - 1) * BS)
            k_e = min(N, (bi + 2) * BS)
            local_mask[q_s:q_e, k_s:k_e] = True
        self.register_buffer("local_window_mask", local_mask)

    def forward(
        self, x: torch.Tensor, is_global: torch.Tensor = None
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        orig_dtype = x.dtype
 
        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)   
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        q = q * self.scale   
        
        content_logits = torch.einsum(
            "bhid,bhjd->bhij", q + self.rel_content_bias, k
        )   

        positions = self.pos_embed                             
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)                       
        rel_k = rel_k.view(-1, H, dk).permute(1, 0, 2)        

        rel_logits = torch.einsum(
            "bhid,hjd->bhij", q + self.rel_pos_bias, rel_k
        )   
        rel_logits = relative_shift(rel_logits)                

        
        logits = content_logits + rel_logits                   

        
        if is_global is not None and is_global.any():
            attn_bool = (
                self.local_window_mask.view(1, 1, N, N)
                | is_global.view(B, 1, N, 1)    
                | is_global.view(B, 1, 1, N)    
            )
        else:
            
            attn_bool = self.local_window_mask.view(1, 1, N, N)

        logits = logits.float()
        logits.masked_fill_(~attn_bool, float("-inf"))

        attn = logits.softmax(dim=-1)

        if self.training and self.dropout_p > 0:
            attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v.float())
        out = out.to(orig_dtype)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)

class BigBirdAttention(nn.Module):
    """
    BigBird attention with ONE learnable global token prepended at position 0.
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
        N    = T - 1
        nb   = N // BS
        drop = self.dropout_p if self.training else 0.0
        assert N % BS == 0, f"Local length N={N} must be divisible by block_size={BS}"

        q = self.to_q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.to_k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, dv).transpose(1, 2)

        out_g = F.scaled_dot_product_attention(
            q[:, :, :1, :], k, v, dropout_p=drop, is_causal=False
        )

        kg, vg = k[:, :, :1, :], v[:, :, :1, :]
        kx, vx = k[:, :, 1:, :], v[:, :, 1:, :]

        qx_blk = q[:, :, 1:, :].view(B, H, nb, BS, dk)
        kx_blk = kx.view(B, H, nb, BS, dk)
        vx_blk = vx.view(B, H, nb, BS, dv)

        zk     = kx.new_zeros(B, H, 1, BS, dk)
        zv     = vx.new_zeros(B, H, 1, BS, dv)
        kx_pad = torch.cat([zk, kx_blk, zk], dim=2)
        vx_pad = torch.cat([zv, vx_blk, zv], dim=2)

        k_local = torch.cat([kx_pad[:, :, 0:nb], kx_pad[:, :, 1:nb+1], kx_pad[:, :, 2:nb+2]], dim=3)
        v_local = torch.cat([vx_pad[:, :, 0:nb], vx_pad[:, :, 1:nb+1], vx_pad[:, :, 2:nb+2]], dim=3)

        kg_exp = kg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        vg_exp = vg.unsqueeze(2).expand(-1, -1, nb, -1, -1)
        k_full = torch.cat([kg_exp, k_local], dim=3)
        v_full = torch.cat([vg_exp, v_local], dim=3)

        q_flat   = qx_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,         dk)
        k_flat   = k_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS, dk)
        v_flat   = v_full.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 1 + 3 * BS, dv)
        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False)

        out_x = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)
        out   = torch.cat([out_g, out_x], dim=2)
        out   = out.transpose(1, 2).contiguous().view(B, T, self.inner_v)
        return self.to_out(out)


class BlockSparseAttention(nn.Module):
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
        B, N, D    = x.shape
        H, BLK     = self.heads, self.block_size
        assert N % BLK == 0, f"Sequence length {N} must be divisible by block_size {BLK}"
        num_blocks = N // BLK
        drop       = self.dropout_p if self.training else 0.0

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q_blk_view = q.view(B, num_blocks, BLK, H, self.dim_key)
        k_blk_view = k.view(B, num_blocks, BLK, H, self.dim_key)
        v_blk_view = v.view(B, num_blocks, BLK, H, self.dim_value)

        q_blk  = q_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        k_blk  = k_blk_view.reshape(B * num_blocks, BLK, H, self.dim_key).transpose(1, 2)
        v_blk  = v_blk_view.reshape(B * num_blocks, BLK, H, self.dim_value).transpose(1, 2)
        out_blk = F.scaled_dot_product_attention(q_blk, k_blk, v_blk, dropout_p=drop, is_causal=False)
        out_blk = out_blk.view(B, num_blocks, H, BLK, self.dim_value)

        q_g   = q_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        k_g   = k_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        v_g   = v_blk_view[:, :, 0, :, :].permute(0, 2, 1, 3)
        out_g = F.scaled_dot_product_attention(q_g, k_g, v_g, dropout_p=drop, is_causal=False)

        out_g_reshaped = out_g.permute(0, 2, 1, 3).unsqueeze(3)
        global_part    = out_blk[:, :, :, 0:1, :] + out_g_reshaped
        local_part     = out_blk[:, :, :, 1:,  :]

        out_combined = torch.cat([global_part, local_part], dim=3)
        out_combined = out_combined.permute(0, 1, 3, 2, 4).reshape(B, N, self.inner_v)
        return self.to_out(out_combined)


class FullAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        dropout: float = 0.0,
        pos_dropout: float = 0.01,
        num_rel_pos_features: int = None,
        max_seq_len: int = 1536,
        **kwargs,
    ):
        super().__init__()
        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.dropout_p  = float(dropout)
        self.inner_v    = heads * dim_value
        self.scale      = dim_key ** -0.5

        # Projections
        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        # Zero-init output projection → transformer blocks start as identity
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # --- Relative positional encoding ---
        if num_rel_pos_features is None:
            num_rel_pos_features = dim // heads
        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(
            num_rel_pos_features, dim_key * heads, bias=False
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, heads, 1, dim_key)
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, heads, 1, dim_key)
        )

        # Dropouts
        self.pos_dropout  = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Precompute positional basis functions
        pos_embed = get_positional_embed(
            max_seq_len, num_rel_pos_features,
            torch.device("cpu"), dtype=torch.float,
        )
        self.register_buffer("pos_embed", pos_embed)   # [2N-1, features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        orig_dtype = x.dtype

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)   # [B, H, N, dk]
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        q = q * self.scale

        content_logits = torch.einsum(
            "bhid,bhjd->bhij", q + self.rel_content_bias, k
        )  


        positions = self.pos_embed                           
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)                       
        rel_k = rel_k.view(-1, H, dk).permute(1, 0, 2)       

        rel_logits = torch.einsum(
            "bhid,hjd->bhij", q + self.rel_pos_bias, rel_k
        )  
        rel_logits = relative_shift(rel_logits)              

        logits = content_logits + rel_logits                 

        attn = logits.softmax(dim=-1)

        if self.training and self.dropout_p > 0:
            attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class BigBirdAttentionAblation(nn.Module):
    """
    BigBird local-window attention — no global tokens.
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
        B, N, _        = x.shape
        H, dk, dv, BS  = self.heads, self.dim_key, self.dim_value, self.block_size
        nb             = N // BS
        drop           = self.dropout_p if self.training else 0.0
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

        q_flat   = q_blk.permute(0, 2, 1, 3, 4).reshape(B * nb, H, BS,     dk)
        k_flat   = k_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dk)
        v_flat   = v_win.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 3 * BS, dv)
        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=drop, is_causal=False)

        out = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)