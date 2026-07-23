import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

#  Relative positional encoding helpers

def get_positional_features_exponential(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    min_half_life: float = 3.0,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions).to(dtype)


def get_positional_features_central_mask(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
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
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]
    num_components = len(feature_functions) * 2
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
    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings.to(dtype)


def relative_shift(x: torch.Tensor) -> torch.Tensor:
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]

#  RELATIVE PE attention classes 

class RelBigBirdCCREAttention(nn.Module):
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
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias     = nn.Parameter(torch.randn(1, heads, 1, dim_key))

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

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
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


class RelFullAttention(nn.Module):
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
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias     = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        self.pos_dropout  = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        pos_embed = get_positional_embed(
            max_seq_len, num_rel_pos_features,
            torch.device("cpu"), dtype=torch.float,
        )
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value

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

        attn = logits.softmax(dim=-1)
        if self.training and self.dropout_p > 0:
            attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)

#  ABSOLUTE PE attention classes 

def _ccre_attention_mask(
    is_global: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the union of local-window and global attention positions."""
    block_ids = torch.arange(seq_len, device=device) // block_size
    local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1

    if is_global is None:
        return local.view(1, 1, seq_len, seq_len)

    if is_global.ndim != 2 or is_global.shape != (batch_size, seq_len):
        raise ValueError(
            "is_global must have shape [batch, sequence], "
            f"got {tuple(is_global.shape)}; expected {(batch_size, seq_len)}"
        )

    is_global = is_global.to(device=device, dtype=torch.bool)
    allowed = (
        local.unsqueeze(0)
        | is_global[:, :, None]
        | is_global[:, None, :]
    )
    return allowed.unsqueeze(1)


def _flex_ccre_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_global: torch.Tensor,
    block_size: int,
    attention_fn,
) -> torch.Tensor:
    """Run cCRE attention after packing global positions into contiguous blocks."""
    batch_size, heads, seq_len, _ = q.shape
    value_dim = v.shape[-1]

    permutation = torch.argsort(~is_global, dim=1, stable=True)
    sorted_is_global = is_global.gather(1, permutation)

    def gather_sequence(tensor):
        indices = permutation[:, None, :, None].expand(
            batch_size, tensor.shape[1], seq_len, tensor.shape[-1]
        )
        return tensor.gather(2, indices)

    q_sorted = gather_sequence(q)
    k_sorted = gather_sequence(k)
    v_sorted = gather_sequence(v)

    def mask_mod(batch, head, query_index, key_index):
        query_position = permutation[batch, query_index]
        key_position = permutation[batch, key_index]
        local = (
            query_position // block_size - key_position // block_size
        ).abs() <= 1
        return (
            local
            | sorted_is_global[batch, query_index]
            | sorted_is_global[batch, key_index]
        )

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=q.device,
        BLOCK_SIZE=block_size,
    )
    out_sorted = attention_fn(
        q_sorted,
        k_sorted,
        v_sorted,
        block_mask=block_mask,
    )

    inverse_permutation = permutation.argsort(dim=1)
    output_indices = inverse_permutation[:, None, :, None].expand(
        batch_size, heads, seq_len, value_dim
    )
    return out_sorted.gather(2, output_indices)


def _local_window_padding_mask(
    batch_size: int,
    num_blocks: int,
    block_size: int,
    device: torch.device,
    *,
    include_global: bool = False,
) -> torch.Tensor:
    """Mask dummy blocks used to batch the first and last local windows."""
    valid = torch.ones(
        num_blocks,
        3 * block_size,
        dtype=torch.bool,
        device=device,
    )
    valid[0, :block_size] = False
    valid[-1, -block_size:] = False

    if include_global:
        valid = torch.cat(
            [
                torch.ones(
                    num_blocks,
                    1,
                    dtype=torch.bool,
                    device=device,
                ),
                valid,
            ],
            dim=-1,
        )

    valid = valid.unsqueeze(0).expand(batch_size, -1, -1)
    return valid.reshape(batch_size * num_blocks, 1, 1, valid.shape[-1])


class FullAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.heads     = heads
        self.dim_key   = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)
        self.inner_v   = heads * dim_value

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        drop = self.dropout_p if self.training else 0.0

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class FullAttentionEinsum(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.heads     = heads
        self.dim_key   = dim_key
        self.dim_value = dim_value
        self.dropout_p = float(dropout)
        self.inner_v   = heads * dim_value
        self.scale     = dim_key ** -0.5

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)   # [B, H, N, dk]
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        # Explicit N×N score matrix — O(N²) memory
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale   # [B, H, N, N]
        attn = scores.softmax(dim=-1)                                   # [B, H, N, N]

        if self.training and self.dropout_p > 0:
            attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)                 # [B, H, N, dv]
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class BigBirdCCREAttentionEinsum(nn.Module):
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
        self.inner_v    = heads * dim_value
        self.scale      = dim_key ** -0.5

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        BS = self.block_size
        assert N % BS == 0, f"N={N} must be divisible by block_size={BS}"

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        attn_mask = _ccre_attention_mask(is_global, B, N, BS, x.device)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        scores.masked_fill_(~attn_mask, float("-inf"))
        attn = scores.softmax(dim=-1)

        if self.training and self.dropout_p > 0:
            attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)


class BigBirdCCREAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        block_size: int = 128,
        dropout: float = 0.0,
        backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        if backend not in {"auto", "flex", "sdpa"}:
            raise ValueError(
                "BigBirdCCREAttention backend must be 'auto', 'flex', or "
                f"'sdpa', got {backend!r}"
            )
        if backend in {"auto", "flex"} and dropout:
            raise ValueError(
                "FlexAttention does not support post-softmax attention dropout; "
                "set attn_dropout=0 or attention_backend='sdpa'"
            )

        self.heads      = heads
        self.dim_key    = dim_key
        self.dim_value  = dim_value
        self.block_size = block_size
        self.backend    = backend
        self.dropout_p  = float(dropout)
        self.inner_v    = heads * dim_value

        self.to_q   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_k   = nn.Linear(dim, heads * dim_key,   bias=False)
        self.to_v   = nn.Linear(dim, heads * dim_value, bias=False)
        self.to_out = nn.Linear(self.inner_v, dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, dk, dv = self.heads, self.dim_key, self.dim_value
        BS = self.block_size
        assert N % BS == 0, f"N={N} must be divisible by block_size={BS}"

        q = self.to_q(x).view(B, N, H, dk).transpose(1, 2)   # [B, H, N, dk]
        k = self.to_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, dv).transpose(1, 2)

        use_flex = self.backend == "flex" or (
            self.backend == "auto" and x.device.type == "cuda"
        )
        if self.backend == "flex" and x.device.type != "cuda":
            raise RuntimeError(
                "attention_backend='flex' requires CUDA; use 'auto' for an "
                "SDPA fallback on CPU/MPS"
            )

        if not use_flex:
            attn_mask = _ccre_attention_mask(is_global, B, N, BS, x.device)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
            return self.to_out(out)

        if is_global is None:
            is_global = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        elif is_global.ndim != 2 or is_global.shape != (B, N):
            raise ValueError(
                "is_global must have shape [batch, sequence], "
                f"got {tuple(is_global.shape)}; expected {(B, N)}"
            )
        else:
            is_global = is_global.to(device=x.device, dtype=torch.bool)

        # Packing globals makes the union block-sparse while retaining every
        # original query and key exactly once.
        out = _flex_ccre_attention(
            q,
            k,
            v,
            is_global,
            BS,
            _compiled_flex_attention,
        )
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
        attn_mask = _local_window_padding_mask(
            B,
            nb,
            BS,
            x.device,
            include_global=True,
        )
        out_flat = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=attn_mask,
            dropout_p=drop,
            is_causal=False,
        )

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


class BigBirdAttentionAblation(nn.Module):
    """BigBird local-window attention — no global tokens."""
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
        attn_mask = _local_window_padding_mask(B, nb, BS, x.device)
        out_flat = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=attn_mask,
            dropout_p=drop,
            is_causal=False,
        )

        out = out_flat.view(B, nb, H, BS, dv).permute(0, 2, 1, 3, 4).reshape(B, H, N, dv)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_v)
        return self.to_out(out)
