"""
src/models/enformer_plus.py

Adapted from enformer-pytorch by lucidrains.
https://github.com/lucidrains/enformer-pytorch

Changes vs original:
  - Added BigBirdAttention, BlockSparseAttention, BigBirdAttentionAblation modes
  - Added "ccre_bigbird" mode: biologically-grounded in-place cCRE global tokens
    from the ENCODE4 SCREEN database. No injector/remover — T=1536 throughout.
  - TransformerBlock class for ccre_bigbird: supports is_global kwarg passthrough.
  - _run_trunk_ccre() method for ccre_bigbird forward path.
  - Enformer.forward() accepts is_global=[B,1536] bool tensor for ccre_bigbird.
"""

import sys
import os
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

from einops import rearrange
from einops.layers.torch import Rearrange

current_dir  = os.path.dirname(os.path.abspath(__file__))
src_folder   = os.path.dirname(current_dir)
project_root = os.path.dirname(src_folder)
sys.path.insert(0, project_root)

from src.utils.data import str_to_one_hot, seq_indices_to_one_hot
from src.utils.config import EnformerConfig
from src.layers.attention import (
    BlockSparseAttention,
    FullAttention,
    BigBirdAttention,
    BigBirdAttentionAblation,
    BigBirdCCREAttention,
)
from transformers import PreTrainedModel

# constants
SEQUENCE_LENGTH = 196_608
TARGET_LENGTH   = 896

# helpers
def exists(val):        return val is not None
def default(val, d):    return val if exists(val) else d
def map_values(fn, d):  return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x): return int(round(x / divisible_by) * divisible_by)
    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps=1e-20):  return torch.log(t.clamp(min=eps))

def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(
        is_distributed,
        dist.is_initialized() and dist.get_world_size() > 1
    )
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


# global-token helpers (used by bigbird / block_sparse modes)

class SingleGlobalTokenInjector(nn.Module):
    """Prepends one learnable global token. Used by 'bigbird' mode."""
    def __init__(self, dim: int, init_std: float = 0.02):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.g, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.g.expand(x.size(0), -1, -1), x], dim=1)


class SingleGlobalTokenRemover(nn.Module):
    """Strips the global token at pos 0. Used by 'bigbird' mode."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:, :]


class ChunkGlobalTokenInjector(nn.Module):
    """
    Inserts C learnable globals interleaved with C equal chunks.
    Used by 'block_sparse' mode.
    """
    def __init__(self, dim: int, num_chunks: int, init_std: float = 0.02):
        super().__init__()
        self.num_chunks = int(num_chunks)
        self.g = nn.Parameter(torch.zeros(1, self.num_chunks, 1, dim))
        nn.init.normal_(self.g, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C       = self.num_chunks
        assert N % C == 0
        n_chunk = N // C
        x_chunks = x.view(B, C, n_chunk, D)
        g = self.g.expand(B, -1, -1, -1)
        interleaved = torch.cat([g, x_chunks], dim=2)
        return interleaved.reshape(B, C * (1 + n_chunk), D)


class ChunkGlobalTokenRemover(nn.Module):
    """Removes interleaved globals. Used by 'block_sparse' mode."""
    def __init__(self, num_chunks: int):
        super().__init__()
        self.num_chunks = int(num_chunks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        C       = self.num_chunks
        assert T % C == 0
        stride  = T // C
        x = x.view(B, C, stride, D)
        return x[:, :, 1:, :].reshape(B, C * (stride - 1), D)


# TransformerBlock for ccre_bigbird mode

class TransformerBlock(nn.Module):
    """
    Single transformer block (pre-norm, attention + FFN with residuals).
    Accepts an optional is_global kwarg that is forwarded to the attention layer.
    Used exclusively by "ccre_bigbird" mode so that is_global can flow through
    gradient checkpointing without modifying nn.Sequential / Residual wrappers.
    """
    def __init__(self, dim: int, attn_layer: nn.Module, dropout_rate: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = attn_layer
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        h = self.attn(h, is_global=is_global)
        x = x + self.drop1(h)
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        return x


# standard building blocks (unchanged)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200_000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn   = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.dirac_(self.to_attn_logits.weight)
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        if remainder > 0:
            x = F.pad(x, (0, remainder), value=0)
        x      = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        attn   = logits.softmax(dim=-1)
        return (x * attn).sum(dim=-1)


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f"sequence length {seq_len} < target length {target_len}")
        start = (seq_len - target_len) // 2
        return x[:, start : start + target_len, :]


def ConvBlock(dim, dim_out=None, kernel_size=1, is_distributed=None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed=is_distributed)
    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


# main Enformer class

class Enformer(PreTrainedModel):
    config_class      = EnformerConfig
    base_model_prefix = "enformer"

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        self.dim              = config.dim
        half_dim              = config.dim // 2
        twice_dim             = config.dim * 2
        self.attention_mode   = config.attention_mode
        self.use_checkpointing = config.use_checkpointing
        self.uses_ccre_globals = (self.attention_mode == "ccre_bigbird")

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=2),
        )

        filter_list = exponential_linspace_int(
            half_dim, config.dim,
            num=config.num_downsamples - 1,
            divisible_by=config.dim_divisible_by,
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=2),
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        # Positional embedding
        self.pos_embedding = SinusoidalPositionalEmbedding(config.dim)

        # seq_len after conv tower (before any global-token injection)
        seq_len_trans = SEQUENCE_LENGTH // (2 ** config.num_downsamples)  # 1536

        # Injector / Remover & Transformer
        if self.attention_mode in ("full", "bigbird_ablation"):
            self.injector = nn.Identity()
            self.remover  = nn.Identity()
            self.transformer = self._build_sequential_transformer(config)

        elif self.attention_mode == "bigbird":
            if seq_len_trans % config.block_size != 0:
                raise ValueError(
                    f"BigBird: seq_len_trans={seq_len_trans} must be divisible by "
                    f"block_size={config.block_size}."
                )
            self.injector = SingleGlobalTokenInjector(config.dim)
            self.remover  = SingleGlobalTokenRemover()
            self.transformer = self._build_sequential_transformer(config)

        elif self.attention_mode == "ccre_bigbird":
            self.injector = nn.Identity()
            self.remover  = nn.Identity()
            self.transformer = nn.ModuleList([
                TransformerBlock(
                    dim=config.dim,
                    attn_layer=BigBirdCCREAttention(
                        dim=config.dim,
                        heads=config.heads,
                        block_size=config.block_size,
                        dim_key=config.attn_dim_key,
                        dim_value=config.attn_dim_value,
                        dropout=config.attn_dropout,
                        max_seq_len=seq_len_trans,
                    ),
                    dropout_rate=config.dropout_rate,
                )
                for _ in range(config.depth)
            ])

        else:  # block_sparse
            if seq_len_trans % config.block_size != 0:
                raise ValueError(
                    f"BlockSparse: seq_len_trans={seq_len_trans} must be divisible by "
                    f"block_size={config.block_size}."
                )
            num_chunks = seq_len_trans // config.block_size
            self.injector = ChunkGlobalTokenInjector(config.dim, num_chunks=num_chunks)
            self.remover  = ChunkGlobalTokenRemover(num_chunks=num_chunks)
            self.transformer = self._build_sequential_transformer(config)

        self.target_length  = config.target_length
        self.crop_final     = TargetLengthCrop(config.target_length)
        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(config.dropout_rate / 8),
            GELU(),
        )

        if not self.uses_ccre_globals:
            self._trunk = nn.Sequential(
                Rearrange("b n d -> b d n"),
                self.stem,
                self.conv_tower,
                Rearrange("b d n -> b n d"),
                self.injector,
                self.pos_embedding,
                self.transformer,
                self.remover,
                self.crop_final,
                self.final_pointwise,
            )

        self.add_heads(**config.output_heads)

    # helper: build nn.Sequential transformer (non-ccre modes)

    def _build_sequential_transformer(self, config):
        blocks = []
        for _ in range(config.depth):
            if self.attention_mode == "full":
                attn_layer = FullAttention(
                    dim=config.dim, heads=config.heads,
                    dim_key=config.attn_dim_key, dim_value=config.attn_dim_value,
                    dropout=config.attn_dropout,
                )
            elif self.attention_mode == "bigbird_ablation":
                attn_layer = BigBirdAttentionAblation(
                    dim=config.dim, heads=config.heads,
                    block_size=config.block_size,
                    dim_key=config.attn_dim_key, dim_value=config.attn_dim_value,
                    dropout=config.attn_dropout,
                )
            elif self.attention_mode == "bigbird":
                attn_layer = BigBirdAttention(
                    dim=config.dim, heads=config.heads,
                    block_size=config.block_size,
                    dim_key=config.attn_dim_key, dim_value=config.attn_dim_value,
                    dropout=config.attn_dropout,
                )
            else:  # block_sparse
                attn_layer = BlockSparseAttention(
                    dim=config.dim, heads=config.heads,
                    block_size=config.block_size + 1,
                    dim_key=config.attn_dim_key, dim_value=config.attn_dim_value,
                    dropout=config.attn_dropout,
                )

            blocks.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    attn_layer,
                    nn.Dropout(config.dropout_rate),
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate),
                )),
            ))
        return nn.Sequential(*blocks)

    # output heads

    def add_heads(self, **kwargs):
        self.output_heads = kwargs
        self._heads = nn.ModuleDict(map_values(
            lambda features: nn.Sequential(nn.Linear(self.dim * 2, features), nn.Softplus()),
            kwargs,
        ))

    def set_target_length(self, target_length):
        self.crop_final.target_length = target_length

    def trunk_checkpointed(self, x):
        """Checkpointed trunk for non-ccre modes."""
        x = rearrange(x, "b n d -> b d n")
        x = self.stem(x)
        x = checkpoint_sequential(self.conv_tower, len(self.conv_tower), x, use_reentrant=False)
        x = rearrange(x, "b d n -> b n d")
        x = self.injector(x)
        x = self.pos_embedding(x)
        x = checkpoint_sequential(self.transformer, len(self.transformer), x, use_reentrant=False)
        x = self.remover(x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def _run_trunk_ccre(self, x: torch.Tensor, is_global: torch.Tensor = None) -> torch.Tensor:
        """
        Forward trunk for ccre_bigbird mode.
        is_global: [B, 1536] bool — passed through to every TransformerBlock.
        """
        x = rearrange(x, "b n d -> b d n")
        x = self.stem(x)
        if self.use_checkpointing:
            x = checkpoint_sequential(self.conv_tower, len(self.conv_tower), x, use_reentrant=False)
        else:
            x = self.conv_tower(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.injector(x)           # nn.Identity for ccre_bigbird
        x = self.pos_embedding(x)

        for block in self.transformer:
            if self.use_checkpointing:
                if is_global is not None:
                    # checkpoint passes is_global as positional arg → forward(x, is_global)
                    x = checkpoint(block, x, is_global, use_reentrant=False)
                else:
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, is_global=is_global)

        x = self.remover(x)            # nn.Identity for ccre_bigbird
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x


    def forward(
        self,
        x,
        is_global=None,
        target=None,
        return_corr_coef=False,
        return_embeddings=False,
        return_only_embeddings=False,
        head=None,
        target_length=None,
    ):
        """
        Args:
            x:         [B, L, 4] one-hot DNA sequence  (L = 196608)
            is_global: [B, 1536] bool cCRE mask.
                       Only used when attention_mode == "ccre_bigbird".
                       Ignored for all other modes.
        """
        if isinstance(x, list):
            x = str_to_one_hot(x)
        elif isinstance(x, torch.Tensor) and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        x = x.to(self.device)

        no_batch = x.ndim == 2
        if no_batch:
            x = rearrange(x, "... -> () ...")

        if exists(target_length):
            self.set_target_length(target_length)

        # Route to the appropriate trunk
        if self.uses_ccre_globals:
            # Pass is_global (may be None for mouse/zero-mask samples)
            _is_global = is_global.to(self.device) if is_global is not None else None
            x = self._run_trunk_ccre(x, is_global=_is_global)
        elif self.use_checkpointing:
            x = self.trunk_checkpointed(x)
        else:
            x = self._trunk(x)

        if no_batch:
            x = rearrange(x, "() ... -> ...")
        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f"head {head} not found"
            out = out[head]

        if exists(target):
            assert exists(head), "head must be passed in if target is given"
            if return_corr_coef:
                return pearson_corr_coef(out, target)
            return poisson_loss(out, target)

        if return_embeddings:
            return out, x
        return out