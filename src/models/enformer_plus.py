"""
    Adapted from enformer-pytorch from lucidrains
    https://github.com/lucidrains/enformer-pytorch

    Modifed as following:
    
    Positional Embedding
        Original Enformer uses relative positional embedding. 
        Since that does not work with BigBird Sparse attention, simple sine PE is used.

    Attention
        Original Enformer uses classic O(N^2) attention.
        That is replaced with BigBird Sparse attention, which is O(N)    
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_folder = os.path.dirname(current_dir)      
project_root = os.path.dirname(src_folder)    
sys.path.insert(0, project_root)

import math
from pathlib import Path

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from src.utils.data import str_to_one_hot, seq_indices_to_one_hot
from src.utils.config import EnformerConfig
from src.layers.attention import BigBirdAttention, FullAttention, HierarchicalAttention

from transformers import PreTrainedModel

# constants

SEQUENCE_LENGTH = 196_608 # maybe save this to config and call it here?
TARGET_LENGTH = 896

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# maybe sync batchnorm, for distributed training

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

#------------- NEW POSITIONAL EMBEDDING -------------
# Original enformers code uses relative positional encoding functions
# However, since that does not fit with BigBird sparse attention,
# absolute Sinusodial PE is used like what original BigBird paper used.

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200000):
        super().__init__()
        self.d_model = d_model
        
        # Create constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]

# classes

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
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        start = (seq_len - target_len) // 2
        end = start + target_len
        return x[:, start:end, :]


def ConvBlock(dim, dim_out = None, kernel_size = 1, is_distributed = None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed = is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

#----------------- ADDED CLASS -----------------
# Deleted original attention class from enformers and created wrapper class
# to use BigBirdAttention as a new attention

class ChunkGlobalTokenInjector(nn.Module):
    """
    Inserts C learnable globals interleaved with C equal chunks:

      x (B,N,D) -> [g0, x0, g1, x1, ..., g(C-1), x(C-1)]

    where N must be divisible by C, and xi has length N/C.
    """
    def __init__(self, dim: int, num_chunks: int, init_std: float = 0.02):
        super().__init__()
        assert num_chunks >= 1
        self.num_chunks = int(num_chunks)

        self.g = nn.Parameter(torch.zeros(1, self.num_chunks, 1, dim))
        nn.init.normal_(self.g, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = self.num_chunks
        assert N % C == 0, f"Local sequence length N={N} must be divisible by num_chunks C={C}"
        n_chunk = N // C

        x_chunks = x.view(B, C, n_chunk, D)
        g = self.g.expand(B, -1, -1, -1)
        interleaved = torch.cat([g, x_chunks], dim=2)

        return interleaved.reshape(B, C * (1 + n_chunk), D)


class ChunkGlobalTokenRemover(nn.Module):
    """
    Removes interleaved globals:

      [g0, x0, g1, x1, ..., g(C-1), x(C-1)] -> [x0, x1, ..., x(C-1)]
    """
    def __init__(self, num_chunks: int):
        super().__init__()
        assert num_chunks >= 1
        self.num_chunks = int(num_chunks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        C = self.num_chunks

        assert T % C == 0, f"T={T} must be divisible by num_chunks C={C}"
        stride = T // C                 # = 1 + n_chunk
        assert stride >= 2, f"Need at least 1 global + 1 local per chunk. Got stride={stride}"
        x = x.view(B, C, stride, D)
        x_locals = x[:, :, 1:, :]       # [B, C, n_chunk, D]

        return x_locals.reshape(B, C * (stride - 1), D)

class GlobalTokenInjector(nn.Module):
    """
    Inserts 2 learnable globals into the middle:
      x -> [g1] + x[:mid] + [g2] + x[mid:]
    where mid = N//2
    """
    def __init__(self, dim: int):
        super().__init__()
        self.g1 = nn.Parameter(torch.zeros(1, 1, dim))
        self.g2 = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.g1, std=0.02)
        nn.init.normal_(self.g2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert N % 2 == 0, f"Local sequence length must be even for 2 chunks. Got N={N}"
        mid = N // 2
        g1 = self.g1.expand(B, -1, -1)
        g2 = self.g2.expand(B, -1, -1)
        return torch.cat([g1, x[:, :mid, :], g2, x[:, mid:, :]], dim=1)


class GlobalTokenRemover(nn.Module):
    """
    Removes the 2 globals from:
      [g1, chunk1, g2, chunk2] -> [chunk1, chunk2]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = T - 2
        assert N % 2 == 0, f"(T-2) must be even. Got T={T}"
        mid = N // 2
        chunk1 = x[:, 1:(1 + mid), :]
        chunk2 = x[:, (2 + mid):, :]
        return torch.cat([chunk1, chunk2], dim=1)


class BigBirdAutoWrapper(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        block_size=128,
        dim_key=64,
        dim_value=64,
        num_chunks=2,
        num_global_tokens=2,
        attn_dropout=0.0,
        local_window_blocks=3,
        layer_idx=0,
        attention_mode="sparse_3",
    ):
        super().__init__()

        if attention_mode == "hierarchical":
            self.layer = HierarchicalAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_chunks=num_chunks,
                block_size=block_size,
                dim_key=dim_key,
                dim_value=dim_value,
                attn_dropout=attn_dropout,
                local_window_blocks=local_window_blocks,
                layer_idx=layer_idx,
            )
        else:
            self.layer = BigBirdAttention(
                d_model=d_model,
                num_heads=num_heads,
                block_size=block_size,
                dim_key=dim_key,
                dim_value=dim_value,
                num_global_tokens=num_global_tokens,
                attn_dropout=attn_dropout,
                local_window_blocks=local_window_blocks,
                layer_idx=layer_idx,
            )

    def forward(self, x):
        return self.layer(x)

# main class with BigBirdAttention

class Enformer(PreTrainedModel):
    """
        Main Enformer Class adpated from original code, 
        but replaced attention with BigBirdAttention
    """
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # create stem

        self.attention_mode = getattr(config, "attention_mode", "sparse_3")
        assert self.attention_mode in ("full", "sparse_2", "sparse_3", "hierarchical"), \
            f"Invalid attention_mode={self.attention_mode}"

        self.use_full_attention = (self.attention_mode == "full")

        if self.attention_mode == "full":
            self.local_window_blocks = None
        elif self.attention_mode == "sparse_2":
            self.local_window_blocks = 2
        elif self.attention_mode == "sparse_3":
            self.local_window_blocks = 3
        elif self.attention_mode == "hierarchical":
            # self.local_window_blocks = getattr(config, "local_window_blocks", 2)  # or 3
            self.local_window_blocks = 3

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # whether to use tensorflow gamma positions

        use_tf_gamma = config.use_tf_gamma
        self.use_tf_gamma = use_tf_gamma

        self.block_size = config.block_size
        self.dna_chunk_len = config.dna_chunk_len
        
        # Calculate how many blocks of DNA are in a chunk (128 / 64 = 2 blocks)
        dna_blocks_per_chunk = self.dna_chunk_len // self.block_size

        # transformer

        transformer = []
        for layer_idx in range(config.depth):
            if self.use_full_attention:
                attn_layer = FullAttention(
                    dim=config.dim,
                    heads=config.heads,
                )
            else:
                attn_layer = BigBirdAutoWrapper(
                    d_model=config.dim,
                    num_heads=config.heads,
                    block_size=self.block_size,
                    dim_key=config.attn_dim_key,
                    dim_value=config.attn_dim_key,
                    num_chunks=getattr(config, "num_chunks", 2),
                    num_global_tokens=config.num_global_tokens,
                    attn_dropout=config.attn_dropout,
                    local_window_blocks=self.local_window_blocks,
                    layer_idx=layer_idx,
                    attention_mode=self.attention_mode, 
                )


            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    attn_layer,
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        self.pos_embedding = SinusoidalPositionalEmbedding(config.dim)

        G = int(getattr(config, "num_global_tokens", 2))

        if self.use_full_attention:
            self.injector = nn.Identity()
            self.remover = nn.Identity()
        elif self.attention_mode == "hierarchical":
            C = int(getattr(config, "num_chunks", 2))
            self.injector = ChunkGlobalTokenInjector(config.dim, num_chunks=C)
            self.remover  = ChunkGlobalTokenRemover(num_chunks=C)
        else:
            assert G == 2, f"For front+end globals, num_global_tokens must be 2, got {G}"
            self.injector = GlobalTokenInjector(config.dim)   
            self.remover  = GlobalTokenRemover()             

        # create trunk sequential module

        self._trunk = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            # Inject CLS tokens before pos_embedding and transformer
            self.injector,
            self.pos_embedding,
            self.transformer,
            # Remove CLS tokens before final layers
            self.remover,
            self.crop_final,
            self.final_pointwise
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = checkpoint_sequential(self.conv_tower, len(self.conv_tower), x, use_reentrant=False)
        x = rearrange(x, 'b d n -> b n d')
        x = self.injector(x)
        x = self.pos_embedding(x)
        x = checkpoint_sequential(self.transformer, len(self.transformer), x, use_reentrant=False)
        x = self.remover(x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None
    ):
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif type(x) == torch.Tensor and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        x = x.to(self.device)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        if exists(target_length):
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, '() ... -> ...')

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f'head {head} not found'
            out = out[head]

        if exists(target):
            assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

            if return_corr_coef:
                return pearson_corr_coef(out, target)

            return poisson_loss(out, target)

        if return_embeddings:
            return out, x

        return out
