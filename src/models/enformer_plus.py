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
from src.layers.attention import BigBirdAttention, FullAttention

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

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

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

class PeriodicTokenInjector(nn.Module):
    def __init__(self, dim, block_size, tokens_per_chunk):
        """
        dim: Model dimension (256)
        block_size: Size of one BigBird block (64)
        tokens_per_chunk: How often to insert
        """
        super().__init__()
        self.block_size = block_size
        self.tokens_per_chunk = tokens_per_chunk
        
        # insert a full block of CLS tokens so the shapes align with BigBird
        # Shape: [1, 1, block_size, dim]
        self.cls_block = nn.Parameter(torch.randn(1, 1, block_size, dim))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        b, seq_len, d = x.shape
        
        # Pad sequence so it is perfectly divisible by tokens_per_chunk
        remainder = seq_len % self.tokens_per_chunk
        if remainder > 0:
            pad_needed = self.tokens_per_chunk - remainder
            x = F.pad(x, (0, 0, 0, pad_needed))
            seq_len = x.shape[1]
            
        # Reshape into chunks
        # [Batch, Num_Chunks, Chunk_Len, Dim]
        num_chunks = seq_len // self.tokens_per_chunk
        x_reshaped = x.view(b, num_chunks, self.tokens_per_chunk, d)
        
        # Expand CLS block to match number of chunks
        # [Batch, Num_Chunks, Block_Size, Dim]
        cls_expanded = self.cls_block.expand(b, num_chunks, -1, -1)
        
        # Concatenate: Put CLS at the START of every chunk
        x_with_cls = torch.cat([cls_expanded, x_reshaped], dim=2)
        
        return x_with_cls.view(b, -1, d)

class PeriodicTokenRemover(nn.Module):
    """
        Remove global cls tokens before the prediction
    """
    def __init__(self, block_size, tokens_per_chunk):
        super().__init__()
        self.block_size = block_size
        self.tokens_per_chunk = tokens_per_chunk

    def forward(self, x):
        # x: [Batch, Mixed_Seq_Len, Dim]
        b, seq_len, d = x.shape
        
        # Calculate the size of a Mixed Chunk (CLS + DNA)
        mixed_chunk_len = self.block_size + self.tokens_per_chunk
        
        # Reshape to separate CLS from DNA
        num_chunks = seq_len // mixed_chunk_len
        x = x.view(b, num_chunks, mixed_chunk_len, d)
        
        # Slice off the CLS block and keep only the DNA part
        x_dna_only = x[:, :, self.block_size:, :]
        
        return x_dna_only.contiguous().view(b, -1, d)
    
class BigBirdAutoWrapper(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64, chunk_size_in_blocks=12):
        """
        Wrapper class that calculates indicies of cls tokens and return BigBirdAttention with those indicies
        chunk_size_in_blocks: Number of DNA in one chunk?
                              If injector used 768 tokens, that is 768/64 = 12 blocks.
        """
        super().__init__()
        self.layer = BigBirdAttention(d_model, num_heads, block_size)
        self.block_size = block_size
        self.chunk_size_in_blocks = chunk_size_in_blocks

    def forward(self, x):
        # x already has CLS tokens inserted.
        # The structure is: [CLS_Block, DNA_Block_1...DNA_Block_12, CLS_Block, ...]
        
        # The period is: 1 global block + N dna blocks
        period = 1 + self.chunk_size_in_blocks
        
        seq_len = x.shape[1]
        num_blocks = seq_len // self.block_size
        
        # Calculate indices: 0, 13, 26, 39...
        global_indices = [i for i in range(0, num_blocks, period)]
        
        return self.layer(x, global_indices=global_indices)
    
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

        self.use_full_attention = getattr(config, 'full_attention', False)

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
        for _ in range(config.depth):

            if self.use_full_attention:
                # Use standard O(N^2) Full Attention
                attn_layer = FullAttention(
                    d_model=config.dim,
                    num_heads=config.heads,
                    # Block size is ignored by FullAttention but passing it doesn't hurt
                    block_size=self.block_size 
                )
            else:
                # Use BigBird Sparse Attention
                attn_layer = BigBirdAutoWrapper(
                    d_model=config.dim,
                    num_heads=config.heads,
                    block_size=self.block_size,
                    chunk_size_in_blocks=dna_blocks_per_chunk 
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

        if self.use_full_attention:
            self.injector = nn.Identity()
            self.remover = nn.Identity()
        else:
            self.injector = PeriodicTokenInjector(
                    dim=config.dim, 
                    block_size=self.block_size, 
                    tokens_per_chunk=self.dna_chunk_len
                )
            
            self.remover = PeriodicTokenRemover(
                    block_size=self.block_size,
                    tokens_per_chunk=self.dna_chunk_len
                )
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
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.injector(x)
        x = self.pos_embedding(x)
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
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
        x.to(self.device)

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
