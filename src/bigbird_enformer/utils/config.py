from pathlib import Path

import yaml
from transformers import PretrainedConfig


def load_experiment_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")
    if not isinstance(config.get("model"), dict):
        raise ValueError(f"experiment config must contain a model mapping: {path}")
    if "training" in config and not isinstance(config["training"], dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return config


class EnformerConfig(PretrainedConfig):
    model_type = "enformer"

    @classmethod
    def from_yaml_file(cls, path):
        experiment_config = load_experiment_config(path)
        return cls(**experiment_config["model"])

    def __init__(
        self,
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=896,
        attn_dim_key=64,
        attn_dim_value=64,
        block_size=128,

        attention_mode="block_sparse",

        dropout_rate=0.3,
        attn_dropout=0.05,
        pos_dropout=0.01,
        use_checkpointing=False,
        use_convnext=False,
        num_downsamples=7,
        dim_divisible_by=128,
        use_tf_gamma=False,

        use_rel_pe=False,
        use_einsum=False,
        attention_backend="auto",

        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.attn_dim_value = attn_dim_value
        self.block_size = block_size

        assert attention_mode in {"block_sparse", "full", "bigbird", "bigbird_ablation", "ccre_bigbird"}, \
            f"attention_mode must be 'block_sparse' or 'full', got {attention_mode}"

        self.attention_mode = attention_mode

        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.use_convnext = use_convnext
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma
        self.use_rel_pe = use_rel_pe

        valid_backends = {"auto", "flex", "sdpa", "einsum"}
        if attention_backend not in valid_backends:
            raise ValueError(
                f"attention_backend must be one of {sorted(valid_backends)}, "
                f"got {attention_backend!r}"
            )
        if use_einsum:
            if attention_backend not in {"auto", "einsum"}:
                raise ValueError(
                    "use_einsum=True conflicts with "
                    f"attention_backend={attention_backend!r}"
                )
            attention_backend = "einsum"

        self.attention_backend = attention_backend
        # Retain the old flag when reading and writing existing checkpoints.
        self.use_einsum = attention_backend == "einsum"

        super().__init__(**kwargs)
