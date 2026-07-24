import pytest
from pathlib import Path

from bigbird_enformer.utils.config import EnformerConfig, load_experiment_config


@pytest.mark.parametrize(
    "mode",
    ["block_sparse", "full", "bigbird", "bigbird_ablation", "ccre_bigbird"],
)
def test_supported_attention_modes(mode):
    assert EnformerConfig(attention_mode=mode).attention_mode == mode


def test_invalid_attention_mode_is_rejected():
    with pytest.raises(AssertionError, match="attention_mode"):
        EnformerConfig(attention_mode="dense")


def test_config_round_trip_preserves_model_settings(tiny_config_factory):
    original = tiny_config_factory(
        attention_mode="ccre_bigbird",
        output_heads={"human": 7},
        use_checkpointing=True,
    )

    restored = EnformerConfig.from_dict(original.to_dict())

    assert restored.attention_mode == "ccre_bigbird"
    assert restored.output_heads == {"human": 7}
    assert restored.use_checkpointing is True
    assert restored.dim == original.dim
    assert restored.heads == original.heads


def test_default_output_heads_are_available():
    config = EnformerConfig()
    assert config.output_heads == {"human": 5313, "mouse": 1643}


def test_default_attention_dimensions_match_published_enformer():
    config = EnformerConfig()

    assert config.attn_dim_key == 64
    assert config.attn_dim_value == 192
    assert config.heads * config.attn_dim_value == config.dim


def test_value_dimension_defaults_to_model_width_per_head():
    config = EnformerConfig(dim=32, heads=4)

    assert config.attn_dim_value == 8


def test_explicit_legacy_value_dimension_is_preserved():
    config = EnformerConfig(attn_dim_value=64)

    assert config.attn_dim_value == 64


@pytest.mark.parametrize("backend", ["auto", "flex", "sdpa", "einsum"])
def test_supported_attention_backends(backend):
    assert EnformerConfig(attention_backend=backend).attention_backend == backend


def test_attention_backend_defaults_to_auto():
    assert EnformerConfig().attention_backend == "auto"


def test_invalid_attention_backend_is_rejected():
    with pytest.raises(ValueError, match="attention_backend"):
        EnformerConfig(attention_backend="flash")


def test_legacy_use_einsum_selects_einsum_backend():
    config = EnformerConfig(use_einsum=True)

    assert config.attention_backend == "einsum"
    assert config.use_einsum is True


def test_repository_ccre_config_loads_with_auto_backend():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ccre_bigbird.yaml"
    )

    config = EnformerConfig.from_yaml_file(config_path)

    assert config.attention_mode == "ccre_bigbird"
    assert config.attention_backend == "auto"
    assert config.attn_dropout == 0.0
    assert config.attn_dim_key == 64
    assert config.attn_dim_value == 192
    assert config.heads * config.attn_dim_value == config.dim


def test_repository_ccre_config_exposes_topk_parameter():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ccre_bigbird.yaml"
    )

    experiment_config = load_experiment_config(config_path)
    topk_k = experiment_config["training"]["mean_ccre_k"]

    assert isinstance(topk_k, int)
    assert topk_k > 0
