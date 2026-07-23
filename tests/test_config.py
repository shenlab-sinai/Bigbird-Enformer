import pytest
from pathlib import Path

from bigbird_enformer.utils.config import EnformerConfig


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
