from pathlib import Path

import numpy as np
import pytest
import torch

from bigbird_enformer.utils.config import EnformerConfig


@pytest.fixture(autouse=True)
def deterministic_random_state():
    """Keep every synthetic test repeatable."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def tiny_config_factory():
    """Build small Enformer configurations suitable for CPU tests."""

    def make(**overrides):
        values = {
            "dim": 16,
            "depth": 1,
            "heads": 2,
            "output_heads": {"human": 3, "mouse": 2},
            "target_length": 4,
            "attn_dim_key": 4,
            "attn_dim_value": 4,
            "block_size": 4,
            "attention_mode": "full",
            "dropout_rate": 0.0,
            "attn_dropout": 0.0,
            "pos_dropout": 0.0,
            "use_checkpointing": False,
            "dim_divisible_by": 8,
            "use_rel_pe": False,
            "use_einsum": False,
        }
        values.update(overrides)
        return EnformerConfig(**values)

    return make


@pytest.fixture
def write_npz():
    """Write a small sequence/target pair using the repository dataset format."""

    def write(
        directory: Path,
        name: str,
        *,
        sequence_length: int = 8,
        target_length: int = 2,
        channels: int = 3,
        value: float = 1.0,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        sequence = np.zeros((sequence_length, 4), dtype=np.float32)
        sequence[:, 0] = value
        target = np.full(
            (target_length, channels),
            value,
            dtype=np.float32,
        )
        np.savez(path, sequence=sequence, target=target)
        return path

    return write
