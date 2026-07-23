import importlib
import os
import sys
import types

import numpy as np
import pytest
import torch

from bigbird_enformer.train_lightning import (
    SingleOrganismDataset,
    ZippedOrganismDataset,
    zipped_collate_fn,
)


def test_single_organism_dataset_loads_split_and_mask(tmp_path, write_npz):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    write_npz(data_dir, "train-000000.npz", value=1.0)
    write_npz(data_dir, "valid-000000.npz", value=2.0)
    mask_dir.mkdir()
    expected_mask = np.array([True, False, True, False])
    np.save(mask_dir / "train-000000.npy", expected_mask)

    dataset = SingleOrganismDataset(data_dir, "train", ccre_mask_dir=mask_dir)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["sequence"].dtype == torch.float32
    assert item["target"].dtype == torch.float32
    torch.testing.assert_close(item["ccre_mask"], torch.from_numpy(expected_mask))


def test_single_organism_dataset_uses_fixed_empty_mask_fallback(tmp_path, write_npz):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    write_npz(data_dir, "test-000000.npz")
    mask_dir.mkdir()

    item = SingleOrganismDataset(data_dir, "test", ccre_mask_dir=mask_dir)[0]

    assert item["ccre_mask"].shape == (1536,)
    assert not item["ccre_mask"].any()


def test_zipped_dataset_cycles_shorter_organism(tmp_path, write_npz):
    human_dir = tmp_path / "human"
    mouse_dir = tmp_path / "mouse"
    write_npz(human_dir, "train-000000.npz", value=1.0)
    write_npz(human_dir, "train-000001.npz", value=2.0)
    write_npz(mouse_dir, "train-000000.npz", channels=2, value=3.0)

    dataset = ZippedOrganismDataset(
        human_dir,
        mouse_dir,
        "train",
        augment=False,
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    torch.testing.assert_close(
        first["mouse"]["sequence"],
        second["mouse"]["sequence"],
    )
    assert not torch.equal(
        first["human"]["sequence"],
        second["human"]["sequence"],
    )


def test_zipped_collate_stacks_sequences_targets_and_masks():
    sample = {
        "human": {
            "sequence": torch.ones(8, 4),
            "target": torch.ones(2, 3),
            "ccre_mask": torch.tensor([True, False]),
        },
        "mouse": {
            "sequence": torch.zeros(8, 4),
            "target": torch.ones(2, 2),
            "ccre_mask": torch.tensor([False, True]),
        },
    }

    batch = zipped_collate_fn([sample, sample])

    assert batch["human"]["sequence"].shape == (2, 8, 4)
    assert batch["human"]["target"].shape == (2, 2, 3)
    assert batch["mouse"]["target"].shape == (2, 2, 2)
    assert batch["mouse"]["ccre_mask"].shape == (2, 2)


@pytest.fixture
def gtex_dataset_module(monkeypatch):
    pytest.importorskip("cyvcf2")
    pytest.importorskip("pyfaidx")

    dependency_name = "bigbird_enformer.utils.ccre_mask_utils"
    module_name = "bigbird_enformer.utils.gtex_dataset"
    stub = types.ModuleType(dependency_name)
    stub.generate_ccre_mask = lambda **kwargs: np.zeros(
        kwargs["seq_len"] // 128,
        dtype=bool,
    )
    monkeypatch.setitem(sys.modules, dependency_name, stub)
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_gtex_one_hot_encoding_and_collation(gtex_dataset_module):
    encoded = gtex_dataset_module.one_hot_encode("ACGTN")
    np.testing.assert_array_equal(encoded[:4], np.eye(4, dtype=np.float32))
    np.testing.assert_array_equal(encoded[4], np.full(4, 0.25, dtype=np.float32))

    item = {
        "hap1": torch.from_numpy(encoded),
        "hap2": torch.from_numpy(encoded.copy()),
        "target": torch.tensor([1.0, 2.0]),
        "ccre_mask": torch.tensor([True, False]),
        "sample": "sample-1",
    }
    batch = gtex_dataset_module.gtex_collate_fn([item, item])

    assert batch["hap1"].shape == (2, 5, 4)
    assert batch["target"].shape == (2, 2)
    assert batch["samples"] == ["sample-1", "sample-1"]


@pytest.mark.integration
def test_external_npz_dataset_can_load_one_sample():
    data_dir = os.environ.get("ATLAS_TEST_DATA")
    if not data_dir:
        pytest.skip("set ATLAS_TEST_DATA to an NPZ dataset directory")

    split = os.environ.get("ATLAS_TEST_SPLIT", "test")
    dataset = SingleOrganismDataset(data_dir, split)
    assert len(dataset) > 0
    item = dataset[0]
    assert item["sequence"].ndim == 2
    assert item["sequence"].shape[-1] == 4
    assert item["target"].ndim == 2
