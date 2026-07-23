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


def test_single_organism_dataset_rejects_missing_mask(tmp_path, write_npz):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    write_npz(data_dir, "test-000000.npz")
    mask_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="missing 1 cCRE mask"):
        SingleOrganismDataset(data_dir, "test", ccre_mask_dir=mask_dir)


def test_single_organism_dataset_validates_every_mask_before_loading(
    tmp_path,
    write_npz,
):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    write_npz(data_dir, "train-000000.npz")
    write_npz(data_dir, "train-000001.npz")
    mask_dir.mkdir()
    np.save(mask_dir / "train-000000.npy", np.zeros(4, dtype=bool))

    with pytest.raises(
        FileNotFoundError,
        match=r"train-000001\.npy",
    ):
        SingleOrganismDataset(data_dir, "train", ccre_mask_dir=mask_dir)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((2, 2), dtype=bool), "expected one dimension"),
        (np.zeros(4, dtype=np.int8), "expected Boolean dtype"),
        (np.zeros(3, dtype=bool), "expected length 4"),
    ],
)
def test_single_organism_dataset_rejects_invalid_mask(
    tmp_path,
    write_npz,
    mask,
    message,
):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    write_npz(data_dir, "train-000000.npz")
    mask_dir.mkdir()
    np.save(mask_dir / "train-000000.npy", mask)

    with pytest.raises(ValueError, match=message):
        SingleOrganismDataset(
            data_dir,
            "train",
            ccre_mask_dir=mask_dir,
            expected_mask_length=4,
        )


def test_zipped_dataset_rejects_missing_mask_before_training(
    tmp_path,
    write_npz,
):
    human_dir = tmp_path / "human"
    mouse_dir = tmp_path / "mouse"
    human_mask_dir = tmp_path / "human_masks"
    mouse_mask_dir = tmp_path / "mouse_masks"
    write_npz(human_dir, "train-000000.npz")
    write_npz(mouse_dir, "train-000000.npz")
    human_mask_dir.mkdir()
    mouse_mask_dir.mkdir()
    np.save(human_mask_dir / "train-000000.npy", np.zeros(4, dtype=bool))

    with pytest.raises(
        FileNotFoundError,
        match=r"mouse_masks/train-000000\.npy",
    ):
        ZippedOrganismDataset(
            human_dir,
            mouse_dir,
            "train",
            human_ccre_mask_dir=human_mask_dir,
            mouse_ccre_mask_dir=mouse_mask_dir,
            augment=False,
        )


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
