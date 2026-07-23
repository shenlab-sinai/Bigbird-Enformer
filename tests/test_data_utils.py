import numpy as np
import torch

from bigbird_enformer.train_lightning import (
    augment_pair,
    random_shift,
    reverse_complement_seq,
)
from bigbird_enformer.utils.data import (
    seq_indices_to_one_hot,
    str_to_one_hot,
    str_to_seq_indices,
)


def test_string_encoders_handle_case_unknown_and_padding():
    upper = str_to_one_hot("ACGTN.")
    lower = str_to_one_hot("acgtn.")

    torch.testing.assert_close(upper, lower)
    torch.testing.assert_close(upper[:4], torch.eye(4))
    torch.testing.assert_close(upper[4], torch.zeros(4))
    torch.testing.assert_close(upper[5], torch.full((4,), 0.25))
    assert str_to_seq_indices("ACGTN.").tolist() == [0, 1, 2, 3, 4, -1]


def test_sequence_indices_handle_unknown_and_padding():
    indices = torch.tensor([0, 1, 2, 3, 4, -1])
    encoded = seq_indices_to_one_hot(indices)

    torch.testing.assert_close(encoded[:4], torch.eye(4))
    torch.testing.assert_close(encoded[4], torch.zeros(4))
    torch.testing.assert_close(encoded[5], torch.full((4,), 0.25))


def test_random_shift_preserves_shape_and_zero_pads():
    sequence = np.arange(20, dtype=np.float32).reshape(5, 4)

    shifted_right = random_shift(sequence, 2)
    shifted_left = random_shift(sequence, -2)

    np.testing.assert_array_equal(shifted_right[:2], np.zeros((2, 4)))
    np.testing.assert_array_equal(shifted_right[2:], sequence[:-2])
    np.testing.assert_array_equal(shifted_left[:-2], sequence[2:])
    np.testing.assert_array_equal(shifted_left[-2:], np.zeros((2, 4)))


def test_reverse_complement_reverses_positions_and_channels():
    sequence = np.arange(20, dtype=np.float32).reshape(5, 4)
    expected = sequence[::-1, ::-1]
    np.testing.assert_array_equal(reverse_complement_seq(sequence), expected)


def test_augment_pair_keeps_sequence_target_and_mask_aligned(monkeypatch):
    sequence = np.arange(24, dtype=np.float32).reshape(6, 4)
    target = np.arange(12, dtype=np.float32).reshape(6, 2)
    mask = np.array([True, False, False, True, False, True])

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    monkeypatch.setattr(np.random, "rand", lambda *args, **kwargs: 0.0)

    aug_sequence, aug_target, aug_mask = augment_pair(
        sequence,
        target,
        mask,
        max_shift=3,
    )

    np.testing.assert_array_equal(aug_sequence, sequence[::-1, ::-1])
    np.testing.assert_array_equal(aug_target, target[::-1])
    np.testing.assert_array_equal(aug_mask, mask[::-1])
