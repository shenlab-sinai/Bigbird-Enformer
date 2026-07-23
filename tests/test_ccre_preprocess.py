import csv

import numpy as np

from bigbird_enformer.utils.ccre_preprocess import (
    HALF_SEQ,
    NUM_BINS,
    build_chrom_index,
    center_expand,
    locus_to_mask,
    parse_ccre_bed,
    parse_seq_bed,
    preprocess,
)


def test_bed_parsers_and_chromosome_index(tmp_path):
    sequence_bed = tmp_path / "sequences.bed"
    sequence_bed.write_text(
        "# comment\nchr1\t0\t196608\ttrain\nchr2\t10\t20\tvalid\n"
    )
    ccre_bed = tmp_path / "ccre.bed"
    ccre_bed.write_text(
        "track name=ccre\nchr1\t300\t400\nchr1\t100\t200\nchr2\t10\t20\n"
    )

    splits = parse_seq_bed(sequence_bed)
    intervals = parse_ccre_bed(ccre_bed)
    index = build_chrom_index(intervals)

    assert splits["train"] == [("chr1", 0, 196608)]
    assert splits["valid"] == [("chr2", 10, 20)]
    assert index["chr1"]["intervals"] == [(100, 200), (300, 400)]
    assert index["chr1"]["starts"] == [100, 300]


def test_center_expand_preserves_locus_center():
    chrom, start, end = center_expand("chr1", 100, 300)
    assert chrom == "chr1"
    assert start == 200 - HALF_SEQ
    assert end == 200 + HALF_SEQ


def test_locus_to_mask_applies_overlap_threshold():
    index = build_chrom_index(
        [
            ("chr1", 1_000, 1_064),
            ("chr1", 1_128, 1_191),
        ]
    )

    mask = locus_to_mask("chr1", 1_000, index, min_overlap_bp=64)

    assert mask.shape == (NUM_BINS,)
    assert mask.dtype == bool
    assert mask[0]
    assert not mask[1]
    assert mask.sum() == 1


def test_preprocess_writes_mask_and_manifest(tmp_path):
    sequence_bed = tmp_path / "sequences.bed"
    sequence_bed.write_text("chr1\t0\t196608\ttrain\n")
    ccre_bed = tmp_path / "ccre.bed"
    ccre_bed.write_text("chr1\t0\t64\n")
    output_dir = tmp_path / "masks"

    preprocess(ccre_bed, sequence_bed, output_dir, min_overlap_bp=64)

    mask = np.load(output_dir / "train-000000.npy")
    assert mask.shape == (NUM_BINS,)
    assert mask.dtype == bool
    assert mask[0]
    assert mask.sum() == 1

    with (output_dir / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "npz_file": "train-000000.npz",
            "mask_file": "train-000000.npy",
            "chrom": "chr1",
            "exp_start": "0",
            "exp_end": "196608",
            "n_ccre_bins": "1",
        }
    ]
