"""
Precompute per-locus cCRE binary masks from ENCODE4 SCREEN BED file.

Usage:
    # Human (GRCh38)
    python src/data/ccre_preprocess.py \
        --ccre_bed   /hpc/users/hongw01/Bigbird-Enformer/data/GRCh38-cCREs.bed \
        --seq_bed    /sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/basenji_data/human/sequences.bed \
        --output_dir /sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/ccre_mask/human_50pct

    # Mouse (GRCm38) 
    python src/data/ccre_preprocess.py \
        --ccre_bed   /hpc/users/hongw01/Bigbird-Enformer/data/mm10-cCREs.bed \
        --seq_bed    /sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/basenji_data/mouse/sequences.bed \
        --output_dir /sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/ccre_mask/mouse_50pct

Output:
    {output_dir}/{split}-{i:06d}.npy   shape [1536] dtype bool
    {output_dir}/manifest.csv
"""

import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQ_LENGTH     = 196_608
HALF_SEQ       = SEQ_LENGTH // 2   # 98304
BIN_SIZE       = 128
NUM_BINS       = SEQ_LENGTH // BIN_SIZE  # 1536
MAX_CCRE_WIDTH = 1_000

DEFAULT_MIN_OVERLAP_BP = BIN_SIZE // 2   # 64


def parse_seq_bed(path: str) -> dict:
    splits = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, start, end, split = parts[0], int(parts[1]), int(parts[2]), parts[3]
            splits[split].append((chrom, start, end))
    return splits


def parse_ccre_bed(path: str) -> list:
    intervals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if (line.startswith("#") or
                    line.startswith("track") or
                    line.startswith("browser")):
                continue
            parts = line.split("\t")
            intervals.append((parts[0], int(parts[1]), int(parts[2])))
    return intervals

def build_chrom_index(intervals: list) -> dict:
    raw = defaultdict(list)
    for chrom, start, end in intervals:
        raw[chrom].append((start, end))

    index = {}
    for chrom, ivs in raw.items():
        ivs.sort()
        index[chrom] = {
            "intervals": ivs,
            "starts":    [iv[0] for iv in ivs],
        }
    return index

def center_expand(chrom: str, start: int, end: int):
    center    = (start + end) // 2
    exp_start = center - HALF_SEQ
    exp_end   = center + HALF_SEQ
    return chrom, exp_start, exp_end

def locus_to_mask(
    chrom:            str,
    locus_start:      int,
    ccre_index:       dict,
    min_overlap_bp:   int = DEFAULT_MIN_OVERLAP_BP,
) -> np.ndarray:
    """
    Compute [NUM_BINS] bool mask using a minimum-overlap threshold.
    """
    coverage = np.zeros(NUM_BINS, dtype=np.int32)

    if chrom not in ccre_index:
        return coverage.astype(bool)

    entry     = ccre_index[chrom]
    intervals = entry["intervals"]
    starts    = entry["starts"]
    locus_end = locus_start + SEQ_LENGTH

    search_start = locus_start - MAX_CCRE_WIDTH
    lo = bisect.bisect_left(starts, search_start)

    for cstart, cend in intervals[lo:]:
        if cstart >= locus_end:
            break
        if cend <= locus_start:
            continue

        # Clip cCRE to locus window
        clipped_start = max(cstart, locus_start)
        clipped_end   = min(cend,   locus_end)

        # Bins this clipped cCRE touches
        bin_lo = (clipped_start - locus_start) // BIN_SIZE
        bin_hi = (clipped_end - 1 - locus_start) // BIN_SIZE
        bin_lo = max(0, bin_lo)
        bin_hi = min(NUM_BINS - 1, bin_hi)

        for b in range(bin_lo, bin_hi + 1):
            bin_start = locus_start + b * BIN_SIZE
            bin_end   = bin_start + BIN_SIZE
            # Overlap of this cCRE with bin b (clamped to bin boundaries)
            overlap = min(cend, bin_end) - max(cstart, bin_start)
            if overlap > 0:
                # Cap at BIN_SIZE so multiple cCREs can't exceed 100% coverage
                coverage[b] = min(BIN_SIZE, coverage[b] + overlap)

    return coverage >= min_overlap_bp

def preprocess(
    ccre_bed:       str,
    seq_bed:        str,
    output_dir:     str,
    min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SCREEN cCREs from {ccre_bed} ...")
    ccre_index = build_chrom_index(parse_ccre_bed(ccre_bed))
    n_ccre = sum(len(v["intervals"]) for v in ccre_index.values())
    print(f"  {n_ccre:,} cCREs across {len(ccre_index)} chromosomes.")
    print(f"  Overlap threshold: {min_overlap_bp} bp "
          f"({min_overlap_bp / BIN_SIZE * 100:.0f}% of {BIN_SIZE}-bp bin)")

    print(f"Loading sequences.bed from {seq_bed} ...")
    splits = parse_seq_bed(seq_bed)
    for sp, rows in splits.items():
        print(f"  {sp}: {len(rows):,} loci")

    total = sum(len(v) for v in splits.values())
    print(f"Processing {total:,} loci total ...")

    manifest_path = output_dir / "manifest.csv"
    n_written = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["npz_file", "mask_file", "chrom",
                          "exp_start", "exp_end", "n_ccre_bins"])

        for split, loci in splits.items():
            for i, (chrom, start, end) in enumerate(loci):
                chrom, exp_start, exp_end = center_expand(chrom, start, end)

                mask      = locus_to_mask(chrom, exp_start, ccre_index,
                                          min_overlap_bp)
                npz_fname = f"{split}-{i:06d}.npz"
                npy_fname = f"{split}-{i:06d}.npy"
                np.save(output_dir / npy_fname, mask)
                writer.writerow([npz_fname, npy_fname, chrom,
                                  exp_start, exp_end, int(mask.sum())])
                n_written += 1

                if n_written % 5_000 == 0:
                    print(f"  {n_written:,} / {total:,} done ...")

    print(f"\nDone. {n_written:,} masks → {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccre_bed",        required=True)
    parser.add_argument("--seq_bed",         required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--min_overlap_bp",  type=int,
                        default=DEFAULT_MIN_OVERLAP_BP,
                        help=f"Min bp of cCRE coverage to mark a bin True "
                             f"(default {DEFAULT_MIN_OVERLAP_BP} = 50%% of bin).")
    args = parser.parse_args()
    preprocess(args.ccre_bed, args.seq_bed, args.output_dir,
               args.min_overlap_bp)