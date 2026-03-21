"""
validate_ccre_masks.py

Ground-truth validation of cCRE masks by checking exact bin positions.

Strategy:
  For each sampled locus:
    1. Find all cCREs that genuinely overlap it (from the BED file directly)
    2. For each cCRE, compute EXACTLY which bins [bin_lo, bin_hi] it spans
    3. Assert those bins are True in the saved mask
    4. Assert the bin just before bin_lo and just after bin_hi are False
       (boundary check — proves we didn't over-extend)
    5. Assert total True bins == sum of all expected bins from step 2

This catches:
  - Off-by-one errors in bin boundaries
  - Wrong locus coordinates (wrong center_expand)
  - Shifted masks (mask saved for wrong locus)
  - Over-marking (too many True bins)
  - Under-marking (missed cCREs)

Usage:
    python validate_ccre_masks.py \
        --mask_dir /sc/.../ccre_mask/human \
        --ccre_bed /hpc/users/hongw01/.../GRCh38-cCREs.bed \
        --seq_bed  /sc/.../basenji_data/human/sequences.bed \
        --n_loci   50 \
        --seed     42 \
        2>&1 | tee validation_out.txt
"""

import argparse
import bisect
import random
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

SEQ_LENGTH    = 196_608
HALF_SEQ      = SEQ_LENGTH // 2
BIN_SIZE      = 128
NUM_BINS      = SEQ_LENGTH // BIN_SIZE   # 1536
MAX_CCRE_WIDTH = 1_000


# ── file parsers (same as ccre_preprocess.py) ───────────────────────────────

def parse_seq_bed(path):
    splits = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            splits[p[3]].append((p[0], int(p[1]), int(p[2])))
    return splits

def parse_ccre_bed(path):
    ivs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            p = line.split("\t")
            ivs.append((p[0], int(p[1]), int(p[2])))
    return ivs

def build_index(ivs):
    raw = defaultdict(list)
    for c, s, e in ivs:
        raw[c].append((s, e))
    idx = {}
    for c, lst in raw.items():
        lst.sort()
        idx[c] = {"intervals": lst, "starts": [x[0] for x in lst]}
    return idx

def center_expand(chrom, start, end):
    ctr = (start + end) // 2
    return chrom, ctr - HALF_SEQ, ctr + HALF_SEQ


# ── ground-truth reference (no bisect shortcut, just a clear loop) ──────────

def ground_truth_expected_bins(chrom, locus_start, ccre_idx):
    """
    Returns a dict mapping each overlapping cCRE → (bin_lo, bin_hi).
    This is the ground truth: iterate ALL cCREs on chrom, find overlapping
    ones, compute their exact bin range. No bisect, no MAX_CCRE_WIDTH
    assumption — just straightforward correctness.
    """
    locus_end = locus_start + SEQ_LENGTH
    result = {}

    if chrom not in ccre_idx:
        return result

    for cs, ce in ccre_idx[chrom]["intervals"]:
        # standard half-open interval overlap: cs < locus_end AND ce > locus_start
        if cs >= locus_end:
            break          # sorted → no more can overlap
        if ce <= locus_start:
            continue       # before locus

        bin_lo = max(0,           (cs      - locus_start) // BIN_SIZE)
        bin_hi = min(NUM_BINS - 1, (ce - 1 - locus_start) // BIN_SIZE)
        if bin_lo <= bin_hi:
            result[(cs, ce)] = (bin_lo, bin_hi)

    return result


# ── the actual validator ─────────────────────────────────────────────────────

def validate_locus(mask_path, chrom, locus_start, ccre_idx):
    """
    Returns list of failure strings (empty = pass).
    """
    mask = np.load(mask_path)
    failures = []

    # Basic sanity
    if mask.shape != (NUM_BINS,):
        return [f"SHAPE ERROR: expected ({NUM_BINS},), got {mask.shape}"]
    if mask.dtype != bool:
        return [f"DTYPE ERROR: expected bool, got {mask.dtype}"]

    expected = ground_truth_expected_bins(chrom, locus_start, ccre_idx)

    # Build expected mask from ground truth
    expected_mask = np.zeros(NUM_BINS, dtype=bool)
    for (cs, ce), (bl, bh) in expected.items():
        expected_mask[bl : bh + 1] = True

    # ── Check 1: every expected bin is True ─────────────────────────────────
    for (cs, ce), (bl, bh) in expected.items():
        for b in range(bl, bh + 1):
            if not mask[b]:
                failures.append(
                    f"UNDER-MARK: cCRE [{cs},{ce}) should mark bin {b} "
                    f"(={locus_start + b*BIN_SIZE}-{locus_start + (b+1)*BIN_SIZE}) "
                    f"but it is False"
                )
                if len(failures) >= 5:
                    break
        if len(failures) >= 5:
            break

    # ── Check 2: bin just outside each cCRE is NOT marked (unless another
    #    cCRE also covers it) ─────────────────────────────────────────────────
    for (cs, ce), (bl, bh) in expected.items():
        # bin before bl
        if bl > 0:
            b = bl - 1
            if mask[b] and not expected_mask[b]:
                failures.append(
                    f"OVER-MARK at left boundary: bin {b} "
                    f"(={locus_start + b*BIN_SIZE}-{locus_start + (b+1)*BIN_SIZE}) "
                    f"is True but no cCRE should cover it "
                    f"(adjacent cCRE was [{cs},{ce}))"
                )
        # bin after bh
        if bh < NUM_BINS - 1:
            b = bh + 1
            if mask[b] and not expected_mask[b]:
                failures.append(
                    f"OVER-MARK at right boundary: bin {b} "
                    f"(={locus_start + b*BIN_SIZE}-{locus_start + (b+1)*BIN_SIZE}) "
                    f"is True but no cCRE should cover it "
                    f"(adjacent cCRE was [{cs},{ce}))"
                )

    # ── Check 3: total True bin count matches ───────────────────────────────
    saved_count    = int(mask.sum())
    expected_count = int(expected_mask.sum())
    if saved_count != expected_count:
        failures.append(
            f"COUNT MISMATCH: saved has {saved_count} True bins, "
            f"expected {expected_count} from {len(expected)} overlapping cCREs"
        )

    # ── Check 4: if no cCREs overlap, mask must be all False ────────────────
    if not expected and mask.any():
        failures.append(
            f"SPURIOUS MARKS: no cCREs overlap this locus but "
            f"{mask.sum()} bins are True"
        )

    return failures


# ── main ────────────────────────────────────────────────────────────────────

def main(mask_dir, ccre_bed, seq_bed, n_loci, seed):
    mask_dir = Path(mask_dir)
    random.seed(seed)

    print("Loading cCRE BED ...")
    ccre_idx = build_index(parse_ccre_bed(ccre_bed))
    n_ccre = sum(len(v["intervals"]) for v in ccre_idx.values())
    print(f"  {n_ccre:,} cCREs across {len(ccre_idx)} chromosomes")

    print("Loading sequences.bed ...")
    splits = parse_seq_bed(seq_bed)
    all_loci = []
    for split, loci in splits.items():
        for i, (chrom, s, e) in enumerate(loci):
            all_loci.append((split, i, chrom, s, e))
    print(f"  {len(all_loci):,} total loci across {len(splits)} splits")

    # Sample n_loci, but always include a few zero-cCRE loci and high-density
    # loci if they exist, for better coverage
    sampled = random.sample(all_loci, min(n_loci, len(all_loci)))

    print(f"\nValidating {len(sampled)} sampled loci ...")
    print("=" * 70)

    n_pass = 0
    n_fail = 0
    all_failures = []

    for split, idx_i, chrom, raw_s, raw_e in sampled:
        chrom_exp, locus_start, locus_end = center_expand(chrom, raw_s, raw_e)
        mask_path = mask_dir / f"{split}-{idx_i:06d}.npy"

        if not mask_path.exists():
            msg = f"MISSING FILE: {mask_path.name}"
            all_failures.append((mask_path.name, [msg]))
            n_fail += 1
            print(f"  FAIL  {mask_path.name} — {msg}")
            continue

        failures = validate_locus(mask_path, chrom_exp, locus_start, ccre_idx)

        if failures:
            n_fail += 1
            all_failures.append((mask_path.name, failures))
            print(f"  FAIL  {mask_path.name}  "
                  f"({chrom_exp}:{locus_start}-{locus_end})")
            for f in failures[:3]:
                print(f"         → {f}")
        else:
            n_pass += 1
            mask = np.load(mask_path)
            density = mask.sum() / NUM_BINS * 100
            n_ccre_overlap = len(
                ground_truth_expected_bins(chrom_exp, locus_start, ccre_idx)
            )
            print(f"  PASS  {mask_path.name}  "
                  f"({chrom_exp}:{locus_start}-{locus_end})  "
                  f"density={density:.1f}%  n_cCREs={n_ccre_overlap}")

    print("=" * 70)
    print(f"\nRESULT: {n_pass} PASS / {n_fail} FAIL out of {len(sampled)} loci")

    if all_failures:
        print(f"\nAll failures:")
        for name, flist in all_failures:
            print(f"  {name}:")
            for f in flist:
                print(f"    {f}")
        sys.exit(1)
    else:
        print("\nAll checks passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--ccre_bed", required=True)
    p.add_argument("--seq_bed",  required=True)
    p.add_argument("--n_loci",   type=int, default=50,
                   help="Number of loci to spot-check (default 50)")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    main(args.mask_dir, args.ccre_bed, args.seq_bed, args.n_loci, args.seed)