import os
import subprocess as sp
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from cyvcf2 import VCF
from pyfaidx import Fasta
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from .ccre_mask_utils import generate_ccre_mask

_GINFO_PATH = Path(
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/genias/data/reference_genomes/hg38_gene_info.csv"
)
_REF_PATH = Path(
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/genias/data/reference_genomes/hg38.ml.fa"
)
_CCRE_BED_PATH = Path(
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/Sparse_Enformer/data/cCREs/full_cCRES/GRCh38-cCREs.bed"
)

_OHE_TABLE = np.full((256, 4), 0.25, dtype=np.float32)
for _idx, _pair in enumerate([
    (ord("A"), ord("a")),
    (ord("C"), ord("c")),
    (ord("G"), ord("g")),
    (ord("T"), ord("t")),
]):
    for _byte in _pair:
        _OHE_TABLE[_byte]       = 0.0
        _OHE_TABLE[_byte, _idx] = 1.0


def one_hot_encode(seq: str) -> np.ndarray:
    return _OHE_TABLE[
        np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    ].copy()                   


class GTExConsensusDataset(Dataset):
    def __init__(
        self,
        bcf_path:       Union[str, Path],
        gene_id:        Optional[str]                       = None,
        gene_name:      Optional[str]                       = None,
        sample_list:    Optional[List[str]]                 = None,
        target_list:    Optional[Union[List, np.ndarray]]   = None,
        ref_path:       Union[str, Path]                    = _REF_PATH,
        ginfo_path:     Union[str, Path]                    = _GINFO_PATH,
        seq_len:        int                                 = 196_608,
        haplotypes:     List[str]                           = ("1pIu", "2pIu"),
        buffer:         int                                 = 1_000,
        ccre_bed_path:  Optional[Union[str, Path]]          = _CCRE_BED_PATH,
        show_progress:  bool                                = True,
    ):
        super().__init__()
        assert gene_id is not None or gene_name is not None, \
            "Provide at least one of gene_id or gene_name."
        assert seq_len % 128 == 0, \
            f"seq_len={seq_len} must be divisible by 128 (bin size)."
        assert len(haplotypes) >= 1, "haplotypes list must not be empty."

        self.seq_len    = seq_len
        self.haplotypes = list(haplotypes)

        bcf_handle      = VCF(str(bcf_path))
        all_bcf_samples = list(bcf_handle.samples)

        if sample_list is not None:
            if target_list is not None and len(sample_list) != len(target_list):
                raise ValueError(
                    f"sample_list length ({len(sample_list)}) must equal "
                    f"target_list length ({len(target_list)})."
                )
            kept_samples: List[str]       = []
            kept_targets: List[np.ndarray] = []
            dropped = 0
            for i, s in enumerate(sample_list):
                if s in all_bcf_samples:
                    kept_samples.append(s)
                    if target_list is not None:
                        kept_targets.append(target_list[i])
                else:
                    dropped += 1
            if dropped:
                warnings.warn(
                    f"{dropped} sample(s) from sample_list not found in BCF; "
                    f"dropped."
                )
            self.samples = kept_samples
            self.targets = (
                np.array(kept_targets, dtype=np.float32) if kept_targets else None
            )
        else:
            self.samples = all_bcf_samples
            if target_list is not None:
                if len(target_list) != len(all_bcf_samples):
                    raise ValueError(
                        f"target_list length ({len(target_list)}) must equal "
                        f"number of BCF samples ({len(all_bcf_samples)})."
                    )
                self.targets = np.array(target_list, dtype=np.float32)
            else:
                self.targets = None

        if not self.samples:
            raise ValueError(
                "No samples remain after BCF filtering. "
                "Check that sample_list IDs match BCF sample names."
            )

        ginfo_df = pd.read_csv(str(ginfo_path))

        if gene_id is not None:
            if gene_id not in ginfo_df["id"].values:
                raise ValueError(f"Gene ID '{gene_id}' not in {ginfo_path}.")
            rows = ginfo_df[ginfo_df["id"] == gene_id]
            if rows["id"].duplicated().any():
                warnings.warn(f"Gene ID {gene_id} matched multiple rows; "
                              f"using the first.")
            row = rows.iloc[0]
        else:
            if gene_name not in ginfo_df["name"].values:
                raise ValueError(f"Gene name '{gene_name}' not in {ginfo_path}.")
            rows = ginfo_df[ginfo_df["name"] == gene_name]
            if rows["name"].duplicated().any():
                warnings.warn(f"Gene name {gene_name} matched multiple rows; "
                              f"using the first.")
            row = rows.iloc[0]

        gid, gname, chrom, tss_1based, tes_1based, strand = tuple(row)
        self.gene_id   = str(gid)
        self.gene_name = str(gname)
        self.chrom     = str(chrom)           
        self.tss       = int(tss_1based) - 1  
        self.strand    = strand

        print(
            f"[GTExConsensusDataset] {gname} ({gid}) | "
            f"chr{self.chrom}:{self.tss} ({strand}) | "
            f"seq_len={seq_len:,} bp | "
            f"n_samples={len(self.samples)}",
            flush=True,
        )

        half     = seq_len // 2
        buf_half = buffer // 2

        ref_fasta   = Fasta(str(ref_path))
        chrom_names = list(ref_fasta.keys())
        ref_prefix  = "chr" if any(c.startswith("chr") for c in chrom_names) else ""
        ref_key     = f"{ref_prefix}{self.chrom}"
        chr_len     = len(ref_fasta[ref_key])

        raw_start = self.tss - half - buf_half
        raw_stop  = self.tss + half + buf_half

        left_pad  = ""
        right_pad = ""
        if raw_start < 0:
            left_pad  = "N" * (-raw_start)
            raw_start = 0
        if raw_stop > chr_len:
            right_pad = "N" * (raw_stop - chr_len)
            raw_stop  = chr_len

        dna_seq = str(ref_fasta[ref_key][raw_start:raw_stop])

        _first_var     = next(VCF(str(bcf_path)))
        bcf_prefix     = "chr" if "chr" in _first_var.CHROM else ""
        self._bcf_path = str(bcf_path)

        self._fasta_str = (
            f">{bcf_prefix}{self.chrom}:{raw_start + 1}-{raw_stop}\n{dna_seq}"
        )
        self._left_pad  = left_pad
        self._right_pad = right_pad

        print(
            f"[GTExConsensusDataset] Running bcftools for "
            f"{len(self.samples)} samples × {len(self.haplotypes)} haplotypes ...",
            flush=True,
        )
        self._sample_seqs: List[Tuple[str, ...]] = []
        for sample in tqdm(
            self.samples, desc=f"bcftools [{gname}]", disable=not show_progress
        ):
            hap_seqs = tuple(
                self._run_bcftools(sample, hap) for hap in self.haplotypes
            )
            self._sample_seqs.append(hap_seqs)

        n_bins = seq_len // 128
        if ccre_bed_path is not None:
            self.ccre_mask: np.ndarray = generate_ccre_mask(
                chrom=self.chrom,
                tss=self.tss,
                seq_len=seq_len,
                ccre_bed_path=str(ccre_bed_path),
            )
            print(
                f"[GTExConsensusDataset] cCRE mask: "
                f"{int(self.ccre_mask.sum())} / {n_bins} bins active "
                f"(density {self.ccre_mask.mean():.3f})",
                flush=True,
            )
        else:
            self.ccre_mask = np.zeros(n_bins, dtype=bool)
            print(
                f"[GTExConsensusDataset] No cCRE BED → all-False mask "
                f"(full-attention baseline mode).",
                flush=True,
            )

    def _run_bcftools(self, sample: str, haplotype: str) -> str:
        cmd = [
            "bcftools", "consensus",
            "-s",  sample,
            "-I",              
            "-H",  haplotype,
            "-f",  "-",      
            self._bcf_path,
        ]
        result = sp.run(
            cmd,
            input=self._fasta_str,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            warnings.warn(
                f"bcftools returned code {result.returncode} for "
                f"sample={sample} hap={haplotype}: "
                f"{result.stderr.strip()[:300]}"
            )

        lines = result.stdout.split("\n")
        seq   = "".join(lines[1:])

        seq = self._left_pad + seq + self._right_pad

        seq = seq.replace("*", "")

        if len(seq) < self.seq_len:
            raise ValueError(
                f"Sequence for sample={sample} hap={haplotype} is shorter than "
                f"seq_len ({len(seq)} < {self.seq_len}). "
                f"Increase `buffer` (currently implied from raw_stop-raw_start)."
            )
        if len(seq) > self.seq_len:
            trim = (len(seq) - self.seq_len) // 2
            seq  = seq[trim : trim + self.seq_len]

        return seq

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        hap_strs = self._sample_seqs[idx]

        hap_tensors = [
            torch.from_numpy(one_hot_encode(s)) for s in hap_strs
        ]

        target = (
            torch.from_numpy(self.targets[idx].copy())
            if self.targets is not None
            else torch.tensor([], dtype=torch.float32)
        )

        return {
            "hap1":      hap_tensors[0],                         
            "hap2":      hap_tensors[1],                          
            "target":    target,                                  
            "ccre_mask": torch.from_numpy(self.ccre_mask.copy()), 
            "sample":    self.samples[idx],
        }

    @property
    def n_tissues(self) -> int:
        if self.targets is None:
            return 0
        return self.targets.shape[1] if self.targets.ndim > 1 else 1

    @property
    def n_bins(self) -> int:
        return self.seq_len // 128

    @property
    def mean_ccre_k(self) -> int:
        return int(self.ccre_mask.sum())

def gtex_collate_fn(batch: List[dict]) -> dict:
    return {
        "hap1":      torch.stack([item["hap1"]      for item in batch]),
        "hap2":      torch.stack([item["hap2"]      for item in batch]),
        "target":    torch.stack([item["target"]    for item in batch]),
        "ccre_mask": torch.stack([item["ccre_mask"] for item in batch]),
        "samples":   [item["sample"] for item in batch],
    }

if __name__ == "__main__":
    import sys

    BCF_PATH = (
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/lis_play_ground/gtex/data"
    "/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_953Indiv.SHAPEIT2_phased.bcf"
    )
    EXP_NPZ = (
        "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
        "/lis_play_ground/gtex/data/GTEx_50tissue_exp.npz"
    )
    GENE_IDS_TXT = (
        "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
        "/lis_play_ground/gtex/data/0_gene_ids.txt"
    )
    SAMPLES_TXT = (
        "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
        "/lis_play_ground/gtex/data/1_samples.txt"
    )

    print("Loading GTEx expression matrix ...")
    exp_mat = np.load(EXP_NPZ)["exp"]
    with open(GENE_IDS_TXT) as f:
        gene_ids = [g.split(".")[0] for g in f.read().splitlines()]
    with open(SAMPLES_TXT) as f:
        sample_names = f.read().splitlines()

    TARGET_GENE = "ENSG00000168903"  
    ix = gene_ids.index(TARGET_GENE)
    exp_df = pd.DataFrame(exp_mat[ix], index=sample_names)
    # Keep tissues with <50% missing
    exp_df = exp_df.loc[:, exp_df.isna().mean() < 0.5]
    print(f"Expression matrix: {exp_df.shape}  (samples × tissues)")

    for seq_len in [196_608, 393_216, 786_432]:
        label = f"{seq_len // 1024}k"
        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len:,} ({label})")
        ds = GTExConsensusDataset(
            bcf_path=BCF_PATH,
            gene_id=TARGET_GENE,
            sample_list=sample_names,
            target_list=exp_df.values,
            seq_len=seq_len,
            show_progress=True,
        )
        print(f"Dataset: {len(ds)} samples | "
              f"n_tissues={ds.n_tissues} | "
              f"n_bins={ds.n_bins} | "
              f"mean_ccre_k={ds.mean_ccre_k}")

        item = ds[0]
        print(f"hap1:      {tuple(item['hap1'].shape)}  dtype={item['hap1'].dtype}")
        print(f"hap2:      {tuple(item['hap2'].shape)}  dtype={item['hap2'].dtype}")
        print(f"target:    {tuple(item['target'].shape)}  "
              f"nan={item['target'].isnan().sum().item()}")
        print(f"ccre_mask: {tuple(item['ccre_mask'].shape)}  "
              f"active={item['ccre_mask'].sum().item()}")
        print(f"sample:    {item['sample']}")

    print("\nSmoke test passed.")
