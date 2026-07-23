import os
import sys
import glob
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torchmetrics import Metric
from pathlib import Path
import pandas as pd

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from bigbird_enformer.train_lightning import BigBirdLightningModule, SingleOrganismDataset
from bigbird_enformer.utils.config import EnformerConfig
from bigbird_enformer.utils.checkpoint_loading import load_atlas_checkpoint

# Configuration

USE_OFFICIAL = False

# CKPT_PATH = os.path.join(
#     project_root,
#     "tb_logs", "ATLAS-Ablation", "ablation-full-192k", "checkpoints",
#     "ccre_cls-best-epoch=139-val_corr_coef=0.6751.ckpt",
# )

CKPT_PATH = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/tb_logs/ATLAS-Ablation/ablation-full-k153/checkpoints/ccre_cls-best-epoch=133-val_corr_coef=0.6751.ckpt"

HUMAN_NPZ_DIR       = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human"
HUMAN_CCRE_MASK_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/ccre_mask/human"

CCRE_MASK_DIR = HUMAN_CCRE_MASK_DIR

run_version = Path(CKPT_PATH).parent.parent.name  
OUTPUT_DIR = os.path.join(project_root, "evaluation_results", run_version)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4

PLOT_TRACKS = {
    "DNASE:CD14 (ENCODE)":     45,
    "H3K27ac:HepG2 (ENCODE)": 694,
    "CAGE:liver":             4828,
}

ASSAY_SLICES = {
    "DNASE": slice(0, 674),
    "CHIP":  slice(674, 4675),
    "CAGE":  slice(4675, 5313),
}


# Dataset

class TestDataModule(pl.LightningDataModule):
    def __init__(self, npz_dir, batch_size=4, ccre_mask_dir=None):
        super().__init__()
        self.npz_dir       = npz_dir
        self.batch_size    = batch_size
        self.ccre_mask_dir = ccre_mask_dir   # None for non-ccre models

    def setup(self, stage=None):
        # Pass ccre_mask_dir so masks are loaded alongside sequences
        self.test_ds = SingleOrganismDataset(
            self.npz_dir,
            split="test",
            ccre_mask_dir=self.ccre_mask_dir,
        )
        has_masks = self.ccre_mask_dir is not None
        print(f"Test set: {len(self.test_ds)} sequences  ccre_masks={'yes' if has_masks else 'no'}")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


# Metric

class PearsonCorrPerChannel(Metric):
    full_state_update = False

    def __init__(self, n_channels: int = 5313):
        super().__init__()
        self.add_state("product",      default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_sum",     default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_sq_sum",  default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_sum",     default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_sq_sum",  default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("count",        default=torch.zeros(n_channels), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.product     += (preds * target).sum(dim=(0, 1))
        self.true_sum    += target.sum(dim=(0, 1))
        self.true_sq_sum += target.pow(2).sum(dim=(0, 1))
        self.pred_sum    += preds.sum(dim=(0, 1))
        self.pred_sq_sum += preds.pow(2).sum(dim=(0, 1))
        self.count       += torch.ones_like(target).sum(dim=(0, 1))

    def compute(self) -> torch.Tensor:
        eps      = 1e-8
        true_mean = self.true_sum / (self.count + eps)
        pred_mean = self.pred_sum / (self.count + eps)
        cov      = (self.product
                    - true_mean * self.pred_sum
                    - pred_mean * self.true_sum
                    + self.count * true_mean * pred_mean)
        true_var = self.true_sq_sum - self.count * true_mean.pow(2)
        pred_var = self.pred_sq_sum - self.count * pred_mean.pow(2)
        return cov / (true_var.sqrt() * pred_var.sqrt() + eps)


# Evaluation wrapper

class EvalWrapper(pl.LightningModule):
    def __init__(self, model, use_official: bool = False):
        super().__init__()
        self.use_official = use_official
        self.model        = model
        self.corr_metric  = PearsonCorrPerChannel(n_channels=5313)
        self._plotted     = False

    @torch.no_grad()
    def _predict(self, seq, ccre_mask=None):
        if self.use_official:
            out = self.model(seq)
            return out["human"] if isinstance(out, dict) else out
        else:
            if self.model.is_ccre_mode and self.model.use_classifier:
                pred, _ = self.model._forward_with_classifiers(seq, "human")
                return pred
            else:
                out = self.model.model(seq, is_global=ccre_mask)
                return out["human"]

    def test_step(self, batch, batch_idx):
        seq    = batch["sequence"]           # [B, L, 4]
        target = batch["target"]             # [B, 896, 5313]

        # Load ccre_mask if present in batch
        ccre_mask = batch.get("ccre_mask", None)
        if ccre_mask is not None:
            ccre_mask = ccre_mask.to(self.device)

        pred = self._predict(seq, ccre_mask=ccre_mask)   # [B, 896, 5313]
        self.corr_metric.update(pred, target)

        if not self._plotted:
            self._plot(pred, target)
            self._plotted = True

    def on_test_epoch_end(self):
        corr = self.corr_metric.compute().cpu().numpy()   # [5313]

        print("\n=== Test Results ===")
        print(f"{'Assay':<8} | {'Mean R':>8} | {'Median R':>10}")
        print("-" * 34)

        results = {}
        for name, sl in ASSAY_SLICES.items():
            subset   = corr[sl]
            mean_r   = float(np.nanmean(subset))
            median_r = float(np.nanmedian(subset))
            results[f"{name}_mean"]   = mean_r
            results[f"{name}_median"] = median_r
            print(f"{name:<8} | {mean_r:>8.4f} | {median_r:>10.4f}")

        global_mean        = float(np.nanmean(corr))
        results["GLOBAL_MEAN"] = global_mean
        print("-" * 34)
        print(f"GLOBAL   | {global_mean:>8.4f}")

        csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
        pd.Series(results, name="value").to_csv(csv_path, header=False)
        print(f"\nSaved results to: {csv_path}")

        self._plot_corr_distribution(corr)

    def _plot(self, pred, target):
        pred_np   = pred.detach().cpu().float().numpy()
        target_np = target.detach().cpu().float().numpy()
        n_tracks  = len(PLOT_TRACKS)
        fig, axes = plt.subplots(n_tracks, 1, figsize=(15, 4 * n_tracks), sharex=True)
        if n_tracks == 1:
            axes = [axes]

        for ax, (track_name, track_idx) in zip(axes, PLOT_TRACKS.items()):
            y_true = target_np[0, :, track_idx]
            y_pred = pred_np[0, :, track_idx]
            r, _   = pearsonr(y_true, y_pred)
            ax.fill_between(range(len(y_true)), y_true, color="black", alpha=0.25, label="Ground truth")
            ax.plot(y_true, color="black", lw=1, alpha=0.5)
            ax.plot(y_pred, color="tab:blue", lw=1.5, label=f"Prediction  R={r:.3f}")
            ax.set_title(f"{track_name}  (track {track_idx})", fontsize=11, loc="left", fontweight="bold")
            ax.set_ylabel("Signal", fontsize=9)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.2, linestyle="--")

        axes[-1].set_xlabel("Genomic bins (128 bp each)", fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "track_predictions.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved track plot: {out_path}")

    def _plot_corr_distribution(self, corr):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        colors = {"DNASE": "tab:blue", "CHIP": "tab:orange", "CAGE": "tab:green"}

        for ax, (name, sl) in zip(axes, ASSAY_SLICES.items()):
            subset = corr[sl]
            ax.hist(subset, bins=50, color=colors[name], alpha=0.8, edgecolor="white")
            ax.axvline(np.nanmean(subset),   color="red",   linestyle="--", label=f"mean={np.nanmean(subset):.3f}")
            ax.axvline(np.nanmedian(subset), color="black", linestyle=":",  label=f"median={np.nanmedian(subset):.3f}")
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Pearson R", fontsize=10)
            ax.set_ylabel("# tracks", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(-0.1, 1.0)

        plt.suptitle("Per-channel Pearson R distribution", fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "corr_distribution.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved distribution plot: {out_path}")


# Entry point

if __name__ == "__main__":
    if USE_OFFICIAL:
        from enformer_pytorch import from_pretrained
        OFFICIAL_ID = os.path.join(project_root, "Official_Enformer")
        print(f"Loading official Enformer from {OFFICIAL_ID}")
        base_model = load_atlas_checkpoint(CKPT_PATH, model_config=config)
        base_model.eval()

    else:
        assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
        print(f"Loading checkpoint: {CKPT_PATH}")

        config = EnformerConfig(
            dim=1536,
            depth=11,
            heads=8,
            output_heads=dict(human=5313, mouse=1643),
            target_length=896,
            attention_mode="ccre_bigbird",
            block_size=128,
            use_checkpointing=True,
            attn_dropout=0.05,
            dropout_rate=0.3,
            pos_dropout=0.01,
            use_rel_pe=False,
            use_einsum = True,
        )

        base_model = load_atlas_checkpoint(CKPT_PATH, model_config=config)
        base_model.eval()

    eval_module = EvalWrapper(base_model, use_official=USE_OFFICIAL)

    dm = TestDataModule(
        HUMAN_NPZ_DIR,
        batch_size=BATCH_SIZE,
        ccre_mask_dir=CCRE_MASK_DIR)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.test(eval_module, datamodule=dm)
