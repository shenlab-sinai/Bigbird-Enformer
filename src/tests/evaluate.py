import os
import sys
import torch
import pytorch_lightning as pl
from torchmetrics import Metric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
sys.path.insert(0, project_root)

# Keep your specific path insertion
sys.path = ['/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/adams_playground/inactive/GTEx_Enformer_Improvement'] + sys.path

from gtex_code.data.enformer import create_enformer_npz_datasets
from src.train_lightning import BigBirdLightningModule
from src.utils.config import EnformerConfig

# --- CONFIGURATION ---
CKPT_FILENAME = "bigbird-best-epoch=27-val_corr_coef=0.5690.ckpt"
CKPT_PATH = os.path.join(project_root, "tb_logs", "bigbird_a100", "version_3", "checkpoints", CKPT_FILENAME)
DATA_DIR = '/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human'
OUTPUT_DIR = os.path.join(project_root, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EnformerDataModule(pl.LightningDataModule):
    def __init__(self, npz_dir, batch_size=1):
        super().__init__()
        self.npz_dir = npz_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = create_enformer_npz_datasets(npz_dir=self.npz_dir)
        print(f"Test Set Size: {len(self.test_ds)} samples (Chromosomes 8 & 13)")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )


class EnformerPearsonCorrCoef(Metric):
    """
    Calculates Pearson R exactly as described in the Enformer Paper methods.
    Critically: Applies Softplus to predictions before correlation.
    """
    full_state_update = False
    
    def __init__(self, n_channels=5313):
        super().__init__()
        self.add_state("product", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_sum", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_squared_sum", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_sum", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_squared_sum", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_channels), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.nn.functional.softplus(preds)
    
        self.product += torch.sum(preds * target, dim=(0, 1))
        self.true_sum += torch.sum(target, dim=(0, 1))
        self.true_squared_sum += torch.sum(torch.square(target), dim=(0, 1))
        self.pred_sum += torch.sum(preds, dim=(0, 1))
        self.pred_squared_sum += torch.sum(torch.square(preds), dim=(0, 1))
        self.count += torch.sum(torch.ones_like(target), dim=(0, 1))

    def compute(self):
        true_mean = self.true_sum / self.count
        pred_mean = self.pred_sum / self.count
        
        covariance = (self.product 
                      - true_mean * self.pred_sum 
                      - pred_mean * self.true_sum 
                      + self.count * true_mean * pred_mean)
        
        true_var = self.true_squared_sum - self.count * torch.square(true_mean)
        pred_var = self.pred_squared_sum - self.count * torch.square(pred_mean)
        
        # Add epsilon to prevent div by zero
        correlation = covariance / (torch.sqrt(true_var) * torch.sqrt(pred_var) + 1e-8)
        return correlation


class TestWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.corr_metric = EnformerPearsonCorrCoef()

    def test_step(self, batch, batch_idx):
        seq = batch['sequence']
        target = batch['target']
        
        with torch.no_grad():
            output = self.model(seq)
            
            if isinstance(output, dict):
                pred = output['human']
            else:
                pred = output
        
        self.corr_metric.update(pred, target)

        if batch_idx == 0:
            self.plot_batch(pred, target)

    def plot_batch(self, pred, target):
        
        pred_counts = torch.nn.functional.softplus(pred)
        
        pred_np = pred_counts.cpu().numpy()
        target_np = target.cpu().numpy()
        
        tracks = {
            "DNase (Open Chromatin)": 45,    # Early index = DNase
            "H3K27ac (Enhancer)": 700,       # Middle index = Histone/ChIP
            "CAGE (Gene Expression)": 5111   # Late index = CAGE
        }
        
        sample_idx = 0  # Plot the first sequence in the batch
        num_tracks = len(tracks)

        fig, axes = plt.subplots(num_tracks, 1, figsize=(15, 4 * num_tracks), sharex=True)
        if num_tracks == 1: axes = [axes]
        
        for i, (track_name, track_id) in enumerate(tracks.items()):
            ax = axes[i]
            
            y_true = target_np[sample_idx, :, track_id]
            y_pred = pred_np[sample_idx, :, track_id]
            
            r, _ = pearsonr(y_true, y_pred)
            
  
            ax.fill_between(range(len(y_true)), y_true, color='black', alpha=0.3, label='Experiment')
            ax.plot(y_true, color='black', lw=1, alpha=0.5)
            
            ax.plot(y_pred, color='tab:blue', lw=2, label=f'Prediction (R={r:.3f})')
            
            ax.set_ylabel("Signal (Counts)", fontsize=10)
            ax.set_title(f"{track_name} | Track {track_id}", fontsize=12, loc='left', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.2, linestyle='--')
            

        plt.xlabel("Genomic Bins (128bp)", fontsize=12)
        plt.tight_layout()
        
        # Save
        filename = os.path.join(OUTPUT_DIR, f"paper_style_plot_sample{sample_idx}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved paper-style plot: {filename}")

    def on_test_epoch_end(self):

        final_corr_tensor = self.corr_metric.compute()
        final_corr_np = final_corr_tensor.cpu().numpy()

        slices = {
            "DNASE": slice(0, 674),
            "CHIP": slice(674, 4675),
            "CAGE": slice(4675, 5313)
        }
        
        results = {}
        print("\n=== Official Test Results by Assay ===")
        print(f"{'Assay':<10} | {'Mean R':<10} | {'Median R':<10}")
        print("-" * 35)
        
        for name, s in slices.items():
            subset = final_corr_np[s]
            avg = np.nanmean(subset)
            med = np.nanmedian(subset)
            results[f"{name}_mean"] = avg
            results[f"{name}_median"] = med
            print(f"{name:<10} | {avg:.4f}     | {med:.4f}")
            

        results["GLOBAL_MEAN"] = np.nanmean(final_corr_np)        
        print("-" * 35)
        print(f"GLOBAL MEAN: {results['GLOBAL_MEAN']:.4f}")
 
        csv_path = os.path.join(OUTPUT_DIR, "evaluation_breakdown_official.csv")
        df = pd.DataFrame([results])
        df.to_csv(csv_path, index=False)
        print(f"\nSaved official breakdown to: {csv_path}")


if __name__ == "__main__":
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        sys.exit(1)

    print(f"Loading Model from: {CKPT_FILENAME}")
    
    base_model = BigBirdLightningModule.load_from_checkpoint(CKPT_PATH, weights_only=False)
    test_module = TestWrapper(base_model)
    
    print(f"Loading Data from: {DATA_DIR}")
    dm = EnformerDataModule(DATA_DIR, batch_size=4)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    trainer.test(test_module, datamodule=dm)