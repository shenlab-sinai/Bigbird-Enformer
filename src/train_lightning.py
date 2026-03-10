import os
import sys
import glob
import time
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import Metric
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models.enformer_plus import Enformer
from src.utils.config import EnformerConfig


# Performance monitor

class PerformanceMonitor(pl.Callback):
    def __init__(self, log_every=50):
        super().__init__()
        self.log_every = log_every
        self._t0 = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.log_every) != 0:
            return
        if trainer.is_global_zero is False:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - self._t0

        local_bs = int(batch["human"]["sequence"].shape[0]) * 2  # human + mouse
        world = int(trainer.world_size) if hasattr(trainer, "world_size") else 1
        thr_local = local_bs / (dt + 1e-8)
        thr_global = thr_local * world

        pl_module.log("perf/throughput_local",  thr_local,  on_step=True, prog_bar=True, sync_dist=False)
        pl_module.log("perf/throughput_global", thr_global, on_step=True, prog_bar=True, sync_dist=False)


# Loss & metrics 

def poisson_loss(pred_rate, target):
    pred_rate = pred_rate.clamp(min=1e-6)
    return F.poisson_nll_loss(pred_rate, target, log_input=False, full=False, reduction="mean")


class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable  = False
    full_state_update  = False
    higher_is_better   = True

    def __init__(self, n_channels: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims = (0, 1)
        self.add_state("product", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_channels), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.clamp(min=1e-6)
        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(target * target, dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(preds * preds, dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / (self.count + 1e-8)
        pred_mean = self.pred / (self.count + 1e-8)
        covariance = (
            self.product
            - true_mean * self.pred
            - pred_mean * self.true
            + self.count * true_mean * pred_mean
        )
        true_var = self.true_squared - self.count * (true_mean * true_mean)
        pred_var = self.pred_squared - self.count * (pred_mean * pred_mean)
        return covariance / (torch.sqrt(true_var) * torch.sqrt(pred_var) + 1e-8)


# Lightning module

class BigBirdLightningModule(pl.LightningModule):
    def __init__(self, model_config, lr=5e-4, warmup_steps=5000):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model = Enformer(model_config)

        self.lr  = float(lr)
        self.warmup_steps = int(warmup_steps)

        self.corr_human = MeanPearsonCorrCoefPerChannel(n_channels=5313)
        self.corr_mouse = MeanPearsonCorrCoefPerChannel(n_channels=1643)

        self.val_loss_sum = 0.0
        self.val_loss_n = 0

    def _forward_organism(self, seq, organism):
        return self.model(seq)[organism]

    # Training 

    def training_step(self, batch, batch_idx):
        seq_h = batch["human"]["sequence"]
        tgt_h = batch["human"]["target"].to(self.device, non_blocking=True)
        seq_m = batch["mouse"]["sequence"]
        tgt_m = batch["mouse"]["target"].to(self.device, non_blocking=True)

        pred_h = self._forward_organism(seq_h, "human")
        pred_m = self._forward_organism(seq_m, "mouse")

        loss_h = poisson_loss(pred_h, tgt_h)
        loss_m = poisson_loss(pred_m, tgt_m)
        loss = (loss_h + loss_m) / 2.0

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {self.global_step}, skipping update")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        self.log("train_loss_human", loss_h, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_mouse", loss_m, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        seq    = batch["sequence"]
        target = batch["target"].to(self.device, non_blocking=True)

        if dataloader_idx == 0:
            pred = self._forward_organism(seq, "human")
            loss = poisson_loss(pred, target)
            self.corr_human.update(preds=pred.detach(), target=target.detach())
        else:
            pred = self._forward_organism(seq, "mouse")
            loss = poisson_loss(pred, target)
            self.corr_mouse.update(preds=pred.detach(), target=target.detach())

        self.val_loss_sum += loss.item() * seq.size(0)
        self.val_loss_n   += seq.size(0)
        return loss

    def on_validation_epoch_end(self):
        corr_h   = self.corr_human.compute().nanmean()
        corr_m   = self.corr_mouse.compute().nanmean()
        val_corr = (corr_h + corr_m) / 2.0
        val_loss = torch.tensor(
            self.val_loss_sum / max(self.val_loss_n, 1),
            device=self.device
        )

        self.log("val_corr_coef",  val_corr, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_loss",       val_loss,  on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_corr_human", corr_h,   on_epoch=True, sync_dist=True)
        self.log("val_corr_mouse", corr_m,   on_epoch=True, sync_dist=True)

        self.corr_human.reset()
        self.corr_mouse.reset()
        self.val_loss_sum = 0.0
        self.val_loss_n   = 0
        torch.cuda.empty_cache()

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log("train_grad_norm", grad_norm, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,   # effectively starts from 0
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        # ConstantLR with factor=1.0 is a no-op — holds LR at target value
        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[self.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# Augmentation helpers

def random_shift(seq: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift a one-hot sequence [L, 4] by `shift` positions.
    Positive shift = shift right (pad left), negative = shift left (pad right).
    """
    L = seq.shape[0]
    if shift == 0:
        return seq
    pad = np.zeros((abs(shift), 4), dtype=seq.dtype)
    if shift > 0:
        return np.concatenate([pad, seq[:-shift]], axis=0)
    else:
        return np.concatenate([seq[-shift:], pad], axis=0)


def reverse_complement_seq(seq: np.ndarray) -> np.ndarray:
    """
    Reverse complement a one-hot sequence [L, 4].
    Columns are assumed to be [A, C, G, T].
    RC = reverse the sequence and swap A↔T (col 0↔3) and C↔G (col 1↔2).
    """
    return seq[::-1, ::-1].copy()


def augment_pair(seq: np.ndarray, target: np.ndarray, max_shift: int = 3):
    """
    Apply Enformer-style training augmentation:
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    seq = random_shift(seq, shift)

    if np.random.rand() < 0.5:
        seq    = reverse_complement_seq(seq)
        target = target[::-1].copy() 

    return seq, target


# Datasets

class SingleOrganismDataset(torch.utils.data.Dataset):
    """Validation/test dataset for a single organism. No augmentation."""
    def __init__(self, npz_dir, split):
        self.files = sorted(glob.glob(str(Path(npz_dir) / f"{split}-*.npz")))
        print(f"[{split}/{Path(npz_dir).name}] {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            "sequence": torch.tensor(data["sequence"], dtype=torch.float32),
            "target":   torch.tensor(data["target"],   dtype=torch.float32),
        }


class ZippedOrganismDataset(torch.utils.data.Dataset):
    """
    Zipped human+mouse training dataset matching the original Enformer training.
    Each item returns one human and one mouse sample, augmented independently.

    Augmentation applied per paper:
    
    Length = max(human, mouse) with wraparound on the shorter dataset.
    """
    def __init__(self, human_dir, mouse_dir, split, augment=True, max_shift=3):
        self.human_files = sorted(glob.glob(str(Path(human_dir) / f"{split}-*.npz")))
        self.mouse_files = sorted(glob.glob(str(Path(mouse_dir) / f"{split}-*.npz")))
        self.augment     = augment
        self.max_shift   = max_shift
        print(f"[{split}] human={len(self.human_files)}  mouse={len(self.mouse_files)}  augment={augment}")

    def __len__(self):
        return max(len(self.human_files), len(self.mouse_files))

    def __getitem__(self, idx):
        h_data = np.load(self.human_files[idx % len(self.human_files)])
        m_data = np.load(self.mouse_files[idx % len(self.mouse_files)])

        h_seq, h_tgt = h_data["sequence"], h_data["target"]
        m_seq, m_tgt = m_data["sequence"], m_data["target"]

        if self.augment:
            h_seq, h_tgt = augment_pair(h_seq, h_tgt, self.max_shift)
            m_seq, m_tgt = augment_pair(m_seq, m_tgt, self.max_shift)

        return {
            "human": {
                "sequence": torch.tensor(h_seq, dtype=torch.float32),
                "target":   torch.tensor(h_tgt, dtype=torch.float32),
            },
            "mouse": {
                "sequence": torch.tensor(m_seq, dtype=torch.float32),
                "target":   torch.tensor(m_tgt, dtype=torch.float32),
            },
        }


def zipped_collate_fn(batch):
    return {
        "human": {
            "sequence": torch.stack([b["human"]["sequence"] for b in batch]),
            "target":   torch.stack([b["human"]["target"]   for b in batch]),
        },
        "mouse": {
            "sequence": torch.stack([b["mouse"]["sequence"] for b in batch]),
            "target":   torch.stack([b["mouse"]["target"]   for b in batch]),
        },
    }


# DataModule 

class EnformerDataModule(pl.LightningDataModule):
    def __init__(self, human_npz_dir, mouse_npz_dir,
                 train_batch_size=8, val_batch_size=8, num_workers=4):
        super().__init__()
        self.human_npz_dir    = human_npz_dir
        self.mouse_npz_dir    = mouse_npz_dir
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size   = int(val_batch_size)
        self.num_workers      = int(num_workers)

    def setup(self, stage=None):
        self.train_ds  = ZippedOrganismDataset(
            self.human_npz_dir, self.mouse_npz_dir, "train",
            augment=True, max_shift=3,
        )
        self.val_human = SingleOrganismDataset(self.human_npz_dir, "valid")
        self.val_mouse = SingleOrganismDataset(self.mouse_npz_dir, "valid")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=zipped_collate_fn,
        )

    def val_dataloader(self):
        def make_loader(ds):
            return torch.utils.data.DataLoader(
                ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                persistent_workers=(self.num_workers > 0),
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        return [make_loader(self.val_human), make_loader(self.val_mouse)]


# Entry point 

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    HUMAN_NPZ_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human"
    MOUSE_NPZ_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/mouse"

    config = EnformerConfig(
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=896,
        attention_mode="bigbird",
        block_size=128,
        use_checkpointing=True,
        attn_dropout=0.05,   
        dropout_rate=0.4,    
        pos_dropout=0.01,   
    )

    dm = EnformerDataModule(
        human_npz_dir=HUMAN_NPZ_DIR,
        mouse_npz_dir=MOUSE_NPZ_DIR,
        train_batch_size=8,
        val_batch_size=8,
        num_workers=4,
    )

    model = BigBirdLightningModule(config, lr=5e-4, warmup_steps=5000)

    logger = TensorBoardLogger(
        save_dir="/hpc/users/hongw01/Bigbird-Enformer/tb_logs",
        name="bigbird_enformer",
        version="new_version",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_corr_coef",
        mode="max",
        save_top_k=1,
        filename="bigbird-best-{epoch:02d}-{val_corr_coef:.4f}",
        save_last=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_corr_coef",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor   = LearningRateMonitor(logging_interval="step")
    perf_monitor = PerformanceMonitor(log_every=50)

    strategy = DDPStrategy(gradient_as_bucket_view=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        num_nodes=2,
        strategy=strategy,
        max_steps=150_000,
        max_epochs=-1,      
        precision="bf16-mixed",
        gradient_clip_val=0.2,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, perf_monitor],
        sync_batchnorm=False,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, dm)