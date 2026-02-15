import os
import sys
import time
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import Metric
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

sys.path = ['/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/adams_playground/inactive/GTEx_Enformer_Improvement'] + sys.path
from gtex_code.data.enformer import create_enformer_npz_datasets

from src.models.enformer_plus import Enformer
from src.utils.config import EnformerConfig


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
            return  # avoid 8x duplicate logs

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - self._t0

        local_bs = int(batch["sequence"].shape[0])
        world = int(trainer.world_size) if hasattr(trainer, "world_size") else 1

        thr_local = local_bs / (dt + 1e-8)
        thr_global = thr_local * world

        pl_module.log("perf/throughput_local", thr_local, on_step=True, prog_bar=True, sync_dist=False)
        pl_module.log("perf/throughput_global", thr_global, on_step=True, prog_bar=True, sync_dist=False)



def poisson_loss(pred_rate, target):
    pred_rate = pred_rate.clamp(min=1e-6)
    return F.poisson_nll_loss(pred_rate, target, log_input=False, full=False, reduction="mean")


class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = True

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


class BigBirdLightningModule(pl.LightningModule):
    def __init__(self, model_config, lr=3e-4, warmup_steps=2000, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model = Enformer(model_config)

        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.weight_decay = float(weight_decay)

        self.corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=5313)

    def forward(self, x):
        return self.model(x)["human"]

    def training_step(self, batch, batch_idx):
        seq, target = batch["sequence"], batch["target"]
        pred = self(seq)
        target = target.to(pred.device, non_blocking=True)
        if torch.isnan(pred).any():
            print(f"WARNING: NaN in predictions at step {self.global_step}")
            return None
        
        loss = poisson_loss(pred, target)
        
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {self.global_step}")
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, target = batch["sequence"], batch["target"]
        pred = self(seq)
        target = target.to(pred.device, non_blocking=True)
        loss = poisson_loss(pred, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.corr_coef.update(preds=pred.detach(), target=target.detach())
        return loss

    def on_validation_epoch_end(self):
        avg_corr = self.corr_coef.compute().nanmean()
        self.log("val_corr_coef", avg_corr, on_epoch=True, sync_dist=True)
        self.corr_coef.reset()

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm=float('inf')
        )
        self.log("train_grad_norm", grad_norm, prog_bar=False)

    def configure_optimizers(self):
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 60000))
        warmup_steps = min(self.warmup_steps, max(1, total_steps // 2))

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        decay_steps = max(1, total_steps - warmup_steps)
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }


class EnformerDataModule(pl.LightningDataModule):
    def __init__(self, npz_dir, train_batch_size=8, val_batch_size=4, num_workers=2):
        super().__init__()
        self.npz_dir = npz_dir
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = create_enformer_npz_datasets(npz_dir=self.npz_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True # Set to false to free up memory

    config = EnformerConfig(
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313),
        target_length=896,
        use_checkpointing=True,
        num_chunks=4,
        attention_mode="hierarchical",
        block_size=64,
        attn_dropout=0.1,
    )

    NPZ_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human"
    dm = EnformerDataModule(npz_dir=NPZ_DIR, train_batch_size=8, val_batch_size=8, num_workers=4)
    model = BigBirdLightningModule(config, lr=5e-4, warmup_steps=5000, weight_decay=1e-7)
    logger = TensorBoardLogger("tb_logs", name="hierarchical_enformer")

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

    lr_monitor = LearningRateMonitor(logging_interval="step")
    perf_monitor = PerformanceMonitor(log_every=50)

    strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        num_nodes=2,
        strategy=strategy,
        max_epochs=70,                    
        precision="bf16-mixed",
        gradient_clip_val=0.2,              
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, perf_monitor],
        #callbacks=[early_stop_callback, lr_monitor, perf_monitor],
        sync_batchnorm=False,
        accumulate_grad_batches=2,          
        log_every_n_steps=10,
        enable_model_summary=False,
    )

    trainer.fit(model, dm)
