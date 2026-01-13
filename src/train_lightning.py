import os
import sys
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics import Metric
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


sys.path = ['/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/adams_playground/inactive/GTEx_Enformer_Improvement'] + sys.path
from gtex_code.data.enformer import create_enformer_npz_datasets

from src.models.enformer_plus import Enformer
from src.utils.config import EnformerConfig

class PerformanceMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.start_time
        
        local_batch_size = batch['sequence'].shape[0]
        local_throughput = local_batch_size / (batch_time + 1e-8)
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(pl_module.device) / (1024 ** 3)
        else:
            peak_mem = 0.0

        pl_module.log("perf/throughput", local_throughput, on_step=True, prog_bar=True, sync_dist=True, reduce_fx='sum')
        pl_module.log("perf/memory_gb", peak_mem, on_step=True, prog_bar=True, sync_dist=True, reduce_fx='max')

def poisson_loss(pred, target):
    return torch.nn.functional.poisson_nll_loss(pred, target, log_input=True, full=True)

class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        super().__init__()
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("true_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("pred_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_channels), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.nn.functional.softplus(preds)
        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count
        covariance = (self.product - true_mean * self.pred - pred_mean * self.true + self.count * true_mean * pred_mean)
        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        return covariance / (torch.sqrt(true_var) * torch.sqrt(pred_var) + 1e-8)


class BigBirdLightningModule(pl.LightningModule):
    def __init__(self, model_config, lr=2e-4, warmup_steps=2000):
        super().__init__()
        self.save_hyperparameters()
        self.model = Enformer(model_config)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=5313)

    def forward(self, x):
        return self.model(x)['human']

    def training_step(self, batch, batch_idx):
        seq, target = batch['sequence'], batch['target']
        pred = self(seq)
        loss = poisson_loss(pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, target = batch['sequence'], batch['target']
        pred = self(seq)
        loss = poisson_loss(pred, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.corr_coef.update(preds=pred, target=target)
        return loss

    def on_validation_epoch_end(self):
        avg_corr = self.corr_coef.compute().nanmean()
        self.log('val_corr_coef', avg_corr, on_epoch=True, sync_dist=True)
        self.corr_coef.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=1e-4, 
            betas=(0.9, 0.999),
            eps=1e-4
        )

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=self.warmup_steps
        )

        decay_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.estimated_stepping_batches - self.warmup_steps, 
            eta_min=1e-6 
        )

        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, decay_scheduler], 
            milestones=[self.warmup_steps] 
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }

class EnformerDataModule(pl.LightningDataModule):
    def __init__(self, npz_dir, batch_size=1):
        super().__init__()
        self.npz_dir = npz_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = create_enformer_npz_datasets(npz_dir=self.npz_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

if __name__ == '__main__':
    config = EnformerConfig(
        dim=1152, depth=11, heads=8, output_heads=dict(human=5313), target_length=896
    )

    NPZ_DIR = '/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human'
    dm = EnformerDataModule(npz_dir=NPZ_DIR, batch_size=1)
    
    model = BigBirdLightningModule(config, lr=2e-4)
    logger = TensorBoardLogger("tb_logs", name="Enformer_Original")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_corr_coef', mode='max', save_top_k=1,
        filename='bigbird-best-{epoch:02d}-{val_corr_coef:.4f}', save_last=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    perf_monitor = PerformanceMonitor()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        num_nodes=1,
        strategy='ddp',
        max_epochs=50,
        precision='16-mixed',
        gradient_clip_val=0.2,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, perf_monitor],
        sync_batchnorm=True,
        accumulate_grad_batches=16,
        log_every_n_steps=10
    )
    
    trainer.fit(model, dm)