import os
import glob
import time
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision, BinaryPrecisionRecallCurve
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .models.enformer_plus import Enformer, SEQUENCE_LENGTH
from .utils.config import EnformerConfig, load_experiment_config

#  Performance monitor

class PerformanceMonitor(pl.Callback):
    def __init__(self, log_every=50):
        super().__init__()
        self.log_every = log_every
        self._step_t0 = None
        self._train_start = None
        self._epoch_start = None
        self._epoch_transformer_ms = []

    def on_train_start(self, trainer, pl_module):
        self._train_start = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()
        self._epoch_transformer_ms = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step % self.log_every == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._step_t0 = time.perf_counter()
            pl_module._time_this_step = True
        else:
            self._step_t0 = None
            pl_module._time_this_step = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._step_t0 is None or not trainer.is_global_zero:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - self._step_t0) * 1000
        local_bs = int(batch["human"]["sequence"].shape[0]) * 2
        samples_per_sec = local_bs / (step_ms / 1000 + 1e-8)
        cum_hours = (time.time() - self._train_start) / 3600
        pl_module.log("time/step_ms", step_ms, on_step=True, prog_bar=True, sync_dist=False)
        pl_module.log("time/samples_per_sec", samples_per_sec, on_step=True, prog_bar=False, sync_dist=False)
        pl_module.log("time/cumulative_hours", cum_hours, on_step=True, prog_bar=False, sync_dist=False)
        if getattr(pl_module, "_last_transformer_ms", None) is not None:
            self._epoch_transformer_ms.append(pl_module._last_transformer_ms)
            pl_module._last_transformer_ms = None

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if self._epoch_start is not None:
            pl_module.log("time/epoch_minutes",
                          (time.time() - self._epoch_start) / 60,
                          on_epoch=True, prog_bar=False, sync_dist=False)
        if self._epoch_transformer_ms:
            arr = np.array(self._epoch_transformer_ms)
            skip = min(5, len(arr) // 2)
            steady = arr[skip:] if len(arr) > skip else arr
            avg_ms = float(steady.mean())
            pl_module.log("time/transformer_only_ms", avg_ms, on_epoch=True, prog_bar=False, sync_dist=False)
            pl_module.log("time/per_block_ms", avg_ms / 11.0, on_epoch=True, prog_bar=False, sync_dist=False)


def poisson_loss(pred_rate, target):
    pred_rate = pred_rate.clamp(min=1e-6)
    return F.poisson_nll_loss(pred_rate, target, log_input=False, full=False, reduction="mean")

class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = True

    def __init__(self, n_channels, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims = (0, 1)
        for name in ("product", "true", "true_squared", "pred", "pred_squared", "count"):
            self.add_state(name, default=torch.zeros(n_channels), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds, target):
        preds = preds.clamp(min=1e-6)
        self.product += (preds * target).sum(dim=self.reduce_dims)
        self.true += target.sum(dim=self.reduce_dims)
        self.true_squared += (target * target).sum(dim=self.reduce_dims)
        self.pred += preds.sum(dim=self.reduce_dims)
        self.pred_squared += (preds * preds).sum(dim=self.reduce_dims)
        self.count += torch.ones_like(target).sum(dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / (self.count + 1e-8)
        pred_mean = self.pred / (self.count + 1e-8)
        covariance = (
            self.product
            - true_mean * self.pred
            - pred_mean * self.true
            + self.count * true_mean * pred_mean
        )
        true_var = self.true_squared - self.count * true_mean * true_mean
        pred_var = self.pred_squared - self.count * pred_mean * pred_mean
        return covariance / (torch.sqrt(true_var) * torch.sqrt(pred_var) + 1e-8)

class CCREClassifierHead(nn.Module):
    def __init__(self, dim, hidden_dims=(384, 96), dropout=0.1):
        super().__init__()
        if isinstance(hidden_dims, int):
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dims),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims, 1),
            )
        else:
            h1, h2 = hidden_dims
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, h1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h2, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def topk_mask(self, x, k):
        logits = self.forward(x)
        k = max(1, min(k, x.shape[1]))
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, logits.topk(k, dim=1).indices, True)
        return mask


class BigBirdLightningModule(pl.LightningModule):
    """
    Enformer + BigBird sparse attention trained with cCRE global tokens.

    classifier_mode="progressive"
        One classifier per every `classifier_every` transformer blocks.
        Each classifier predicts the attention mask for its own block group
        on-the-fly during the forward pass.
    """

    def __init__(
        self,
        model_config,
        lr=5e-4,
        warmup_steps=5000,
        use_classifier=True,
        classifier_hidden_dims=(384, 96),
        classifier_dropout=0.1,
        bce_weight=0.1,
        mean_ccre_k=None,
        ccre_condition=None,
        classifier_every=3,
        mix_steps=100_000,
        weight_decay=1e-4,
        classifier_mode="progressive",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        if classifier_mode != "progressive":
            raise ValueError(
                "classifier_mode must be 'progressive'; cached mode was removed "
                "because its logits were not keyed by sequence identity, got "
                f"{classifier_mode!r}"
            )
        assert classifier_every >= 1, "classifier_every must be >= 1"

        self.model = Enformer(model_config)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.is_ccre_mode = (model_config.attention_mode == "ccre_bigbird")
        self.use_classifier = use_classifier and self.is_ccre_mode
        self.bce_weight = float(bce_weight)
        self.mean_ccre_k = (
            None if mean_ccre_k is None else int(mean_ccre_k)
        )
        self.ccre_condition = ccre_condition
        if self.use_classifier and (
            self.mean_ccre_k is None or self.mean_ccre_k < 1
        ):
            raise ValueError(
                "mean_ccre_k must be a positive user-configured value "
                "when the cCRE classifier is enabled"
            )
        self.classifier_every = int(classifier_every)
        self.mix_steps = int(mix_steps)
        self.weight_decay = float(weight_decay)
        self.classifier_mode = classifier_mode
        self.depth = int(model_config.depth)

        self.n_classifiers = (
            self.depth + self.classifier_every - 1
        ) // self.classifier_every

        if self.use_classifier:
            self.classifiers = nn.ModuleList([
                CCREClassifierHead(
                    dim=model_config.dim,
                    hidden_dims=classifier_hidden_dims,
                    dropout=classifier_dropout,
                )
                for _ in range(self.n_classifiers)
            ])
            self.val_auprc = nn.ModuleList([
                BinaryAveragePrecision() for _ in range(self.n_classifiers)
            ])
            self.val_pr = nn.ModuleList([
                BinaryPrecisionRecallCurve() for _ in range(self.n_classifiers)
            ])
            print(
                f"[BigBird] {self.n_classifiers} classifier(s)  "
                f"mode={classifier_mode}  every={classifier_every}  depth={self.depth}"
            )

        self._time_this_step = False
        self._last_transformer_ms = None
        self._transformer_ms_accum = []

        self.corr_human = MeanPearsonCorrCoefPerChannel(n_channels=5313)
        self.corr_mouse = MeanPearsonCorrCoefPerChannel(n_channels=1643)
        self.val_loss_sum = 0.0
        self.val_loss_n = 0

    def _build_topk_mask(self, logits):
        B, N = logits.shape
        k = max(1, min(self.mean_ccre_k, N))
        mask = torch.zeros(B, N, dtype=torch.bool, device=logits.device)
        mask.scatter_(1, logits.topk(k, dim=1).indices, True)
        return mask

    def _forward_organism(self, seq, organism, ccre_mask=None):
        do_time = self._time_this_step and torch.cuda.is_available()
        if do_time:
            self.model._time_transformer = True

        out = self.model(seq, is_global=ccre_mask)[organism]

        if do_time:
            self.model._time_transformer = False
            if self.model._transformer_ms is not None:
                self._transformer_ms_accum.append(self.model._transformer_ms)
                self.model._transformer_ms = None
        return out

    def _forward_progressive(self, seq, organism, gt_mask=None):
        encoded = self.model._encode_ccre(seq)
        x = encoded
        all_logits = []
        do_time = self._time_this_step and torch.cuda.is_available()
        transformer_total = 0.0

        if self.training and gt_mask is not None:
            mix_ratio = min(self.global_step / self.mix_steps, 0.8)
            gt_logits = gt_mask.float() * 20.0 - 10.0
        else:
            mix_ratio = 1.0
            gt_logits = None

        for i, classifier in enumerate(self.classifiers):
            logits = classifier(x.detach())
            all_logits.append(logits)

            if self.training and gt_logits is not None:
                clamped = logits.clamp(-10.0, 10.0)
                mixed = (
                    (1 - mix_ratio) * gt_logits + mix_ratio * clamped
                    if clamped.shape == gt_logits.shape
                    else clamped
                )
            else:
                mixed = logits

            attn_mask = self._build_topk_mask(mixed)

            if do_time:
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            start = i * self.classifier_every
            end = min(start + self.classifier_every, self.depth)
            x = self.model._run_transformer_group(x, start, end, is_global=attn_mask)

            if do_time:
                torch.cuda.synchronize()
                transformer_total += (time.perf_counter() - t0) * 1000

        if do_time:
            self._transformer_ms_accum.append(transformer_total)

        pred = self.model._heads[organism](self.model._crop_and_pointwise(x))
        return pred, all_logits

    def training_step(self, batch, batch_idx):
        self._transformer_ms_accum = []

        seq_h = batch["human"]["sequence"]
        tgt_h = batch["human"]["target"].to(self.device, non_blocking=True)
        seq_m = batch["mouse"]["sequence"]
        tgt_m = batch["mouse"]["target"].to(self.device, non_blocking=True)

        if self.is_ccre_mode and self.use_classifier:
            mask_h = batch["human"]["ccre_mask"].to(self.device, non_blocking=True)
            mask_m = batch["mouse"]["ccre_mask"].to(self.device, non_blocking=True)

            pred_h, logits_h = self._forward_progressive(
                seq_h, "human", gt_mask=mask_h
            )
            pred_m, logits_m = self._forward_progressive(
                seq_m, "mouse", gt_mask=mask_m
            )

            loss_h = poisson_loss(pred_h, tgt_h)
            loss_m = poisson_loss(pred_m, tgt_m)
            enformer_loss = (loss_h + loss_m) / 2.0

            pos_frac_h = mask_h.float().mean().clamp(0.01, 0.99)
            pos_frac_m = mask_m.float().mean().clamp(0.01, 0.99)
            pos_w_h = (1 - pos_frac_h) / pos_frac_h
            pos_w_m = (1 - pos_frac_m) / pos_frac_m

            bce_h_per_cls = [
                F.binary_cross_entropy_with_logits(lg, mask_h.float(), pos_weight=pos_w_h)
                for lg in logits_h
            ]
            bce_m_per_cls = [
                F.binary_cross_entropy_with_logits(lg, mask_m.float(), pos_weight=pos_w_m)
                for lg in logits_m
            ]
            bce_h = sum(bce_h_per_cls) / len(bce_h_per_cls)
            bce_m = sum(bce_m_per_cls) / len(bce_m_per_cls)
            bce_loss = (bce_h + bce_m) / 2.0
            total_loss = enformer_loss + self.bce_weight * bce_loss

            self.log("train_bce_loss", bce_loss, on_step=True, prog_bar=False, sync_dist=False)
            self.log("train_mix_ratio",
                     min(self.global_step / self.mix_steps, 0.8),
                     on_step=True, prog_bar=False, sync_dist=False)
            for i, (bh, bm) in enumerate(zip(bce_h_per_cls, bce_m_per_cls)):
                self.log(f"train_bce_cls{i}", (bh + bm) / 2.0,
                         on_step=True, prog_bar=False, sync_dist=False)

        else:
            mask_h = batch["human"].get("ccre_mask", None)
            mask_m = batch["mouse"].get("ccre_mask", None)
            if mask_h is not None:
                mask_h = mask_h.to(self.device, non_blocking=True)
            if mask_m is not None:
                mask_m = mask_m.to(self.device, non_blocking=True)

            pred_h = self._forward_organism(seq_h, "human", ccre_mask=mask_h)
            pred_m = self._forward_organism(seq_m, "mouse", ccre_mask=mask_m)
            loss_h = poisson_loss(pred_h, tgt_h)
            loss_m = poisson_loss(pred_m, tgt_m)
            enformer_loss = (loss_h + loss_m) / 2.0
            total_loss = enformer_loss

        if self._transformer_ms_accum:
            self._last_transformer_ms = float(np.mean(self._transformer_ms_accum))

        if torch.isnan(total_loss):
            print(f"WARNING: NaN loss at step {self.global_step}, skipping optimizer step")
            return None

        self.log("train_loss", enformer_loss, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        self.log("train_loss_human", loss_h, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_mouse", loss_m, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train_loss_epoch", enformer_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        seq = batch["sequence"]
        target = batch["target"].to(self.device, non_blocking=True)
        organism = "human" if dataloader_idx == 0 else "mouse"

        if self.is_ccre_mode and self.use_classifier:
            pred, all_logits = self._forward_progressive(seq, organism)
            gt_mask = batch.get("ccre_mask", None)
            if gt_mask is not None and dataloader_idx == 0:
                gt_mask = gt_mask.to(self.device, non_blocking=True)
                labels_flat = gt_mask.flatten().int()
                for i, logits in enumerate(all_logits):
                    preds_flat = logits.detach().flatten().float()
                    self.val_auprc[i].update(preds_flat, labels_flat)
                    self.val_pr[i].update(preds_flat, labels_flat)
        else:
            mask = batch.get("ccre_mask", None)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            pred = self._forward_organism(seq, organism, ccre_mask=mask)

        loss = poisson_loss(pred, target)
        if dataloader_idx == 0:
            self.corr_human.update(preds=pred.detach(), target=target.detach())
        else:
            self.corr_mouse.update(preds=pred.detach(), target=target.detach())
        self.val_loss_sum += loss.item() * seq.size(0)
        self.val_loss_n += seq.size(0)
        return loss

    def on_validation_epoch_end(self):
        corr_h = self.corr_human.compute().nanmean()
        corr_m = self.corr_mouse.compute().nanmean()
        val_corr = (corr_h + corr_m) / 2.0
        val_loss = torch.tensor(
            self.val_loss_sum / max(self.val_loss_n, 1),
            device=self.device,
        )
        self.log("val_corr_coef", val_corr, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_corr_human", corr_h, on_epoch=True, sync_dist=True)
        self.log("val_corr_mouse", corr_m, on_epoch=True, sync_dist=True)

        if self.is_ccre_mode and self.use_classifier:
            self._log_pr_auprc()

        self.corr_human.reset()
        self.corr_mouse.reset()
        self.val_loss_sum = 0.0
        self.val_loss_n = 0
        torch.cuda.empty_cache()

    def _log_pr_auprc(self):
        auprcs = []
        curves = []

        for i in range(self.n_classifiers):
            try:
                auprc = self.val_auprc[i].compute()
                p, r, _ = self.val_pr[i].compute()
                auprcs.append(float(auprc.item()))
                self.log(f"val_cls{i}_auprc", auprc, on_epoch=True, sync_dist=False,
                         prog_bar=(i == self.n_classifiers - 1))
                curves.append((r.detach().cpu().numpy(), p.detach().cpu().numpy()))
            except Exception as e:
                print(f"WARNING: Could not compute PR for classifier {i}: {e}")
                auprcs.append(float("nan"))
                curves.append(None)
            self.val_auprc[i].reset()
            self.val_pr[i].reset()

        valid_auprcs = [a for a in auprcs if not (a != a)]
        if valid_auprcs:
            mean_auprc = float(np.mean(valid_auprcs))
            self.log("val_auprc_mean", mean_auprc, on_epoch=True, sync_dist=False, prog_bar=True)

        if self.trainer.is_global_zero and self.logger is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(self.n_classifiers, 2)))
            for i, curve in enumerate(curves):
                if curve is None:
                    continue
                recall, precision = curve
                if len(recall) > 2000:
                    idx = np.linspace(0, len(recall) - 1, 2000).astype(int)
                    recall, precision = recall[idx], precision[idx]
                ax.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f"Classifier {i} (AUPRC={auprcs[i]:.3f})")
            ax.set_xlabel("Recall", fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_title(f"cCRE Classifier PR Curves  (step {self.global_step})",
                         fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left", fontsize=10)
            fig.tight_layout()
            try:
                self.logger.experiment.add_figure(
                    "val_pr_curves", fig, global_step=self.global_step
                )
            except Exception as e:
                print(f"WARNING: Could not log PR figure to TensorBoard: {e}")
            plt.close(fig)

    def on_before_optimizer_step(self, optimizer):
        has_nan_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad.zero_()
                has_nan_grad = True
        if has_nan_grad:
            print(f"WARNING: NaN gradient detected at step {self.global_step}, zeroed")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.2)
        self.log("train_grad_norm", grad_norm, prog_bar=False, sync_dist=False)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
                    or "rel_content_bias" in name or "rel_pos_bias" in name):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.trainer.max_steps if self.trainer is not None else 150_000
        cosine_steps = max(1, total_steps - self.warmup_steps)

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=self.lr * 0.1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

def random_shift(seq, shift):
    if shift == 0:
        return seq
    pad = np.zeros((abs(shift), 4), dtype=seq.dtype)
    if shift > 0:
        return np.concatenate([pad, seq[:-shift]], axis=0)
    else:
        return np.concatenate([seq[-shift:], pad], axis=0)


def reverse_complement_seq(seq):
    return seq[::-1, ::-1].copy()


def augment_pair(seq, target, ccre_mask=None, max_shift=3):
    shift = np.random.randint(-max_shift, max_shift + 1)
    seq = random_shift(seq, shift)
    if np.random.rand() < 0.5:
        seq = reverse_complement_seq(seq)
        target = target[::-1].copy()
        if ccre_mask is not None:
            ccre_mask = ccre_mask[::-1].copy()
    return seq, target, ccre_mask


def _mask_path(mask_dir, npz_path):
    return mask_dir / Path(npz_path).with_suffix(".npy").name


def _validate_mask_files(
    files,
    mask_dir,
    dataset_name,
    expected_mask_length=None,
):
    if mask_dir is None:
        return
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"cCRE mask directory does not exist for {dataset_name}: {mask_dir}"
        )

    mask_paths = [_mask_path(mask_dir, npz_path) for npz_path in files]
    missing = [path for path in mask_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        remainder = len(missing) - 5
        if remainder > 0:
            preview += f", ... and {remainder} more"
        raise FileNotFoundError(
            f"missing {len(missing)} cCRE mask file(s) for {dataset_name}: "
            f"{preview}"
        )

    validated_length = expected_mask_length
    invalid = []
    for mask_path in mask_paths:
        try:
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            invalid.append(f"{mask_path}: cannot load ({exc})")
            continue

        if mask.ndim != 1:
            invalid.append(
                f"{mask_path}: expected one dimension, got shape {mask.shape}"
            )
            continue
        if mask.dtype != np.bool_:
            invalid.append(
                f"{mask_path}: expected Boolean dtype, got {mask.dtype}"
            )
            continue
        if mask.shape[0] == 0:
            invalid.append(f"{mask_path}: mask must not be empty")
            continue

        if validated_length is None:
            validated_length = mask.shape[0]
        elif mask.shape[0] != validated_length:
            invalid.append(
                f"{mask_path}: expected length {validated_length}, "
                f"got {mask.shape[0]}"
            )

    if invalid:
        preview = "; ".join(invalid[:5])
        remainder = len(invalid) - 5
        if remainder > 0:
            preview += f"; ... and {remainder} more"
        raise ValueError(
            f"invalid cCRE mask file(s) for {dataset_name}: {preview}"
        )


class SingleOrganismDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        npz_dir,
        split,
        ccre_mask_dir=None,
        expected_mask_length=None,
    ):
        self.files = sorted(glob.glob(str(Path(npz_dir) / f"{split}-*.npz")))
        self.ccre_mask_dir = Path(ccre_mask_dir) if ccre_mask_dir else None
        _validate_mask_files(
            self.files,
            self.ccre_mask_dir,
            f"{split}/{Path(npz_dir).name}",
            expected_mask_length,
        )
        print(f"[{split}/{Path(npz_dir).name}] {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        item = {
            "sequence": torch.tensor(data["sequence"], dtype=torch.float32),
            "target": torch.tensor(data["target"], dtype=torch.float32),
        }
        if self.ccre_mask_dir is not None:
            mask_path = _mask_path(self.ccre_mask_dir, self.files[idx])
            item["ccre_mask"] = torch.from_numpy(
                np.load(mask_path, allow_pickle=False)
            )
        return item


class ZippedOrganismDataset(torch.utils.data.Dataset):
    def __init__(self, human_dir, mouse_dir, split,
                 human_ccre_mask_dir=None, mouse_ccre_mask_dir=None,
                 augment=True, max_shift=3, expected_mask_length=None):
        self.human_files = sorted(glob.glob(str(Path(human_dir) / f"{split}-*.npz")))
        self.mouse_files = sorted(glob.glob(str(Path(mouse_dir) / f"{split}-*.npz")))
        self.augment = augment
        self.max_shift = max_shift
        self.human_mask_dir = Path(human_ccre_mask_dir) if human_ccre_mask_dir else None
        self.mouse_mask_dir = Path(mouse_ccre_mask_dir) if mouse_ccre_mask_dir else None
        _validate_mask_files(
            self.human_files,
            self.human_mask_dir,
            f"{split}/{Path(human_dir).name}",
            expected_mask_length,
        )
        _validate_mask_files(
            self.mouse_files,
            self.mouse_mask_dir,
            f"{split}/{Path(mouse_dir).name}",
            expected_mask_length,
        )
        print(
            f"[{split}] human={len(self.human_files)}  mouse={len(self.mouse_files)}  "
            f"augment={augment}  "
            f"human_masks={'yes' if self.human_mask_dir else 'no'}  "
            f"mouse_masks={'yes' if self.mouse_mask_dir else 'no'}"
        )

    def __len__(self):
        return max(len(self.human_files), len(self.mouse_files))

    def _load_mask(self, mask_dir, npz_path):
        if mask_dir is None:
            return None
        return np.load(
            _mask_path(mask_dir, npz_path),
            allow_pickle=False,
        )

    def __getitem__(self, idx):
        h_path = self.human_files[idx % len(self.human_files)]
        m_path = self.mouse_files[idx % len(self.mouse_files)]

        h_data = np.load(h_path)
        m_data = np.load(m_path)

        h_seq, h_tgt = h_data["sequence"], h_data["target"]
        m_seq, m_tgt = m_data["sequence"], m_data["target"]
        h_mask = self._load_mask(self.human_mask_dir, h_path)
        m_mask = self._load_mask(self.mouse_mask_dir, m_path)

        if self.augment:
            h_seq, h_tgt, h_mask = augment_pair(h_seq, h_tgt, h_mask, self.max_shift)
            m_seq, m_tgt, m_mask = augment_pair(m_seq, m_tgt, m_mask, self.max_shift)

        human_item = {
            "sequence": torch.tensor(h_seq, dtype=torch.float32),
            "target": torch.tensor(h_tgt, dtype=torch.float32),
        }
        mouse_item = {
            "sequence": torch.tensor(m_seq, dtype=torch.float32),
            "target": torch.tensor(m_tgt, dtype=torch.float32),
        }
        if h_mask is not None:
            human_item["ccre_mask"] = torch.tensor(h_mask, dtype=torch.bool)
        if m_mask is not None:
            mouse_item["ccre_mask"] = torch.tensor(m_mask, dtype=torch.bool)

        return {"human": human_item, "mouse": mouse_item}


def zipped_collate_fn(batch):
    result = {
        "human": {
            "sequence": torch.stack([b["human"]["sequence"] for b in batch]),
            "target": torch.stack([b["human"]["target"] for b in batch]),
        },
        "mouse": {
            "sequence": torch.stack([b["mouse"]["sequence"] for b in batch]),
            "target": torch.stack([b["mouse"]["target"] for b in batch]),
        },
    }
    if "ccre_mask" in batch[0]["human"]:
        result["human"]["ccre_mask"] = torch.stack([b["human"]["ccre_mask"] for b in batch])
    if "ccre_mask" in batch[0]["mouse"]:
        result["mouse"]["ccre_mask"] = torch.stack([b["mouse"]["ccre_mask"] for b in batch])
    return result

class EnformerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        human_npz_dir,
        mouse_npz_dir,
        human_ccre_mask_dir=None,
        mouse_ccre_mask_dir=None,
        train_batch_size=8,
        val_batch_size=8,
        num_workers=4,
        expected_mask_length=None,
    ):
        super().__init__()
        self.human_npz_dir = human_npz_dir
        self.mouse_npz_dir = mouse_npz_dir
        self.human_ccre_mask_dir = human_ccre_mask_dir
        self.mouse_ccre_mask_dir = mouse_ccre_mask_dir
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.expected_mask_length = expected_mask_length

    def setup(self, stage=None):
        self.train_ds = ZippedOrganismDataset(
            self.human_npz_dir, self.mouse_npz_dir, "train",
            human_ccre_mask_dir=self.human_ccre_mask_dir,
            mouse_ccre_mask_dir=self.mouse_ccre_mask_dir,
            augment=True, max_shift=3,
            expected_mask_length=self.expected_mask_length,
        )
        self.val_human = SingleOrganismDataset(
            self.human_npz_dir, "valid",
            ccre_mask_dir=self.human_ccre_mask_dir,
            expected_mask_length=self.expected_mask_length,
        )
        self.val_mouse = SingleOrganismDataset(
            self.mouse_npz_dir, "valid",
            ccre_mask_dir=self.mouse_ccre_mask_dir,
            expected_mask_length=self.expected_mask_length,
        )

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

if __name__ == "__main__":
    default_config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "ccre_bigbird.yaml"
    )
    config_path = Path(
        os.environ.get("ENFORMER_CONFIG", default_config_path)
    )
    experiment_config = load_experiment_config(config_path)
    config = EnformerConfig(**experiment_config["model"])
    training_config = experiment_config["training"]
    classifier_config = training_config["classifier"]
    dataloader_config = training_config["dataloader"]
    trainer_config = dict(training_config["trainer"])

    pl.seed_everything(training_config["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    HUMAN_NPZ_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/human"
    MOUSE_NPZ_DIR = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/enformer-pytorch/data/enformer_flat_npy/mouse"

    CONDITION = training_config["ccre_condition"]
    MEAN_CCRE_K = int(training_config["mean_ccre_k"])
    CLASSIFIER_EVERY = int(classifier_config["every"])
    CLASSIFIER_MODE = classifier_config["mode"]

    ABLATION_BASE = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/ccre_mask/ablation"
    HUMAN_CCRE_MASK_DIR = f"{ABLATION_BASE}/{CONDITION}/human"
    MOUSE_CCRE_MASK_DIR = f"{ABLATION_BASE}/{CONDITION}/mouse"

    print(f"MODEL_CONFIG  = {config_path}")
    print(f"ATTN_BACKEND  = {config.attention_backend}")
    print(f"CONDITION     = {CONDITION}")
    print(f"MEAN_CCRE_K   = {MEAN_CCRE_K}")
    print(f"CLASSIFIER_EVERY = {CLASSIFIER_EVERY}")
    print(f"CLASSIFIER_MODE  = {CLASSIFIER_MODE}")

    dm = EnformerDataModule(
        human_npz_dir=HUMAN_NPZ_DIR,
        mouse_npz_dir=MOUSE_NPZ_DIR,
        human_ccre_mask_dir=HUMAN_CCRE_MASK_DIR,
        mouse_ccre_mask_dir=MOUSE_CCRE_MASK_DIR,
        expected_mask_length=(
            SEQUENCE_LENGTH // (2 ** config.num_downsamples)
        ),
        **dataloader_config,
    )

    model = BigBirdLightningModule(
        config,
        lr=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        use_classifier=classifier_config["enabled"],
        classifier_hidden_dims=classifier_config["hidden_dims"],
        classifier_dropout=classifier_config["dropout"],
        bce_weight=classifier_config["bce_weight"],
        mean_ccre_k=MEAN_CCRE_K,
        ccre_condition=CONDITION,
        classifier_every=CLASSIFIER_EVERY,
        mix_steps=training_config["mix_steps"],
        weight_decay=training_config["weight_decay"],
        classifier_mode=CLASSIFIER_MODE,
    )

    # logger = TensorBoardLogger(
    #     save_dir="/hpc/users/hongw01/Bigbird-Enformer/tb_logs",
    #     name="ATLAS-Einsum",
    #     version=f"cls-every{CLASSIFIER_EVERY}-k{MEAN_CCRE_K}-{CLASSIFIER_MODE}-75pct-ccre",
    # )

    logger = TensorBoardLogger(
        save_dir="/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/tb_logs",
        name="ATLAS-Ablation",
        version=f"ablation-{CONDITION}-k{MEAN_CCRE_K}",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor = "val_corr_coef",
        mode = "max",
        save_top_k = 1,
        filename = "ccre_cls-best-{epoch:02d}-{val_corr_coef:.4f}",
        save_last = False,
    )

    strategy = DDPStrategy(**training_config["strategy"])

    trainer = pl.Trainer(
        **trainer_config,
        strategy=strategy,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            PerformanceMonitor(
                log_every=training_config["performance_log_every"]
            ),
        ],
    )

    trainer.fit(model, dm)
