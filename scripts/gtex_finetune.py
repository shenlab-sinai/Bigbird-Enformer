import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from bigbird_enformer.models.enformer_plus import Enformer
from bigbird_enformer.train_lightning import CCREClassifierHead
from bigbird_enformer.utils.config import EnformerConfig
from bigbird_enformer.utils.gtex_dataset import GTExConsensusDataset, gtex_collate_fn


CKPT_PATH = (
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/Sparse_Enformer/tb_logs/ATLAS-Ablation"
    "/ablation-full-k153/checkpoints"
    "/ccre_cls-best-epoch=133-val_corr_coef=0.6751.ckpt"
)

attention_mode = "ccre_bigbird"
seq_len        = 196_608
mean_ccre_k    = 153

gene_id = "ENSG00000168903"   # BTNL3

N_FOLDS = 5
FOLD    = int(os.environ.get("GTEX_FOLD", 0))

head_only_steps    = 1000
head_lr            = 1e-3
backbone_lr        = 5e-6
warmup_steps       = 100
n_epochs           = 50
contrastive_weight = 0.01
bce_weight         = 0.05
head_dropout       = 0.3
weight_decay       = 1e-4
grad_clip          = 0.1
batch_size         = 4
num_workers        = 2
seed               = 8888

log_dir = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/gtex_finetune"

BCF_PATH     = (
    "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp"
    "/lis_play_ground/gtex/data"
    "/GTEx_Analysis_2021-02-11_v9_WholeGenomeSeq_953Indiv.SHAPEIT2_phased.bcf"
)
EXP_NPZ      = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/lis_play_ground/gtex/data/GTEx_50tissue_exp.npz"
GENE_IDS_TXT = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/lis_play_ground/gtex/data/0_gene_ids.txt"
SAMPLES_TXT  = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/lis_play_ground/gtex/data/1_samples.txt"
GINFO_PATH   = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/genias/data/reference_genomes/hg38_gene_info.csv"
REF_PATH     = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/genias/data/reference_genomes/hg38.ml.fa"

CCRE_CONDITION = "full"

_RAW  = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/data/cCREs/raw_cCRES"
_FULL = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/data/cCREs/full_cCRES"
_COMB = "/sc/arion/projects/Nestlerlab/shenl03_ml/gene_exp/Sparse_Enformer/data/cCREs/combined"

CCRE_BED_MAP = {
    "full":      f"{_FULL}/GRCh38-cCREs.bed",
    "no_pls":    f"{_COMB}/GRCh38-no_pls.bed",
    "no_els":    f"{_COMB}/GRCh38-no_els.bed",
    "no_ctcf":   f"{_COMB}/GRCh38-no_ctcf.bed",
    "no_tf":     f"{_COMB}/GRCh38-no_tf.bed",
    "no_ca":     f"{_COMB}/GRCh38-no_ca.bed",
    "pls_only":  f"{_RAW}/GRCh38-cCREs.PLS.bed",
    "els_only":  f"{_RAW}/GRCh38-cCREs.ELS.bed",
    "ctcf_only": f"{_RAW}/GRCh38-cCREs.CTCF-bound.bed",
    "tf_only":   f"{_RAW}/GRCh38-cCREs.TF.bed",
    "ca_only":   f"{_RAW}/GRCh38-cCREs.CA.bed",
}

CCRE_BED = CCRE_BED_MAP[CCRE_CONDITION]

run_name = (
    f"GTEx-finetune-v3-{gene_id}-{attention_mode}"
    f"-{seq_len//1024}k-{mean_ccre_k}-fold{FOLD}"
)

_MODEL_CONFIG = dict(
    dim=1536, depth=11, heads=8,
    output_heads=dict(human=5313, mouse=1643),
    target_length=896,
    block_size=128,
    use_checkpointing=True,
    attn_dropout=0.05,
    dropout_rate=0.3,
    pos_dropout=0.01,
    use_rel_pe=False,
    use_einsum=True,
)


def mse_ignore_nan(pred, target):
    mask = torch.isfinite(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])


def contrastive_loss(pred, target):
    B, T = pred.shape
    if B < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    true_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pred_diff = pred.unsqueeze(1)   - pred.unsqueeze(0)
    diff_diff = pred_diff - true_diff
    ix   = torch.triu_indices(B, B, offset=1, device=pred.device)
    triu = diff_diff[ix[0], ix[1], :]
    mask = torch.isfinite(triu)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return (triu[mask] ** 2).mean()

#  MODEL

class GTExHead(nn.Module):
    def __init__(self, in_dim, n_tissues, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, n_tissues),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        center = x[:, x.shape[1] // 2, :]
        return self.net(center)


class GTExFinetuneModule(pl.LightningModule):
    def __init__(self, model, classifiers, n_tissues, ccre_mask):
        super().__init__()
        self.model       = model
        self.classifiers = classifiers
        self.is_ccre     = (attention_mode == "ccre_bigbird")
        self.gtex_head   = GTExHead(_MODEL_CONFIG["dim"] * 2, n_tissues, head_dropout)
        self.register_buffer("ccre_mask", ccre_mask.bool())

        self._val_preds      = []
        self._val_targets    = []
        self._stage2_enabled = False

        self._set_backbone_grad(False)
        print("[GTExFT] Stage 1: backbone FROZEN")

    def _set_backbone_grad(self, requires_grad):
        for p in self.model.parameters():
            p.requires_grad = requires_grad
        if self.classifiers is not None:
            for p in self.classifiers.parameters():
                p.requires_grad = requires_grad

    def _embed(self, seq):
        if self.is_ccre:
            encoded   = self.model._encode_ccre(seq)
            local_out = self.model._run_transformer_blocks(encoded, is_global=None)
            clf_logits = self.classifiers[-1](
                local_out.detach() if not self._stage2_enabled else local_out
            )
            pred_mask = self.classifiers[-1].topk_mask(local_out, k=mean_ccre_k)
            final_out = self.model._run_transformer_blocks(encoded, is_global=pred_mask)
            return self.model._crop_and_pointwise(final_out), clf_logits
        else:
            emb = self.model(seq, return_only_embeddings=True)
            return emb, None

    def on_train_batch_start(self, batch, batch_idx):
        if not self._stage2_enabled and self.global_step >= head_only_steps:
            self._set_backbone_grad(True)
            self._stage2_enabled = True
            print(f"\n[GTExFT] Stage 2: backbone UNFROZEN at step {self.global_step}")
            self.trainer.strategy.setup_optimizers(self.trainer)

    def training_step(self, batch, batch_idx):
        hap1, hap2, target = batch["hap1"], batch["hap2"], batch["target"]
        B = hap1.shape[0]
        gt_mask = self.ccre_mask.unsqueeze(0).expand(B, -1)

        emb1, clf1 = self._embed(hap1)
        emb2, clf2 = self._embed(hap2)
        pred = (self.gtex_head(emb1) + self.gtex_head(emb2)) / 2

        loss_mse = mse_ignore_nan(pred, target)
        loss_con = contrastive_loss(pred, target)

        loss_bce = torch.tensor(0.0, device=self.device)
        if self.is_ccre and self._stage2_enabled and clf1 is not None:
            pos_frac = gt_mask.float().mean().clamp(0.01, 0.99)
            pos_w    = (1 - pos_frac) / pos_frac
            gt_f     = gt_mask.float()
            loss_bce = (
                F.binary_cross_entropy_with_logits(clf1, gt_f, pos_weight=pos_w)
                + F.binary_cross_entropy_with_logits(clf2, gt_f, pos_weight=pos_w)
            ) / 2

        total = loss_mse + contrastive_weight * loss_con + bce_weight * loss_bce

        if torch.isnan(total):
            print(f"WARNING: NaN loss at step {self.global_step}")
            return None

        self.log("train_loss",        loss_mse, on_step=True, prog_bar=True,  sync_dist=False)
        self.log("train_contrastive", loss_con, on_step=True, prog_bar=False, sync_dist=False)
        self.log("train_bce",         loss_bce, on_step=True, prog_bar=False, sync_dist=False)
        self.log("train_total",       total,    on_step=True, prog_bar=False, sync_dist=False)
        return total

    def validation_step(self, batch, batch_idx):
        hap1, hap2, target = batch["hap1"], batch["hap2"], batch["target"]
        with torch.no_grad():
            emb1, _ = self._embed(hap1)
            emb2, _ = self._embed(hap2)
        pred = (self.gtex_head(emb1) + self.gtex_head(emb2)) / 2
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(target.detach().cpu())
        return mse_ignore_nan(pred, target)

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds   = torch.cat(self._val_preds).float().numpy()
        targets = torch.cat(self._val_targets).float().numpy()
        self._val_preds.clear()
        self._val_targets.clear()

        flat_mask = np.isfinite(targets.ravel())
        r2      = float(r2_score(targets.ravel()[flat_mask], preds.ravel()[flat_mask]))
        pearson = float(pearsonr(targets.ravel()[flat_mask], preds.ravel()[flat_mask])[0])

        per_sample = []
        for i in range(preds.shape[0]):
            mask = np.isfinite(targets[i])
            if mask.sum() > 1:
                per_sample.append(pearsonr(preds[i][mask], targets[i][mask])[0])

        per_tissue = []
        for t in range(preds.shape[1]):
            mask = np.isfinite(targets[:, t])
            if mask.sum() > 1:
                per_tissue.append(pearsonr(preds[:, t][mask], targets[:, t][mask])[0])

        pearson_sample = float(np.nanmean(per_sample)) if per_sample else float("nan")
        pearson_tissue = float(np.nanmean(per_tissue)) if per_tissue else float("nan")

        self.log("val_r2",             r2,             on_epoch=True, prog_bar=True,  sync_dist=False)
        self.log("val_pearson",        pearson,        on_epoch=True, prog_bar=True,  sync_dist=False)
        self.log("val_pearson_sample", pearson_sample, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("val_pearson_tissue", pearson_tissue, on_epoch=True, prog_bar=True,  sync_dist=False)
        print(f"\n[Val fold={FOLD}] R²={r2:.4f}  Pearson={pearson:.4f}  "
              f"per-sample={pearson_sample:.4f}  per-tissue={pearson_tissue:.4f}")

    def on_before_optimizer_step(self, optimizer):
        for p in self.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                p.grad.zero_()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
        self.log("train_grad_norm", grad_norm, prog_bar=False, sync_dist=False)

    def configure_optimizers(self):
        head_params     = list(self.gtex_head.parameters())
        backbone_params = (
            list(self.model.parameters()) +
            (list(self.classifiers.parameters()) if self.classifiers else [])
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params,     "lr": head_lr,     "weight_decay": weight_decay},
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )
        total_steps  = self.trainer.max_steps if self.trainer else 5_000
        cosine_steps = max(1, total_steps - warmup_steps)
        sched = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                         total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=head_lr * 0.01),
            ],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

def load_pretrained(ckpt_path, config):
    print(f"[ckpt] Loading {ckpt_path} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["state_dict"]

    model    = Enformer(config)
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"  [ckpt] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    clf_keys    = [k for k in sd if k.startswith("classifiers.")]
    classifiers = None
    if clf_keys and attention_mode == "ccre_bigbird":
        n_cls      = max(int(k.split(".")[1]) for k in clf_keys) + 1
        clf_sd     = {k[len("classifiers."):]: v for k, v in sd.items()
                      if k.startswith("classifiers.")}
        hidden_dim = clf_sd["0.net.1.weight"].shape[0]
        classifiers = nn.ModuleList([
            CCREClassifierHead(dim=config.dim, hidden_dims=hidden_dim, dropout=0.1)
            for _ in range(n_cls)
        ])
        classifiers.load_state_dict(clf_sd, strict=True)
        print(f"  [ckpt] {n_cls} classifiers loaded  hidden_dim={hidden_dim}")

    return model, classifiers


def load_gtex_expression(gid):
    exp_mat = np.load(EXP_NPZ)["exp"]
    with open(GENE_IDS_TXT) as f:
        gene_ids = [g.split(".")[0] for g in f.read().splitlines()]
    with open(SAMPLES_TXT) as f:
        sample_names = f.read().splitlines()
    if gid not in gene_ids:
        raise ValueError(f"Gene {gid} not found.")
    ix     = gene_ids.index(gid)
    exp_df = pd.DataFrame(exp_mat[ix], index=sample_names)
    exp_df = exp_df.loc[:, exp_df.isna().mean() < 0.5]
    print(f"[data] {gid}: {exp_df.shape[0]} samples, {exp_df.shape[1]} tissues", flush=True)
    return exp_df.index.tolist(), exp_df.values.astype(np.float32), exp_df.shape[1]


def build_dataloaders_kfold(dataset, fold, n_folds=5):
    n  = len(dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(range(n)))
    train_val_idx, test_idx = splits[fold]
    train_val_idx = train_val_idx.tolist()
    test_idx      = test_idx.tolist()

    n_val     = int(len(train_val_idx) * 0.15)
    val_idx   = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    print(f"[fold {fold}/{n_folds}] "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}", flush=True)

    def make_loader(idx, shuffle):
        return DataLoader(
            Subset(dataset, idx),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=gtex_collate_fn,
        )

    return (make_loader(train_idx, True),
            make_loader(val_idx,   False),
            make_loader(test_idx,  False),
            test_idx)


def evaluate(ft_model, test_loader, test_idx, out_path):
    ft_model.eval()
    ft_model = ft_model.cuda()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            hap1, hap2 = batch["hap1"].cuda(), batch["hap2"].cuda()
            target      = batch["target"]
            emb1, _     = ft_model._embed(hap1)
            emb2, _     = ft_model._embed(hap2)
            pred        = (ft_model.gtex_head(emb1) + ft_model.gtex_head(emb2)) / 2
            all_preds.append(pred.cpu().float())
            all_targets.append(target)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    flat_mask = np.isfinite(targets.ravel())
    test_r2   = float(r2_score(targets.ravel()[flat_mask], preds.ravel()[flat_mask]))
    test_pear = float(pearsonr(targets.ravel()[flat_mask], preds.ravel()[flat_mask])[0])

    per_tissue = []
    for t in range(preds.shape[1]):
        mask = np.isfinite(targets[:, t])
        if mask.sum() > 1:
            per_tissue.append(pearsonr(preds[:, t][mask], targets[:, t][mask])[0])
    test_pear_tissue = float(np.nanmean(per_tissue))

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS  fold={FOLD}  ({len(test_idx)} samples)")
    print(f"  R²                  : {test_r2:.4f}")
    print(f"  Pearson (overall)   : {test_pear:.4f}")
    print(f"  Pearson (per-tissue): {test_pear_tissue:.4f}")
    print(f"{'='*60}\n")

    out_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "gene_id":             gene_id,
        "attention_mode":      attention_mode,
        "fold":                FOLD,
        "n_folds":             N_FOLDS,
        "n_tissues":           preds.shape[1],
        "n_test_samples":      len(test_idx),
        "test_r2":             test_r2,
        "test_pearson":        test_pear,
        "test_pearson_tissue": test_pear_tissue,
        "contrastive_weight":  contrastive_weight,
        "bce_weight":          bce_weight,
        "mean_ccre_k":         mean_ccre_k,
        "seq_len":             seq_len,
        "ccre_condition":      CCRE_CONDITION,
    }]).to_csv(out_path / f"test_results_fold{FOLD}.csv", index=False)
    print(f"[Saved] {out_path}/test_results_fold{FOLD}.csv")


def main():
    pl.seed_everything(seed + FOLD, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    config = EnformerConfig(**{**_MODEL_CONFIG, "attention_mode": attention_mode})
    model, classifiers = load_pretrained(CKPT_PATH, config)

    sample_names, exp_values, n_tissues = load_gtex_expression(gene_id)

    ccre_bed = CCRE_BED if attention_mode == "ccre_bigbird" else None
    dataset  = GTExConsensusDataset(
        bcf_path=BCF_PATH,
        gene_id=gene_id,
        sample_list=sample_names,
        target_list=exp_values,
        ref_path=REF_PATH,
        ginfo_path=GINFO_PATH,
        seq_len=seq_len,
        ccre_bed_path=ccre_bed,
        show_progress=True,
    )
    ccre_mask = torch.from_numpy(dataset.ccre_mask)

    train_loader, val_loader, test_loader, test_idx = \
        build_dataloaders_kfold(dataset, fold=FOLD, n_folds=N_FOLDS)

    ft_model = GTExFinetuneModule(
        model=model,
        classifiers=classifiers,
        n_tissues=n_tissues,
        ccre_mask=ccre_mask,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="GTEx-Finetune",
        version=run_name,
    )
    callbacks = [
        ModelCheckpoint(
            monitor="val_pearson_tissue",
            mode="max",
            save_top_k=1,
            save_last=False,
            filename=f"fold{FOLD}" + "-{epoch:03d}-{val_pearson_tissue:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=n_epochs,
        precision="bf16-mixed",
        gradient_clip_val=grad_clip,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=0.25,
        enable_model_summary=False,
        num_sanity_val_steps=2,
    )

    trainer.fit(ft_model, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"[eval] Loading best checkpoint: {best_ckpt}")

    out_path = Path(log_dir) / "GTEx-Finetune" / run_name

    ft_model = GTExFinetuneModule.load_from_checkpoint(
        best_ckpt,
        model=model,
        classifiers=classifiers,
        n_tissues=n_tissues,
        ccre_mask=ccre_mask,
        strict=False,
    )
    evaluate(ft_model, test_loader, test_idx, out_path)

    # Delete checkpoint after eval to save storage
    import os
    os.remove(best_ckpt)
    print(f"Deleted {best_ckpt}")


if __name__ == "__main__":
    main()
