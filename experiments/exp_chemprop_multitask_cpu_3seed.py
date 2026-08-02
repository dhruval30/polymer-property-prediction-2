"""
exp_chemprop_multitask_cpu_3seed.py — 5-fold × 3-seed bagged Chemprop D-MPNN multitask on Mac CPU.

============================================================================
WHAT THIS IS
============================================================================

Extends `exp_chemprop_multitask_cpu.py` (LB 0.887) with:
  1. **3-seed bagging** — per fold, train 3 models with different random seeds,
     average their val predictions for OOF. Refit on full-train: 3 models with
     different seeds, average their test predictions. Total: 5 folds × 3 seeds
     + 3 refits = 18 model training runs.
  2. **Longer training** — max_epochs=60 (up from 40), patience=10 (up from 8).
     In the single-seed run, 3 of 5 folds hit the 40-epoch ceiling still
     improving. Extension should let them converge properly.
  3. **Identical fold assignment** — SPLIT_SEED=42, matching the single-seed
     run so this script's OOF can be per-target NNLS-blended with the earlier
     single-seed Chemprop OOFs, LGB OOFs, and CAT OOFs.

The Round-1 canonical Chemprop recipe from CLAUDE.md: "5-fold × 3-seed bag =
15 models". This script implements that.

============================================================================
REQUIREMENTS
============================================================================

  - poly2-venv (Python 3.11) with chemprop already installed
  - Data: ppp-round-2/{train,test}.csv

============================================================================
WALL TIME
============================================================================

Realistic estimate on Mac M-series CPU:
  - Per model (60 epochs × ~15 sec/epoch, or early stop): 10–15 min
  - 15 CV models: 150–225 min
  - 3 refit models: 45–75 min
  - Total: **3.5–5 hours**

CLAUDE.md flags: Mac MPS thermal degradation and Chemprop's silent-hang risk
without per-epoch logging. This script uses CPU only and wires an EpochLogger.
Ctrl+C safe — per-fold checkpointing bundles all 3 seeds per fold. If you
interrupt mid-fold, that fold restarts from scratch on rerun (all 3 seeds
retrain). Completed folds are skipped.

============================================================================
OUTPUTS  (under results/exp_chemprop_multitask_cpu_3seed/)
============================================================================

  run.log                          — per-epoch train/val loss, per-fold-per-seed R², final summary
  oof.csv                          — averaged OOF predictions (canon, target_type, y_true, y_pred)
  submission.csv                   — averaged test predictions (id, target)
  cv_summary.json                  — per-target R², per-seed R², timing, config
  checkpoint_fold_{k}.pkl.gz       — per-fold bundle (all 3 seeds' val preds + best_epochs)
  refit_test_preds.pkl.gz          — cached 3-seed averaged test predictions

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu_3seed.py

Then submit results/exp_chemprop_multitask_cpu_3seed/submission.csv to Kaggle.

Expected LB (based on single-seed Chemprop LB 0.887 and Round-1 bagging Δ):
  0.890–0.895 solo. If blended into the 2-way NNLS as a replacement for
  single-seed Chemprop, expected 0.896–0.899.

============================================================================
"""
from __future__ import annotations

# --- stdlib ---
import gzip
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

# --- third-party ---
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from chemprop import data, featurizers, nn
from chemprop.models import MPNN

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_chemprop_multitask_cpu_3seed"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SPLIT_SEED = 42                 # MUST match single-seed run for OOF alignment
MODEL_SEEDS = (42, 43, 44)      # 3 seeds per fold for bagging

# Model config (unchanged from single-seed)
D_H = 300
DEPTH = 4
MP_DROPOUT = 0.05
FFN_HIDDEN = 300
FFN_LAYERS = 2
FFN_DROPOUT = 0.05
BATCH_NORM = True

# Trainer config — LONGER TRAINING
MAX_EPOCHS = 60                 # was 40
PATIENCE = 10                   # was 8
BATCH_SIZE = 64
GRAD_CLIP = 1.0
LR_INIT = 1e-3
LR_MAX = 1e-3
LR_FINAL = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS = 0

DEVICE = "cpu"

REFIT_ITER_MULTIPLIER = 1.10


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(exp_dir: Path) -> logging.Logger:
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "run.log"
    logger = logging.getLogger(EXP_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")   # append so resume keeps history
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


class EpochLogger(L.Callback):
    """Per-epoch train/val loss printer. Prevents silent 9h hangs (Round 1 lesson).
       ctx label identifies fold + seed for grep-ability."""
    def __init__(self, logger, ctx: str):
        self.logger = logger
        self.ctx = ctx
        self.t_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        if self.t_start is None:
            self.t_start = time.time()
        self._epoch_t = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        m = trainer.callback_metrics
        train_loss = float(m.get("train_loss", float("nan")))
        val_loss = float(m.get("val_loss", float("nan")))
        epoch_time = time.time() - self._epoch_t
        elapsed = time.time() - self.t_start
        self.logger.info(
            f"[{self.ctx}] epoch {epoch:>3d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"epoch_time={epoch_time:.1f}s  elapsed={elapsed/60:.1f}min"
        )


# ============================================================================
# DATA + CANONICALIZATION
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_and_canonicalize(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading train.csv / test.csv")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log.info(f"train raw: {tr.shape}   test raw: {te.shape}")

    all_smi = pd.concat([tr["smiles"], te["smiles"]]).unique()
    log.info(f"canonicalizing {len(all_smi)} unique raw SMILES")
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical", ncols=100)}
    tr["canon"] = tr["smiles"].map(canon_map)
    te["canon"] = te["smiles"].map(canon_map)

    dupes = tr.groupby(["canon", "target_type"]).size()
    n_dup_rows = int((dupes[dupes > 1] - 1).sum())
    if n_dup_rows:
        log.info(f"collapsing {n_dup_rows} duplicate (canon, target_type) rows in train by mean")
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean"),
                 smiles=("smiles", "first")))
    log.info(f"train after dedup: {tr.shape}")
    log.info(f"per-target train counts: {tr['target_type'].value_counts().to_dict()}")
    log.info(f"per-target test  counts: {te['target_type'].value_counts().to_dict()}")
    return tr, te


def build_wide_train(tr: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[list(TARGETS)]
    canons = wide.index.tolist()
    y_matrix = wide.values.astype(np.float32)
    return canons, y_matrix


# ============================================================================
# CV
# ============================================================================

def group_kfold_splits(
    canon_arr: list[str] | np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = SPLIT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# MODEL BUILDING + DATASETS
# ============================================================================

def build_model(output_transform=None) -> MPNN:
    mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH, dropout=MP_DROPOUT)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        n_tasks=N_TARGETS,
        input_dim=D_H,
        hidden_dim=FFN_HIDDEN,
        n_layers=FFN_LAYERS,
        dropout=FFN_DROPOUT,
        output_transform=output_transform,
    )
    return MPNN(
        mp, agg, ffn,
        batch_norm=BATCH_NORM,
        init_lr=LR_INIT, max_lr=LR_MAX, final_lr=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
    )


def make_train_val_datasets(
    canons: list[str],
    y: np.ndarray,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    featurizer,
) -> tuple[data.MoleculeDataset, data.MoleculeDataset, object]:
    """Build fresh train + val datasets and normalize targets on train fold."""
    def _mk(idxs):
        pts = []
        for i in idxs:
            m = Chem.MolFromSmiles(canons[i])
            if m is None: continue
            pts.append(data.MoleculeDatapoint(mol=m, y=y[i]))
        return pts

    train_pts = _mk(train_idxs)
    val_pts = _mk(val_idxs)
    train_dset = data.MoleculeDataset(train_pts, featurizer=featurizer)
    val_dset = data.MoleculeDataset(val_pts, featurizer=featurizer)
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)
    return train_dset, val_dset, scaler


def make_full_dataset(
    canons: list[str],
    y: np.ndarray,
    featurizer,
) -> tuple[data.MoleculeDataset, object]:
    all_pts = []
    for i, smi in enumerate(canons):
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        all_pts.append(data.MoleculeDatapoint(mol=m, y=y[i]))
    full_dset = data.MoleculeDataset(all_pts, featurizer=featurizer)
    scaler = full_dset.normalize_targets()
    return full_dset, scaler


# ============================================================================
# TRAIN ONE MODEL (single seed)
# ============================================================================

def train_single_model_cv(
    train_dset,
    val_dset,
    scaler,
    seed: int,
    ctx: str,
    log: logging.Logger,
) -> tuple[np.ndarray, int, float]:
    """Train one Chemprop model on a fold. Return (val_preds, best_epoch, wall_time_min)."""
    L.seed_everything(seed, workers=True)
    train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = data.build_dataloader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_model(output_transform=output_transform)

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=False)
    epoch_logger = EpochLogger(log, ctx)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[early_stop, epoch_logger],
        deterministic=False,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    wall = (time.time() - t0) / 60

    model.eval()
    preds_list = trainer.predict(model, val_loader)
    val_preds = torch.cat(preds_list, dim=0).cpu().numpy()

    best_epoch = trainer.current_epoch - PATIENCE if early_stop.stopped_epoch > 0 else trainer.current_epoch
    log.info(f"[{ctx}] done. time={wall:.1f}min  final_epoch={trainer.current_epoch}  approx_best_epoch={best_epoch}")
    return val_preds, int(best_epoch), float(wall)


def train_single_model_refit(
    full_dset,
    scaler,
    seed: int,
    n_epochs: int,
    test_canon_unique: list[str],
    featurizer,
    ctx: str,
    log: logging.Logger,
) -> tuple[np.ndarray, float]:
    """Train one Chemprop model on full train, return (test_preds, wall_time_min)."""
    L.seed_everything(seed, workers=True)
    full_loader = data.build_dataloader(full_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_model(output_transform=output_transform)
    epoch_logger = EpochLogger(log, ctx)

    trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator=DEVICE,
        devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[epoch_logger],
        deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, full_loader)
    wall = (time.time() - t0) / 60
    log.info(f"[{ctx}] refit done. time={wall:.1f}min")

    # Predict test
    test_pts = []
    valid_mask = []
    for smi in test_canon_unique:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            test_pts.append(data.MoleculeDatapoint(mol=m, y=np.zeros(N_TARGETS, dtype=np.float32)))

    test_dset = data.MoleculeDataset(test_pts, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds_list = trainer.predict(model, test_loader)
    test_preds_valid = torch.cat(preds_list, dim=0).cpu().numpy()

    aligned = np.zeros((len(test_canon_unique), N_TARGETS), dtype=np.float32)
    j = 0
    for i, v in enumerate(valid_mask):
        if v:
            aligned[i] = test_preds_valid[j]; j += 1
        else:
            aligned[i] = np.nan
    return aligned, wall


# ============================================================================
# TRAIN ONE FOLD (3-seed bag)
# ============================================================================

def train_fold_3seed(
    fold_k: int,
    canons: list[str],
    y: np.ndarray,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    featurizer,
    log: logging.Logger,
) -> dict:
    log.info(f"=" * 60)
    log.info(f"FOLD {fold_k}  ({len(MODEL_SEEDS)}-seed bag)")
    log.info(f"n_train_canon={len(train_idxs)}   n_val_canon={len(val_idxs)}")

    val_preds_per_seed = []
    best_epochs = []
    wall_times = []
    fold_r2_per_seed = []

    for si, seed in enumerate(MODEL_SEEDS):
        ctx = f"fold {fold_k} seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
        log.info(f"[{ctx}] starting...")
        # Fresh datasets per seed (normalize_targets is in-place)
        train_dset, val_dset, scaler = make_train_val_datasets(canons, y, train_idxs, val_idxs, featurizer)
        val_preds, best_epoch, wall = train_single_model_cv(train_dset, val_dset, scaler, seed, ctx, log)
        val_preds_per_seed.append(val_preds)
        best_epochs.append(best_epoch)
        wall_times.append(wall)

        # Per-seed R²
        val_true = y[val_idxs]
        r2s = {}
        for t_idx, tgt in enumerate(TARGETS):
            mask = ~np.isnan(val_true[:, t_idx])
            if mask.sum() < 5:
                r2s[tgt] = None; continue
            r2s[tgt] = float(r2_score(val_true[mask, t_idx], val_preds[mask, t_idx]))
        fold_r2_per_seed.append(r2s)
        log.info(f"[{ctx}] per-target R²: "
                 + "  ".join([f"{t}={r2s[t]:.3f}" if r2s[t] is not None else f"{t}=n/a" for t in TARGETS]))

    # Average across seeds
    val_preds_avg = np.mean(np.stack(val_preds_per_seed, axis=0), axis=0)

    val_true = y[val_idxs]
    fold_r2_avg = {}
    for t_idx, tgt in enumerate(TARGETS):
        mask = ~np.isnan(val_true[:, t_idx])
        if mask.sum() < 5:
            fold_r2_avg[tgt] = None; continue
        fold_r2_avg[tgt] = float(r2_score(val_true[mask, t_idx], val_preds_avg[mask, t_idx]))
    log.info(f"[fold {fold_k}] BAG per-target R² (avg of {len(MODEL_SEEDS)} seeds): "
             + "  ".join([f"{t}={fold_r2_avg[t]:.4f}" if fold_r2_avg[t] is not None else f"{t}=n/a" for t in TARGETS]))
    log.info(f"[fold {fold_k}] total time: {sum(wall_times):.1f}min "
             f"(per-seed: {[f'{w:.1f}' for w in wall_times]})")

    return {
        "fold_k": fold_k,
        "val_idxs": val_idxs,
        "val_preds_per_seed": val_preds_per_seed,   # list of arrays
        "val_preds_avg": val_preds_avg,
        "val_true": val_true,
        "best_epochs_per_seed": best_epochs,
        "wall_times_per_seed_min": wall_times,
        "fold_time_min": sum(wall_times),
        "fold_r2_avg": fold_r2_avg,
        "fold_r2_per_seed": fold_r2_per_seed,
        "seeds_used": list(MODEL_SEEDS),
    }


def save_fold_checkpoint(exp_dir: Path, fold_k: int, fold_result: dict, log: logging.Logger):
    path = exp_dir / f"checkpoint_fold_{fold_k}.pkl.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(fold_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"[fold {fold_k}] wrote checkpoint {path.name}")


def load_fold_checkpoint(exp_dir: Path, fold_k: int) -> dict | None:
    path = exp_dir / f"checkpoint_fold_{fold_k}.pkl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# 3-SEED REFIT + TEST PREDICTIONS
# ============================================================================

def refit_3seed_and_predict_test(
    canons: list[str],
    y: np.ndarray,
    test_canon_unique: list[str],
    featurizer,
    n_epochs: int,
    log: logging.Logger,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Refit on full train × 3 seeds, average test predictions. Return (avg, per_seed, wall_times)."""
    test_preds_per_seed = []
    wall_times = []
    for si, seed in enumerate(MODEL_SEEDS):
        ctx = f"REFIT seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
        log.info(f"[{ctx}] starting refit for {n_epochs} epochs")
        # Fresh dataset per seed (normalize_targets is in-place)
        full_dset, scaler = make_full_dataset(canons, y, featurizer)
        test_preds, wall = train_single_model_refit(
            full_dset, scaler, seed, n_epochs, test_canon_unique, featurizer, ctx, log,
        )
        test_preds_per_seed.append(test_preds)
        wall_times.append(wall)

    test_preds_avg = np.mean(np.stack(test_preds_per_seed, axis=0), axis=0)
    return test_preds_avg, test_preds_per_seed, wall_times


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info("=" * 60)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: device={DEVICE}  n_splits={N_SPLITS}  split_seed={SPLIT_SEED}  "
             f"model_seeds={MODEL_SEEDS}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  batch_size={BATCH_SIZE}")
    log.info(f"MODEL: d_h={D_H} depth={DEPTH} mp_dropout={MP_DROPOUT}  "
             f"ffn_hidden={FFN_HIDDEN} ffn_layers={FFN_LAYERS} ffn_dropout={FFN_DROPOUT}  "
             f"batch_norm={BATCH_NORM}")
    log.info(f"BAG: {len(MODEL_SEEDS)}-seed × {N_SPLITS}-fold = {len(MODEL_SEEDS)*N_SPLITS} CV models + "
             f"{len(MODEL_SEEDS)} refit models = {len(MODEL_SEEDS)*(N_SPLITS+1)} total")

    random.seed(SPLIT_SEED); np.random.seed(SPLIT_SEED); torch.manual_seed(SPLIT_SEED)
    L.seed_everything(SPLIT_SEED, workers=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log.info(f"torch threads: {torch.get_num_threads()}")

    t_start = time.time()

    # ---- Load + prepare ----
    tr, te = load_and_canonicalize(log)
    canons, y = build_wide_train(tr)
    log.info(f"wide train: n_canon={len(canons)}  y shape={y.shape}  NaN fraction={100*np.isnan(y).mean():.1f}%")

    test_canon_unique = te["canon"].drop_duplicates().tolist()
    log.info(f"unique test SMILES: {len(test_canon_unique)}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # ---- 5-fold CV × 3-seed with per-fold checkpointing ----
    splits = group_kfold_splits(canons, N_SPLITS, SPLIT_SEED)
    fold_results = []
    for k, (tri, vai) in enumerate(splits):
        cached = load_fold_checkpoint(EXP_DIR, k)
        if cached is not None:
            log.info(f"[fold {k}] loaded checkpoint (skipping training)")
            fold_results.append(cached)
            continue
        result = train_fold_3seed(k, canons, y, tri, vai, featurizer, log)
        save_fold_checkpoint(EXP_DIR, k, result, log)
        fold_results.append(result)

    # ---- Assemble OOF (average per fold, already computed) ----
    oof_preds = np.full((len(canons), N_TARGETS), np.nan, dtype=np.float32)
    for r in fold_results:
        oof_preds[r["val_idxs"]] = r["val_preds_avg"]

    log.info("=" * 60)
    log.info("PER-TARGET OOF R²  (5-fold × 3-seed bagged, no blending)")
    log.info("=" * 60)
    per_target = {}
    for t_idx, tgt in enumerate(TARGETS):
        mask = ~np.isnan(y[:, t_idx])
        y_true = y[mask, t_idx]
        y_pred = oof_preds[mask, t_idx]
        r2 = float(r2_score(y_true, y_pred))
        per_target[tgt] = {
            "n_train": int(mask.sum()),
            "oof_r2": r2,
            "fold_r2s_bag": [r["fold_r2_avg"].get(tgt) for r in fold_results],
            "fold_r2s_per_seed": [r["fold_r2_per_seed"] for r in fold_results],  # nested
        }
        fold_r2_str = [f'{v:.3f}' if v is not None else 'n/a' for v in per_target[tgt]['fold_r2s_bag']]
        log.info(f"  {tgt:>4s}  n={int(mask.sum()):>5d}  OOF R²={r2:.4f}   fold-bag={fold_r2_str}")
    mean_r2 = float(np.mean([per_target[t]["oof_r2"] for t in TARGETS]))
    log.info(f"  MEAN R² = {mean_r2:.4f}")

    # ---- Refit epochs ----
    all_best_epochs = [e for r in fold_results for e in r["best_epochs_per_seed"]]
    refit_epochs = max(15, int(np.median(all_best_epochs) * REFIT_ITER_MULTIPLIER))
    log.info(f"all fold×seed best_epochs = {all_best_epochs}   →   refit for {refit_epochs} epochs")

    # ---- Refit + test predictions with cache ----
    refit_cache = EXP_DIR / "refit_test_preds.pkl.gz"
    if refit_cache.exists():
        log.info(f"loading cached refit test predictions from {refit_cache.name}")
        with gzip.open(refit_cache, "rb") as f:
            refit_cache_data = pickle.load(f)
        test_preds_avg = refit_cache_data["test_preds_avg"]
        refit_wall_times = refit_cache_data.get("wall_times", [None]*len(MODEL_SEEDS))
    else:
        test_preds_avg, test_preds_per_seed, refit_wall_times = refit_3seed_and_predict_test(
            canons, y, test_canon_unique, featurizer, refit_epochs, log,
        )
        with gzip.open(refit_cache, "wb") as f:
            pickle.dump({
                "test_preds_avg": test_preds_avg,
                "test_preds_per_seed": test_preds_per_seed,
                "wall_times": refit_wall_times,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached refit test predictions to {refit_cache.name}")

    test_preds_by_canon = {smi: test_preds_avg[i] for i, smi in enumerate(test_canon_unique)}

    # ---- Build OOF DataFrame ----
    oof_rows = []
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    for _, row in tr.iterrows():
        c = row["canon"]; t = row["target_type"]
        i = canon_to_idx.get(c)
        if i is None: continue
        t_idx = TARGET_IDX[t]
        oof_rows.append({
            "canon": c, "target_type": t,
            "y_true": float(row["target"]),
            "y_pred": float(oof_preds[i, t_idx]) if not np.isnan(oof_preds[i, t_idx]) else np.nan,
        })
    oof_df = pd.DataFrame(oof_rows)
    oof_path = EXP_DIR / "oof.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_df)}")

    # ---- Build submission ----
    sub_rows = []
    for _, row in te.iterrows():
        c = row["canon"]; t = row["target_type"]
        t_idx = TARGET_IDX[t]
        preds = test_preds_by_canon.get(c)
        if preds is None:
            sub_rows.append({"id": int(row["id"]), "target": float("nan")})
        else:
            sub_rows.append({"id": int(row["id"]), "target": float(preds[t_idx])})
    sub_df = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_df)}")

    # ---- Summary JSON ----
    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "per_target":     per_target,
        "config": {
            "device":         DEVICE,
            "n_splits":       N_SPLITS,
            "split_seed":     SPLIT_SEED,
            "model_seeds":    list(MODEL_SEEDS),
            "d_h":            D_H, "depth": DEPTH, "mp_dropout": MP_DROPOUT,
            "ffn_hidden":     FFN_HIDDEN, "ffn_layers": FFN_LAYERS, "ffn_dropout": FFN_DROPOUT,
            "batch_norm":     BATCH_NORM,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
            "batch_size":     BATCH_SIZE,
            "grad_clip":      GRAD_CLIP,
            "lr":             {"init": LR_INIT, "max": LR_MAX, "final": LR_FINAL, "warmup_epochs": WARMUP_EPOCHS},
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "refit_epochs":   refit_epochs,
        },
        "fold_times_min": [r.get("fold_time_min") for r in fold_results],
        "all_best_epochs": all_best_epochs,
        "refit_wall_times_min": refit_wall_times,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = EXP_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    # ---- Final comparison vs single-seed Chemprop reference ----
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (3-seed bag)  vs single-seed Chemprop reference")
    log.info("=" * 60)
    single_ref = {"eea": 0.8883, "egb": 0.9251, "egc": 0.8830,
                  "ei": 0.7804, "eps": 0.7584, "nc": 0.8599, "tg": 0.8934}
    log.info(f"  {'target':>6s}  {'3-seed':>10s}  {'single':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = per_target[t]['oof_r2']
        ref = single_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    single_mean = float(np.mean(list(single_ref.values())))
    log.info(f"  {'MEAN':>6s}  {mean_r2:>10.4f}  {single_mean:>10.4f}  {mean_r2 - single_mean:>+8.4f}")
    log.info(f"  (single-seed Chemprop LB reference: 0.887)")
    log.info(f"total wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
