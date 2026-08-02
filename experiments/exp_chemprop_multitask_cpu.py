"""
exp_chemprop_multitask_cpu.py — D-MPNN multitask over 7 targets on Mac CPU.

Uses Chemprop 2.x. Single shared MPNN encoder + one regression head with 7 output
tasks, trained jointly on multi-target long-format data pivoted to wide (canonical
SMILES × 7 targets with NaN for unlabeled). NaN targets are masked from the loss
so each row contributes only to the targets it has labels for.

Configuration (from Round 1 winning recipe):
- BondMessagePassing(d_h=300, depth=4, dropout=0.05)
- MeanAggregation
- RegressionFFN(n_tasks=7, hidden_dim=300, n_layers=2, dropout=0.05)
- batch_norm=True, gradient_clip_val=1.0
- max_epochs=40, patience=8, batch_size=64
- Per-target standardization (train fold only)
- 5-fold GroupKFold on canonical SMILES

Device: CPU only (Mac MPS caused >8h fold-5 thermal degradation in Round 1;
CPU is slow but predictable). Wall time on Mac M-series: 2-5 hours expected.

Per-fold checkpointing: OOF predictions saved after each fold. If interrupted
(Ctrl+C), rerun and completed folds are skipped.

Per-epoch logging: EpochLogger callback prints train_loss + val_loss to file
and stdout every epoch. Do NOT let this run silently — always tail the log.

Runs on Mac M-series CPU end-to-end. Fully self-contained; no shared utils.

Requirements:
  poly2-venv/bin/pip install chemprop

Outputs (under results/exp_chemprop_multitask_cpu/):
  run.log            — per-epoch train/val loss, per-fold R², final summary
  oof.csv            — OOF predictions: canon, target_type, y_true, y_pred
  submission.csv     — Kaggle format id, target
  cv_summary.json    — per-target R², timing, config
  checkpoint_fold_{k}.pkl.gz  — per-fold OOF + model state (for resume)
  refit_test_preds.pkl.gz     — test predictions after full-train refit

Usage:
  poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py
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
EXP_NAME = "exp_chemprop_multitask_cpu"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42

# Model config (Round 1 winning recipe)
D_H = 300
DEPTH = 4
MP_DROPOUT = 0.05
FFN_HIDDEN = 300
FFN_LAYERS = 2
FFN_DROPOUT = 0.05
BATCH_NORM = True

# Trainer config (tuned down slightly from Round 1 for CPU wall time)
MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 64
GRAD_CLIP = 1.0
LR_INIT = 1e-3
LR_MAX = 1e-3
LR_FINAL = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS = 0  # CPU — multiprocessing overhead not worth it

DEVICE = "cpu"

# Refit config: pool best-epoch counts across folds, refit on full train for that many
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
    fh = logging.FileHandler(log_path, mode="a")  # append so resumes don't wipe history
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


class EpochLogger(L.Callback):
    """Per-epoch train/val loss printer. CRITICAL — Round 1 hung 9h silently
       without this. Do NOT remove."""
    def __init__(self, logger, fold_k):
        self.logger = logger
        self.fold_k = fold_k
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
            f"[fold {self.fold_k}] epoch {epoch:>3d}  "
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
    """Return (canon_list, y_matrix) where y_matrix is shape (n_canon, 7) with NaN for unlabeled."""
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    # Ensure all 7 target columns exist in the right order
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
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic shuffle-then-mod GroupKFold on canonical SMILES."""
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(output_transform=None) -> MPNN:
    """Build the multitask MPNN from Round-1 config."""
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
    model = MPNN(
        mp, agg, ffn,
        batch_norm=BATCH_NORM,
        init_lr=LR_INIT, max_lr=LR_MAX, final_lr=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
    )
    return model


def make_datasets(
    canons: list[str],
    y: np.ndarray,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    featurizer,
    log: logging.Logger,
) -> tuple[data.MoleculeDataset, data.MoleculeDataset, object]:
    """Build train + val MoleculeDatasets, normalize targets on train, apply to val."""
    def _mk(idxs):
        pts = []
        for i in idxs:
            smi = canons[i]
            yi = y[i]  # shape (7,) with possible NaN
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            pts.append(data.MoleculeDatapoint(mol=m, y=yi))
        return pts

    train_pts = _mk(train_idxs)
    val_pts = _mk(val_idxs)
    log.info(f"    train_pts={len(train_pts)}  val_pts={len(val_pts)}")

    train_dset = data.MoleculeDataset(train_pts, featurizer=featurizer)
    val_dset = data.MoleculeDataset(val_pts, featurizer=featurizer)

    # Per-target standardization on train fold only
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)
    log.info(f"    target scaler fit on train fold (per-target mean/std)")

    return train_dset, val_dset, scaler


# ============================================================================
# TRAINING PER FOLD
# ============================================================================

def train_fold(
    fold_k: int,
    canons: list[str],
    y: np.ndarray,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    featurizer,
    log: logging.Logger,
) -> dict:
    """Train one fold, return val OOF predictions + best_epoch."""
    log.info(f"[fold {fold_k}] starting.  n_train_canon={len(train_idxs)}  n_val_canon={len(val_idxs)}")
    L.seed_everything(SEED + fold_k, workers=True)

    train_dset, val_dset, scaler = make_datasets(canons, y, train_idxs, val_idxs, featurizer, log)

    train_loader = data.build_dataloader(
        train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = data.build_dataloader(
        val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_model(output_transform=output_transform)
    log.info(f"[fold {fold_k}] model built.  n_params={sum(p.numel() for p in model.parameters()):,}")

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=False)
    epoch_logger = EpochLogger(log, fold_k)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False,   # tqdm inside Lightning gets noisy; rely on EpochLogger
        enable_checkpointing=False,   # we save our own
        logger=False,
        callbacks=[early_stop, epoch_logger],
        deterministic=False,          # deterministic + MPS/CPU can slow things
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    fold_time = time.time() - t0
    best_epoch = trainer.current_epoch - PATIENCE if early_stop.stopped_epoch > 0 else trainer.current_epoch
    log.info(f"[fold {fold_k}] training done.  time={fold_time/60:.1f}min  final_epoch={trainer.current_epoch}  approx_best_epoch={best_epoch}")

    # Predict OOF (val fold) — targets automatically un-normalized via output_transform
    log.info(f"[fold {fold_k}] predicting val OOF...")
    model.eval()
    preds_list = trainer.predict(model, val_loader)
    val_preds = torch.cat(preds_list, dim=0).cpu().numpy()  # shape (n_val, 7)
    log.info(f"[fold {fold_k}] val_preds shape={val_preds.shape}")

    # Compute per-target OOF R² for this fold
    val_true = y[val_idxs]  # (n_val, 7) with NaN
    fold_r2 = {}
    for t_idx, tgt in enumerate(TARGETS):
        mask = ~np.isnan(val_true[:, t_idx])
        if mask.sum() < 5:
            fold_r2[tgt] = None
            continue
        r2 = r2_score(val_true[mask, t_idx], val_preds[mask, t_idx])
        fold_r2[tgt] = float(r2)
    log.info(f"[fold {fold_k}] per-target val R²: " +
             "  ".join([f"{t}={fold_r2[t]:.4f}" if fold_r2[t] is not None else f"{t}=n/a" for t in TARGETS]))

    return {
        "fold_k": fold_k,
        "val_idxs": val_idxs,
        "val_preds": val_preds,     # (n_val, 7)
        "val_true": val_true,        # (n_val, 7)
        "best_epoch": int(best_epoch),
        "final_epoch": int(trainer.current_epoch),
        "fold_time_min": fold_time / 60,
        "fold_r2": fold_r2,
        "target_mean": scaler.mean_.tolist(),
        "target_std": scaler.scale_.tolist(),
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
# REFIT ON FULL TRAIN + TEST PREDICTIONS
# ============================================================================

def refit_and_predict_test(
    canons: list[str],
    y: np.ndarray,
    test_canon_unique: list[str],
    featurizer,
    n_epochs: int,
    log: logging.Logger,
) -> np.ndarray:
    """Refit on full training set for n_epochs, then predict all 7 targets for each unique test SMILES.
       Returns predictions of shape (n_test_unique, 7).
    """
    log.info(f"[REFIT] fitting on full train for {n_epochs} epochs")
    L.seed_everything(SEED, workers=True)

    all_pts = []
    for i, smi in enumerate(canons):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        all_pts.append(data.MoleculeDatapoint(mol=m, y=y[i]))
    log.info(f"[REFIT] full train datapoints: {len(all_pts)}")

    full_dset = data.MoleculeDataset(all_pts, featurizer=featurizer)
    scaler = full_dset.normalize_targets()
    full_loader = data.build_dataloader(
        full_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    log.info(f"[REFIT] scaler mean={scaler.mean_}  scale={scaler.scale_}")

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_model(output_transform=output_transform)
    epoch_logger = EpochLogger(log, fold_k="REFIT")

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
    log.info(f"[REFIT] done.  time={(time.time()-t0)/60:.1f}min")

    # Predict test set (all 7 targets per unique SMILES)
    log.info(f"[REFIT] predicting {len(test_canon_unique)} unique test SMILES")
    test_pts = []
    for smi in test_canon_unique:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            # placeholder: return zeros for un-parseable (shouldn't happen — earlier check passed)
            test_pts.append(None)
        else:
            test_pts.append(data.MoleculeDatapoint(mol=m, y=np.zeros(N_TARGETS, dtype=np.float32)))

    valid_mask = [p is not None for p in test_pts]
    if not all(valid_mask):
        log.info(f"[REFIT] WARNING: {sum(not v for v in valid_mask)} un-parseable test SMILES")
    test_valid_pts = [p for p in test_pts if p is not None]
    test_dset = data.MoleculeDataset(test_valid_pts, featurizer=featurizer)
    # Do NOT normalize test targets (they're placeholder zeros); rely on output_transform in model
    test_loader = data.build_dataloader(
        test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    preds_list = trainer.predict(model, test_loader)
    test_preds = torch.cat(preds_list, dim=0).cpu().numpy()

    # Re-align to include any un-parseable placeholders (shouldn't happen)
    aligned = np.zeros((len(test_canon_unique), N_TARGETS), dtype=np.float32)
    j = 0
    for i, valid in enumerate(valid_mask):
        if valid:
            aligned[i] = test_preds[j]; j += 1
        else:
            aligned[i] = np.nan
    return aligned


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info("=" * 60)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: device={DEVICE}  n_splits={N_SPLITS}  seed={SEED}  "
             f"max_epochs={MAX_EPOCHS}  patience={PATIENCE}  batch_size={BATCH_SIZE}")
    log.info(f"MODEL: d_h={D_H} depth={DEPTH} mp_dropout={MP_DROPOUT}  "
             f"ffn_hidden={FFN_HIDDEN} ffn_layers={FFN_LAYERS} ffn_dropout={FFN_DROPOUT}  "
             f"batch_norm={BATCH_NORM}")

    # Seeds
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    L.seed_everything(SEED, workers=True)

    # Force CPU (defensive)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log.info(f"torch threads: {torch.get_num_threads()}")

    t_start = time.time()

    # Load + canonicalize
    tr, te = load_and_canonicalize(log)
    canons, y = build_wide_train(tr)
    log.info(f"wide train: n_canon={len(canons)}  y shape={y.shape}  "
             f"NaN fraction={100*np.isnan(y).mean():.1f}%")
    log.info(f"per-target labeled counts: " +
             "  ".join([f"{t}={int((~np.isnan(y[:, i])).sum())}" for i, t in enumerate(TARGETS)]))

    # Test uniques
    test_canon_unique = te["canon"].drop_duplicates().tolist()
    log.info(f"unique test SMILES to predict: {len(test_canon_unique)}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # 5-fold CV with per-fold checkpointing
    splits = group_kfold_splits(canons, N_SPLITS, SEED)
    fold_results: list[dict] = []
    for k, (tri, vai) in enumerate(splits):
        cached = load_fold_checkpoint(EXP_DIR, k)
        if cached is not None:
            log.info(f"[fold {k}] loaded checkpoint (skipping training)")
            fold_results.append(cached)
            continue
        log.info("=" * 60)
        log.info(f"FOLD {k} of {N_SPLITS}")
        log.info("=" * 60)
        result = train_fold(k, canons, y, tri, vai, featurizer, log)
        save_fold_checkpoint(EXP_DIR, k, result, log)
        fold_results.append(result)

    # Assemble OOF predictions
    oof_preds = np.full((len(canons), N_TARGETS), np.nan, dtype=np.float32)
    for r in fold_results:
        oof_preds[r["val_idxs"]] = r["val_preds"]

    # Per-target OOF R²
    log.info("=" * 60)
    log.info("PER-TARGET OOF R² (multitask Chemprop, no blending)")
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
            "fold_r2s": [r["fold_r2"].get(tgt) for r in fold_results],
        }
        log.info(f"  {tgt:>4s}  n={int(mask.sum()):>5d}  OOF R²={r2:.4f}  "
                 f"folds={[f'{v:.3f}' if v is not None else 'n/a' for v in per_target[tgt]['fold_r2s']]}")
    mean_r2 = float(np.mean([per_target[t]["oof_r2"] for t in TARGETS]))
    log.info(f"  MEAN R² = {mean_r2:.4f}")

    # Determine refit epochs — median best_epoch across folds × multiplier, floor at 15
    best_epochs = [r["best_epoch"] for r in fold_results if r.get("best_epoch")]
    refit_epochs = max(15, int(np.median(best_epochs) * REFIT_ITER_MULTIPLIER))
    log.info(f"fold best_epochs = {best_epochs}   →   refit for {refit_epochs} epochs")

    # Refit + test predictions (with cache)
    refit_cache = EXP_DIR / "refit_test_preds.pkl.gz"
    if refit_cache.exists():
        log.info(f"loading cached refit test predictions from {refit_cache.name}")
        with gzip.open(refit_cache, "rb") as f:
            test_preds_by_canon = pickle.load(f)
    else:
        test_preds_aligned = refit_and_predict_test(
            canons, y, test_canon_unique, featurizer, refit_epochs, log,
        )
        test_preds_by_canon = {smi: test_preds_aligned[i] for i, smi in enumerate(test_canon_unique)}
        with gzip.open(refit_cache, "wb") as f:
            pickle.dump(test_preds_by_canon, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached refit test predictions to {refit_cache.name}")

    # Build OOF DataFrame (one row per (canon, target_type) in train)
    oof_rows = []
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    for _, row in tr.iterrows():
        c = row["canon"]
        t = row["target_type"]
        i = canon_to_idx.get(c)
        if i is None:
            continue
        t_idx = TARGET_IDX[t]
        oof_rows.append({
            "canon": c,
            "target_type": t,
            "y_true": float(row["target"]),
            "y_pred": float(oof_preds[i, t_idx]) if not np.isnan(oof_preds[i, t_idx]) else np.nan,
        })
    oof_df = pd.DataFrame(oof_rows)
    oof_path = EXP_DIR / "oof.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_df)}")

    # Build submission (one row per test id)
    sub_rows = []
    for _, row in te.iterrows():
        c = row["canon"]
        t = row["target_type"]
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

    # Summary JSON
    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "per_target":     per_target,
        "config": {
            "device":         DEVICE,
            "n_splits":       N_SPLITS,
            "seed":           SEED,
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
        "fold_times_min": [r["fold_time_min"] for r in fold_results if "fold_time_min" in r],
        "fold_best_epochs": best_epochs,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = EXP_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    log.info("=" * 60)
    log.info(f"DONE.  mean OOF R² = {mean_r2:.4f}  total wall time = {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
