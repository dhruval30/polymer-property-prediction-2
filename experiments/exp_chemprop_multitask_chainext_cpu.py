"""
exp_chemprop_multitask_chainext_cpu.py — Chemprop D-MPNN multitask on TRIMER SMILES (not monomer).

============================================================================
WHY THIS EXISTS
============================================================================

The chain-extension trick that gave LGB its biggest single-experiment LB
jump (0.860 → 0.894, +0.034) applied to the tree model. Now apply it to
Chemprop.

Reasoning:
  - Chemprop 3-seed monomer  : LB 0.892 (current strongest neural base)
  - LGB monomer + Maxwell    : LB 0.860
  - LGB TRIMER + Maxwell     : LB 0.894  (+0.034 from chain extension)

If chain-ext gives Chemprop the same *proportional* lift, expected LB is
0.895-0.905 solo. Even a fraction of the LGB lift (say +0.005) puts
Chemprop 3-seed at 0.897 solo — matching our current best ensemble but as
a single model.

More importantly for the blend problem: chain-ext Chemprop is more likely
to complement mono-LGB than chain-ext LGB was (chain-ext LGB failed to
blend because both it and Chemprop learned polymer-context features from
different angles → too correlated). A chain-ext CHEMPROP + mono-LGB blend
might restore the diversity that mono-Chemprop + mono-LGB had.

Research reference: research doc §5.1 (chain extension, 3rd-place NeurIPS
2025) and §4 (multitask Chemprop, Round 1 winner). This script combines
the two.

============================================================================
WHAT'S DIFFERENT FROM exp_chemprop_multitask_cpu_3seed.py
============================================================================

Every SMILES (train AND test) is expanded from monomer `*A*` to a trimer
`*AAA*` via the polymer_to_multimer function (copied verbatim from
exp_chain_ext_lgbm.py) BEFORE being handed to Chemprop's featurizer. Chemprop
sees each polymer as a chain 3× longer than the monomer.

The rest — 5-fold GroupKFold, per-target NNLS-ready OOF format, EpochLogger
to prevent silent hangs, per-fold checkpointing, CPU device — is unchanged.

Split seed (42) and fold assignment are identical to the monomer 3-seed
script so this OOF can be per-target NNLS-blended with the existing OOFs
(mono Chemprop, mono LGB, chain-ext LGB, CAT).

============================================================================
WALL TIME — READ BEFORE STARTING
============================================================================

Trimer graphs have ~3× the edges of monomer graphs. D-MPNN message passing
scales O(edges × depth × d_h). So per-epoch time scales ~3×.

Ballpark on Mac M-series CPU:
  - Monomer 3-seed (60 epochs, 5 folds + 3 refits) took **225 min**.
  - Trimer 3-seed same config estimated **~10-12 hours**.

Options (set at top of file):
  MODE = "full"   → 3 seeds × 60 epochs. Fair comparison to monomer. ~10-12h.
  MODE = "medium" → 2 seeds × 60 epochs. ~7-8h. Recommended for first run.
  MODE = "fast"   → 2 seeds × 40 epochs. ~4-5h. Quick check.

**Ctrl+C is safe** between folds — per-fold checkpointing writes each fold's
3-seed bag once complete. Interrupting mid-fold restarts that fold from
scratch on next run.

Default here: `MODE="medium"` (2-seed × 60 epochs, ~7-8h). Change to "full"
if you want fair 3-seed comparison with the monomer baseline.

============================================================================
DEPENDENCIES
============================================================================

  - poly2-venv (Python 3.11) with chemprop 2.x + rdkit already installed
  - Data: ppp-round-2/{train,test}.csv

============================================================================
OUTPUTS  (under results/exp_chemprop_multitask_chainext_cpu/)
============================================================================

  run.log                          — per-epoch train/val loss, per-fold-per-seed R², final summary
  oof.csv                          — averaged OOF predictions (canon, target_type, y_true, y_pred)
                                     `canon` = MONOMER canonical SMILES (aligns with mono OOFs for blending!)
  submission.csv                   — averaged test predictions (id, target)
  cv_summary.json                  — per-target R², per-seed R², timing, config
  checkpoint_fold_{k}.pkl.gz       — per-fold bundle
  refit_test_preds.pkl.gz          — cached refit test predictions

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chemprop_multitask_chainext_cpu.py

Then submit results/exp_chemprop_multitask_chainext_cpu/submission.csv to Kaggle.

============================================================================
EXPECTED
============================================================================

Solo LB:
  - Optimistic: +0.005 to +0.010 over monomer 3-seed (0.892 → 0.897-0.902).
  - Pessimistic: 0 (chain extension helps LGB more than graph models because
    LGB was missing chain context that Chemprop already learns via message passing).

Blend LB (chain-ext Chemprop + mono-LGB, 2-way):
  - Optimistic: 0.900-0.905 (new best).
  - Pessimistic: 0.897 (matches current best but no lift).

Blend LB (chain-ext Chemprop + mono-LGB + chain-ext LGB, 3-way):
  - Optimistic: 0.902-0.907 if NNLS finds orthogonal signal across all 3.
  - Pessimistic: 0.897 (same ceiling).

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
EXP_NAME = "exp_chemprop_multitask_chainext_cpu"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Chain extension
CHAIN_N_UNITS = 3            # trimer, matches exp_chain_ext_lgbm.py

# CV — MUST match exp_chemprop_multitask_cpu_3seed.py for OOF alignment
N_SPLITS = 5
SPLIT_SEED = 42

# Bag mode — change here to trade compute vs coverage
MODE = "full"              # "full" | "medium" | "fast"

if MODE == "full":
    MODEL_SEEDS = (42, 43, 44)
    MAX_EPOCHS = 60
elif MODE == "medium":
    MODEL_SEEDS = (42, 43)
    MAX_EPOCHS = 60
elif MODE == "fast":
    MODEL_SEEDS = (42, 43)
    MAX_EPOCHS = 40
else:
    raise ValueError(f"unknown MODE={MODE}")

# Model config (unchanged from monomer Chemprop 3-seed)
D_H = 300
DEPTH = 4
MP_DROPOUT = 0.05
FFN_HIDDEN = 300
FFN_LAYERS = 2
FFN_DROPOUT = 0.05
BATCH_NORM = True

# Trainer config
PATIENCE = 10
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
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


class EpochLogger(L.Callback):
    """Per-epoch train/val loss printer. Prevents silent hangs (Round 1 lesson)."""
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
# POLYMER CHAIN EXTENSION  (copied verbatim from exp_chain_ext_lgbm.py)
# ============================================================================

def polymer_to_multimer(smi: str, n_units: int = CHAIN_N_UNITS) -> str:
    """Extend polymer *A* SMILES to n-mer chain (head-to-tail). Return canonical SMILES.

    Falls back to the original SMILES on any structural issue so featurization
    always succeeds. Handles the standard 2-wildcard polymer convention.
    """
    if n_units <= 1:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi

    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        return smi

    star_a, star_b = stars
    a_bonds = m.GetAtomWithIdx(star_a).GetBonds()
    b_bonds = m.GetAtomWithIdx(star_b).GetBonds()
    if len(a_bonds) != 1 or len(b_bonds) != 1:
        return smi

    connect_a = a_bonds[0].GetOtherAtomIdx(star_a)
    connect_b = b_bonds[0].GetOtherAtomIdx(star_b)
    bond_type_a = a_bonds[0].GetBondType()
    bond_type_b = b_bonds[0].GetBondType()

    editable = Chem.RWMol(m)
    for idx in sorted(stars, reverse=True):
        editable.RemoveAtom(idx)

    def adjust(orig_idx: int, removed_sorted: list[int]) -> int:
        return orig_idx - sum(1 for r in removed_sorted if r < orig_idx)

    removed_sorted = sorted(stars)
    ca = adjust(connect_a, removed_sorted)
    cb = adjust(connect_b, removed_sorted)
    core = editable.GetMol()
    n_atoms_core = core.GetNumAtoms()
    if n_atoms_core == 0:
        return smi

    result = Chem.RWMol(core)
    prev_cb = cb
    first_ca = ca
    for i in range(1, n_units):
        result = Chem.RWMol(Chem.CombineMols(result, core))
        offset = result.GetNumAtoms() - n_atoms_core
        new_ca = offset + ca
        new_cb = offset + cb
        result.AddBond(prev_cb, new_ca, bond_type_a)
        prev_cb = new_cb

    left_star = result.AddAtom(Chem.Atom(0))
    right_star = result.AddAtom(Chem.Atom(0))
    result.AddBond(first_ca, left_star, bond_type_a)
    result.AddBond(prev_cb, right_star, bond_type_b)

    try:
        final = result.GetMol()
        Chem.SanitizeMol(final)
        return Chem.MolToSmiles(final, canonical=True)
    except Exception:
        return smi


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


def build_trimer_map(canons_unique: list[str], log: logging.Logger) -> dict[str, str]:
    """Map monomer canonical SMILES → trimer canonical SMILES. Cache-friendly."""
    log.info(f"expanding {len(canons_unique)} unique monomer SMILES → {CHAIN_N_UNITS}-mer chains")
    mapping = {}
    n_fallback = 0
    for s in tqdm(canons_unique, desc=f"mono→{CHAIN_N_UNITS}mer", ncols=100):
        tri = polymer_to_multimer(s, CHAIN_N_UNITS)
        mapping[s] = tri
        if tri == s:
            n_fallback += 1
    log.info(f"  chain extension: {len(canons_unique) - n_fallback} extended, "
             f"{n_fallback} fell back to original ({100*n_fallback/max(1,len(canons_unique)):.1f}%)")
    return mapping


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
# CV — MUST match exp_chemprop_multitask_cpu_3seed for OOF alignment
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
    trimer_map: dict[str, str],
    featurizer,
) -> tuple[data.MoleculeDataset, data.MoleculeDataset, object]:
    """Build fresh train + val datasets on TRIMER SMILES, normalize targets on train fold."""
    def _mk(idxs):
        pts = []
        for i in idxs:
            tri_smi = trimer_map[canons[i]]
            m = Chem.MolFromSmiles(tri_smi)
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
    trimer_map: dict[str, str],
    featurizer,
) -> tuple[data.MoleculeDataset, object]:
    all_pts = []
    for i, smi in enumerate(canons):
        tri_smi = trimer_map[smi]
        m = Chem.MolFromSmiles(tri_smi)
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
    trimer_map: dict[str, str],
    featurizer,
    ctx: str,
    log: logging.Logger,
) -> tuple[np.ndarray, float]:
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

    # Predict test on TRIMER SMILES
    test_pts = []
    valid_mask = []
    for smi in test_canon_unique:
        tri_smi = trimer_map[smi]
        m = Chem.MolFromSmiles(tri_smi)
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
# TRAIN ONE FOLD (N-seed bag)
# ============================================================================

def train_fold_bag(
    fold_k: int,
    canons: list[str],
    y: np.ndarray,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    trimer_map: dict[str, str],
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
        train_dset, val_dset, scaler = make_train_val_datasets(
            canons, y, train_idxs, val_idxs, trimer_map, featurizer,
        )
        val_preds, best_epoch, wall = train_single_model_cv(train_dset, val_dset, scaler, seed, ctx, log)
        val_preds_per_seed.append(val_preds)
        best_epochs.append(best_epoch)
        wall_times.append(wall)

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
        "val_preds_per_seed": val_preds_per_seed,
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
# N-SEED REFIT + TEST PREDICTIONS
# ============================================================================

def refit_bag_and_predict_test(
    canons: list[str],
    y: np.ndarray,
    test_canon_unique: list[str],
    trimer_map: dict[str, str],
    featurizer,
    n_epochs: int,
    log: logging.Logger,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    test_preds_per_seed = []
    wall_times = []
    for si, seed in enumerate(MODEL_SEEDS):
        ctx = f"REFIT seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
        log.info(f"[{ctx}] starting refit for {n_epochs} epochs")
        full_dset, scaler = make_full_dataset(canons, y, trimer_map, featurizer)
        test_preds, wall = train_single_model_refit(
            full_dset, scaler, seed, n_epochs,
            test_canon_unique, trimer_map, featurizer, ctx, log,
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
    log.info(f"MODE={MODE}   CHAIN_N_UNITS={CHAIN_N_UNITS}")
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

    # ---- Build monomer → trimer map over ALL canonical SMILES (train + test) ----
    all_canons_unique = list(set(canons) | set(test_canon_unique))
    trimer_map = build_trimer_map(all_canons_unique, log)

    # Sanity check: report a couple example expansions
    sample_examples = 3
    log.info(f"trimer expansion examples ({sample_examples}):")
    for i, mono in enumerate(all_canons_unique[:sample_examples]):
        tri = trimer_map[mono]
        log.info(f"  mono[{i}]: {mono[:70]}{'...' if len(mono)>70 else ''}")
        log.info(f"  tri [{i}]: {tri[:70]}{'...' if len(tri)>70 else ''}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # ---- 5-fold CV × N-seed with per-fold checkpointing ----
    splits = group_kfold_splits(canons, N_SPLITS, SPLIT_SEED)
    fold_results = []
    for k, (tri, vai) in enumerate(splits):
        cached = load_fold_checkpoint(EXP_DIR, k)
        if cached is not None:
            log.info(f"[fold {k}] loaded checkpoint (skipping training)")
            fold_results.append(cached)
            continue
        result = train_fold_bag(k, canons, y, tri, vai, trimer_map, featurizer, log)
        save_fold_checkpoint(EXP_DIR, k, result, log)
        fold_results.append(result)

    # ---- Assemble OOF ----
    oof_preds = np.full((len(canons), N_TARGETS), np.nan, dtype=np.float32)
    for r in fold_results:
        oof_preds[r["val_idxs"]] = r["val_preds_avg"]

    log.info("=" * 60)
    log.info(f"PER-TARGET OOF R²  (5-fold × {len(MODEL_SEEDS)}-seed bagged on trimer SMILES)")
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
            "fold_r2s_per_seed": [r["fold_r2_per_seed"] for r in fold_results],
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
        test_preds_avg, test_preds_per_seed, refit_wall_times = refit_bag_and_predict_test(
            canons, y, test_canon_unique, trimer_map, featurizer, refit_epochs, log,
        )
        with gzip.open(refit_cache, "wb") as f:
            pickle.dump({
                "test_preds_avg": test_preds_avg,
                "test_preds_per_seed": test_preds_per_seed,
                "wall_times": refit_wall_times,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached refit test predictions to {refit_cache.name}")

    test_preds_by_canon = {smi: test_preds_avg[i] for i, smi in enumerate(test_canon_unique)}

    # ---- Build OOF DataFrame using MONOMER canonical SMILES (for blend alignment) ----
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
        "mode":           MODE,
        "chain_n_units":  CHAIN_N_UNITS,
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

    # ---- Comparison vs monomer 3-seed Chemprop reference ----
    log.info("=" * 60)
    log.info("PER-TARGET OOF R²  (trimer Chemprop)  vs monomer 3-seed Chemprop reference")
    log.info("=" * 60)
    mono_ref = {"eea": 0.9082, "egb": 0.9305, "egc": 0.9070,
                "ei":  0.7766, "eps": 0.7916, "nc":  0.8681, "tg":  0.9083}
    log.info(f"  {'target':>6s}  {'trimer':>10s}  {'mono 3s':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = per_target[t]['oof_r2']
        ref = mono_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    mono_mean = float(np.mean(list(mono_ref.values())))
    log.info(f"  {'MEAN':>6s}  {mean_r2:>10.4f}  {mono_mean:>10.4f}  {mean_r2 - mono_mean:>+8.4f}")
    log.info(f"  (monomer Chemprop 3-seed LB reference: 0.892)")
    log.info(f"total wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
