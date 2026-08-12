"""
reproduce.py — end-to-end pipeline producing the LB 0.902 submission.

============================================================================
WHAT THIS DOES (in order)
============================================================================

1. **LGB (mono + Maxwell physics prior)** — LightGBM per-target on monomer
   fingerprints + RDKit descriptors + 14 aux matrix-completion features.
   Maxwell EPS↔Nc post-fit as physics prior. Solo LB reference: 0.860.

2. **Chemprop 3-seed multitask** — 5-fold × 3-seed bagged D-MPNN with
   shared graph encoder and 7 regression heads (one per target). CPU-only,
   ~4h wall time. Solo LB reference: 0.892.

3. **2-way NNLS blend** — per-target non-negative-least-squares blend of the
   Chemprop 3-seed and LGB+Maxwell predictions. Chemprop weight floor 0.40
   + additive bias +0.15 to correct for OOF-LB gap disparity between the
   two bases. Blend LB reference: 0.897.

4. **Koopmans bandgap post-processor** — physics rule `Egc ≈ Ei − Eea`
   applied as a per-target OOF-fit blend weight to the 3 bandgap-related
   targets (Egc, Ei, Eea). Uses Chemprop OOF for α tuning (n=2028/222/221
   per target) and Chemprop refit predictions for the physics term at test.
   Final LB: **0.902**.

============================================================================
REQUIREMENTS
============================================================================

  Python 3.11 (required for Chemprop 2.x)
  pip packages:
      numpy pandas scipy scikit-learn tqdm
      rdkit
      lightgbm
      torch lightning chemprop

============================================================================
DATA LAYOUT
============================================================================

  Adjust DATA_DIR at the top of this file to point to a directory containing:
      train.csv    (columns: id, smiles, target_type, target)
      test.csv     (columns: id, smiles, target_type)

  Local run (default):    ../ppp-round-2/
  Kaggle notebook:        /kaggle/input/<comp-name>/ppp-round-2/
                          (or wherever host mounts the dataset)

============================================================================
OUTPUTS
============================================================================

  work_0902/                     — all intermediates + logs
      reproduce.log              — full training log (also stdout)
      lgb_maxwell/               — Phase 1 outputs
          oof.csv, submission.csv, feature_cache.pkl
      chemprop_3seed/            — Phase 2 outputs
          checkpoint_fold_{0..4}.pkl.gz
          refit_test_preds.pkl.gz
          oof.csv, submission.csv
      blend_nnls/                — Phase 3 outputs
          submission.csv, blend_summary.json
      koopmans/                  — Phase 4 outputs
          submission.csv, koopmans_summary.json
  submission.csv                 — FINAL submission (LB 0.902) — copy of Phase 4

============================================================================
CHECKPOINTING
============================================================================

Each phase checks for its output files first. If present, skips the phase.
- Phase 1 (LGB, ~20 min): skipped if lgb_maxwell/submission.csv exists.
- Phase 2 (Chemprop, ~4h): resumes per fold via checkpoint_fold_k.pkl.gz;
  skipped entirely if refit_test_preds.pkl.gz already exists.
- Phases 3-4: cheap, always rerun.

Delete work_0902/<phase>/ to force re-run of that phase.

============================================================================
WALL TIME (fresh run, no cache)
============================================================================

  Phase 1 (LGB features + train):  ~15-20 min
  Phase 2 (Chemprop 3-seed):       ~3.5-4 h  (CPU)
                                    ~1-1.5 h (GPU, if available)
  Phase 3 (blend NNLS):             <1 sec
  Phase 4 (Koopmans post-fit):      <5 sec

  Total: ~4 hours CPU, ~1.5-2 hours GPU. Kaggle-notebook-compatible.

============================================================================
"""
from __future__ import annotations

# --- stdlib ---
import gzip
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

# --- third-party ---
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from chemprop import data, featurizers, nn
from chemprop.models import MPNN

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG — adjust DATA_DIR for your environment
# ============================================================================

# For local run: DATA_DIR = <path to dir containing train.csv, test.csv>
# For Kaggle:    DATA_DIR = Path("/kaggle/input/<comp>/ppp-round-2")
DATA_DIR = Path(__file__).resolve().parent.parent / "ppp-round-2"

# Where to write intermediates and final output
WORK_DIR = Path(__file__).resolve().parent / "work_0902"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SPLIT_SEED = 42

# ---- LGB (Phase 1) ----
MORGAN2_NBITS = 2048
MORGAN3_NBITS = 2048
ATOMPAIR_NBITS = 2048
TOPTORSION_NBITS = 2048
AVALON_NBITS = 512

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    feature_fraction=0.5,
    bagging_fraction=0.85,
    bagging_freq=1,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=-1,
    seed=SPLIT_SEED,
)
N_ESTIMATORS = 4000
EARLY_STOP_ROUNDS = 200
REFIT_ITER_MULTIPLIER = 1.10
BLEND_W_GRID = np.linspace(0.0, 1.0, 201)

# ---- Chemprop (Phase 2) ----
MODEL_SEEDS = (42, 43, 44, 45, 46, 47, 48)   # 7-seed bag (LB 0.901 variant, selected as robustness backup)
D_H = 300
DEPTH = 4
MP_DROPOUT = 0.05
FFN_HIDDEN = 300
FFN_LAYERS = 2
FFN_DROPOUT = 0.05
BATCH_NORM = True
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
GRAD_CLIP = 1.0
LR_INIT = 1e-3
LR_MAX = 1e-3
LR_FINAL = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS = 0
DEVICE = "cpu"                       # override to "gpu" on Kaggle if desired

# ---- Blend (Phase 3) ----
CHEMPROP_WEIGHT_FLOOR = 0.40
APPLY_CHEMPROP_BIAS = 0.15

# ---- Koopmans (Phase 4) ----
PHYSICS_TARGETS = ("egc", "ei", "eea")
ALPHA_GRID = np.arange(0.5, 1.001, 0.025)
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),   # Koopmans: bandgap = IE - EA
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(work_dir: Path) -> logging.Logger:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "reproduce.log"
    logger = logging.getLogger("reproduce_0902")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


class EpochLogger(L.Callback):
    """Per-epoch train/val loss printer — critical for Chemprop CPU runs
    (prevents silent hangs that plagued Round 1)."""
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
# DATA LOADING + CANONICALIZATION
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_and_canonicalize(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"loading train.csv / test.csv from {DATA_DIR}")
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
            .agg(target=("target", "mean"), smiles=("smiles", "first")))
    log.info(f"train after dedup: {tr.shape}")
    log.info(f"per-target train counts: {tr['target_type'].value_counts().to_dict()}")
    log.info(f"per-target test  counts: {te['target_type'].value_counts().to_dict()}")
    return tr, te


# ============================================================================
# LGB FEATURE COMPUTATION (monomer only — no chain extension)
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def _mol(smi: str):
    return Chem.MolFromSmiles(_cap(smi))


def compute_rdkit_desc(smi: str) -> dict | None:
    m = _mol(smi)
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))


def _count_fp_to_arr(fp, nbits: int) -> np.ndarray:
    out = np.zeros(nbits, dtype=np.int32)
    for k, v in fp.GetNonzeroElements().items():
        out[k] = v
    return out


def compute_morgan_count(smi: str, radius: int, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(AllChem.GetHashedMorganFingerprint(m, radius, nBits=nbits), nbits)


def compute_maccs(smi: str) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(167, dtype=np.int8)
    return np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8)


def compute_atompair_count(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=nbits), nbits)


def compute_toptorsion_count(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=nbits), nbits)


def compute_avalon(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int8)
    return np.array(pyAvalonTools.GetAvalonFP(m, nBits=nbits), dtype=np.int8)


def _sanitize_desc_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        med = df[c].median()
        if pd.isna(med): med = 0.0
        df[c] = df[c].fillna(med)
    for c in df.columns:
        lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
        if lo == hi or not np.isfinite(lo) or not np.isfinite(hi): continue
        df[c] = df[c].clip(lo, hi)
    df = df.clip(-1e10, 1e10)   # hard float32-safety cap
    dropped = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=dropped)
    return df, dropped


def build_feature_bundle_lgb(canon_smiles: list[str], log: logging.Logger) -> dict:
    """Full monomer-only feature stack for LGB (matches exp_maxwell_prior_lgbm)."""
    smis = list(dict.fromkeys(canon_smiles))
    log.info(f"unique canonical SMILES: {len(smis)}")

    parts, families_slice, cursor = [], {}, 0
    def _add(name: str, arr: np.ndarray):
        nonlocal cursor
        parts.append(arr)
        families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    log.info("computing RDKit descriptors")
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis, desc="rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X = df_desc.values.astype(np.float32)
    _add("desc", X)
    log.info(f"  desc: {X.shape}  dropped={len(dropped)}  time={time.time()-t0:.1f}s")

    for name, fn in [
        ("morgan2c",     lambda: np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis, desc="morgan-r2", ncols=100)])),
        ("morgan3c",     lambda: np.stack([compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis, desc="morgan-r3", ncols=100)])),
        ("maccs",        lambda: np.stack([compute_maccs(s) for s in tqdm(smis, desc="maccs", ncols=100)])),
        ("atompair_c",   lambda: np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis, desc="atom-pair", ncols=100)])),
        ("toptorsion_c", lambda: np.stack([compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis, desc="top-torsion", ncols=100)])),
        ("avalon",       lambda: np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis, desc="avalon", ncols=100)])),
    ]:
        t0 = time.time()
        X = fn().astype(np.float32)
        _add(name, X)
        log.info(f"  {name}: {X.shape}  time={time.time()-t0:.1f}s")

    X_full = np.concatenate(parts, axis=1)
    log.info(f"FEATURE MATRIX TOTAL: {X_full.shape}  size≈{X_full.nbytes/1e6:.1f}MB")
    return {
        "X": X_full,
        "smiles_index": {s: i for i, s in enumerate(smis)},
        "families_slice": families_slice,
    }


def get_or_build_features_lgb(all_canon: list[str], cache_path: Path, log: logging.Logger) -> dict:
    key = hashlib.md5(
        (str(sorted(set(all_canon))) +
         f"m2={MORGAN2_NBITS};m3={MORGAN3_NBITS};ap={ATOMPAIR_NBITS};tt={TOPTORSION_NBITS};av={AVALON_NBITS}"
         ).encode()
    ).hexdigest()[:12]

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded LGB feature cache: {cache_path.name}  key={key}")
                return bundle
            log.info("LGB feature cache key mismatch; rebuilding")
        except Exception as e:
            log.info(f"failed to load LGB cache ({e}); rebuilding")

    bundle = build_feature_bundle_lgb(all_canon, log)
    bundle["_key"] = key
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached LGB features to {cache_path.name}")
    return bundle


def slice_smiles_features(bundle: dict, canon_series: pd.Series) -> np.ndarray:
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# ============================================================================
# AUX (matrix completion) FEATURES for LGB
# ============================================================================

def build_aux_lookup(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    lookup: dict[str, np.ndarray] = {}
    grouped = train_df.groupby("canon")
    for canon, g in tqdm(grouped, desc="build aux lookup", ncols=100, total=grouped.ngroups):
        row = empty.copy()
        for tt, gg in g.groupby("target_type"):
            if tt in TARGET_IDX:
                idx = TARGET_IDX[tt]
                row[idx] = float(gg["target"].mean())
                row[idx + N_TARGETS] = 1.0
        lookup[canon] = row
    return lookup


def aux_features_for_target(canon_series: pd.Series, target: str,
                             lookup: dict[str, np.ndarray]) -> np.ndarray:
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


# ============================================================================
# CV SPLITS  (identical fold assignment for LGB and Chemprop)
# ============================================================================

def group_kfold_splits(canon_arr, n_splits: int = N_SPLITS,
                       seed: int = SPLIT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# MAXWELL PHYSICS PRIOR (Phase 1 helpers)
# ============================================================================

def fit_maxwell_forward(nc_values, eps_values):
    x = nc_values ** 2
    y = eps_values
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b), float(r2_score(y, a * x + b))


def fit_maxwell_reverse(eps_values, nc_values):
    x = eps_values
    y = nc_values ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred_nc = np.sqrt(np.clip(a * x + b, 1e-9, None))
    return float(a), float(b), float(r2_score(nc_values, pred_nc))


def apply_maxwell_forward(nc_values, a, b):
    return a * (nc_values ** 2) + b


def apply_maxwell_reverse(eps_values, a, b):
    return np.sqrt(np.clip(a * eps_values + b, 1e-9, None))


def search_blend_weight(y_true, y_ml, y_prior, grid=BLEND_W_GRID):
    r2s = np.array([r2_score(y_true, w * y_ml + (1 - w) * y_prior) for w in grid])
    best_i = int(np.argmax(r2s))
    baseline_r2 = float(r2_score(y_true, y_ml))
    return float(grid[best_i]), float(r2s[best_i]), baseline_r2


def build_effective_value_lookup(train_df, oof_results, target):
    lookup: dict[str, float] = {}
    tr_t = train_df[train_df["target_type"] == target]
    for _, row in tr_t.iterrows():
        lookup[row["canon"]] = float(row["target"])
    if target in oof_results:
        oof_df = oof_results[target]["oof"]
        for _, row in oof_df.iterrows():
            if row["canon"] not in lookup:
                lookup[row["canon"]] = float(row["y_pred"])
    return lookup


# ============================================================================
# PHASE 1: LGB TRAINING (Maxwell mono)  — target LB 0.860 solo
# ============================================================================

def train_lgb_one_target(target: str, tr: pd.DataFrame, te: pd.DataFrame,
                         bundle: dict, aux_lookup: dict, log: logging.Logger) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"])
    X_te_smi = slice_smiles_features(bundle, g_te["canon"])
    X_tr_aux = aux_features_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_features_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)

    log.info(f"[LGB {target}] train rows={len(g_tr)}   test rows={len(g_te)}   X shape={X_tr.shape}")

    splits = group_kfold_splits(g_tr["canon"].values)
    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    for k, (tri, vai) in enumerate(splits):
        d_tr = lgb.Dataset(X_tr[tri], y[tri])
        d_va = lgb.Dataset(X_tr[vai], y[vai], reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS, d_tr,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[d_va], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred_va = booster.predict(X_tr[vai], num_iteration=booster.best_iteration)
        oof[vai] = pred_va
        best_iters.append(int(booster.best_iteration))
        r2 = float(r2_score(y[vai], pred_va))
        fold_r2s.append(r2)
        log.info(f"[LGB {target}] fold {k}: best_iter={booster.best_iteration}  R²={r2:.4f}  n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[LGB {target}] OOF R² (pre-Maxwell) = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[LGB {target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y)
    full_booster = lgb.train(LGB_PARAMS, d_full, num_boost_round=refit_iters,
                              callbacks=[lgb.log_evaluation(0)])
    test_pred = full_booster.predict(X_te)

    return {
        "target": target,
        "oof": pd.DataFrame({
            "canon": g_tr["canon"].values, "target_type": target,
            "y_true": y, "y_pred": oof,
        }),
        "test_pred": pd.DataFrame({
            "id": g_te["id"].values, "canon": g_te["canon"].values,
            "target_type": target, "target": test_pred,
        }),
        "oof_r2": oof_r2,
        "best_iters": best_iters,
        "refit_iters": refit_iters,
    }


def phase1_lgb_maxwell(tr: pd.DataFrame, te: pd.DataFrame, log: logging.Logger) -> Path:
    """Full LGB + Maxwell pipeline. Returns directory containing oof.csv + submission.csv."""
    phase_dir = WORK_DIR / "lgb_maxwell"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"
    oof_path = phase_dir / "oof.csv"

    if sub_path.exists() and oof_path.exists():
        log.info(f"[PHASE 1] found existing outputs at {phase_dir} — skipping LGB training")
        return phase_dir

    log.info("=" * 60)
    log.info("PHASE 1: LGB (mono + Maxwell)  — target solo LB 0.860")
    log.info("=" * 60)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features_lgb(all_canon, phase_dir / "feature_cache.pkl", log)
    log.info(f"aux lookup over {tr['canon'].nunique()} unique train canons")
    aux_lookup = build_aux_lookup(tr)

    results: dict[str, dict] = {}
    for tgt in tqdm(TARGETS, desc="[LGB] targets", ncols=100):
        results[tgt] = train_lgb_one_target(tgt, tr, te, bundle, aux_lookup, log)

    # ---- Maxwell physics blend on EPS/Nc ----
    log.info("MAXWELL RELATION POST-FIT (EPS ↔ Nc)")
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    co = wide.dropna(subset=["eps", "nc"])
    log.info(f"co-labeled train molecules: n={len(co)}")

    a_fwd, b_fwd, r2_fwd = fit_maxwell_forward(co["nc"].values, co["eps"].values)
    a_rev, b_rev, r2_rev = fit_maxwell_reverse(co["eps"].values, co["nc"].values)
    log.info(f"forward EPS = {a_fwd:.4f}·Nc² + {b_fwd:.4f}   R²={r2_fwd:.4f}")
    log.info(f"reverse Nc² = {a_rev:.4f}·EPS + {b_rev:.4f}   R²(on Nc)={r2_rev:.4f}")

    canon_to_nc = build_effective_value_lookup(tr, results, "nc")
    canon_to_eps = build_effective_value_lookup(tr, results, "eps")

    # EPS OOF blend
    eps_oof = results["eps"]["oof"].copy()
    nc_eff = eps_oof["canon"].map(canon_to_nc).values.astype(float)
    eps_max = apply_maxwell_forward(nc_eff, a_fwd, b_fwd)
    m = np.isnan(eps_max); eps_max[m] = eps_oof["y_pred"].values[m]
    best_w_eps, best_r2_eps, base_r2_eps = search_blend_weight(
        eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_max)
    log.info(f"eps blend: LGB R²={base_r2_eps:.4f}  best w={best_w_eps:.3f}  "
             f"blend R²={best_r2_eps:.4f}   Δ={best_r2_eps - base_r2_eps:+.4f}")
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred"].values + (1 - best_w_eps) * eps_max
    results["eps"]["oof"] = eps_oof

    # Nc OOF blend
    nc_oof = results["nc"]["oof"].copy()
    eps_eff = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    nc_max = apply_maxwell_reverse(eps_eff, a_rev, b_rev)
    m = np.isnan(nc_max); nc_max[m] = nc_oof["y_pred"].values[m]
    best_w_nc, best_r2_nc, base_r2_nc = search_blend_weight(
        nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_max)
    log.info(f"nc blend: LGB R²={base_r2_nc:.4f}  best w={best_w_nc:.3f}  "
             f"blend R²={best_r2_nc:.4f}   Δ={best_r2_nc - base_r2_nc:+.4f}")
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_max
    results["nc"]["oof"] = nc_oof

    # Apply Maxwell to test predictions
    canon_to_nc_test = dict(zip(results["nc"]["test_pred"]["canon"], results["nc"]["test_pred"]["target"]))
    canon_to_eps_test = dict(zip(results["eps"]["test_pred"]["canon"], results["eps"]["test_pred"]["target"]))

    def get_nc_test(c):
        if c in canon_to_nc: return canon_to_nc[c]
        return canon_to_nc_test.get(c, float("nan"))

    def get_eps_test(c):
        if c in canon_to_eps: return canon_to_eps[c]
        return canon_to_eps_test.get(c, float("nan"))

    eps_te = results["eps"]["test_pred"].copy()
    nc_eff_te = np.array([get_nc_test(c) for c in eps_te["canon"]], dtype=float)
    eps_max_te = apply_maxwell_forward(nc_eff_te, a_fwd, b_fwd)
    m = np.isnan(eps_max_te); eps_max_te[m] = eps_te["target"].values[m]
    eps_te["target"] = best_w_eps * eps_te["target"].values + (1 - best_w_eps) * eps_max_te
    results["eps"]["test_pred"] = eps_te

    nc_te = results["nc"]["test_pred"].copy()
    eps_eff_te = np.array([get_eps_test(c) for c in nc_te["canon"]], dtype=float)
    nc_max_te = apply_maxwell_reverse(eps_eff_te, a_rev, b_rev)
    m = np.isnan(nc_max_te); nc_max_te[m] = nc_te["target"].values[m]
    nc_te["target"] = best_w_nc * nc_te["target"].values + (1 - best_w_nc) * nc_max_te
    results["nc"]["test_pred"] = nc_te

    # ---- Write outputs ----
    oof_all = pd.concat([results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                         for t in TARGETS], ignore_index=True)
    sub_all = pd.concat([results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                        ignore_index=True).sort_values("id").reset_index(drop=True)

    oof_all.to_csv(oof_path, index=False)
    sub_all.to_csv(sub_path, index=False)
    log.info(f"[PHASE 1] wrote {oof_path}  ({len(oof_all)} rows)")
    log.info(f"[PHASE 1] wrote {sub_path}  ({len(sub_all)} rows)")
    return phase_dir


# ============================================================================
# PHASE 2: CHEMPROP MULTITASK 3-SEED  — target solo LB 0.892
# ============================================================================

def build_wide_train_chemprop(tr: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[list(TARGETS)]
    canons = wide.index.tolist()
    y_matrix = wide.values.astype(np.float32)
    return canons, y_matrix


def build_chemprop_model(output_transform=None) -> MPNN:
    mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH, dropout=MP_DROPOUT)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        n_tasks=N_TARGETS, input_dim=D_H, hidden_dim=FFN_HIDDEN,
        n_layers=FFN_LAYERS, dropout=FFN_DROPOUT,
        output_transform=output_transform,
    )
    return MPNN(
        mp, agg, ffn, batch_norm=BATCH_NORM,
        init_lr=LR_INIT, max_lr=LR_MAX, final_lr=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
    )


def make_train_val_datasets_chemprop(canons, y, train_idxs, val_idxs, featurizer):
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


def make_full_dataset_chemprop(canons, y, featurizer):
    all_pts = []
    for i, smi in enumerate(canons):
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        all_pts.append(data.MoleculeDatapoint(mol=m, y=y[i]))
    full_dset = data.MoleculeDataset(all_pts, featurizer=featurizer)
    scaler = full_dset.normalize_targets()
    return full_dset, scaler


def train_chemprop_cv_model(train_dset, val_dset, scaler, seed, ctx, log):
    L.seed_everything(seed, workers=True)
    train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = data.build_dataloader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_chemprop_model(output_transform=output_transform)
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=False)
    epoch_logger = EpochLogger(log, ctx)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, accelerator=DEVICE, devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        callbacks=[early_stop, epoch_logger], deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    wall = (time.time() - t0) / 60

    model.eval()
    preds_list = trainer.predict(model, val_loader)
    val_preds = torch.cat(preds_list, dim=0).cpu().numpy()

    best_epoch = trainer.current_epoch - PATIENCE if early_stop.stopped_epoch > 0 else trainer.current_epoch
    log.info(f"[{ctx}] done. time={wall:.1f}min  best_epoch≈{best_epoch}")
    return val_preds, int(best_epoch), float(wall)


def train_chemprop_refit_model(full_dset, scaler, seed, n_epochs, test_canon_unique,
                                featurizer, ctx, log):
    L.seed_everything(seed, workers=True)
    full_loader = data.build_dataloader(full_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_chemprop_model(output_transform=output_transform)
    epoch_logger = EpochLogger(log, ctx)

    trainer = L.Trainer(
        max_epochs=n_epochs, accelerator=DEVICE, devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        callbacks=[epoch_logger], deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, full_loader)
    wall = (time.time() - t0) / 60
    log.info(f"[{ctx}] refit done. time={wall:.1f}min")

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


def train_chemprop_fold_3seed(fold_k, canons, y, train_idxs, val_idxs, featurizer, log):
    log.info("=" * 60)
    log.info(f"CHEMPROP FOLD {fold_k}  ({len(MODEL_SEEDS)}-seed bag)")
    log.info(f"n_train_canon={len(train_idxs)}   n_val_canon={len(val_idxs)}")

    val_preds_per_seed, best_epochs, wall_times = [], [], []
    for si, seed in enumerate(MODEL_SEEDS):
        ctx = f"CP fold {fold_k} seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
        log.info(f"[{ctx}] starting...")
        train_dset, val_dset, scaler = make_train_val_datasets_chemprop(canons, y, train_idxs, val_idxs, featurizer)
        val_preds, best_epoch, wall = train_chemprop_cv_model(train_dset, val_dset, scaler, seed, ctx, log)
        val_preds_per_seed.append(val_preds)
        best_epochs.append(best_epoch)
        wall_times.append(wall)

    val_preds_avg = np.mean(np.stack(val_preds_per_seed, axis=0), axis=0)
    val_true = y[val_idxs]
    return {
        "fold_k": fold_k, "val_idxs": val_idxs,
        "val_preds_per_seed": val_preds_per_seed,
        "val_preds_avg": val_preds_avg,
        "val_true": val_true,
        "best_epochs_per_seed": best_epochs,
        "wall_times_per_seed_min": wall_times,
        "seeds_used": list(MODEL_SEEDS),
    }


def phase2_chemprop_3seed(tr: pd.DataFrame, te: pd.DataFrame, log: logging.Logger) -> Path:
    """Full Chemprop 3-seed pipeline with per-fold checkpointing."""
    phase_dir = WORK_DIR / "chemprop_3seed"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"
    oof_path = phase_dir / "oof.csv"
    refit_path = phase_dir / "refit_test_preds.pkl.gz"

    if sub_path.exists() and oof_path.exists() and refit_path.exists():
        log.info(f"[PHASE 2] found existing outputs at {phase_dir} — skipping Chemprop training")
        return phase_dir

    log.info("=" * 60)
    log.info("PHASE 2: Chemprop 3-seed multitask  — target solo LB 0.892")
    log.info("=" * 60)

    random.seed(SPLIT_SEED); np.random.seed(SPLIT_SEED); torch.manual_seed(SPLIT_SEED)
    L.seed_everything(SPLIT_SEED, workers=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log.info(f"torch threads: {torch.get_num_threads()}, device: {DEVICE}")

    canons, y = build_wide_train_chemprop(tr)
    test_canon_unique = te["canon"].drop_duplicates().tolist()
    log.info(f"wide train: n_canon={len(canons)}   n_test_canons={len(test_canon_unique)}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # ---- 5-fold CV × 3-seed with per-fold checkpointing ----
    splits = group_kfold_splits(canons)
    fold_results = []
    for k, (tri, vai) in enumerate(splits):
        cp_path = phase_dir / f"checkpoint_fold_{k}.pkl.gz"
        if cp_path.exists():
            log.info(f"[CP fold {k}] loading checkpoint (skipping training)")
            with gzip.open(cp_path, "rb") as f:
                fold_results.append(pickle.load(f))
            continue
        result = train_chemprop_fold_3seed(k, canons, y, tri, vai, featurizer, log)
        with gzip.open(cp_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"[CP fold {k}] wrote checkpoint {cp_path.name}")
        fold_results.append(result)

    # ---- Assemble OOF ----
    oof_preds = np.full((len(canons), N_TARGETS), np.nan, dtype=np.float32)
    for r in fold_results:
        oof_preds[r["val_idxs"]] = r["val_preds_avg"]

    log.info("PER-TARGET OOF R² (Chemprop 3-seed)")
    for t_idx, tgt in enumerate(TARGETS):
        mask = ~np.isnan(y[:, t_idx])
        r2 = float(r2_score(y[mask, t_idx], oof_preds[mask, t_idx]))
        log.info(f"  {tgt:>4s}  n={int(mask.sum()):>5d}  OOF R²={r2:.4f}")

    # ---- Refit epochs ----
    all_best_epochs = [e for r in fold_results for e in r["best_epochs_per_seed"]]
    refit_epochs = max(15, int(np.median(all_best_epochs) * REFIT_ITER_MULTIPLIER))
    log.info(f"refit epochs (median * 1.10): {refit_epochs}")

    # ---- Refit + test predictions with cache ----
    if refit_path.exists():
        log.info(f"loading cached refit test predictions from {refit_path.name}")
        with gzip.open(refit_path, "rb") as f:
            refit_cache = pickle.load(f)
        test_preds_avg = refit_cache["test_preds_avg"]
    else:
        test_preds_per_seed = []
        for si, seed in enumerate(MODEL_SEEDS):
            ctx = f"CP REFIT seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
            log.info(f"[{ctx}] starting refit for {refit_epochs} epochs")
            full_dset, scaler = make_full_dataset_chemprop(canons, y, featurizer)
            test_preds, wall = train_chemprop_refit_model(
                full_dset, scaler, seed, refit_epochs, test_canon_unique, featurizer, ctx, log,
            )
            test_preds_per_seed.append(test_preds)
        test_preds_avg = np.mean(np.stack(test_preds_per_seed, axis=0), axis=0)
        with gzip.open(refit_path, "wb") as f:
            pickle.dump({"test_preds_avg": test_preds_avg,
                          "test_preds_per_seed": test_preds_per_seed}, f,
                         protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached refit test predictions to {refit_path.name}")

    test_preds_by_canon = {smi: test_preds_avg[i] for i, smi in enumerate(test_canon_unique)}

    # ---- Build OOF and submission DataFrames ----
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    oof_rows = []
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
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False)
    log.info(f"[PHASE 2] wrote {oof_path}  ({len(oof_rows)} rows)")

    sub_rows = []
    for _, row in te.iterrows():
        c = row["canon"]; t = row["target_type"]
        t_idx = TARGET_IDX[t]
        preds = test_preds_by_canon.get(c)
        sub_rows.append({"id": int(row["id"]),
                          "target": float(preds[t_idx]) if preds is not None else float("nan")})
    sub_df = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)
    sub_df.to_csv(sub_path, index=False)
    log.info(f"[PHASE 2] wrote {sub_path}  ({len(sub_df)} rows)")

    return phase_dir


# ============================================================================
# PHASE 3: 2-WAY NNLS BLEND  — target LB 0.897
# ============================================================================

def fit_target_weights_nnls(y_true, y_c, y_l, log, target):
    """Fit NNLS on [y_chemprop, y_lgb], normalize to sum=1, apply bias mitigations."""
    A = np.vstack([y_c, y_l]).T
    x, _ = nnls(A, y_true)
    w_c_raw, w_l_raw = float(x[0]), float(x[1])
    s = w_c_raw + w_l_raw
    if s < 1e-9:
        log.warning(f"[BLEND {target}] NNLS collapsed to zero; using 50/50")
        w_c_norm, w_l_norm = 0.5, 0.5
    else:
        w_c_norm, w_l_norm = w_c_raw / s, w_l_raw / s

    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_bias = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        w_l_bias = max(0.0, 1.0 - w_c_bias)
    else:
        w_c_bias, w_l_bias = w_c_norm, w_l_norm

    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        w_c_final, w_l_final = CHEMPROP_WEIGHT_FLOOR, 1.0 - CHEMPROP_WEIGHT_FLOOR
    else:
        w_c_final, w_l_final = w_c_bias, w_l_bias

    log.info(f"[BLEND {target}] NNLS raw: w_c={w_c_raw:.3f} w_l={w_l_raw:.3f}   "
             f"final (bias + floor): w_c={w_c_final:.3f} w_l={w_l_final:.3f}")
    return w_c_final, w_l_final


def phase3_nnls_blend(chemprop_dir: Path, lgb_dir: Path, log: logging.Logger) -> Path:
    """Per-target NNLS blend of Chemprop 3-seed + LGB Maxwell OOFs.
    Applies Chemprop-weight-floor 0.40 and additive bias +0.15."""
    phase_dir = WORK_DIR / "blend_nnls"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"

    log.info("=" * 60)
    log.info("PHASE 3: 2-way NNLS blend (Chemprop 3-seed + LGB Maxwell)  — target LB 0.897")
    log.info("=" * 60)
    log.info(f"  CHEMPROP_WEIGHT_FLOOR = {CHEMPROP_WEIGHT_FLOOR}")
    log.info(f"  APPLY_CHEMPROP_BIAS   = {APPLY_CHEMPROP_BIAS}")

    # Load OOFs
    oof_c = pd.read_csv(chemprop_dir / "oof.csv")
    oof_l = pd.read_csv(lgb_dir / "oof.csv")
    oof = (oof_c.rename(columns={"y_pred": "y_pred_chemprop"})
           .merge(oof_l.rename(columns={"y_pred": "y_pred_lgb"})[["canon", "target_type", "y_pred_lgb"]],
                  on=["canon", "target_type"], how="inner"))
    log.info(f"aligned OOF: {oof.shape}")

    # Load subs
    sub_c = pd.read_csv(chemprop_dir / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_l = pd.read_csv(lgb_dir / "submission.csv").rename(columns={"target": "target_lgb"})
    te = pd.read_csv(DATA_DIR / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_c, on="id", how="left").merge(sub_l, on="id", how="left")

    # Per-target NNLS
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(subset=["y_true", "y_pred_chemprop", "y_pred_lgb"])
        y_true, y_c, y_l = g["y_true"].values, g["y_pred_chemprop"].values, g["y_pred_lgb"].values
        r2_c = float(r2_score(y_true, y_c))
        r2_l = float(r2_score(y_true, y_l))
        log.info(f"[BLEND {target}] Chemprop OOF R²={r2_c:.4f}  LGB OOF R²={r2_l:.4f}")
        w_c, w_l = fit_target_weights_nnls(y_true, y_c, y_l, log, target)
        y_blend = w_c * y_c + w_l * y_l
        r2_blend = float(r2_score(y_true, y_blend))
        log.info(f"[BLEND {target}] blend OOF R²={r2_blend:.4f}   Δ vs better={r2_blend - max(r2_c, r2_l):+.4f}")

        mask = sub["target_type"] == target
        sub.loc[mask, "target_blend"] = (
            w_c * sub.loc[mask, "target_chemprop"] + w_l * sub.loc[mask, "target_lgb"]
        )
        per_target[target] = {"w_chemprop": w_c, "w_lgb": w_l, "r2_blend": r2_blend}

    # Write
    sub_out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_out.to_csv(sub_path, index=False)
    log.info(f"[PHASE 3] wrote {sub_path}  ({len(sub_out)} rows)")

    with open(phase_dir / "blend_summary.json", "w") as f:
        json.dump({"config": {"chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
                              "apply_chemprop_bias": APPLY_CHEMPROP_BIAS},
                   "per_target": per_target}, f, indent=2, default=str)
    return phase_dir


# ============================================================================
# PHASE 4: KOOPMANS BANDGAP POST-FIT  — FINAL LB 0.902
# ============================================================================

def load_chemprop_oof_matrix(canons, chemprop_dir: Path, log) -> np.ndarray:
    """Load per-fold Chemprop checkpoints, assemble full (n_canons, 7) OOF matrix."""
    n_canons = len(canons)
    oof = np.full((n_canons, 7), np.nan, dtype=np.float32)
    for k in range(N_SPLITS):
        with gzip.open(chemprop_dir / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            fold_result = pickle.load(f)
        oof[fold_result["val_idxs"]] = fold_result["val_preds_avg"]
    log.info(f"[KOOPMANS] Chemprop OOF matrix: {oof.shape}   "
             f"missing rows: {int(np.isnan(oof).all(axis=1).sum())}")
    return oof


def tune_alpha_koopmans(target, oof, y_matrix, log) -> dict:
    src_a_name, src_b_name, combine = PHYSICS_RECIPES[target]
    t_own = TARGET_IDX[target]
    t_a = TARGET_IDX[src_a_name]
    t_b = TARGET_IDX[src_b_name]

    mask = ~np.isnan(y_matrix[:, t_own])
    own_pred = oof[mask, t_own]; src_a = oof[mask, t_a]; src_b = oof[mask, t_b]
    y_true = y_matrix[mask, t_own]
    valid = ~(np.isnan(own_pred) | np.isnan(src_a) | np.isnan(src_b))
    own_pred = own_pred[valid]; src_a = src_a[valid]; src_b = src_b[valid]
    y_true = y_true[valid]

    physics = combine(src_a, src_b)
    r2_baseline = float(r2_score(y_true, own_pred))
    r2_pure_phys = float(r2_score(y_true, physics))

    best_r2, best_alpha = -np.inf, 1.0
    for alpha in ALPHA_GRID:
        blend = alpha * own_pred + (1 - alpha) * physics
        r2 = float(r2_score(y_true, blend))
        if r2 > best_r2:
            best_r2, best_alpha = r2, float(alpha)

    log.info(f"[KOOPMANS {target}] baseline R²={r2_baseline:.4f}   pure-phys R²={r2_pure_phys:.4f}   "
             f"best α={best_alpha:.3f}   blend R²={best_r2:.4f}   Δ={best_r2 - r2_baseline:+.4f}")
    return {"best_alpha": best_alpha, "r2_baseline": r2_baseline,
            "r2_blend": best_r2, "delta_r2": best_r2 - r2_baseline}


def phase4_koopmans_postfit(blend_sub_dir: Path, chemprop_dir: Path,
                              tr: pd.DataFrame, te: pd.DataFrame, log: logging.Logger) -> Path:
    """Apply Koopmans-theorem physics post-processor to the 2-way blend submission."""
    phase_dir = WORK_DIR / "koopmans"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"

    log.info("=" * 60)
    log.info("PHASE 4: Koopmans bandgap post-fit (Egc ≈ Ei - Eea)  — FINAL LB 0.902")
    log.info("=" * 60)

    # Rebuild train canon list + y_matrix (same as Chemprop's phase used)
    canons, y_matrix = build_wide_train_chemprop(tr)
    oof = load_chemprop_oof_matrix(canons, chemprop_dir, log)

    # α tuning per target
    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = tune_alpha_koopmans(tgt, oof, y_matrix, log)
    alphas = {tgt: alpha_results[tgt]["best_alpha"] for tgt in PHYSICS_TARGETS}
    total_delta = sum(alpha_results[t]["delta_r2"] for t in PHYSICS_TARGETS)
    log.info(f"[KOOPMANS] sum ΔR² = {total_delta:+.4f}   expected 7-target mean uplift = {total_delta/7:+.4f}")

    # Load blend submission
    sub = pd.read_csv(blend_sub_dir / "submission.csv")
    te_lookup = te[["id", "canon", "target_type"]].copy()
    sub_full = sub.merge(te_lookup, on="id", how="left")

    # Load Chemprop refit test preds (all 7 targets per canon)
    with gzip.open(chemprop_dir / "refit_test_preds.pkl.gz", "rb") as f:
        cache = pickle.load(f)
    chemprop_test = cache["test_preds_avg"]
    test_canons = te["canon"].drop_duplicates().tolist()
    canon_to_idx = {c: i for i, c in enumerate(test_canons)}

    for tgt in PHYSICS_TARGETS:
        alpha = alphas[tgt]
        src_a, src_b, combine = PHYSICS_RECIPES[tgt]
        src_a_idx, src_b_idx = TARGET_IDX[src_a], TARGET_IDX[src_b]
        mask = sub_full["target_type"] == tgt
        rows = sub_full[mask].copy()
        canon_idx = np.array([canon_to_idx[c] for c in rows["canon"]])
        chem_a = chemprop_test[canon_idx, src_a_idx]
        chem_b = chemprop_test[canon_idx, src_b_idx]
        physics_test = combine(chem_a, chem_b)
        own_test = rows["target"].values
        new_pred = alpha * own_test + (1 - alpha) * physics_test
        diffs = np.abs(new_pred - own_test)
        log.info(f"[KOOPMANS {tgt}] applying α={alpha:.3f}   "
                 f"n_rows={len(rows)}  mean |Δ|={diffs.mean():.4f}  max |Δ|={diffs.max():.4f}")
        sub_full.loc[mask, "target"] = new_pred

    sub_out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    sub_out.to_csv(sub_path, index=False)
    log.info(f"[PHASE 4] wrote {sub_path}  ({len(sub_out)} rows)")

    with open(phase_dir / "koopmans_summary.json", "w") as f:
        json.dump({"alphas": alphas, "alpha_results": alpha_results,
                   "sum_delta_r2": total_delta,
                   "expected_mean_uplift": total_delta / 7}, f, indent=2, default=str)
    return phase_dir


# ============================================================================
# MAIN — orchestrate all 4 phases
# ============================================================================

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(WORK_DIR)
    log.info("=" * 60)
    log.info(" REPRODUCTION PIPELINE FOR LB 0.902 SUBMISSION")
    log.info("=" * 60)
    log.info(f"DATA_DIR: {DATA_DIR}")
    log.info(f"WORK_DIR: {WORK_DIR}")
    log.info(f"DEVICE:   {DEVICE}   (adjust top of file to use GPU on Kaggle)")

    t_total = time.time()

    tr, te = load_and_canonicalize(log)

    lgb_dir      = phase1_lgb_maxwell(tr, te, log)
    chemprop_dir = phase2_chemprop_3seed(tr, te, log)
    blend_dir    = phase3_nnls_blend(chemprop_dir, lgb_dir, log)
    koopmans_dir = phase4_koopmans_postfit(blend_dir, chemprop_dir, tr, te, log)

    # ---- Copy final submission to root ----
    final_sub_src = koopmans_dir / "submission.csv"
    final_sub_dst = WORK_DIR / "submission.csv"
    shutil.copy2(final_sub_src, final_sub_dst)
    log.info("=" * 60)
    log.info(f"FINAL SUBMISSION: {final_sub_dst}")
    log.info(f"                  (also at {final_sub_src})")
    log.info(f"Total wall time: {(time.time() - t_total) / 60:.1f} min")
    log.info(f"Expected LB: 0.902")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
