"""
exp_maxwell_prior_catboost.py — CatBoost mirror of `exp_maxwell_prior_lgbm.py`.

============================================================================
PURPOSE
============================================================================

Standalone CatBoost pipeline for direct head-to-head comparison with LightGBM.
Everything is identical to `exp_maxwell_prior_lgbm.py` (LB 0.860) EXCEPT the
tree learner: CatBoost instead of LightGBM. Same features, same CV, same
aux-augmented matrix completion, same Maxwell EPS↔Nc post-fit, same seed.

Once submitted, this gives us a clean LB comparison of the same recipe with
a different tree family. Then we can 3-way blend LGB + Chemprop + CatBoost
in a follow-up.

============================================================================
DEPENDENCIES
============================================================================

  - Data: ppp-round-2/{train,test}.csv
  - Venv: poly2-venv (Python 3.11)
  - Packages: catboost, rdkit, numpy, pandas, scikit-learn, tqdm
  - INSTALL CATBOOST FIRST if not present:
      poly2-venv/bin/pip install catboost

============================================================================
METHOD
============================================================================

1. Load train + test, canonicalize SMILES, dedup (collapses 4 tg dupes to
   mean).
2. Featurize each unique canonical SMILES with the full Round-1 stack:
   RDKit descriptors (~207) + Morgan-r2 count FP (2048) + Morgan-r3 count FP
   (2048) + MACCS (167) + Atom-Pair count FP (2048) + Topological-Torsion
   count FP (2048) + Avalon FP (512) = ~9,038 SMILES features. Wildcards
   `*` replaced with C before featurization.
3. Add 14 aux cross-target features per row (7 mean-target values on same
   canonical SMILES from full train + 7 masks). Target-being-predicted slot
   always masked (NaN + 0). Total 9,052 features per row.
4. For each of the 7 targets: 5-fold GroupKFold on canonical SMILES,
   aux-augmented CV. Train CatBoost per fold with early stopping on val
   loss. Refit on full train at 1.1× median-best-iter for test predictions.
5. Maxwell EPS↔Nc post-fit on 134 co-labeled train molecules (same as LGB
   version): `EPS = a·Nc² + b` forward and `Nc² = a'·EPS + b'` reverse.
   Search optimal per-target blend weight on OOF. Apply to test.

CatBoost hyperparams (from Round-1 winning recipe):
  iterations=4000, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
  early_stopping_rounds=200, rsm=0.5 (feature subsampling per tree),
  bootstrap_type='Bernoulli', subsample=0.85 (row subsampling),
  nan_mode='Min' (handles NaN in aux features natively).

============================================================================
OUTPUTS  (under results/exp_maxwell_prior_catboost/)
============================================================================

  run.log            — per-fold R², best iters, Maxwell fit stats, timings
  oof.csv            — OOF predictions: canon, target_type, y_true, y_pred
  submission.csv     — Kaggle format id, target (Maxwell-corrected eps/nc rows)
  cv_summary.json    — per-target R², fold R²s, Maxwell fit params, blend weights
  feature_cache.pkl  — reusable SMILES feature bundle (excluded from git via *.pkl)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_maxwell_prior_catboost.py

Then submit results/exp_maxwell_prior_catboost/submission.csv to Kaggle.

============================================================================
EXPECTED WALL TIME
============================================================================

~20-35 min on Mac M-series CPU (CatBoost is ~1.5-2× slower per iteration
than LGB but the same features + folds). First run rebuilds feature cache
(~5 min); subsequent runs skip that.

============================================================================
EXPECTED LB
============================================================================

Round 1 experience: CatBoost typically edges LGB by +0.005 to +0.015 on
tabular chemistry regression, but not always. Realistic range: LB 0.858 to
0.870. Blend value is where CatBoost really shines — different bias profile
from LGB means the 3-way blend (LGB + CAT + Chemprop) should push above
the current best 0.894.
"""
from __future__ import annotations

# --- stdlib ---
import hashlib
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
from catboost import CatBoostRegressor, Pool
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_maxwell_prior_catboost"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42

# Fingerprint sizes (identical to exp_maxwell_prior_lgbm.py for direct comparison)
MORGAN2_NBITS = 2048
MORGAN3_NBITS = 2048
ATOMPAIR_NBITS = 2048
TOPTORSION_NBITS = 2048
AVALON_NBITS = 512

# CatBoost hyperparams (Round-1 winning recipe)
CAT_PARAMS = dict(
    iterations=4000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    rsm=0.5,                    # random subspace method (feature_fraction analog)
    bootstrap_type="Bernoulli",
    subsample=0.85,             # row subsampling (bagging_fraction analog)
    random_seed=SEED,
    thread_count=-1,
    verbose=False,
    allow_writing_files=False,  # prevents cluttering disk with catboost_info/
    nan_mode="Min",             # NaN aux features → min-value branch
    loss_function="RMSE",
    eval_metric="RMSE",
)
EARLY_STOP_ROUNDS = 200
REFIT_ITER_MULTIPLIER = 1.10

# Grid resolution for Maxwell blend-weight search
BLEND_W_GRID = np.linspace(0.0, 1.0, 201)


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
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


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


# ============================================================================
# FINGERPRINT / DESCRIPTOR COMPUTATION (identical to LGB version)
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
        if lo == hi: continue
        df[c] = df[c].clip(lo, hi)
    dropped = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=dropped)
    return df, dropped


def build_feature_bundle(canon_smiles: list[str], log: logging.Logger) -> dict:
    smis = list(dict.fromkeys(canon_smiles))
    parts, families_slice, cursor = [], {}, 0

    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis, desc="rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X = df_desc.values.astype(np.float32); parts.append(X)
    families_slice["desc"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"desc: shape={X.shape}  dropped={len(dropped)} const cols  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis, desc="morgan-r2", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["morgan2c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"morgan2c: shape={X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis, desc="morgan-r3", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["morgan3c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"morgan3c: shape={X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_maccs(s) for s in tqdm(smis, desc="maccs", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["maccs"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"maccs: shape={X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis, desc="atom-pair", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["atompair_c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"atompair_c: shape={X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis, desc="top-torsion", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["toptorsion_c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"toptorsion_c: shape={X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_avalon(s, AVALON_NBITS) for s in tqdm(smis, desc="avalon", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["avalon"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"avalon: shape={X.shape}  time={time.time()-t0:.1f}s")

    X_full = np.concatenate(parts, axis=1)
    log.info(f"SMILES feature matrix TOTAL: {X_full.shape}  size≈{X_full.nbytes/1e6:.1f}MB")

    return {
        "X": X_full,
        "smiles_index": {s: i for i, s in enumerate(smis)},
        "families_slice": families_slice,
    }


def get_or_build_features(all_canon: list[str], log: logging.Logger) -> dict:
    key = hashlib.md5(
        (str(sorted(set(all_canon))) +
         f"m2={MORGAN2_NBITS};m3={MORGAN3_NBITS};ap={ATOMPAIR_NBITS};tt={TOPTORSION_NBITS};av={AVALON_NBITS}"
         ).encode()
    ).hexdigest()[:12]

    if FEATURE_CACHE_PATH.exists():
        try:
            with open(FEATURE_CACHE_PATH, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded feature cache: {FEATURE_CACHE_PATH.name}  key={key}")
                return bundle
            log.info(f"feature cache key mismatch; rebuilding")
        except Exception as e:
            log.info(f"failed to load cache ({e}); rebuilding")

    bundle = build_feature_bundle(all_canon, log)
    bundle["_key"] = key
    with open(FEATURE_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached features to {FEATURE_CACHE_PATH.name}")
    return bundle


def slice_smiles_features(bundle: dict, canon_series: pd.Series) -> np.ndarray:
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# ============================================================================
# AUX (cross-target) FEATURES
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


def aux_features_for_target(
    canon_series: pd.Series,
    target: str,
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


# ============================================================================
# CV
# ============================================================================

def group_kfold_splits(
    canon_arr: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# PER-TARGET TRAIN + PREDICT (CatBoost)
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    bundle: dict,
    aux_lookup: dict,
    log: logging.Logger,
) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"])
    X_te_smi = slice_smiles_features(bundle, g_te["canon"])
    X_tr_aux = aux_features_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_features_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    fold_bar = tqdm(splits, desc=f"[{target}] folds", ncols=100, leave=False)
    for k, (tri, vai) in enumerate(fold_bar):
        train_pool = Pool(X_tr[tri], y[tri])
        val_pool = Pool(X_tr[vai], y[vai])
        model = CatBoostRegressor(**CAT_PARAMS)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            use_best_model=True,
            verbose=False,
        )
        pred_va = model.predict(X_tr[vai])
        oof[vai] = pred_va
        best_iter = int(model.get_best_iteration())
        best_iters.append(best_iter)
        r2 = r2_score(y[vai], pred_va)
        fold_r2s.append(float(r2))
        fold_bar.set_postfix(fold=k, best_iter=best_iter, r2=f"{r2:.4f}")
        log.info(f"[{target}] fold {k}: best_iter={best_iter:>4d}   "
                 f"R²={r2:.4f}   n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² (CatBoost only) = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} iterations")
    refit_params = {**CAT_PARAMS, "iterations": refit_iters}
    full_model = CatBoostRegressor(**refit_params)
    full_model.fit(X_tr, y, verbose=False)
    test_pred = full_model.predict(X_te)

    return {
        "target": target,
        "n_train": int(len(g_tr)),
        "n_test":  int(len(g_te)),
        "oof": pd.DataFrame({
            "canon":       g_tr["canon"].values,
            "target_type": target,
            "y_true":      y,
            "y_pred":      oof,
        }),
        "test_pred": pd.DataFrame({
            "id":          g_te["id"].values,
            "canon":       g_te["canon"].values,
            "target_type": target,
            "target":      test_pred,
        }),
        "oof_r2":       oof_r2,
        "fold_r2s":     fold_r2s,
        "best_iters":   best_iters,
        "refit_iters":  refit_iters,
    }


# ============================================================================
# MAXWELL RELATION POST-FIT (identical logic to LGB version)
# ============================================================================

def fit_maxwell_forward(nc_values: np.ndarray, eps_values: np.ndarray) -> tuple[float, float, float]:
    """Fit `EPS = a * Nc² + b` by ordinary least squares. Returns (a, b, R²)."""
    x = nc_values ** 2
    y = eps_values
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * x + b
    r2 = r2_score(y, pred)
    return float(a), float(b), float(r2)


def fit_maxwell_reverse(eps_values: np.ndarray, nc_values: np.ndarray) -> tuple[float, float, float]:
    """Fit `Nc² = a' * EPS + b'`. Returns (a', b', R²) with R² measured on Nc, not Nc²."""
    x = eps_values
    y = nc_values ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred_nc2 = a * x + b
    pred_nc = np.sqrt(np.clip(pred_nc2, 1e-9, None))
    r2 = r2_score(nc_values, pred_nc)
    return float(a), float(b), float(r2)


def apply_maxwell_forward(nc_values: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (nc_values ** 2) + b


def apply_maxwell_reverse(eps_values: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.sqrt(np.clip(a * eps_values + b, 1e-9, None))


def search_blend_weight(
    y_true: np.ndarray,
    y_ml: np.ndarray,
    y_prior: np.ndarray,
    grid: np.ndarray = BLEND_W_GRID,
) -> tuple[float, float, float]:
    r2s = np.array([r2_score(y_true, w * y_ml + (1 - w) * y_prior) for w in grid])
    best_i = int(np.argmax(r2s))
    baseline_r2 = float(r2_score(y_true, y_ml))
    return float(grid[best_i]), float(r2s[best_i]), baseline_r2


def build_effective_value_lookup(
    train_df: pd.DataFrame,
    oof_results: dict,
    target: str,
) -> dict[str, float]:
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
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} "
             f"morgan2={MORGAN2_NBITS} morgan3={MORGAN3_NBITS} "
             f"atompair={ATOMPAIR_NBITS} toptorsion={TOPTORSION_NBITS} avalon={AVALON_NBITS} "
             f"early_stop={EARLY_STOP_ROUNDS}")
    log.info(f"CV mode: aux-augmented (aux lookup from full train, target slot always masked)")
    log.info(f"Post-fit: Maxwell relation EPS↔Nc physics prior blend")
    log.info(f"CAT_PARAMS = {CAT_PARAMS}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    # ---- Load + featurize ----
    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"SMILES feature families: {fam_str}")
    log.info(f"SMILES feature TOTAL columns: {bundle['X'].shape[1]}")

    log.info(f"building aux lookup over {tr['canon'].nunique()} unique canonical SMILES in train")
    aux_lookup = build_aux_lookup(tr)
    log.info(f"aux lookup built.  aux features per row: {2*N_TARGETS}")

    # ---- Train 7 targets ----
    results: dict[str, dict] = {}
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        results[tgt] = train_one_target(tgt, tr, te, bundle, aux_lookup, log)

    baseline_mean_r2 = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info(f"CatBoost-only per-target OOF R²  (before Maxwell)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}")
    log.info(f"  MEAN R² (CatBoost only) = {baseline_mean_r2:.4f}")

    # ============================================================
    # MAXWELL RELATION POST-FIT
    # ============================================================
    log.info("=" * 60)
    log.info("MAXWELL RELATION POST-FIT (EPS ↔ Nc)")
    log.info("=" * 60)

    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    co = wide.dropna(subset=["eps", "nc"])
    log.info(f"co-labeled train molecules (both EPS and Nc): n={len(co)}")

    raw_r_eps_nc = float(np.corrcoef(co["eps"], co["nc"])[0, 1])
    raw_r_eps_nc2 = float(np.corrcoef(co["eps"], co["nc"] ** 2)[0, 1])
    log.info(f"raw Pearson r(EPS, Nc)  = {raw_r_eps_nc:.4f}")
    log.info(f"raw Pearson r(EPS, Nc²) = {raw_r_eps_nc2:.4f}")

    a_fwd, b_fwd, r2_fwd = fit_maxwell_forward(co["nc"].values, co["eps"].values)
    log.info(f"Maxwell FORWARD fit (EPS = a·Nc² + b): "
             f"a={a_fwd:.4f}  b={b_fwd:.4f}  R²={r2_fwd:.4f}  (on {len(co)} points)")

    a_rev, b_rev, r2_rev = fit_maxwell_reverse(co["eps"].values, co["nc"].values)
    log.info(f"Maxwell REVERSE fit (Nc² = a'·EPS + b'): "
             f"a'={a_rev:.4f}  b'={b_rev:.4f}  R²(on Nc)={r2_rev:.4f}  (on {len(co)} points)")

    # EPS blend
    canon_to_nc = build_effective_value_lookup(tr, results, "nc")
    log.info(f"eps blend: effective-Nc lookup covers {len(canon_to_nc)} canons")

    eps_oof = results["eps"]["oof"].copy()
    nc_eff_for_eps_oof = eps_oof["canon"].map(canon_to_nc).values.astype(float)
    n_missing_nc_eps = int(np.isnan(nc_eff_for_eps_oof).sum())
    log.info(f"eps OOF rows with no effective Nc available: {n_missing_nc_eps}")

    eps_maxwell_oof = apply_maxwell_forward(nc_eff_for_eps_oof, a_fwd, b_fwd)
    missing_mask = np.isnan(eps_maxwell_oof)
    eps_maxwell_oof[missing_mask] = eps_oof["y_pred"].values[missing_mask]

    best_w_eps, best_r2_eps, baseline_r2_eps = search_blend_weight(
        eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_maxwell_oof
    )
    delta_eps = best_r2_eps - baseline_r2_eps
    pure_maxwell_r2 = float(r2_score(eps_oof["y_true"].values, eps_maxwell_oof))
    log.info(f"eps blend: baseline OOF R² = {baseline_r2_eps:.4f}")
    log.info(f"eps blend: pure-Maxwell (w=0) OOF R² = {pure_maxwell_r2:.4f}")
    log.info(f"eps blend: best  w = {best_w_eps:.3f}   (w=1 is CAT-only, w=0 is Maxwell-only)")
    log.info(f"eps blend: best OOF R² = {best_r2_eps:.4f}   Δ = {delta_eps:+.4f}")

    # Nc blend
    canon_to_eps = build_effective_value_lookup(tr, results, "eps")
    log.info(f"nc blend: effective-EPS lookup covers {len(canon_to_eps)} canons")

    nc_oof = results["nc"]["oof"].copy()
    eps_eff_for_nc_oof = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    n_missing_eps_nc = int(np.isnan(eps_eff_for_nc_oof).sum())
    log.info(f"nc OOF rows with no effective EPS available: {n_missing_eps_nc}")

    nc_maxwell_oof = apply_maxwell_reverse(eps_eff_for_nc_oof, a_rev, b_rev)
    missing_mask_nc = np.isnan(nc_maxwell_oof)
    nc_maxwell_oof[missing_mask_nc] = nc_oof["y_pred"].values[missing_mask_nc]

    best_w_nc, best_r2_nc, baseline_r2_nc = search_blend_weight(
        nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_maxwell_oof
    )
    delta_nc = best_r2_nc - baseline_r2_nc
    pure_maxwell_nc_r2 = float(r2_score(nc_oof["y_true"].values, nc_maxwell_oof))
    log.info(f"nc blend: baseline OOF R² = {baseline_r2_nc:.4f}")
    log.info(f"nc blend: pure-Maxwell (w=0) OOF R² = {pure_maxwell_nc_r2:.4f}")
    log.info(f"nc blend: best  w = {best_w_nc:.3f}")
    log.info(f"nc blend: best OOF R² = {best_r2_nc:.4f}   Δ = {delta_nc:+.4f}")

    # Update OOF DataFrames with blended predictions
    eps_oof["y_pred_ml"] = eps_oof["y_pred"].values
    eps_oof["y_pred_maxwell"] = eps_maxwell_oof
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred_ml"] + (1 - best_w_eps) * eps_oof["y_pred_maxwell"]
    results["eps"]["oof"] = eps_oof
    results["eps"]["oof_r2"] = best_r2_eps

    nc_oof["y_pred_ml"] = nc_oof["y_pred"].values
    nc_oof["y_pred_maxwell"] = nc_maxwell_oof
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred_ml"] + (1 - best_w_nc) * nc_oof["y_pred_maxwell"]
    results["nc"]["oof"] = nc_oof
    results["nc"]["oof_r2"] = best_r2_nc

    # Apply Maxwell to TEST predictions
    canon_to_nc_test = dict(zip(results["nc"]["test_pred"]["canon"],
                                 results["nc"]["test_pred"]["target"]))
    canon_to_eps_test = dict(zip(results["eps"]["test_pred"]["canon"],
                                  results["eps"]["test_pred"]["target"]))

    def get_nc_for_test(canon: str) -> float:
        if canon in canon_to_nc:
            tr_val = tr[(tr["canon"] == canon) & (tr["target_type"] == "nc")]["target"]
            if len(tr_val) > 0:
                return float(tr_val.mean())
        if canon in canon_to_nc_test:
            return float(canon_to_nc_test[canon])
        return float("nan")

    def get_eps_for_test(canon: str) -> float:
        if canon in canon_to_eps:
            tr_val = tr[(tr["canon"] == canon) & (tr["target_type"] == "eps")]["target"]
            if len(tr_val) > 0:
                return float(tr_val.mean())
        if canon in canon_to_eps_test:
            return float(canon_to_eps_test[canon])
        return float("nan")

    eps_test = results["eps"]["test_pred"].copy()
    nc_eff_for_eps_test = np.array([get_nc_for_test(c) for c in eps_test["canon"]], dtype=float)
    n_test_eps_have_train_nc = int(sum(1 for c in eps_test["canon"]
                                        if len(tr[(tr["canon"] == c) & (tr["target_type"] == "nc")]) > 0))
    log.info(f"test eps rows: {n_test_eps_have_train_nc}/{len(eps_test)} have train nc label available")

    eps_maxwell_test = apply_maxwell_forward(nc_eff_for_eps_test, a_fwd, b_fwd)
    missing_test = np.isnan(eps_maxwell_test)
    eps_maxwell_test[missing_test] = eps_test["target"].values[missing_test]

    eps_test["target_ml"] = eps_test["target"].values
    eps_test["target_maxwell"] = eps_maxwell_test
    eps_test["target"] = best_w_eps * eps_test["target_ml"] + (1 - best_w_eps) * eps_test["target_maxwell"]
    results["eps"]["test_pred"] = eps_test

    nc_test = results["nc"]["test_pred"].copy()
    eps_eff_for_nc_test = np.array([get_eps_for_test(c) for c in nc_test["canon"]], dtype=float)
    n_test_nc_have_train_eps = int(sum(1 for c in nc_test["canon"]
                                        if len(tr[(tr["canon"] == c) & (tr["target_type"] == "eps")]) > 0))
    log.info(f"test nc rows: {n_test_nc_have_train_eps}/{len(nc_test)} have train eps label available")

    nc_maxwell_test = apply_maxwell_reverse(eps_eff_for_nc_test, a_rev, b_rev)
    missing_test_nc = np.isnan(nc_maxwell_test)
    nc_maxwell_test[missing_test_nc] = nc_test["target"].values[missing_test_nc]

    nc_test["target_ml"] = nc_test["target"].values
    nc_test["target_maxwell"] = nc_maxwell_test
    nc_test["target"] = best_w_nc * nc_test["target_ml"] + (1 - best_w_nc) * nc_test["target_maxwell"]
    results["nc"]["test_pred"] = nc_test

    # ---- Write outputs ----
    oof_all = pd.concat([results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                         for t in TARGETS], ignore_index=True)
    sub_all = pd.concat([results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                        ignore_index=True)

    oof_path = EXP_DIR / "oof.csv"
    oof_all.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_all)}")

    sub_out = sub_all.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    per_target = {t: {
        "n_train":    results[t]["n_train"],
        "n_test":     results[t]["n_test"],
        "oof_r2":     results[t]["oof_r2"],
        "fold_r2s":   results[t]["fold_r2s"],
        "best_iters": results[t]["best_iters"],
        "refit_iters": results[t]["refit_iters"],
    } for t in TARGETS}
    mean_r2 = float(np.mean([per_target[t]["oof_r2"] for t in TARGETS]))

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "mean_r2_cat_only": baseline_mean_r2,
        "mean_r2_delta":  mean_r2 - baseline_mean_r2,
        "per_target":     per_target,
        "maxwell": {
            "n_co_labeled":         int(len(co)),
            "raw_pearson_eps_nc":   raw_r_eps_nc,
            "raw_pearson_eps_nc2":  raw_r_eps_nc2,
            "forward_fit": {"a": a_fwd, "b": b_fwd, "r2": r2_fwd},
            "reverse_fit": {"a": a_rev, "b": b_rev, "r2_on_nc": r2_rev},
            "eps_blend": {
                "baseline_r2":       baseline_r2_eps,
                "pure_maxwell_r2":   pure_maxwell_r2,
                "best_w":            best_w_eps,
                "best_blend_r2":     best_r2_eps,
                "delta":             delta_eps,
            },
            "nc_blend": {
                "baseline_r2":       baseline_r2_nc,
                "pure_maxwell_r2":   pure_maxwell_nc_r2,
                "best_w":            best_w_nc,
                "best_blend_r2":     best_r2_nc,
                "delta":             delta_nc,
            },
            "test_coverage": {
                "n_test_eps":                    len(eps_test),
                "n_test_eps_with_train_nc":      n_test_eps_have_train_nc,
                "n_test_nc":                     len(nc_test),
                "n_test_nc_with_train_eps":      n_test_nc_have_train_eps,
            },
        },
        "config": {
            "n_splits":         N_SPLITS,
            "seed":             SEED,
            "morgan2_nbits":    MORGAN2_NBITS,
            "morgan3_nbits":    MORGAN3_NBITS,
            "atompair_nbits":   ATOMPAIR_NBITS,
            "toptorsion_nbits": TOPTORSION_NBITS,
            "avalon_nbits":     AVALON_NBITS,
            "early_stop":       EARLY_STOP_ROUNDS,
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "cat_params":       CAT_PARAMS,
            "smiles_families":  {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_smiles_features": bundle["X"].shape[1],
            "n_aux_features":   2 * N_TARGETS,
            "n_features_total": bundle["X"].shape[1] + 2 * N_TARGETS,
            "cv_mode":          "aux-augmented",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = EXP_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (CatBoost + Maxwell)  vs LGB reference")
    log.info("=" * 60)
    # LGB+Maxwell reference (exp_maxwell_prior_lgbm, LB 0.860)
    lgb_ref = {"eea": 0.8543, "egb": 0.9050, "egc": 0.8966,
               "ei": 0.7944, "eps": 0.8186, "nc": 0.8603, "tg": 0.9026}
    log.info(f"  {'target':>6s}  {'CAT':>10s}  {'LGB (ref)':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = per_target[t]["oof_r2"]
        ref = lgb_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    lgb_mean = float(np.mean(list(lgb_ref.values())))
    log.info(f"  {'MEAN':>6s}  {mean_r2:>10.4f}  {lgb_mean:>10.4f}  {mean_r2 - lgb_mean:>+8.4f}")
    log.info(f"  (LGB LB reference: 0.860)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
