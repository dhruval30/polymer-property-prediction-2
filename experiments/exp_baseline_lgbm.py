"""
exp_baseline_lgbm.py — Round 2 first baseline.

Per-target LightGBM on a slim feature stack (RDKit 2D descriptors + Morgan-r2 count FP + MACCS).
GroupKFold(5) by canonical SMILES, same folds across all targets.
Identity target transform for all 7 targets (tg has negatives so no log1p).

Runs on Mac M-series CPU end-to-end. Fully self-contained; no shared utils.

Outputs (under results/exp_baseline_lgbm/):
  run.log            — full log (file + stdout mirror)
  oof.csv            — OOF predictions: canon, target_type, y_true, y_pred, fold
  submission.csv     — test predictions in Kaggle format: id, target
  cv_summary.json    — per-target R², mean R², timing, hyperparams, feature-family sizes
  feature_cache.pkl  — reusable feature bundle (regenerated if missing)

Usage:
  poly2-venv/bin/python experiments/exp_baseline_lgbm.py
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
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_baseline_lgbm"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_SPLITS = 5
SEED = 42
MORGAN_NBITS = 2048

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
    seed=SEED,
)
N_ESTIMATORS = 4000
EARLY_STOP_ROUNDS = 200
REFIT_ITER_MULTIPLIER = 1.10   # refit on full-train at 1.1x median-best-iter


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
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
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

    # collapse (canon, target_type) duplicates in train by mean target
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
# FEATURIZATION
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def compute_rdkit_desc(smi: str) -> dict | None:
    m = Chem.MolFromSmiles(_cap(smi))
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))


def compute_morgan_count(smi: str, radius: int = 2, nbits: int = MORGAN_NBITS) -> np.ndarray:
    m = Chem.MolFromSmiles(_cap(smi))
    out = np.zeros(nbits, dtype=np.int32)
    if m is None: return out
    fp = AllChem.GetHashedMorganFingerprint(m, radius, nBits=nbits)
    for k, v in fp.GetNonzeroElements().items():
        out[k] = v
    return out


def compute_maccs(smi: str) -> np.ndarray:
    m = Chem.MolFromSmiles(_cap(smi))
    if m is None: return np.zeros(167, dtype=np.int8)
    return np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8)


def _sanitize_desc_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """inf/NaN → column median; 0.5/99.5 clip; drop constant columns."""
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
    """Featurize a list of canonical SMILES and return an in-memory bundle."""
    smis = list(dict.fromkeys(canon_smiles))  # unique, order-preserving

    # RDKit descriptors
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis, desc="rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X_desc = df_desc.values.astype(np.float32)
    log.info(f"desc: shape={X_desc.shape}  dropped={len(dropped)} const cols  time={time.time()-t0:.1f}s")

    # Morgan-r2 count FP
    t0 = time.time()
    arrs = [compute_morgan_count(s, 2, MORGAN_NBITS) for s in tqdm(smis, desc="morgan-r2", ncols=100)]
    X_morgan = np.stack(arrs).astype(np.float32)
    log.info(f"morgan2c: shape={X_morgan.shape}  time={time.time()-t0:.1f}s")

    # MACCS
    t0 = time.time()
    arrs = [compute_maccs(s) for s in tqdm(smis, desc="maccs", ncols=100)]
    X_maccs = np.stack(arrs).astype(np.float32)
    log.info(f"maccs: shape={X_maccs.shape}  time={time.time()-t0:.1f}s")

    X = np.concatenate([X_desc, X_morgan, X_maccs], axis=1)
    columns = (
        [f"desc__{c}" for c in df_desc.columns]
        + [f"morgan2c__{i}" for i in range(X_morgan.shape[1])]
        + [f"maccs__{i}" for i in range(X_maccs.shape[1])]
    )
    families_slice = {
        "desc":     slice(0, X_desc.shape[1]),
        "morgan2c": slice(X_desc.shape[1], X_desc.shape[1] + X_morgan.shape[1]),
        "maccs":    slice(X_desc.shape[1] + X_morgan.shape[1], X.shape[1]),
    }
    log.info(f"feature matrix total: {X.shape}  size≈{X.nbytes/1e6:.1f}MB")

    return {
        "X": X,
        "smiles_index": {s: i for i, s in enumerate(smis)},
        "columns": columns,
        "families_slice": families_slice,
    }


def get_or_build_features(all_canon: list[str], log: logging.Logger) -> dict:
    """Load bundle from disk if the SMILES set + config match, else build + save."""
    key = hashlib.md5(
        (str(sorted(set(all_canon))) + str(MORGAN_NBITS)).encode()
    ).hexdigest()[:12]

    if FEATURE_CACHE_PATH.exists():
        try:
            with open(FEATURE_CACHE_PATH, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded feature cache: {FEATURE_CACHE_PATH.name}  key={key}")
                return bundle
            log.info(f"feature cache key mismatch (was {bundle.get('_key')}, need {key}); rebuilding")
        except Exception as e:
            log.info(f"failed to load cache ({e}); rebuilding")

    bundle = build_feature_bundle(all_canon, log)
    bundle["_key"] = key
    with open(FEATURE_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached features to {FEATURE_CACHE_PATH.name}")
    return bundle


def slice_features(bundle: dict, canon_series: pd.Series) -> np.ndarray:
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# ============================================================================
# CV — GroupKFold by canonical SMILES, shuffled deterministically
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
# PER-TARGET TRAIN + PREDICT
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    bundle: dict,
    log: logging.Logger,
) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values
    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")

    X_tr = slice_features(bundle, g_tr["canon"])
    X_te = slice_features(bundle, g_te["canon"])

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters = []
    fold_r2s = []
    fold_bar = tqdm(splits, desc=f"[{target}] folds", ncols=100, leave=False)
    for k, (tri, vai) in enumerate(fold_bar):
        d_tr = lgb.Dataset(X_tr[tri], y[tri])
        d_va = lgb.Dataset(X_tr[vai], y[vai], reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS,
            d_tr,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[d_va],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred_va = booster.predict(X_tr[vai], num_iteration=booster.best_iteration)
        oof[vai] = pred_va
        best_iters.append(int(booster.best_iteration))
        fold_r2 = r2_score(y[vai], pred_va)
        fold_r2s.append(float(fold_r2))
        fold_bar.set_postfix(fold=k, best_iter=booster.best_iteration, r2=f"{fold_r2:.4f}")
        log.info(f"[{target}] fold {k}: best_iter={booster.best_iteration:>4d}   "
                 f"R²={fold_r2:.4f}   n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    # Refit on full train at 1.1 * median-best-iter
    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y)
    full_booster = lgb.train(
        LGB_PARAMS,
        d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred = full_booster.predict(X_te)

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
            "target_type": target,
            "target":      test_pred,
        }),
        "oof_r2":     oof_r2,
        "fold_r2s":   fold_r2s,
        "best_iters": best_iters,
        "refit_iters": refit_iters,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} morgan_nbits={MORGAN_NBITS} "
             f"n_estimators={N_ESTIMATORS} early_stop={EARLY_STOP_ROUNDS}")
    log.info(f"LGB_PARAMS = {LGB_PARAMS}")

    random.seed(SEED)
    np.random.seed(SEED)

    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    log.info(f"feature families: "
             + ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items()))

    results = []
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        r = train_one_target(tgt, tr, te, bundle, log)
        results.append(r)

    # Concatenate outputs
    oof_all = pd.concat([r["oof"] for r in results], ignore_index=True)
    sub_all = pd.concat([r["test_pred"] for r in results], ignore_index=True)

    oof_path = EXP_DIR / "oof.csv"
    oof_all.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_all)}")

    # submission.csv: id, target — Kaggle format, sorted by id
    sub_out = sub_all[["id", "target"]].sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    # Summary + metrics
    per_target = {r["target"]: {
        "n_train":     r["n_train"],
        "n_test":      r["n_test"],
        "oof_r2":      r["oof_r2"],
        "fold_r2s":    r["fold_r2s"],
        "best_iters":  r["best_iters"],
        "refit_iters": r["refit_iters"],
    } for r in results}
    mean_r2 = float(np.mean([r["oof_r2"] for r in results]))

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "per_target":     per_target,
        "config": {
            "n_splits":         N_SPLITS,
            "seed":             SEED,
            "morgan_nbits":     MORGAN_NBITS,
            "n_estimators":     N_ESTIMATORS,
            "early_stop":       EARLY_STOP_ROUNDS,
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "lgb_params":       LGB_PARAMS,
            "feature_families": {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_features_total": bundle["X"].shape[1],
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = EXP_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    log.info("=" * 60)
    log.info("PER-TARGET OOF R²")
    log.info("=" * 60)
    for tgt in TARGETS:
        log.info(f"  {tgt:>4s}   n={per_target[tgt]['n_train']:>5d}   R²={per_target[tgt]['oof_r2']:.4f}")
    log.info(f"  MEAN R² = {mean_r2:.4f}")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
