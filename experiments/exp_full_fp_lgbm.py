"""
exp_full_fp_lgbm.py — matrix-completion pipeline + full Round-1 fingerprint stack.

Same LightGBM per-target setup and aux-augmented matrix-completion features as
exp_matrix_completion_lgbm.py, but with the SMILES feature stack expanded to the
full Round-1 recipe:
  - RDKit 2D descriptors (~207)
  - Morgan-r2 count FP (2048)
  - Morgan-r3 count FP (2048)      [NEW]
  - MACCS (167)
  - Atom-Pair count FP (2048)      [NEW]
  - Topological-Torsion count FP (2048)  [NEW]
  - Avalon FP (512)                [NEW]
  Total SMILES features: ~9,038
  + 14 aux cross-target features = ~9,052 total

CV mode: aux-augmented (same as exp_matrix_completion_lgbm.py — full-train lookup,
target-being-predicted slot always masked).

Runs on Mac M-series CPU end-to-end. Fully self-contained; no shared utils.

Outputs (under results/exp_full_fp_lgbm/):
  run.log            — full log (file + stdout mirror)
  oof.csv            — OOF predictions
  submission.csv     — Kaggle format id, target
  cv_summary.json    — per-target R², feature family sizes, hyperparams, aux stats
  feature_cache.pkl  — reusable SMILES feature bundle (excluded from git via *.pkl)

Usage:
  poly2-venv/bin/python experiments/exp_full_fp_lgbm.py
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
EXP_NAME = "exp_full_fp_lgbm"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42

# Fingerprint sizes
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
    seed=SEED,
)
N_ESTIMATORS = 4000
EARLY_STOP_ROUNDS = 200
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
# FINGERPRINT / DESCRIPTOR COMPUTATION
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

    # RDKit descriptors
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis, desc="rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X_desc = df_desc.values.astype(np.float32)
    parts.append(X_desc)
    families_slice["desc"] = slice(cursor, cursor + X_desc.shape[1]); cursor += X_desc.shape[1]
    log.info(f"desc: shape={X_desc.shape}  dropped={len(dropped)} const cols  time={time.time()-t0:.1f}s")

    # Morgan-r2 count
    t0 = time.time()
    arrs = [compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis, desc="morgan-r2", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["morgan2c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"morgan2c: shape={X.shape}  time={time.time()-t0:.1f}s")

    # Morgan-r3 count
    t0 = time.time()
    arrs = [compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis, desc="morgan-r3", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["morgan3c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"morgan3c: shape={X.shape}  time={time.time()-t0:.1f}s")

    # MACCS
    t0 = time.time()
    arrs = [compute_maccs(s) for s in tqdm(smis, desc="maccs", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["maccs"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"maccs: shape={X.shape}  time={time.time()-t0:.1f}s")

    # Atom-Pair count
    t0 = time.time()
    arrs = [compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis, desc="atom-pair", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["atompair_c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"atompair_c: shape={X.shape}  time={time.time()-t0:.1f}s")

    # Topological-Torsion count
    t0 = time.time()
    arrs = [compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis, desc="top-torsion", ncols=100)]
    X = np.stack(arrs).astype(np.float32); parts.append(X)
    families_slice["toptorsion_c"] = slice(cursor, cursor + X.shape[1]); cursor += X.shape[1]
    log.info(f"toptorsion_c: shape={X.shape}  time={time.time()-t0:.1f}s")

    # Avalon
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
# AUX (cross-target) FEATURES — same as exp_matrix_completion_lgbm.py
# ============================================================================

def build_aux_lookup(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """canon -> np.array of length 2*N_TARGETS: [values..., masks...]."""
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
    """(n_rows, 2*N_TARGETS). Target-being-predicted slot is always masked."""
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


def aux_column_names() -> list[str]:
    return ([f"aux_val_{t}" for t in TARGETS] + [f"aux_mask_{t}" for t in TARGETS])


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
# PER-TARGET TRAIN + PREDICT
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

    other_mask_ids = [TARGET_IDX[t] + N_TARGETS for t in TARGETS if t != target]
    aux_train_known = (X_tr_aux[:, other_mask_ids] > 0).sum(axis=1)
    aux_test_known  = (X_te_aux[:, other_mask_ids] > 0).sum(axis=1)
    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}   "
             f"(SMILES={X_tr_smi.shape[1]} + aux={X_tr_aux.shape[1]})")
    log.info(f"[{target}] aux coverage — train {(aux_train_known>0).sum()}/{len(aux_train_known)} "
             f"({100*(aux_train_known>0).mean():.1f}%);  "
             f"test {(aux_test_known>0).sum()}/{len(aux_test_known)} "
             f"({100*(aux_test_known>0).mean():.1f}%)")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    fold_bar = tqdm(splits, desc=f"[{target}] folds", ncols=100, leave=False)
    for k, (tri, vai) in enumerate(fold_bar):
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
        r2 = r2_score(y[vai], pred_va)
        fold_r2s.append(float(r2))
        fold_bar.set_postfix(fold=k, best_iter=booster.best_iteration, r2=f"{r2:.4f}")
        log.info(f"[{target}] fold {k}: best_iter={booster.best_iteration:>4d}   "
                 f"R²={r2:.4f}   n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y)
    full_booster = lgb.train(
        LGB_PARAMS, d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred = full_booster.predict(X_te)

    # Per-family gain breakdown (does the new FP stack pull weight?)
    imp = full_booster.feature_importance(importance_type="gain")
    n_smi = X_tr_smi.shape[1]
    family_gains: dict[str, int] = {}
    for fam, sl in bundle["families_slice"].items():
        family_gains[fam] = int(imp[sl].sum())
    aux_gain = int(imp[n_smi:].sum())
    total_gain = int(imp.sum())
    log.info(f"[{target}] gain by family (out of {total_gain}):  " +
             "  ".join([f"{k}={v}({100*v/max(1,total_gain):.1f}%)" for k, v in family_gains.items()]) +
             f"  aux={aux_gain}({100*aux_gain/max(1,total_gain):.1f}%)")

    # Top aux features
    aux_imp_ranked = sorted(zip(aux_column_names(), imp[n_smi:].tolist()), key=lambda x: -x[1])
    log.info(f"[{target}] top-6 aux: " + ", ".join([f"{n}={int(v)}" for n, v in aux_imp_ranked[:6]]))

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
        "oof_r2":       oof_r2,
        "fold_r2s":     fold_r2s,
        "best_iters":   best_iters,
        "refit_iters":  refit_iters,
        "family_gains": family_gains,
        "aux_gain_share_pct": float(100 * aux_gain / max(1, total_gain)),
        "top_aux_gains": [(n, int(v)) for n, v in aux_imp_ranked[:6]],
        "aux_coverage_train_pct": float(100 * (aux_train_known > 0).mean()),
        "aux_coverage_test_pct":  float(100 * (aux_test_known > 0).mean()),
    }


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
             f"n_estimators={N_ESTIMATORS} early_stop={EARLY_STOP_ROUNDS}")
    log.info(f"CV mode: aux-augmented (aux lookup from full train, target slot always masked)")
    log.info(f"LGB_PARAMS = {LGB_PARAMS}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"SMILES feature families: {fam_str}")
    log.info(f"SMILES feature TOTAL columns: {bundle['X'].shape[1]}")

    log.info(f"building aux lookup over {tr['canon'].nunique()} unique canonical SMILES in train")
    aux_lookup = build_aux_lookup(tr)
    log.info(f"aux lookup built.  aux features per row: {2*N_TARGETS}")

    log.info("=" * 60)
    log.info("AUX COVERAGE BY TARGET (test rows)")
    log.info("=" * 60)
    for t in TARGETS:
        g_te = te[te["target_type"] == t]
        aux_te = aux_features_for_target(g_te["canon"], t, aux_lookup)
        mask_cols = [TARGET_IDX[o] + N_TARGETS for o in TARGETS if o != t]
        n_any = int((aux_te[:, mask_cols] > 0).any(axis=1).sum())
        log.info(f"  {t:>4s}: {n_any}/{len(g_te)} ({100*n_any/max(1,len(g_te)):.1f}%)")

    results = []
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        r = train_one_target(tgt, tr, te, bundle, aux_lookup, log)
        results.append(r)

    oof_all = pd.concat([r["oof"] for r in results], ignore_index=True)
    sub_all = pd.concat([r["test_pred"] for r in results], ignore_index=True)

    oof_path = EXP_DIR / "oof.csv"
    oof_all.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_all)}")

    sub_out = sub_all[["id", "target"]].sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    per_target = {r["target"]: {
        "n_train":                r["n_train"],
        "n_test":                 r["n_test"],
        "oof_r2":                 r["oof_r2"],
        "fold_r2s":               r["fold_r2s"],
        "best_iters":             r["best_iters"],
        "refit_iters":            r["refit_iters"],
        "family_gains":           r["family_gains"],
        "aux_gain_share_pct":     r["aux_gain_share_pct"],
        "top_aux_gains":          r["top_aux_gains"],
        "aux_coverage_train_pct": r["aux_coverage_train_pct"],
        "aux_coverage_test_pct":  r["aux_coverage_test_pct"],
    } for r in results}
    mean_r2 = float(np.mean([r["oof_r2"] for r in results]))

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "per_target":     per_target,
        "config": {
            "n_splits":         N_SPLITS,
            "seed":             SEED,
            "morgan2_nbits":    MORGAN2_NBITS,
            "morgan3_nbits":    MORGAN3_NBITS,
            "atompair_nbits":   ATOMPAIR_NBITS,
            "toptorsion_nbits": TOPTORSION_NBITS,
            "avalon_nbits":     AVALON_NBITS,
            "n_estimators":     N_ESTIMATORS,
            "early_stop":       EARLY_STOP_ROUNDS,
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "lgb_params":       LGB_PARAMS,
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
    log.info("PER-TARGET OOF R²  (vs matrix_completion reference)")
    log.info("=" * 60)
    matcomp_ref = {"eea": 0.8543, "egb": 0.9050, "egc": 0.8966,
                   "ei": 0.7944, "eps": 0.7931, "nc": 0.8228, "tg": 0.9026}
    for tgt in TARGETS:
        r2 = per_target[tgt]['oof_r2']
        base = matcomp_ref[tgt]
        delta = r2 - base
        log.info(f"  {tgt:>4s}   n={per_target[tgt]['n_train']:>5d}   "
                 f"R²={r2:.4f}   (matcomp {base:.4f}, Δ={delta:+.4f})")
    baseline_mean = float(np.mean(list(matcomp_ref.values())))
    log.info(f"  MEAN R² = {mean_r2:.4f}   (matcomp mean {baseline_mean:.4f}, "
             f"Δ={mean_r2 - baseline_mean:+.4f})")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
