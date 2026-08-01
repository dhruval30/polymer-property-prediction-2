"""
exp_matrix_completion_lgbm.py — Track B: per-target LightGBM with cross-target aux features.

Same structure as exp_baseline_lgbm.py, but each row's feature vector is augmented with
14 aux features derived from OTHER targets' values on the SAME canonical SMILES:
  - 7 aux values: mean target value for each of the 7 target_types on this SMILES in train
  - 7 aux masks:  1 if that value is known, 0 if NaN
  - The slot corresponding to the target being predicted is ALWAYS masked (NaN value + 0 mask)
    so a molecule's own T value never leaks into its own T prediction.

CV mode: **aux-augmented.** The aux lookup is built from the FULL training set (not per fold).
Both fold-train and fold-val rows draw their aux from this global lookup. This mirrors the
test-time scenario where X's train other-target values are legitimately available. No label
leakage because we always mask out the target being predicted.

Rationale: under naïve fold-only aux, val rows have no aux (GroupKFold puts all X's rows in
the same fold), so OOF gain would be near zero. Aux-augmented OOF correctly reflects what LB
will see. See docs/08_eda_deep.md § S5 and docs/09_data_exploration.md § 5 for the cross-
target correlation evidence motivating this approach.

Runs on Mac M-series CPU end-to-end. Fully self-contained; no shared utils.

Outputs (under results/exp_matrix_completion_lgbm/):
  run.log            — full log (file + stdout mirror)
  oof.csv            — OOF predictions: canon, target_type, y_true, y_pred, fold
  submission.csv     — test predictions: id, target
  cv_summary.json    — per-target R², mean R², timing, hyperparams, aux stats
  feature_cache.pkl  — reusable SMILES feature bundle

Usage:
  poly2-venv/bin/python experiments/exp_matrix_completion_lgbm.py
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
EXP_NAME = "exp_matrix_completion_lgbm"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

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
# SMILES FEATURIZATION (same as baseline)
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

    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis, desc="rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X_desc = df_desc.values.astype(np.float32)
    log.info(f"desc: shape={X_desc.shape}  dropped={len(dropped)} const cols  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_morgan_count(s, 2, MORGAN_NBITS) for s in tqdm(smis, desc="morgan-r2", ncols=100)]
    X_morgan = np.stack(arrs).astype(np.float32)
    log.info(f"morgan2c: shape={X_morgan.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    arrs = [compute_maccs(s) for s in tqdm(smis, desc="maccs", ncols=100)]
    X_maccs = np.stack(arrs).astype(np.float32)
    log.info(f"maccs: shape={X_maccs.shape}  time={time.time()-t0:.1f}s")

    X = np.concatenate([X_desc, X_morgan, X_maccs], axis=1)
    families_slice = {
        "desc":     slice(0, X_desc.shape[1]),
        "morgan2c": slice(X_desc.shape[1], X_desc.shape[1] + X_morgan.shape[1]),
        "maccs":    slice(X_desc.shape[1] + X_morgan.shape[1], X.shape[1]),
    }
    log.info(f"SMILES feature matrix: {X.shape}  size≈{X.nbytes/1e6:.1f}MB")

    return {
        "X": X,
        "smiles_index": {s: i for i, s in enumerate(smis)},
        "families_slice": families_slice,
        "n_desc": X_desc.shape[1],
    }


def get_or_build_features(all_canon: list[str], log: logging.Logger) -> dict:
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
# AUX (cross-target) FEATURES — the actual Track B logic
# ============================================================================

def build_aux_lookup(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """canon -> np.array of length 2*N_TARGETS.
       Layout: [val_eea, val_egb, val_egc, val_ei, val_eps, val_nc, val_tg,
                mask_eea, mask_egb, mask_egc, mask_ei, mask_eps, mask_nc, mask_tg]
       NaN value + 0 mask when target isn't measured.
    """
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
    """(n_rows, 2 * N_TARGETS) matrix. The slot for `target` is masked (NaN + 0).
       For unknown SMILES: entire row is NaN/0 (no cross-target info)."""
    n = len(canon_series)
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    # ALWAYS mask the target being predicted so a molecule's own T value never leaks.
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


def aux_column_names() -> list[str]:
    return ([f"aux_val_{t}" for t in TARGETS] + [f"aux_mask_{t}" for t in TARGETS])


# ============================================================================
# CV — GroupKFold by canonical SMILES, deterministic shuffle
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

    # SMILES features
    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"])
    X_te_smi = slice_smiles_features(bundle, g_te["canon"])
    # Aux features
    X_tr_aux = aux_features_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_features_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)

    # Coverage stats
    other_slot_ids = [TARGET_IDX[t] for t in TARGETS if t != target]
    aux_known_per_row = (X_tr_aux[:, [i + N_TARGETS for i in other_slot_ids]] > 0).sum(axis=1)
    aux_te_known_per_row = (X_te_aux[:, [i + N_TARGETS for i in other_slot_ids]] > 0).sum(axis=1)
    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] aux coverage — train rows with ≥1 other-target aux known: "
             f"{(aux_known_per_row > 0).sum()}/{len(aux_known_per_row)} "
             f"({100*(aux_known_per_row>0).mean():.1f}%);  "
             f"mean_known/row={aux_known_per_row.mean():.2f}")
    log.info(f"[{target}] aux coverage — test  rows with ≥1 other-target aux known: "
             f"{(aux_te_known_per_row > 0).sum()}/{len(aux_te_known_per_row)} "
             f"({100*(aux_te_known_per_row>0).mean():.1f}%);  "
             f"mean_known/row={aux_te_known_per_row.mean():.2f}")

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

    # Refit on full train
    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y)
    full_booster = lgb.train(
        LGB_PARAMS, d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred = full_booster.predict(X_te)

    # Also log aux feature importances to see whether they matter
    imp = full_booster.feature_importance(importance_type="gain")
    n_smi = X_tr_smi.shape[1]
    aux_imp = imp[n_smi:]
    aux_names = aux_column_names()
    aux_imp_ranked = sorted(zip(aux_names, aux_imp), key=lambda x: -x[1])
    top_aux = ", ".join([f"{n}={int(v)}" for n, v in aux_imp_ranked[:6]])
    log.info(f"[{target}] top aux-feature gains (out of {len(aux_names)} aux feats): {top_aux}")
    aux_total_gain = int(aux_imp.sum())
    all_total_gain = int(imp.sum())
    log.info(f"[{target}] aux gain share: {aux_total_gain}/{all_total_gain} = "
             f"{100*aux_total_gain/max(1,all_total_gain):.1f}%")

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
        "aux_coverage_train_pct": float(100 * (aux_known_per_row > 0).mean()),
        "aux_coverage_test_pct":  float(100 * (aux_te_known_per_row > 0).mean()),
        "aux_gain_share_pct":     float(100 * aux_total_gain / max(1, all_total_gain)),
        "top_aux_gains":          [(n, int(v)) for n, v in aux_imp_ranked[:6]],
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
    log.info(f"CV mode: aux-augmented (aux lookup built from full train; target-being-predicted "
             f"slot always masked to prevent self-label leak)")
    log.info(f"LGB_PARAMS = {LGB_PARAMS}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    # SMILES features
    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    log.info(f"SMILES feature families: "
             + ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items()))

    # Aux lookup (built once from full train — see docstring for CV correctness argument)
    log.info(f"building aux lookup over {tr['canon'].nunique()} unique canonical SMILES in train")
    aux_lookup = build_aux_lookup(tr)
    log.info(f"aux lookup built.  aux features per row: {2*N_TARGETS} "
             f"({N_TARGETS} values + {N_TARGETS} masks)")

    # Global aux-coverage report before training loop
    log.info("=" * 60)
    log.info("AUX COVERAGE BY TARGET (from full-train lookup)")
    log.info("=" * 60)
    for t in TARGETS:
        g_te = te[te["target_type"] == t]
        aux_te = aux_features_for_target(g_te["canon"], t, aux_lookup)
        other_slot_ids = [TARGET_IDX[o] for o in TARGETS if o != t]
        mask_cols = [i + N_TARGETS for i in other_slot_ids]
        n_any = int((aux_te[:, mask_cols] > 0).any(axis=1).sum())
        log.info(f"  {t:>4s}: {n_any}/{len(g_te)} test rows have ≥1 aux value "
                 f"({100*n_any/max(1,len(g_te)):.1f}%)")

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
        "n_train":               r["n_train"],
        "n_test":                r["n_test"],
        "oof_r2":                r["oof_r2"],
        "fold_r2s":              r["fold_r2s"],
        "best_iters":            r["best_iters"],
        "refit_iters":           r["refit_iters"],
        "aux_coverage_train_pct": r["aux_coverage_train_pct"],
        "aux_coverage_test_pct":  r["aux_coverage_test_pct"],
        "aux_gain_share_pct":     r["aux_gain_share_pct"],
        "top_aux_gains":          r["top_aux_gains"],
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
            "smiles_families":  {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_smiles_features": bundle["X"].shape[1],
            "n_aux_features":   2 * N_TARGETS,
            "n_features_total": bundle["X"].shape[1] + 2 * N_TARGETS,
            "cv_mode":          "aux-augmented (lookup from full train, target slot always masked)",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = EXP_DIR / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    log.info("=" * 60)
    log.info("PER-TARGET OOF R²  (vs baseline reference)")
    log.info("=" * 60)
    baseline_ref = {"eea": 0.8587, "egb": 0.8917, "egc": 0.8948,
                    "ei": 0.7730, "eps": 0.7392, "nc": 0.7814, "tg": 0.9026}
    for tgt in TARGETS:
        r2 = per_target[tgt]['oof_r2']
        base = baseline_ref[tgt]
        delta = r2 - base
        sign = "+" if delta >= 0 else ""
        log.info(f"  {tgt:>4s}   n={per_target[tgt]['n_train']:>5d}   "
                 f"R²={r2:.4f}   (baseline {base:.4f}, Δ={sign}{delta:+.4f})   "
                 f"aux gain share={per_target[tgt]['aux_gain_share_pct']:.1f}%")
    baseline_mean = float(np.mean(list(baseline_ref.values())))
    log.info(f"  MEAN R² = {mean_r2:.4f}   (baseline mean {baseline_mean:.4f}, "
             f"Δ={mean_r2 - baseline_mean:+.4f})")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
