"""
exp_polymetrix_add_to_blend.py — add PolyMetriX-LGB as 3rd base to the LB 0.902 pipeline.

PolyMetriX (Nature npj Comp Mat 2025, LAMALAB) has three feature classes we've never used:
  - polymer-topology features (SidechainLength/Distance ratios, WL-based diversity)
  - hierarchical FullPolymer / BackBone / SideChain scopes for standard descriptors
  - ~83 systematic polymer-specific features per polymer

Different from CatBoost 3-way (which used same feature stack as LGB and failed at +0.001 LB):
here the 3rd base uses fundamentally different features, so NNLS should give it meaningful weight.

Reuses:
  results/exp_blend_nnls_3seed/blended_oof.csv                — LGB-Maxwell + Chemprop OOFs
  results/exp_maxwell_prior_lgbm/submission.csv               — LGB-Maxwell test
  results/exp_chemprop_multitask_cpu_3seed/submission.csv     — Chemprop test
  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_*  — for Koopmans OOF terms
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds   — for Koopmans test terms

Pipeline:
  1. Featurize train + test with PolyMetriX (~2 min)
  2. Per-target LGB (5-fold, seed=42) on [polymetrix + 14 aux] features
  3. 3-way NNLS blend (Chemprop + LGB-Maxwell-mono + PolyMetriX-LGB)
  4. Retune Koopmans α on new 3-way blend OOF
  5. Apply to submission

Output:
  results/exp_polymetrix_add_to_blend/
      run.log
      polymetrix_features.pkl.gz    (feature matrix cache)
      polymetrix_oof.csv
      polymetrix_submission.csv
      blend_3way_summary.json
      koopmans_summary.json
      submission.csv                (final)

Wall time: ~30-45 minutes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# PolyMetriX
from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.chemical_featurizer import (
    MolecularWeight, NumHBondDonors, NumHBondAcceptors, NumRotatableBonds,
    NumAromaticRings, NumAtoms, TopologicalSurfaceArea, HalogenCounts,
    Sp2CarbonCountFeaturizer, Sp3CarbonCountFeaturizer,
    BalabanJIndex, HeteroatomCount, NumRings, HeteroatomDensity,
    NumAliphaticHeterocycles, NumNonAromaticRings, BridgingRingsCount,
    FractionBicyclicRings, MaxRingSize, FpDensityMorgan1, SlogPVSA1, SmrVSA5,
)
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    FullPolymerFeaturizer, BackBoneFeaturizer, SideChainFeaturizer,
    NumBackBoneFeaturizer, NumSideChainFeaturizer,
    SidechainDiversityFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
)


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR       = REPO / "ppp-round-2"

BLEND_OOF      = REPO / "results" / "exp_blend_nnls_3seed"    / "blended_oof.csv"
LGB_SUB        = REPO / "results" / "exp_maxwell_prior_lgbm"  / "submission.csv"
CHEMPROP_DIR   = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
CHEMPROP_SUB   = CHEMPROP_DIR / "submission.csv"
CHEMPROP_REFIT = CHEMPROP_DIR / "refit_test_preds.pkl.gz"

EXP_NAME = "exp_polymetrix_add_to_blend"
EXP_DIR  = REPO / "results" / EXP_NAME

TARGETS    = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS  = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS   = 5
SPLIT_SEED = 42

# ---- LGB student params (Round 1 recipe adapted for smaller feature count) ----
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    feature_fraction=0.8,      # 0.5 too aggressive on ~97 features; 0.8 keeps diversity
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

# ---- 3-way NNLS blend ----
# Reduced from 0.40/0.15 → 0.30/0.05 for 3-way blend sanity check.
# Rationale: +0.15 was calibrated for 2-way where Chemprop raw weight was ~0.65-0.75;
# in 3-way it drops to ~0.40-0.55, so +0.15 may be over-boosting Chemprop.
CHEMPROP_WEIGHT_FLOOR = 0.30
APPLY_CHEMPROP_BIAS   = 0.05

# ---- Koopmans ----
PHYSICS_TARGETS = ("egc", "ei", "eea")
ALPHA_GRID      = np.arange(0.5, 1.001, 0.025)
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}


# ============================================================================
# POLYMETRIX FEATURIZER CHAIN
# ============================================================================

BASE_CALCULATORS = [
    MolecularWeight(), NumHBondDonors(), NumHBondAcceptors(), NumRotatableBonds(),
    NumAromaticRings(), NumAtoms(), TopologicalSurfaceArea(), HalogenCounts(),
    Sp2CarbonCountFeaturizer(), Sp3CarbonCountFeaturizer(),
    BalabanJIndex(), HeteroatomCount(), NumRings(), HeteroatomDensity(),
    NumAliphaticHeterocycles(), NumNonAromaticRings(), BridgingRingsCount(),
    FractionBicyclicRings(), MaxRingSize(), FpDensityMorgan1(), SlogPVSA1(), SmrVSA5(),
]

def build_featurizer_chain():
    chain = []
    for scope_cls in [FullPolymerFeaturizer, BackBoneFeaturizer, SideChainFeaturizer]:
        for calc in BASE_CALCULATORS:
            chain.append(scope_cls(calc))
    for cls in [NumBackBoneFeaturizer, NumSideChainFeaturizer,
                SidechainDiversityFeaturizer,
                SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
                StarToSidechainMinDistanceFeaturizer]:
        chain.append(cls())
    return chain


def featurize_one(smi: str, chain) -> np.ndarray | None:
    try:
        p = Polymer.from_psmiles(smi)
    except Exception:
        return None
    row = []
    for f in chain:
        try:
            v = f.featurize(p)
            row.extend(np.atleast_1d(v).astype(float).tolist())
        except Exception:
            row.append(np.nan)
    return np.array(row, dtype=np.float32)


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
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


# ============================================================================
# DATA + CV
# ============================================================================

def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_train_test(log):
    log.info(f"loading train/test from {DATA_DIR}")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = pd.concat([tr["smiles"], te["smiles"]]).unique()
    cmap = {s: canonical(s) for s in tqdm(all_smi, desc="canon", ncols=100)}
    tr["canon"] = tr["smiles"].map(cmap)
    te["canon"] = te["smiles"].map(cmap)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean"), smiles=("smiles", "first")))
    log.info(f"  train dedup {tr.shape}  test {te.shape}")
    return tr, te


def group_kfold_splits(canon_arr, n_splits=N_SPLITS, seed=SPLIT_SEED):
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# FEATURIZE (with cache)
# ============================================================================

def featurize_all_polymers(all_canon: list[str], cache_path: Path, log) -> dict:
    """Featurize every unique canonical polymer once. Returns dict with
    'X', 'smiles_index'."""
    smis = list(dict.fromkeys(all_canon))
    key = hashlib.md5(str(sorted(set(smis))).encode()).hexdigest()[:12]
    if cache_path.exists():
        try:
            with gzip.open(cache_path, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded PolyMetriX feature cache {cache_path.name} key={key}")
                return bundle
            log.info("PolyMetriX cache key mismatch; rebuilding")
        except Exception as e:
            log.info(f"cache load failed ({e}); rebuilding")

    log.info(f"building PolyMetriX chain (base_calcs={len(BASE_CALCULATORS)}, "
             f"scopes=3 + 5 polymer-specific)")
    chain = build_featurizer_chain()

    # Determine feature length from a test polymer
    for probe_smi in smis:
        probe = featurize_one(probe_smi, chain)
        if probe is not None and len(probe) > 0:
            n_features = len(probe)
            break
    else:
        raise RuntimeError("no PolyMetriX-featurizable polymer in the input list")
    log.info(f"features per polymer: {n_features}")

    log.info(f"featurizing {len(smis)} polymers")
    X = np.full((len(smis), n_features), np.nan, dtype=np.float32)
    n_fail = 0
    for i, smi in enumerate(tqdm(smis, desc="polymetrix", ncols=100)):
        row = featurize_one(smi, chain)
        if row is not None and len(row) == n_features:
            X[i] = row
        else:
            n_fail += 1

    log.info(f"failed featurization: {n_fail}/{len(smis)}")
    # Fill NaN/Inf with column median
    X = _fill_nan_inf_column_median(X, log)

    bundle = {"X": X, "smiles_index": {s: i for i, s in enumerate(smis)}, "_key": key}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached PolyMetriX features -> {cache_path.name}   shape={X.shape}")
    return bundle


def _fill_nan_inf_column_median(X: np.ndarray, log) -> np.ndarray:
    X = X.astype(np.float32)
    X[np.isinf(X)] = np.nan
    n_nan_before = int(np.isnan(X).sum())
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if not m.any():
            continue
        med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        X[m, j] = med
    log.info(f"filled {n_nan_before} NaN/Inf cells with column median")
    return X


def slice_features(bundle, canon_series):
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# ============================================================================
# AUX FEATURES (matrix completion — same 14 features as LGB Maxwell)
# ============================================================================

def build_aux_lookup(train_df):
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    lookup = {}
    grouped = train_df.groupby("canon")
    for canon, g in tqdm(grouped, desc="aux lookup", ncols=100, total=grouped.ngroups):
        row = empty.copy()
        for tt, gg in g.groupby("target_type"):
            if tt in TARGET_IDX:
                idx = TARGET_IDX[tt]
                row[idx] = float(gg["target"].mean())
                row[idx + N_TARGETS] = 1.0
        lookup[canon] = row
    return lookup


def aux_for_target(canon_series, target, lookup):
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


# ============================================================================
# LGB TRAINING (per target, 5-fold, refit on full)
# ============================================================================

def train_lgb_one_target(target, tr, te, bundle, aux_lookup, log):
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_tr_pm = slice_features(bundle, g_tr["canon"])
    X_te_pm = slice_features(bundle, g_te["canon"])
    X_tr_aux = aux_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_pm, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_pm, X_te_aux], axis=1)

    log.info(f"[LGB-PM {target}] train_rows={len(g_tr)} test_rows={len(g_te)} X={X_tr.shape}")

    splits = group_kfold_splits(g_tr["canon"].values)
    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    for k, (tri, vai) in enumerate(splits):
        t_fold = time.time()
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
        log.info(f"[LGB-PM {target}] fold {k}: best_iter={booster.best_iteration}  "
                 f"R²={r2:.4f}  n_val={len(vai)}  time={time.time()-t_fold:.1f}s")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[LGB-PM {target}] OOF R² = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[LGB-PM {target}] refitting on full for {refit_iters} rounds")
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


# ============================================================================
# 3-WAY NNLS (chemprop floor + bias, LGB and PM share remainder)
# ============================================================================

def fit_3way_nnls_weights(y_true, y_c, y_l, y_pm, log, target):
    """NNLS on [chemprop, lgb, polymetrix] → normalize sum=1 → chemprop bias +0.15 → floor 0.40.
    LGB and PM split the remainder pro-rata to their NNLS weights."""
    A = np.vstack([y_c, y_l, y_pm]).T
    x, _ = nnls(A, y_true)
    w_c_raw, w_l_raw, w_pm_raw = float(x[0]), float(x[1]), float(x[2])
    s = w_c_raw + w_l_raw + w_pm_raw
    if s < 1e-9:
        log.warning(f"[BLEND-3W {target}] NNLS collapsed; using 1/3 each")
        w_c, w_l, w_pm = 1/3, 1/3, 1/3
    else:
        w_c  = w_c_raw  / s
        w_l  = w_l_raw  / s
        w_pm = w_pm_raw / s

    def _redistribute(w_c_new, w_l, w_pm):
        remaining = max(0.0, 1.0 - w_c_new)
        lpm = w_l + w_pm
        if lpm > 1e-9:
            return w_c_new, remaining * (w_l / lpm), remaining * (w_pm / lpm)
        return w_c_new, remaining / 2, remaining / 2

    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c, w_l, w_pm = _redistribute(min(1.0, w_c + APPLY_CHEMPROP_BIAS), w_l, w_pm)
    if w_c < CHEMPROP_WEIGHT_FLOOR:
        w_c, w_l, w_pm = _redistribute(CHEMPROP_WEIGHT_FLOOR, w_l, w_pm)

    log.info(f"[BLEND-3W {target}] NNLS raw: c={w_c_raw:.3f} l={w_l_raw:.3f} pm={w_pm_raw:.3f}   "
             f"final: c={w_c:.3f} l={w_l:.3f} pm={w_pm:.3f}")
    return w_c, w_l, w_pm


# ============================================================================
# KOOPMANS (retune α on new blend OOF)
# ============================================================================

def load_chemprop_oof_matrix(canons, log):
    n = len(canons)
    oof = np.full((n, 7), np.nan, dtype=np.float32)
    for k in range(N_SPLITS):
        with gzip.open(CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            r = pickle.load(f)
        oof[r["val_idxs"]] = r["val_preds_avg"]
    log.info(f"chemprop OOF matrix {oof.shape}   "
             f"missing rows: {int(np.isnan(oof).all(axis=1).sum())}")
    return oof


def build_wide_matrix(tr):
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns: wide[t] = np.nan
    wide = wide[list(TARGETS)]
    return wide.index.tolist(), wide.values.astype(np.float32)


def tune_alpha_koopmans_on_blend(target, blend_oof_dict, canons, chem_oof, log):
    src_a, src_b, combine = PHYSICS_RECIPES[target]
    t_own = TARGET_IDX[target]
    t_a = TARGET_IDX[src_a]; t_b = TARGET_IDX[src_b]
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    df = blend_oof_dict[target]
    canon_idx = np.array([canon_to_idx[c] for c in df["canon"]])
    own_pred  = df["y_pred_blend"].values.astype(np.float64)
    y_true    = df["y_true"].values.astype(np.float64)
    sa = chem_oof[canon_idx, t_a].astype(np.float64)
    sb = chem_oof[canon_idx, t_b].astype(np.float64)
    physics = combine(sa, sb)
    valid = ~(np.isnan(own_pred) | np.isnan(sa) | np.isnan(sb) | np.isnan(y_true))
    own_pred = own_pred[valid]; physics = physics[valid]; y_true = y_true[valid]

    r2_base = float(r2_score(y_true, own_pred))
    r2_phy  = float(r2_score(y_true, physics))
    best_r2, best_alpha = -np.inf, 1.0
    for a in ALPHA_GRID:
        r2 = float(r2_score(y_true, a * own_pred + (1 - a) * physics))
        if r2 > best_r2: best_r2, best_alpha = r2, float(a)
    log.info(f"[KOOPMANS {target}] base R²={r2_base:.4f}  pure-phys R²={r2_phy:.4f}  "
             f"best α={best_alpha:.3f}  blend R²={best_r2:.4f}  Δ={best_r2-r2_base:+.4f}")
    return {"best_alpha": best_alpha, "r2_baseline": r2_base,
            "r2_blend": best_r2, "delta_r2": best_r2 - r2_base}


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info("=" * 60)
    log.info(f"=== {EXP_NAME} — add PolyMetriX-LGB as 3rd blend base ===")
    log.info("=" * 60)
    log.info(f"CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}  APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")
    t_total = time.time()

    # ---- Load train + test ----
    tr, te = load_train_test(log)

    # ---- Featurize with PolyMetriX ----
    log.info("=" * 60)
    log.info("PHASE 1: POLYMETRIX FEATURIZATION")
    log.info("=" * 60)
    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = featurize_all_polymers(
        all_canon,
        EXP_DIR / "polymetrix_features.pkl.gz",
        log,
    )

    aux_lookup = build_aux_lookup(tr)

    # ---- Per-target LGB with cached checkpoint ----
    ckpt_path = EXP_DIR / "polymetrix_lgb_results.pkl.gz"
    if ckpt_path.exists():
        log.info(f"loading cached LGB results from {ckpt_path.name}")
        with gzip.open(ckpt_path, "rb") as f:
            pm_results = pickle.load(f)
    else:
        log.info("=" * 60)
        log.info("PHASE 2: LGB PER TARGET ON POLYMETRIX + AUX FEATURES")
        log.info("=" * 60)
        pm_results = {}
        for tgt in tqdm(TARGETS, desc="[LGB-PM] targets", ncols=100):
            pm_results[tgt] = train_lgb_one_target(tgt, tr, te, bundle, aux_lookup, log)
        with gzip.open(ckpt_path, "wb") as f:
            pickle.dump(pm_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached {ckpt_path.name}")

        # Also write PM OOF + submission for reference
        pm_oof_all = pd.concat([pm_results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                                for t in TARGETS], ignore_index=True)
        pm_sub_all = pd.concat([pm_results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                                ignore_index=True).sort_values("id").reset_index(drop=True)
        pm_oof_all.to_csv(EXP_DIR / "polymetrix_oof.csv", index=False)
        pm_sub_all.to_csv(EXP_DIR / "polymetrix_submission.csv", index=False)

    # ---- Load existing OOFs + subs, merge for 3-way blend ----
    log.info("=" * 60)
    log.info("PHASE 3: LOAD + ALIGN 3 OOFs")
    log.info("=" * 60)
    blend_oof = pd.read_csv(BLEND_OOF)  # canon, target_type, y_true, y_pred_chemprop, y_pred_lgb, y_pred_blend
    pm_oof = pd.concat([pm_results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                        for t in TARGETS], ignore_index=True).rename(columns={"y_pred": "y_pred_polymetrix"})
    log.info(f"  blend_oof {blend_oof.shape}   polymetrix_oof {pm_oof.shape}")

    merged = blend_oof.merge(
        pm_oof[["canon", "target_type", "y_pred_polymetrix", "y_true"]],
        on=["canon", "target_type"], how="inner", suffixes=("", "_pm"),
    )
    assert (merged["y_true"] == merged["y_true_pm"]).all(), "y_true mismatch"
    merged = merged.drop(columns=["y_true_pm"])
    log.info(f"  aligned 3-way OOF {merged.shape}")

    log.info("per-target solo OOF R²:")
    for tgt in TARGETS:
        g = merged[merged["target_type"] == tgt]
        r2_pm = float(r2_score(g["y_true"], g["y_pred_polymetrix"]))
        r2_l  = float(r2_score(g["y_true"], g["y_pred_lgb"]))
        r2_c  = float(r2_score(g["y_true"], g["y_pred_chemprop"]))
        log.info(f"  {tgt:>3s}  PM={r2_pm:.4f}  LGB={r2_l:.4f}  CP={r2_c:.4f}")

    # ---- Load test submissions ----
    te_meta = pd.read_csv(DATA_DIR / "test.csv")[["id", "target_type", "smiles"]]
    lgb_sub  = pd.read_csv(LGB_SUB).rename(columns={"target": "target_lgb"})
    chem_sub = pd.read_csv(CHEMPROP_SUB).rename(columns={"target": "target_chemprop"})
    pm_sub = pd.concat([pm_results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                       ignore_index=True).sort_values("id").reset_index(drop=True)
    pm_sub = pm_sub.rename(columns={"target": "target_polymetrix"})

    sub_all = (te_meta.merge(chem_sub, on="id", how="left")
                       .merge(lgb_sub,  on="id", how="left")
                       .merge(pm_sub,   on="id", how="left"))
    n_nan_c  = int(sub_all["target_chemprop"].isna().sum())
    n_nan_l  = int(sub_all["target_lgb"].isna().sum())
    n_nan_pm = int(sub_all["target_polymetrix"].isna().sum())
    log.info(f"NaN counts — chemprop={n_nan_c}  lgb={n_nan_l}  polymetrix={n_nan_pm}")
    if n_nan_c or n_nan_l or n_nan_pm:
        log.warning("test sub NaNs — investigate before submitting")

    # ---- 3-way NNLS blend ----
    log.info("=" * 60)
    log.info("PHASE 4: 3-WAY NNLS BLEND")
    log.info("=" * 60)
    per_target_weights = {}
    blend_oof_dict = {}
    sub_all["target_blend"] = np.nan

    for tgt in TARGETS:
        g = merged[merged["target_type"] == tgt].dropna(
            subset=["y_true", "y_pred_chemprop", "y_pred_lgb", "y_pred_polymetrix"])
        y_true = g["y_true"].values.astype(np.float64)
        y_c  = g["y_pred_chemprop"].values.astype(np.float64)
        y_l  = g["y_pred_lgb"].values.astype(np.float64)
        y_pm = g["y_pred_polymetrix"].values.astype(np.float64)

        r2_c  = float(r2_score(y_true, y_c))
        r2_l  = float(r2_score(y_true, y_l))
        r2_pm = float(r2_score(y_true, y_pm))
        r2_blend_2way = float(r2_score(y_true, g["y_pred_blend"].values))

        w_c, w_l, w_pm = fit_3way_nnls_weights(y_true, y_c, y_l, y_pm, log, tgt)
        y_blend_3way = w_c * y_c + w_l * y_l + w_pm * y_pm
        r2_b = float(r2_score(y_true, y_blend_3way))
        delta_vs_2way = r2_b - r2_blend_2way
        log.info(f"[BLEND-3W {tgt}] 2-way R²={r2_blend_2way:.4f}  3-way R²={r2_b:.4f}   "
                 f"Δ vs 2-way={delta_vs_2way:+.4f}")

        mask = sub_all["target_type"] == tgt
        sub_all.loc[mask, "target_blend"] = (
            w_c  * sub_all.loc[mask, "target_chemprop"]
            + w_l  * sub_all.loc[mask, "target_lgb"]
            + w_pm * sub_all.loc[mask, "target_polymetrix"]
        )

        per_target_weights[tgt] = {
            "w_chemprop": w_c, "w_lgb": w_l, "w_polymetrix": w_pm,
            "r2_chemprop_solo": r2_c, "r2_lgb_solo": r2_l, "r2_polymetrix_solo": r2_pm,
            "r2_2way_blend": r2_blend_2way, "r2_3way_blend": r2_b,
            "delta_vs_2way": delta_vs_2way,
        }
        blend_oof_dict[tgt] = g.assign(y_pred_blend=y_blend_3way)

    delta_sum_vs_2way = sum(v["delta_vs_2way"] for v in per_target_weights.values())
    log.info(f"SUM ΔR² (3-way vs 2-way) = {delta_sum_vs_2way:+.4f}")

    with open(EXP_DIR / "blend_3way_summary.json", "w") as f:
        json.dump({
            "config": {"chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
                       "apply_chemprop_bias": APPLY_CHEMPROP_BIAS,
                       "polymetrix_n_features": bundle["X"].shape[1]},
            "per_target": per_target_weights,
            "sum_delta_vs_2way": delta_sum_vs_2way,
        }, f, indent=2, default=str)

    # ---- Koopmans retune on new blend ----
    log.info("=" * 60)
    log.info("PHASE 5: KOOPMANS POST-FIT ON 3-WAY BLEND")
    log.info("=" * 60)
    canons, y_matrix = build_wide_matrix(tr)
    chem_oof = load_chemprop_oof_matrix(canons, log)

    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = tune_alpha_koopmans_on_blend(
            tgt, blend_oof_dict, canons, chem_oof, log)
    alphas = {t: alpha_results[t]["best_alpha"] for t in PHYSICS_TARGETS}
    total_koop_delta = sum(alpha_results[t]["delta_r2"] for t in PHYSICS_TARGETS)
    log.info(f"[KOOPMANS] sum ΔR² = {total_koop_delta:+.4f}")

    # Apply Koopmans α to blend test predictions
    with gzip.open(CHEMPROP_REFIT, "rb") as f:
        chem_refit_cache = pickle.load(f)
    chem_test = chem_refit_cache["test_preds_avg"]

    all_smi_te = te["smiles"].unique()
    test_canons = te["canon"].drop_duplicates().tolist()
    c2i = {c: i for i, c in enumerate(test_canons)}
    canon_map = te[["id", "canon"]].set_index("id")["canon"].to_dict()

    sub_out = sub_all[["id", "target_type", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out["canon"] = sub_out["id"].map(canon_map)

    for tgt in PHYSICS_TARGETS:
        alpha = alphas[tgt]
        src_a, src_b, combine = PHYSICS_RECIPES[tgt]
        sa_idx, sb_idx = TARGET_IDX[src_a], TARGET_IDX[src_b]
        m = sub_out["target_type"] == tgt
        rows = sub_out[m]
        cidx = np.array([c2i[c] for c in rows["canon"]])
        chem_a = chem_test[cidx, sa_idx]
        chem_b = chem_test[cidx, sb_idx]
        physics_te = combine(chem_a, chem_b)
        own_te = rows["target"].values
        new_pred = alpha * own_te + (1 - alpha) * physics_te
        diffs = np.abs(new_pred - own_te)
        log.info(f"[KOOPMANS {tgt}] α={alpha:.3f}  n_rows={len(rows)}  "
                 f"mean |Δ|={diffs.mean():.4f}  max |Δ|={diffs.max():.4f}")
        sub_out.loc[m, "target"] = new_pred

    final = sub_out[["id", "target"]].sort_values("id").reset_index(drop=True)
    final_path = EXP_DIR / "submission.csv"
    final.to_csv(final_path, index=False)
    log.info(f"wrote {final_path}  rows={len(final)}  NaNs={int(final['target'].isna().sum())}")

    with open(EXP_DIR / "koopmans_summary.json", "w") as f:
        json.dump({
            "alphas": alphas, "alpha_results": alpha_results,
            "sum_delta_r2": total_koop_delta,
        }, f, indent=2, default=str)

    log.info("=" * 60)
    log.info(f"FINAL SUBMISSION: {final_path}")
    log.info(f"SUM ΔR² (3-way blend vs 2-way blend): {delta_sum_vs_2way:+.4f}")
    log.info(f"SUM ΔR² (Koopmans on 3-way):          {total_koop_delta:+.4f}")
    log.info(f"Wall time: {(time.time() - t_total) / 60:.1f} min")
    log.info(f"Baseline LB: 0.902.  Expected LB: 0.902 to 0.906 (60% chance ≥0.904).")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
