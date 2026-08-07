"""
exp_catboost_add_to_blend.py — add existing CatBoost (chain-ext) as 3rd base
to the LB 0.902 pipeline. NO TRAINING — just blend + Koopmans.

CatBoost was already trained at:
  results/exp_chain_ext_catboost/
      oof.csv         (7405 rows, Maxwell already applied, mean R² 0.860)
      submission.csv  (4940 rows)

Reuses:
  results/exp_chain_ext_catboost/{oof.csv, submission.csv}    — CatBoost
  results/exp_blend_nnls_3seed/blended_oof.csv                — LGB-Maxwell + Chemprop OOFs
  results/exp_maxwell_prior_lgbm/submission.csv               — LGB-Maxwell test
  results/exp_chemprop_multitask_cpu_3seed/submission.csv     — Chemprop test
  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_*  — for Koopmans OOF terms
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds   — for Koopmans test terms

Pipeline:
  1. Load 3 aligned OOFs (Chemprop, LGB-Max, CatBoost-Max)
  2. Per-target 3-way NNLS blend  (chemprop floor 0.40 + bias +0.15,
     LGB and CatBoost share the remainder pro-rata)
  3. Retune Koopmans α on the new 3-way blend OOF
  4. Apply Koopmans to 3-way blend test predictions
  5. Write submission

Output:
  results/exp_catboost_add_to_blend/
      run.log
      blend_3way_summary.json
      koopmans_summary.json
      submission.csv                (final)

Wall time: <30 seconds.
"""
from __future__ import annotations

import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR       = REPO / "ppp-round-2"

CB_OOF         = REPO / "results" / "exp_chain_ext_catboost" / "oof.csv"
CB_SUB         = REPO / "results" / "exp_chain_ext_catboost" / "submission.csv"
BLEND_OOF      = REPO / "results" / "exp_blend_nnls_3seed"    / "blended_oof.csv"
LGB_SUB        = REPO / "results" / "exp_maxwell_prior_lgbm"  / "submission.csv"
CHEMPROP_DIR   = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
CHEMPROP_SUB   = CHEMPROP_DIR / "submission.csv"
CHEMPROP_REFIT = CHEMPROP_DIR / "refit_test_preds.pkl.gz"

EXP_NAME = "exp_catboost_add_to_blend"
EXP_DIR  = REPO / "results" / EXP_NAME

TARGETS    = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS  = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5

# 3-way NNLS blend (chemprop floor + bias — same tricks as 0.902 pipeline)
CHEMPROP_WEIGHT_FLOOR = 0.40
APPLY_CHEMPROP_BIAS   = 0.15

# Koopmans post-fit
PHYSICS_TARGETS = ("egc", "ei", "eea")
ALPHA_GRID      = np.arange(0.5, 1.001, 0.025)
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}


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
# DATA (train canons + Chemprop OOF/refit for Koopmans)
# ============================================================================

def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_train_canons(log):
    log.info(f"loading train.csv from {DATA_DIR}")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    all_smi = tr["smiles"].unique()
    cmap = {s: canonical(s) for s in tqdm(all_smi, desc="canon", ncols=100)}
    tr["canon"] = tr["smiles"].map(cmap)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean")))
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns: wide[t] = np.nan
    wide = wide[list(TARGETS)]
    return tr, wide.index.tolist(), wide.values.astype(np.float32)


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


# ============================================================================
# 3-WAY NNLS BLEND (chemprop floor + bias, LGB and CB share remainder)
# ============================================================================

def fit_3way_nnls_weights(y_true, y_c, y_l, y_cb, log, target):
    """NNLS on [chemprop, lgb, catboost] → normalize sum=1 → chemprop bias
    (+0.15) then floor (0.40). L and CB split the remainder pro-rata to their
    NNLS weights."""
    A = np.vstack([y_c, y_l, y_cb]).T
    x, _ = nnls(A, y_true)
    w_c_raw, w_l_raw, w_cb_raw = float(x[0]), float(x[1]), float(x[2])
    s = w_c_raw + w_l_raw + w_cb_raw
    if s < 1e-9:
        log.warning(f"[BLEND-3W {target}] NNLS collapsed to zero; using 1/3 each")
        w_c, w_l, w_cb = 1/3, 1/3, 1/3
    else:
        w_c  = w_c_raw  / s
        w_l  = w_l_raw  / s
        w_cb = w_cb_raw / s

    def _redistribute_chemprop(w_c_new, w_l, w_cb):
        remaining = max(0.0, 1.0 - w_c_new)
        lcb = w_l + w_cb
        if lcb > 1e-9:
            return w_c_new, remaining * (w_l / lcb), remaining * (w_cb / lcb)
        return w_c_new, remaining / 2, remaining / 2

    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c, w_l, w_cb = _redistribute_chemprop(min(1.0, w_c + APPLY_CHEMPROP_BIAS), w_l, w_cb)
    if w_c < CHEMPROP_WEIGHT_FLOOR:
        w_c, w_l, w_cb = _redistribute_chemprop(CHEMPROP_WEIGHT_FLOOR, w_l, w_cb)

    log.info(f"[BLEND-3W {target}] NNLS raw: c={w_c_raw:.3f} l={w_l_raw:.3f} cb={w_cb_raw:.3f}   "
             f"final: c={w_c:.3f} l={w_l:.3f} cb={w_cb:.3f}")
    return w_c, w_l, w_cb


# ============================================================================
# KOOPMANS POST-FIT (retune α on new blend OOF)
# ============================================================================

def tune_alpha_koopmans_on_blend(target, blend_oof_dict, y_matrix, canons, chem_oof, log):
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
    log.info(f"=== {EXP_NAME} — add existing CatBoost as 3rd blend base ===")
    log.info("=" * 60)
    log.info(f"CatBoost source: {CB_OOF.parent}   (chain-ext features, mean R²=0.860)")
    log.info(f"CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")
    t0 = time.time()

    # ---- Load train canons for Koopmans ----
    tr, canons, y_matrix = load_train_canons(log)
    log.info(f"train canons: {len(canons)}")

    # ---- Load 3 OOFs, align on (canon, target_type, y_true) ----
    log.info("=" * 60)
    log.info("LOAD + ALIGN 3 OOFs")
    log.info("=" * 60)
    blend = pd.read_csv(BLEND_OOF)  # has y_pred_chemprop, y_pred_lgb, y_pred_blend
    cb = pd.read_csv(CB_OOF).rename(columns={"y_pred": "y_pred_catboost"})
    log.info(f"  blend_oof {blend.shape}   catboost_oof {cb.shape}")

    merged = blend.merge(
        cb[["canon", "target_type", "y_pred_catboost", "y_true"]],
        on=["canon", "target_type"], how="inner", suffixes=("", "_cb"),
    )
    assert (merged["y_true"] == merged["y_true_cb"]).all(), "y_true mismatch — CV splits differ"
    merged = merged.drop(columns=["y_true_cb"])
    log.info(f"  aligned 3-way OOF {merged.shape}")

    # ---- Per-target CatBoost solo R² (sanity) ----
    log.info("per-target CatBoost solo OOF R²:")
    for tgt in TARGETS:
        g = merged[merged["target_type"] == tgt]
        r2_cb = float(r2_score(g["y_true"], g["y_pred_catboost"]))
        r2_l  = float(r2_score(g["y_true"], g["y_pred_lgb"]))
        r2_c  = float(r2_score(g["y_true"], g["y_pred_chemprop"]))
        log.info(f"  {tgt:>3s}  CB={r2_cb:.4f}  LGB={r2_l:.4f}  CP={r2_c:.4f}")

    # ---- Load 3 test submissions ----
    te_meta = pd.read_csv(DATA_DIR / "test.csv")[["id", "target_type", "smiles"]]
    lgb_sub  = pd.read_csv(LGB_SUB).rename(columns={"target": "target_lgb"})
    chem_sub = pd.read_csv(CHEMPROP_SUB).rename(columns={"target": "target_chemprop"})
    cb_sub   = pd.read_csv(CB_SUB).rename(columns={"target": "target_catboost"})
    log.info(f"test subs — LGB {lgb_sub.shape}  Chemprop {chem_sub.shape}  CatBoost {cb_sub.shape}")

    sub_all = (te_meta.merge(chem_sub, on="id", how="left")
                       .merge(lgb_sub,  on="id", how="left")
                       .merge(cb_sub,   on="id", how="left"))
    n_nan_c  = int(sub_all["target_chemprop"].isna().sum())
    n_nan_l  = int(sub_all["target_lgb"].isna().sum())
    n_nan_cb = int(sub_all["target_catboost"].isna().sum())
    log.info(f"NaN counts after merge — chemprop={n_nan_c}  lgb={n_nan_l}  catboost={n_nan_cb}")
    if n_nan_c or n_nan_l or n_nan_cb:
        log.warning("test sub NaNs — investigate before submitting")

    # ---- 3-way NNLS blend ----
    log.info("=" * 60)
    log.info("3-WAY NNLS BLEND")
    log.info("=" * 60)
    per_target_weights = {}
    blend_oof_dict = {}
    sub_all["target_blend"] = np.nan

    for tgt in TARGETS:
        g = merged[merged["target_type"] == tgt].dropna(
            subset=["y_true", "y_pred_chemprop", "y_pred_lgb", "y_pred_catboost"])
        y_true = g["y_true"].values.astype(np.float64)
        y_c  = g["y_pred_chemprop"].values.astype(np.float64)
        y_l  = g["y_pred_lgb"].values.astype(np.float64)
        y_cb = g["y_pred_catboost"].values.astype(np.float64)

        r2_c  = float(r2_score(y_true, y_c))
        r2_l  = float(r2_score(y_true, y_l))
        r2_cb = float(r2_score(y_true, y_cb))
        r2_blend_2way = float(r2_score(y_true, g["y_pred_blend"].values))

        w_c, w_l, w_cb = fit_3way_nnls_weights(y_true, y_c, y_l, y_cb, log, tgt)
        y_blend_3way = w_c * y_c + w_l * y_l + w_cb * y_cb
        r2_b = float(r2_score(y_true, y_blend_3way))
        delta_vs_2way = r2_b - r2_blend_2way
        log.info(f"[BLEND-3W {tgt}] 2-way R²={r2_blend_2way:.4f}  3-way R²={r2_b:.4f}   "
                 f"Δ vs 2-way={delta_vs_2way:+.4f}")

        mask = sub_all["target_type"] == tgt
        sub_all.loc[mask, "target_blend"] = (
            w_c  * sub_all.loc[mask, "target_chemprop"]
            + w_l  * sub_all.loc[mask, "target_lgb"]
            + w_cb * sub_all.loc[mask, "target_catboost"]
        )

        per_target_weights[tgt] = {
            "w_chemprop": w_c, "w_lgb": w_l, "w_catboost": w_cb,
            "r2_chemprop_solo": r2_c, "r2_lgb_solo": r2_l, "r2_catboost_solo": r2_cb,
            "r2_2way_blend": r2_blend_2way, "r2_3way_blend": r2_b,
            "delta_vs_2way": delta_vs_2way,
        }
        blend_oof_dict[tgt] = g.assign(y_pred_blend=y_blend_3way)

    delta_sum_vs_2way = sum(v["delta_vs_2way"] for v in per_target_weights.values())
    log.info(f"SUM ΔR² (3-way vs 2-way) = {delta_sum_vs_2way:+.4f}")

    with open(EXP_DIR / "blend_3way_summary.json", "w") as f:
        json.dump({
            "config": {"chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
                       "apply_chemprop_bias": APPLY_CHEMPROP_BIAS},
            "per_target": per_target_weights,
            "sum_delta_vs_2way": delta_sum_vs_2way,
        }, f, indent=2, default=str)

    # ---- Koopmans retune α on new blend OOF ----
    log.info("=" * 60)
    log.info("KOOPMANS POST-FIT ON 3-WAY BLEND (retune α on new OOF)")
    log.info("=" * 60)
    chem_oof = load_chemprop_oof_matrix(canons, log)
    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = tune_alpha_koopmans_on_blend(
            tgt, blend_oof_dict, y_matrix, canons, chem_oof, log,
        )
    alphas = {t: alpha_results[t]["best_alpha"] for t in PHYSICS_TARGETS}
    total_koop_delta = sum(alpha_results[t]["delta_r2"] for t in PHYSICS_TARGETS)
    log.info(f"[KOOPMANS] sum ΔR² = {total_koop_delta:+.4f}   "
             f"expected 7-target mean uplift from Koopmans = {total_koop_delta/7:+.4f}")

    # ---- Apply Koopmans α to 3-way blend test predictions ----
    log.info("=" * 60)
    log.info("APPLY KOOPMANS TO 3-WAY BLEND TEST PREDICTIONS")
    log.info("=" * 60)
    with gzip.open(CHEMPROP_REFIT, "rb") as f:
        chem_refit_cache = pickle.load(f)
    chem_test = chem_refit_cache["test_preds_avg"]

    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    cmap = {s: canonical(s) for s in all_smi}
    te["canon"] = te["smiles"].map(cmap)
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
            "expected_mean_uplift": total_koop_delta / 7,
        }, f, indent=2, default=str)

    log.info("=" * 60)
    log.info(f"FINAL SUBMISSION: {final_path}")
    log.info(f"SUM ΔR² (3-way blend vs 2-way blend): {delta_sum_vs_2way:+.4f}")
    log.info(f"SUM ΔR² (Koopmans on 3-way): {total_koop_delta:+.4f}")
    log.info(f"Wall time: {time.time() - t0:.1f}s")
    log.info(f"Baseline LB: 0.902.  If SUM Δ vs 2-way is +0.005 or more → likely LB gain.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
