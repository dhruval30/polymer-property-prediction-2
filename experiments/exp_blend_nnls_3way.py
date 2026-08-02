"""
exp_blend_nnls_3way.py — per-target NNLS blend of Chemprop + LGB(Maxwell) + CatBoost(Maxwell).

============================================================================
DEPENDENCIES (must exist BEFORE running this script)
============================================================================

Three prior experiments' outputs are required:

  1. results/exp_chemprop_multitask_cpu/
       - oof.csv, submission.csv
     Script: experiments/exp_chemprop_multitask_cpu.py     LB solo: 0.887

  2. results/exp_maxwell_prior_lgbm/
       - oof.csv, submission.csv
     Script: experiments/exp_maxwell_prior_lgbm.py         LB solo: 0.860

  3. results/exp_maxwell_prior_catboost/
       - oof.csv, submission.csv
     Script: experiments/exp_maxwell_prior_catboost.py     LB solo: ~0.860 (est.)

Any missing directory → script exits with clear error and reproduction commands.

Current best 2-way blend (Chemprop + LGB only): LB 0.894, rank 5.
See docs/best-ensemble.md for the full reproduction chain.

============================================================================
METHOD
============================================================================

Per target:
  1. Load OOF rows from all 3 sources. Align by (canon, target_type).
  2. NNLS on A = [y_chemprop, y_lgb, y_cat] vs b = y_true → weights (w_c, w_l, w_x).
  3. Normalize weights to sum=1.
  4. Apply LB-vs-OOF-bias mitigations (same rationale as 2-way blend):
     - CHEMPROP_WEIGHT_FLOOR: Chemprop weight ≥ this. Prevents blend from
       over-trusting aux-inflated LGB/CAT OOFs.
     - APPLY_CHEMPROP_BIAS: shift Chemprop weight up by fixed offset post-normalize.
     - LGB/CAT weights renormalized proportionally after Chemprop adjustment.
  5. Report per-target: individual R²s, blend R², weights, deltas.
  6. Apply weights to test predictions.

============================================================================
WHY 3-WAY MIGHT (OR MIGHT NOT) BEAT 2-WAY
============================================================================

CAT and LGB have per-target complementarity even though their means are equal:
  - CAT wins solo on eea (+0.011), egc (+0.007), tg (+0.006)
  - LGB wins solo on egb (-0.005), eps (-0.021), nc (-0.007)

If NNLS splits the "tree family" allocation by per-target skill, we get modest
per-target improvements. Total mean gain vs 2-way: expected +0.001 to +0.005 OOF.

Downside case: LGB and CAT errors are highly correlated (both trees on same
features) → NNLS could just average them, adding no real signal. In that case
blend LB ≈ 2-way blend LB (0.894).

============================================================================
OUTPUTS  (under results/exp_blend_nnls_3way/)
============================================================================

  run.log              — full log
  blend_summary.json   — per-target weights, R²s, deltas, config
  submission.csv       — blended predictions (Kaggle format)
  blended_oof.csv      — per-row blended OOFs (for future 4-way blending)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_nnls_3way.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import r2_score


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
EXP_NAME = "exp_blend_nnls_3way"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu"
LGB_DIR      = REPO / "results" / "exp_maxwell_prior_lgbm"
CAT_DIR      = REPO / "results" / "exp_maxwell_prior_catboost"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

# LB-bias mitigations (same rationale as 2-way blend — see best-ensemble.md)
CHEMPROP_WEIGHT_FLOOR = 0.40   # Chemprop always keeps ≥40% weight
APPLY_CHEMPROP_BIAS   = 0.15   # shift Chemprop weight up by this after NNLS-norm


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
# INPUT VALIDATION
# ============================================================================

def check_inputs(log: logging.Logger) -> None:
    missing = []
    for label, d in [("Chemprop", CHEMPROP_DIR), ("LGB+Maxwell", LGB_DIR), ("CatBoost+Maxwell", CAT_DIR)]:
        for fname in ["oof.csv", "submission.csv"]:
            p = d / fname
            if not p.exists():
                missing.append(f"  - {p}   ({label} source)")
    if missing:
        log.error("MISSING INPUT FILES:")
        for m in missing:
            log.error(m)
        log.error("")
        log.error("Run the missing experiment scripts first:")
        log.error("  poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py")
        log.error("  poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py")
        log.error("  poly2-venv/bin/python experiments/exp_maxwell_prior_catboost.py")
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  Chemprop dir: {CHEMPROP_DIR}")
    log.info(f"  LGB+Max dir : {LGB_DIR}")
    log.info(f"  CAT+Max dir : {CAT_DIR}")


# ============================================================================
# LOAD + ALIGN
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (oof_wide, sub_wide) with columns:
       oof_wide:  canon, target_type, y_true, y_pred_chemprop, y_pred_lgb, y_pred_cat
       sub_wide:  id, target_type, target_chemprop, target_lgb, target_cat
    """
    log.info("loading OOF from all three sources...")
    oof_c = pd.read_csv(CHEMPROP_DIR / "oof.csv")
    oof_l = pd.read_csv(LGB_DIR / "oof.csv")
    oof_x = pd.read_csv(CAT_DIR / "oof.csv")
    log.info(f"  Chemprop OOF: {oof_c.shape}")
    log.info(f"  LGB+Max  OOF: {oof_l.shape}")
    log.info(f"  CAT+Max  OOF: {oof_x.shape}")

    oof = (
        oof_c.rename(columns={"y_pred": "y_pred_chemprop"})
        .merge(oof_l.rename(columns={"y_pred": "y_pred_lgb"})[["canon", "target_type", "y_pred_lgb"]],
               on=["canon", "target_type"], how="inner")
        .merge(oof_x.rename(columns={"y_pred": "y_pred_cat"})[["canon", "target_type", "y_pred_cat"]],
               on=["canon", "target_type"], how="inner")
    )
    log.info(f"  aligned OOF: {oof.shape}  (train rows with all 3 predictions)")

    for col in ["y_true", "y_pred_chemprop", "y_pred_lgb", "y_pred_cat"]:
        n_nan = int(oof[col].isna().sum())
        if n_nan:
            log.warning(f"  {col} has {n_nan} NaN rows — dropped per-target")

    log.info("loading test submissions from all three sources...")
    sub_c = pd.read_csv(CHEMPROP_DIR / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_l = pd.read_csv(LGB_DIR / "submission.csv").rename(columns={"target": "target_lgb"})
    sub_x = pd.read_csv(CAT_DIR / "submission.csv").rename(columns={"target": "target_cat"})
    log.info(f"  Chemprop sub: {sub_c.shape}")
    log.info(f"  LGB+Max  sub: {sub_l.shape}")
    log.info(f"  CAT+Max  sub: {sub_x.shape}")

    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = (
        te.merge(sub_c, on="id", how="left")
          .merge(sub_l, on="id", how="left")
          .merge(sub_x, on="id", how="left")
    )
    log.info(f"  aligned sub: {sub.shape}")

    for col in ["target_chemprop", "target_lgb", "target_cat"]:
        n_nan = int(sub[col].isna().sum())
        if n_nan:
            log.warning(f"  sub {col} has {n_nan} NaN rows")

    return oof, sub


# ============================================================================
# PER-TARGET 3-WAY NNLS
# ============================================================================

def fit_target_weights_3way(
    y_true: np.ndarray, y_c: np.ndarray, y_l: np.ndarray, y_x: np.ndarray,
    log: logging.Logger, target: str,
) -> tuple[float, float, float]:
    """Fit non-negative weights minimizing MSE( w_c·y_c + w_l·y_l + w_x·y_x , y_true ).
       Normalize to sum=1. Apply Chemprop floor + bias mitigations."""
    A = np.vstack([y_c, y_l, y_x]).T   # (n, 3)
    b = y_true
    x, _ = nnls(A, b)
    w_c_raw, w_l_raw, w_x_raw = float(x[0]), float(x[1]), float(x[2])

    s = w_c_raw + w_l_raw + w_x_raw
    if s < 1e-9:
        log.warning(f"[{target}] NNLS collapsed to zero; falling back to equal thirds")
        w_c_norm, w_l_norm, w_x_norm = 1/3, 1/3, 1/3
    else:
        w_c_norm = w_c_raw / s
        w_l_norm = w_l_raw / s
        w_x_norm = w_x_raw / s

    # Apply Chemprop bias — increase w_c, redistribute (l, x) proportionally
    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_bias = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        rem = 1.0 - w_c_bias
        lx_sum = w_l_norm + w_x_norm
        if lx_sum < 1e-9:
            w_l_bias, w_x_bias = rem / 2, rem / 2
        else:
            w_l_bias = rem * (w_l_norm / lx_sum)
            w_x_bias = rem * (w_x_norm / lx_sum)
    else:
        w_c_bias, w_l_bias, w_x_bias = w_c_norm, w_l_norm, w_x_norm

    # Enforce Chemprop weight floor
    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        w_c_final = CHEMPROP_WEIGHT_FLOOR
        rem = 1.0 - CHEMPROP_WEIGHT_FLOOR
        lx_sum = w_l_bias + w_x_bias
        if lx_sum < 1e-9:
            w_l_final, w_x_final = rem / 2, rem / 2
        else:
            w_l_final = rem * (w_l_bias / lx_sum)
            w_x_final = rem * (w_x_bias / lx_sum)
    else:
        w_c_final, w_l_final, w_x_final = w_c_bias, w_l_bias, w_x_bias

    log.info(f"  [{target}] NNLS raw:   w_c={w_c_raw:.4f}  w_l={w_l_raw:.4f}  w_x={w_x_raw:.4f}  (sum={s:.4f})")
    log.info(f"  [{target}] normalized: w_c={w_c_norm:.4f}  w_l={w_l_norm:.4f}  w_x={w_x_norm:.4f}")
    if APPLY_CHEMPROP_BIAS != 0.0:
        log.info(f"  [{target}] + bias {APPLY_CHEMPROP_BIAS:+.3f}: "
                 f"w_c={w_c_bias:.4f}  w_l={w_l_bias:.4f}  w_x={w_x_bias:.4f}")
    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        log.info(f"  [{target}] floor-clipped (Chemprop floor {CHEMPROP_WEIGHT_FLOOR}): "
                 f"w_c={w_c_final:.4f}  w_l={w_l_final:.4f}  w_x={w_x_final:.4f}")
    return w_c_final, w_l_final, w_x_final


def blend_all_targets(oof: pd.DataFrame, sub: pd.DataFrame, log: logging.Logger) -> dict:
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    log.info("=" * 60)
    log.info("PER-TARGET 3-WAY NNLS BLEND")
    log.info("=" * 60)
    log.info(f"CONFIG: CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   "
             f"APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(
            subset=["y_true", "y_pred_chemprop", "y_pred_lgb", "y_pred_cat"]
        )
        if len(g) < 10:
            log.warning(f"[{target}] only {len(g)} OOF rows — falling back to pure Chemprop")
            per_target[target] = {"n_oof": int(len(g)), "skipped": True}
            mask = sub["target_type"] == target
            sub.loc[mask, "target_blend"] = sub.loc[mask, "target_chemprop"]
            continue

        y_true = g["y_true"].values
        y_c = g["y_pred_chemprop"].values
        y_l = g["y_pred_lgb"].values
        y_x = g["y_pred_cat"].values

        log.info(f"[{target}] n_oof={len(g)}")
        r2_c = float(r2_score(y_true, y_c))
        r2_l = float(r2_score(y_true, y_l))
        r2_x = float(r2_score(y_true, y_x))
        log.info(f"  [{target}] individual OOF R²:  "
                 f"Chemprop={r2_c:.4f}   LGB+Max={r2_l:.4f}   CAT+Max={r2_x:.4f}")

        w_c, w_l, w_x = fit_target_weights_3way(y_true, y_c, y_l, y_x, log, target)
        y_blend = w_c * y_c + w_l * y_l + w_x * y_x
        r2_blend = float(r2_score(y_true, y_blend))

        best_solo = max(r2_c, r2_l, r2_x)
        best_solo_name = ["chemprop", "lgb", "cat"][int(np.argmax([r2_c, r2_l, r2_x]))]
        delta_vs_best = r2_blend - best_solo
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   "
                 f"Δ vs best solo ({best_solo_name}: {best_solo:.4f}) = {delta_vs_best:+.4f}")

        # Apply weights to submission
        mask = sub["target_type"] == target
        sub.loc[mask, "target_blend"] = (
            w_c * sub.loc[mask, "target_chemprop"]
            + w_l * sub.loc[mask, "target_lgb"]
            + w_x * sub.loc[mask, "target_cat"]
        )

        per_target[target] = {
            "n_oof":           int(len(g)),
            "r2_chemprop":     r2_c,
            "r2_lgb":          r2_l,
            "r2_cat":          r2_x,
            "r2_blend":        r2_blend,
            "w_chemprop":      w_c,
            "w_lgb":           w_l,
            "w_cat":           w_x,
            "best_solo":       best_solo_name,
            "delta_vs_best":   delta_vs_best,
        }

    # Blended OOF for later 4-way stacking
    oof = oof.copy()
    oof["y_pred_blend"] = np.nan
    for target in TARGETS:
        info = per_target[target]
        if info.get("skipped"):
            continue
        mask = oof["target_type"] == target
        oof.loc[mask, "y_pred_blend"] = (
            info["w_chemprop"] * oof.loc[mask, "y_pred_chemprop"]
            + info["w_lgb"] * oof.loc[mask, "y_pred_lgb"]
            + info["w_cat"] * oof.loc[mask, "y_pred_cat"]
        )

    return {"per_target": per_target, "sub": sub, "oof": oof}


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    check_inputs(log)

    t0 = time.time()
    oof, sub = load_and_align(log)
    result = blend_all_targets(oof, sub, log)
    per_target = result["per_target"]
    sub = result["sub"]
    oof_out = result["oof"]

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY: per-target OOF R² (individual vs 3-way blend)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'chemprop':>10s}  {'lgb+max':>10s}  {'cat+max':>10s}  "
             f"{'blend':>10s}  {'w_c':>6s}  {'w_l':>6s}  {'w_x':>6s}  {'best':>8s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_chemprop']:>10.4f}  {info['r2_lgb']:>10.4f}  "
            f"{info['r2_cat']:>10.4f}  {info['r2_blend']:>10.4f}  "
            f"{info['w_chemprop']:>6.3f}  {info['w_lgb']:>6.3f}  {info['w_cat']:>6.3f}  "
            f"{info['best_solo']:>8s}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_c = float(np.mean([per_target[t]["r2_chemprop"] for t in valid]))
    mean_l = float(np.mean([per_target[t]["r2_lgb"] for t in valid]))
    mean_x = float(np.mean([per_target[t]["r2_cat"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_c:>10.4f}  {mean_l:>10.4f}  {mean_x:>10.4f}  {mean_b:>10.4f}")
    log.info(f"  BLEND OOF R² lift vs Chemprop solo: {mean_b - mean_c:+.4f}")
    log.info(f"  BLEND OOF R² lift vs LGB solo    : {mean_b - mean_l:+.4f}")
    log.info(f"  BLEND OOF R² lift vs CAT solo    : {mean_b - mean_x:+.4f}")

    # 2-way blend reference (from prior best-ensemble)
    blend_2way_ref = 0.8828
    log.info(f"  2-way blend OOF R² (ref): {blend_2way_ref:.4f}")
    log.info(f"  3-way ΔOOF vs 2-way blend: {mean_b - blend_2way_ref:+.4f}")

    # Write outputs
    sub_out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    oof_path = EXP_DIR / "blended_oof.csv"
    oof_out[["canon", "target_type", "y_true",
             "y_pred_chemprop", "y_pred_lgb", "y_pred_cat", "y_pred_blend"]].to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_out)}")

    summary = {
        "exp_name":  EXP_NAME,
        "mean_r2_chemprop_oof": mean_c,
        "mean_r2_lgb_oof":      mean_l,
        "mean_r2_cat_oof":      mean_x,
        "mean_r2_blend_oof":    mean_b,
        "blend_lift_vs_chemprop":       mean_b - mean_c,
        "blend_lift_vs_lgb":            mean_b - mean_l,
        "blend_lift_vs_cat":            mean_b - mean_x,
        "blend_lift_vs_2way_blend_ref": mean_b - blend_2way_ref,
        "per_target": per_target,
        "config": {
            "chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
            "apply_chemprop_bias":   APPLY_CHEMPROP_BIAS,
            "chemprop_source":       str(CHEMPROP_DIR),
            "lgb_source":            str(LGB_DIR),
            "cat_source":            str(CAT_DIR),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(EXP_DIR / "blend_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'blend_summary.json'}")

    # Decision guidance
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"  Chemprop solo LB (ref):        0.887   rank 9")
    log.info(f"  LGB+Max solo LB (ref):         0.860")
    log.info(f"  CAT+Max solo LB (est):         ~0.860")
    log.info(f"  2-way (Chem+LGB) blend LB:     0.894   rank 5   (current best)")
    log.info(f"  3-way blend OOF R² mean:       {mean_b:.4f}  (2-way was {blend_2way_ref:.4f})")
    log.info("")
    log.info("Interpretation rules:")
    log.info("  - Blend OOF > 2-way OOF (0.883) by >0.003:  likely LB win. Submit.")
    log.info("  - Blend OOF ≈ 2-way OOF:  weak signal; expect LB within ±0.003 of 0.894.")
    log.info("  - Blend OOF < 2-way OOF:  DON'T submit — CAT is degrading the blend.")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
