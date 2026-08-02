"""
exp_blend_nnls.py — per-target NNLS blend of Chemprop + LGB(Maxwell) OOFs and test predictions.

============================================================================
DEPENDENCIES (must exist BEFORE running this script)
============================================================================

This script reads OOF predictions and test submissions from two prior experiments
and blends them per target. You must have run BOTH of these first:

  1. results/exp_chemprop_multitask_cpu/
       - oof.csv           (canon, target_type, y_true, y_pred)
       - submission.csv    (id, target)
     Produced by: experiments/exp_chemprop_multitask_cpu.py
     LB reference: 0.887 (rank 9)

  2. results/exp_maxwell_prior_lgbm/
       - oof.csv           (canon, target_type, y_true, y_pred)
       - submission.csv    (id, target)
     Produced by: experiments/exp_maxwell_prior_lgbm.py
     LB reference: 0.860

If either directory is missing, the script exits with a clear error.

============================================================================
METHOD
============================================================================

For each of the 7 target_types independently:
  1. Load OOF rows for that target from both experiments.
  2. Align by canonical SMILES so y_true, y_pred_chemprop, y_pred_lgb match up.
  3. Run scipy.optimize.nnls to find non-negative weights (w_c, w_l) minimizing
     MSE( w_c · y_c + w_l · y_l , y_true ).
  4. Normalize weights to sum=1 (defensive — NNLS may not produce sum=1 automatically).
  5. Compute blend OOF R² and compare vs pure-Chemprop OOF R² and pure-LGB OOF R².
  6. Apply the same weights to test predictions (aligned by id / canon).

Then write a single blended submission covering all 7 targets.

The script prints per-target diagnostics: individual R²s, blend weights, blend R²,
delta vs the better individual model. Use these to decide whether to submit.

============================================================================
IMPORTANT CAVEAT (read before submitting)
============================================================================

LGB OOFs from `exp_maxwell_prior_lgbm` used aux-augmented CV that INFLATES
OOF R² by ~0.006 relative to true LB skill. Chemprop OOFs are honest.
Historical LB−OOF gaps:
   Chemprop:  LB 0.887 − OOF 0.856 = +0.031  (OOF UNDER-estimates)
   LGB+Max:   LB 0.860 − OOF 0.866 = -0.006  (OOF OVER-estimates)

Therefore per-target NNLS on OOF will SYSTEMATICALLY OVER-WEIGHT LGB
compared to what's optimal on LB. Interpret the printed blend R² as an
upper bound on LGB's true contribution.

Two mitigations exist in this script (configurable at the top):
  - CHEMPROP_WEIGHT_FLOOR:  minimum weight for Chemprop per target (default 0.4).
    Prevents blend from fully abandoning Chemprop even if OOF says so.
  - APPLY_CHEMPROP_BIAS:  shift NNLS weights toward Chemprop by a fixed offset
    to reflect its proven LB advantage (default +0.15).

Set both to 0.0 for pure OOF-optimal NNLS.

============================================================================
OUTPUTS  (under results/exp_blend_nnls/)
============================================================================

  run.log             — full log
  blend_summary.json  — per-target weights, R²s, deltas, config
  submission.csv      — blended predictions (Kaggle format id, target)
  blended_oof.csv     — per-row blended OOF predictions (for later stacking)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_nnls.py

Then submit results/exp_blend_nnls/submission.csv to Kaggle.
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
EXP_NAME = "exp_blend_nnls"
EXP_DIR = REPO / "results" / EXP_NAME

# Source experiment directories
CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu"
LGB_DIR      = REPO / "results" / "exp_maxwell_prior_lgbm"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

# Mitigations for the OOF-vs-LB bias discussed in the module docstring.
# Set both to 0.0 for pure OOF-optimal NNLS with no LB-based prior.
CHEMPROP_WEIGHT_FLOOR = 0.40   # each target's Chemprop weight ≥ this
APPLY_CHEMPROP_BIAS   = 0.15   # after NNLS, add this to w_chemprop, then renormalize


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
    for label, d in [("Chemprop", CHEMPROP_DIR), ("LGB+Maxwell", LGB_DIR)]:
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
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  Chemprop dir: {CHEMPROP_DIR}")
    log.info(f"  LGB+Max dir : {LGB_DIR}")


# ============================================================================
# LOAD + ALIGN OOF / SUBMISSION FROM BOTH SOURCES
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (oof_wide, sub_wide) with columns:
       oof_wide:  canon, target_type, y_true, y_pred_chemprop, y_pred_lgb
       sub_wide:  id, target_type, target_chemprop, target_lgb
    """
    log.info("loading OOF from both sources...")
    oof_c = pd.read_csv(CHEMPROP_DIR / "oof.csv")
    oof_l = pd.read_csv(LGB_DIR / "oof.csv")
    log.info(f"  Chemprop OOF: {oof_c.shape}  columns={list(oof_c.columns)}")
    log.info(f"  LGB+Max  OOF: {oof_l.shape}  columns={list(oof_l.columns)}")

    # OOF alignment
    oof = oof_c.rename(columns={"y_pred": "y_pred_chemprop"}).merge(
        oof_l.rename(columns={"y_pred": "y_pred_lgb"})[["canon", "target_type", "y_pred_lgb"]],
        on=["canon", "target_type"], how="inner",
    )
    log.info(f"  aligned OOF: {oof.shape} (train rows with predictions from both sources)")

    # Sanity: NaN check
    for col in ["y_true", "y_pred_chemprop", "y_pred_lgb"]:
        n_nan = int(oof[col].isna().sum())
        if n_nan:
            log.warning(f"  {col} has {n_nan} NaN rows — will be dropped per-target")

    # Submission alignment
    log.info("loading test submissions from both sources...")
    sub_c = pd.read_csv(CHEMPROP_DIR / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_l = pd.read_csv(LGB_DIR / "submission.csv").rename(columns={"target": "target_lgb"})
    log.info(f"  Chemprop sub: {sub_c.shape}")
    log.info(f"  LGB+Max  sub: {sub_l.shape}")

    # Need target_type for the sub too — reload test.csv to attach it
    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_c, on="id", how="left").merge(sub_l, on="id", how="left")
    log.info(f"  aligned sub: {sub.shape}")

    # Sanity: NaN check on sub
    for col in ["target_chemprop", "target_lgb"]:
        n_nan = int(sub[col].isna().sum())
        if n_nan:
            log.warning(f"  sub {col} has {n_nan} NaN rows")

    return oof, sub


# ============================================================================
# PER-TARGET NNLS BLEND
# ============================================================================

def fit_target_weights(y_true: np.ndarray, y_c: np.ndarray, y_l: np.ndarray,
                        log: logging.Logger, target: str) -> tuple[float, float]:
    """Fit non-negative weights minimizing MSE( w_c·y_c + w_l·y_l , y_true ).
       Normalize to sum=1. Apply configured floor + bias."""
    A = np.vstack([y_c, y_l]).T   # (n, 2)
    b = y_true                    # (n,)
    x, _ = nnls(A, b)             # x = [w_c_raw, w_l_raw], both >= 0
    w_c_raw, w_l_raw = float(x[0]), float(x[1])

    # Normalize to sum=1 (defensive — NNLS may not enforce this)
    s = w_c_raw + w_l_raw
    if s < 1e-9:
        # Degenerate: NNLS collapsed to zero (rare). Fall back to 50/50.
        log.warning(f"[{target}] NNLS collapsed to zero weights; falling back to 50/50")
        w_c_norm, w_l_norm = 0.5, 0.5
    else:
        w_c_norm, w_l_norm = w_c_raw / s, w_l_raw / s

    # Apply Chemprop-bias shift (mitigation for LGB OOF inflation)
    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_bias = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        w_l_bias = max(0.0, 1.0 - w_c_bias)
    else:
        w_c_bias, w_l_bias = w_c_norm, w_l_norm

    # Enforce Chemprop weight floor
    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        w_c_final = CHEMPROP_WEIGHT_FLOOR
        w_l_final = 1.0 - CHEMPROP_WEIGHT_FLOOR
    else:
        w_c_final, w_l_final = w_c_bias, w_l_bias

    log.info(f"  [{target}] NNLS raw:   w_c={w_c_raw:.4f}  w_l={w_l_raw:.4f}  (sum={s:.4f})")
    log.info(f"  [{target}] normalized: w_c={w_c_norm:.4f}  w_l={w_l_norm:.4f}")
    if APPLY_CHEMPROP_BIAS != 0.0:
        log.info(f"  [{target}] + bias {APPLY_CHEMPROP_BIAS:+.3f}: w_c={w_c_bias:.4f}  w_l={w_l_bias:.4f}")
    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        log.info(f"  [{target}] floor-clipped (Chemprop floor {CHEMPROP_WEIGHT_FLOOR}): "
                 f"w_c={w_c_final:.4f}  w_l={w_l_final:.4f}")
    return w_c_final, w_l_final


def blend_all_targets(oof: pd.DataFrame, sub: pd.DataFrame, log: logging.Logger) -> dict:
    """Run per-target NNLS + apply to submission. Returns per-target diagnostics
       and the augmented sub DataFrame with a `target_blend` column."""
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    log.info("=" * 60)
    log.info("PER-TARGET NNLS BLEND")
    log.info("=" * 60)
    log.info(f"CONFIG: CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   "
             f"APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(subset=["y_true", "y_pred_chemprop", "y_pred_lgb"])
        if len(g) < 10:
            log.warning(f"[{target}] only {len(g)} OOF rows — skipping blend (using Chemprop directly)")
            per_target[target] = {"n_oof": int(len(g)), "skipped": True}
            mask = sub["target_type"] == target
            sub.loc[mask, "target_blend"] = sub.loc[mask, "target_chemprop"]
            continue

        y_true = g["y_true"].values
        y_c    = g["y_pred_chemprop"].values
        y_l    = g["y_pred_lgb"].values

        log.info(f"[{target}] n_oof={len(g)}")
        r2_c = float(r2_score(y_true, y_c))
        r2_l = float(r2_score(y_true, y_l))
        log.info(f"  [{target}] individual OOF R²:  Chemprop={r2_c:.4f}   LGB+Max={r2_l:.4f}")

        w_c, w_l = fit_target_weights(y_true, y_c, y_l, log, target)
        y_blend = w_c * y_c + w_l * y_l
        r2_blend = float(r2_score(y_true, y_blend))

        better_solo = max(r2_c, r2_l)
        delta_vs_better = r2_blend - better_solo
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   "
                 f"Δ vs better solo = {delta_vs_better:+.4f}   "
                 f"(better solo was {'Chemprop' if r2_c >= r2_l else 'LGB'})")

        # Apply weights to submission
        mask = sub["target_type"] == target
        sub.loc[mask, "target_blend"] = (
            w_c * sub.loc[mask, "target_chemprop"] + w_l * sub.loc[mask, "target_lgb"]
        )

        per_target[target] = {
            "n_oof":           int(len(g)),
            "r2_chemprop":     r2_c,
            "r2_lgb":          r2_l,
            "r2_blend":        r2_blend,
            "w_chemprop":      w_c,
            "w_lgb":           w_l,
            "delta_vs_better": delta_vs_better,
            "better_solo":     "chemprop" if r2_c >= r2_l else "lgb",
        }

    # Also compute + write blended OOF back
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

    # Per-target and mean R² summaries
    log.info("=" * 60)
    log.info("SUMMARY: per-target OOF R²  (individual vs blend)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'chemprop':>10s}  {'lgb+max':>10s}  {'blend':>10s}  "
             f"{'w_chem':>7s}  {'w_lgb':>7s}  {'better':>7s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_chemprop']:>10.4f}  {info['r2_lgb']:>10.4f}  "
            f"{info['r2_blend']:>10.4f}  {info['w_chemprop']:>7.3f}  "
            f"{info['w_lgb']:>7.3f}  {info['better_solo']:>7s}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_c = float(np.mean([per_target[t]["r2_chemprop"] for t in valid]))
    mean_l = float(np.mean([per_target[t]["r2_lgb"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_c:>10.4f}  {mean_l:>10.4f}  {mean_b:>10.4f}")
    log.info(f"  BLEND OOF R² lift vs pure Chemprop: {mean_b - mean_c:+.4f}")
    log.info(f"  BLEND OOF R² lift vs pure LGB+Max : {mean_b - mean_l:+.4f}")

    # Write outputs
    sub_out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    oof_path = EXP_DIR / "blended_oof.csv"
    oof_out[["canon", "target_type", "y_true", "y_pred_chemprop", "y_pred_lgb", "y_pred_blend"]] \
        .to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_out)}")

    summary = {
        "exp_name":  EXP_NAME,
        "mean_r2_chemprop_oof": mean_c,
        "mean_r2_lgb_oof":      mean_l,
        "mean_r2_blend_oof":    mean_b,
        "blend_lift_vs_chemprop": mean_b - mean_c,
        "blend_lift_vs_lgb":      mean_b - mean_l,
        "per_target":  per_target,
        "config": {
            "chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
            "apply_chemprop_bias":   APPLY_CHEMPROP_BIAS,
            "chemprop_source":       str(CHEMPROP_DIR),
            "lgb_source":            str(LGB_DIR),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    summary_path = EXP_DIR / "blend_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {summary_path}")

    # Decision guidance
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Chemprop-only LB (reference): 0.887   rank 9")
    log.info(f"LGB+Max-only LB (reference):  0.860")
    log.info(f"Blend OOF R² mean: {mean_b:.4f}  (Chemprop OOF was {mean_c:.4f})")
    log.info("")
    log.info("Rule of thumb:")
    log.info("  - If blend OOF beats pure Chemprop by >0.005:  likely LB win. Submit.")
    log.info("  - If blend OOF is ~equal to pure Chemprop:  weak signal, probably safe to submit")
    log.info("    but expect LB within ±0.003 of 0.887.")
    log.info("  - If blend OOF is worse than pure Chemprop:  do NOT submit, revisit config")
    log.info("    (raise CHEMPROP_WEIGHT_FLOOR or APPLY_CHEMPROP_BIAS at top of script).")
    log.info("")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
