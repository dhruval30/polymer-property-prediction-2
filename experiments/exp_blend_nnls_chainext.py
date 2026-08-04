"""
exp_blend_nnls_chainext.py — 2-way NNLS blend of 3-seed Chemprop + chain-ext LGB.

============================================================================
WHAT THIS IS
============================================================================

Same structure as `exp_blend_nnls_3seed.py` (LB 0.897, current best ensemble)
but with the LGB base UPGRADED from mono-only Maxwell (`exp_maxwell_prior_lgbm`,
LB 0.860 solo) to the chain-extended trimer variant (`exp_chain_ext_lgbm`,
LB 0.894 solo).

The two bases (Chemprop 3-seed LB 0.892 + chain-ext LGB LB 0.894) are now
near-tied in solo LB skill — a total flip from the prior blend where LGB
was 0.032 LB behind Chemprop. NNLS should find a much more balanced weighting
per target than the prior 0.70/0.30 Chemprop-dominant split.

Expected LB: 0.900–0.905 (up from 0.897).

============================================================================
DEPENDENCIES (must exist BEFORE running)
============================================================================

Two prior experiments' outputs are required:

  1. results/exp_chemprop_multitask_cpu_3seed/
       - oof.csv, submission.csv
     Script: experiments/exp_chemprop_multitask_cpu_3seed.py
     LB solo: 0.892   (5-fold × 3-seed bag, wall time ~225 min)

  2. results/exp_chain_ext_lgbm/
       - oof.csv, submission.csv
     Script: experiments/exp_chain_ext_lgbm.py
     LB solo: 0.894   (mono + trimer features + Maxwell, wall time ~37 min)

============================================================================
METHOD (identical to exp_blend_nnls_3seed.py)
============================================================================

Per target:
  1. Load OOF from both. Align by (canon, target_type).
  2. NNLS on A = [y_chemprop_3seed, y_lgb_chainext] vs y_true → non-negative weights.
  3. Normalize to sum=1.
  4. Apply LB-bias mitigations (CHANGED — see below).
  5. Report per-target R² and blend R².
  6. Apply weights to test predictions.

============================================================================
OOF-LB BIAS CAVEAT — RELAXED FROM PRIOR BLENDS
============================================================================

Prior mono-LGB had OOF-LB gap of -0.006 (aux-augmented CV OVERSTATED true skill).
Chain-ext LGB has OOF-LB gap of +0.028 (OOF 0.866 → LB 0.894 — LB now UNDERSTATED!).

This is a total flip. Chain-ext features generalize to test-set polymers much
better than fold-CV suggests. The prior justification for CHEMPROP_WEIGHT_FLOOR=0.40
and APPLY_CHEMPROP_BIAS=+0.15 (LGB OOF-inflated needs penalty) no longer holds.

Chemprop 3-seed OOF-LB gap: +0.022 (also understated).
Chain-ext LGB OOF-LB gap: +0.028 (slightly MORE understated than Chemprop).

**Both bases under-report on OOF. Applying a Chemprop bias would OVER-correct.**

New defaults:
  - CHEMPROP_WEIGHT_FLOOR = 0.30  (relaxed from 0.40 — still prevents 100% LGB collapse)
  - APPLY_CHEMPROP_BIAS   = 0.00  (removed — no evidence LGB is over-weighted anymore)

If pure NNLS still over-weights one base and blend LB is < 0.897, revisit and
try floor 0.40 / bias +0.05.

============================================================================
OUTPUTS  (under results/exp_blend_nnls_chainext/)
============================================================================

  run.log             — full log with decision guidance
  blend_summary.json  — per-target weights, R²s, deltas, config
  submission.csv      — blended predictions (Kaggle format id, target)
  blended_oof.csv     — per-row blended OOFs (for future stacking)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_nnls_chainext.py

Then submit results/exp_blend_nnls_chainext/submission.csv to Kaggle.

============================================================================
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
EXP_NAME = "exp_blend_nnls_chainext"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
LGB_DIR      = REPO / "results" / "exp_chain_ext_lgbm"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

# LB-bias mitigations — RELAXED because chain-ext LGB LB > OOF (was < before)
CHEMPROP_WEIGHT_FLOOR = 0.30
APPLY_CHEMPROP_BIAS   = 0.00


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
    for label, d in [("Chemprop 3-seed", CHEMPROP_DIR), ("LGB chain-ext", LGB_DIR)]:
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
        log.error("  poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu_3seed.py")
        log.error("  poly2-venv/bin/python experiments/exp_chain_ext_lgbm.py")
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  Chemprop 3-seed dir : {CHEMPROP_DIR}")
    log.info(f"  LGB chain-ext dir   : {LGB_DIR}")


# ============================================================================
# LOAD + ALIGN
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading OOF from both sources...")
    oof_c = pd.read_csv(CHEMPROP_DIR / "oof.csv")
    oof_l = pd.read_csv(LGB_DIR / "oof.csv")
    log.info(f"  Chemprop 3-seed OOF: {oof_c.shape}")
    log.info(f"  LGB chain-ext   OOF: {oof_l.shape}")

    oof = (
        oof_c.rename(columns={"y_pred": "y_pred_chemprop"})
        .merge(oof_l.rename(columns={"y_pred": "y_pred_lgb"})[["canon", "target_type", "y_pred_lgb"]],
               on=["canon", "target_type"], how="inner")
    )
    log.info(f"  aligned OOF: {oof.shape}  (train rows with both predictions)")

    for col in ["y_true", "y_pred_chemprop", "y_pred_lgb"]:
        n_nan = int(oof[col].isna().sum())
        if n_nan:
            log.warning(f"  {col} has {n_nan} NaN rows — dropped per-target")

    log.info("loading test submissions...")
    sub_c = pd.read_csv(CHEMPROP_DIR / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_l = pd.read_csv(LGB_DIR / "submission.csv").rename(columns={"target": "target_lgb"})
    log.info(f"  Chemprop 3-seed sub: {sub_c.shape}")
    log.info(f"  LGB chain-ext   sub: {sub_l.shape}")

    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_c, on="id", how="left").merge(sub_l, on="id", how="left")
    log.info(f"  aligned sub: {sub.shape}")

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
    A = np.vstack([y_c, y_l]).T
    b = y_true
    x, _ = nnls(A, b)
    w_c_raw, w_l_raw = float(x[0]), float(x[1])

    s = w_c_raw + w_l_raw
    if s < 1e-9:
        log.warning(f"[{target}] NNLS collapsed to zero; falling back to 50/50")
        w_c_norm, w_l_norm = 0.5, 0.5
    else:
        w_c_norm, w_l_norm = w_c_raw / s, w_l_raw / s

    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_bias = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        w_l_bias = max(0.0, 1.0 - w_c_bias)
    else:
        w_c_bias, w_l_bias = w_c_norm, w_l_norm

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
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    log.info("=" * 60)
    log.info("PER-TARGET NNLS BLEND  (Chemprop 3-seed + LGB chain-ext)")
    log.info("=" * 60)
    log.info(f"CONFIG: CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   "
             f"APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    # Reference: 2-way with mono LGB (exp_blend_nnls_3seed, LB 0.897)
    ref_prior = {"eea": 0.9125, "egb": 0.9351, "egc": 0.9170, "ei": 0.8073,
                 "eps": 0.8362, "nc": 0.8858, "tg": 0.9170}
    ref_prior_w_chem = {"eea": 0.88, "egb": 0.83, "egc": 0.71, "ei": 0.56,
                        "eps": 0.54, "nc": 0.69, "tg": 0.68}

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(subset=["y_true", "y_pred_chemprop", "y_pred_lgb"])
        if len(g) < 10:
            log.warning(f"[{target}] only {len(g)} OOF rows — falling back to Chemprop")
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
        log.info(f"  [{target}] individual OOF R²:  Chemprop_3seed={r2_c:.4f}   LGB_chainext={r2_l:.4f}")

        w_c, w_l = fit_target_weights(y_true, y_c, y_l, log, target)
        y_blend = w_c * y_c + w_l * y_l
        r2_blend = float(r2_score(y_true, y_blend))

        better_solo = max(r2_c, r2_l)
        delta_vs_better = r2_blend - better_solo
        ref_r2 = ref_prior[target]
        delta_vs_ref = r2_blend - ref_r2
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   "
                 f"Δ vs better solo = {delta_vs_better:+.4f}   "
                 f"Δ vs prior 2-way (mono-LGB) blend = {delta_vs_ref:+.4f}")

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
            "delta_vs_prior":  delta_vs_ref,
            "better_solo":     "chemprop" if r2_c >= r2_l else "lgb",
            "prev_2way_w_chem": ref_prior_w_chem[target],
        }

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

    log.info("=" * 60)
    log.info("SUMMARY: per-target OOF R² (individual vs blend)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'chemp_3s':>10s}  {'lgb_ext':>10s}  {'blend':>10s}  "
             f"{'w_c':>6s}  {'w_l':>6s}  {'w_c(prev)':>10s}  {'prior_ref':>10s}  {'Δ_ref':>8s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_chemprop']:>10.4f}  {info['r2_lgb']:>10.4f}  "
            f"{info['r2_blend']:>10.4f}  {info['w_chemprop']:>6.3f}  "
            f"{info['w_lgb']:>6.3f}  {info['prev_2way_w_chem']:>10.3f}  "
            f"{info['r2_blend'] - info['delta_vs_prior']:>10.4f}  "
            f"{info['delta_vs_prior']:>+8.4f}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_c = float(np.mean([per_target[t]["r2_chemprop"] for t in valid]))
    mean_l = float(np.mean([per_target[t]["r2_lgb"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_c:>10.4f}  {mean_l:>10.4f}  {mean_b:>10.4f}")

    ref_prior_mean = 0.8873   # exp_blend_nnls_3seed OOF (LB 0.897)
    log.info(f"  BLEND OOF vs Chemprop 3-seed solo: {mean_b - mean_c:+.4f}")
    log.info(f"  BLEND OOF vs LGB chain-ext solo  : {mean_b - mean_l:+.4f}")
    log.info(f"  BLEND OOF vs prior 2-way blend (0.8873, LB 0.897): {mean_b - ref_prior_mean:+.4f}")

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
        "mean_r2_chemprop_3seed_oof": mean_c,
        "mean_r2_lgb_chainext_oof":   mean_l,
        "mean_r2_blend_oof":          mean_b,
        "blend_lift_vs_chemprop_3seed":  mean_b - mean_c,
        "blend_lift_vs_lgb_chainext":    mean_b - mean_l,
        "blend_lift_vs_prior_2way":      mean_b - ref_prior_mean,
        "per_target": per_target,
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

    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Chemprop 3-seed solo LB (ref):    0.892")
    log.info(f"LGB chain-ext solo LB (ref):      0.894")
    log.info(f"Prior 2-way blend LB (mono-LGB):  0.897  (current best ensemble)")
    log.info(f"THIS blend OOF R² mean: {mean_b:.4f}  (prior 2-way was 0.8873)")
    log.info("")
    log.info("Interpretation rules:")
    log.info(f"  - Blend OOF > 0.8920:  strong signal, LB likely >= 0.900. SUBMIT.")
    log.info(f"  - Blend OOF 0.8890-0.8920:  moderate, LB likely 0.898-0.902. SUBMIT.")
    log.info(f"  - Blend OOF 0.8873-0.8890:  weak, LB likely 0.897-0.900. Marginal.")
    log.info(f"  - Blend OOF < 0.8873:  chain-ext hurt the blend — investigate weights.")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
