"""
exp_blend_nnls_3seed.py — 2-way NNLS blend of 3-seed Chemprop + LGB(Maxwell).

============================================================================
WHAT THIS IS
============================================================================

Same structure as `exp_blend_nnls.py` (the original 2-way blend, LB 0.894)
but with the Chemprop base UPGRADED from single-seed to the 5-fold × 3-seed
bagged version (`exp_chemprop_multitask_cpu_3seed`, LB 0.892 solo — up from
0.887 for single-seed).

The 3-seed Chemprop's OOF is ~+0.015 mean R² above single-seed. If that OOF
gain propagates through the NNLS blend the same way it did before (LB gain
was ~2/3 the OOF gain last time), we should see a blend LB in the
**0.897–0.900 range** (up from 0.894 for the single-seed 2-way blend).

============================================================================
DEPENDENCIES (must exist BEFORE running)
============================================================================

Two prior experiments' outputs are required:

  1. results/exp_chemprop_multitask_cpu_3seed/
       - oof.csv, submission.csv
     Script: experiments/exp_chemprop_multitask_cpu_3seed.py
     LB solo: 0.892   (5-fold × 3-seed bag, wall time ~225 min)

  2. results/exp_maxwell_prior_lgbm/
       - oof.csv, submission.csv
     Script: experiments/exp_maxwell_prior_lgbm.py
     LB solo: 0.860

============================================================================
METHOD (identical to exp_blend_nnls.py)
============================================================================

Per target:
  1. Load OOF from both. Align by (canon, target_type).
  2. NNLS on A = [y_chemprop_3seed, y_lgb] vs y_true → non-negative weights.
  3. Normalize to sum=1.
  4. Apply LB-bias mitigations: CHEMPROP_WEIGHT_FLOOR=0.40, APPLY_CHEMPROP_BIAS=+0.15.
  5. Report per-target R² and blend R².
  6. Apply weights to test predictions.

Then write a single blended submission covering all 7 targets.

============================================================================
OOF-LB BIAS CAVEAT (refined for 3-seed)
============================================================================

Single-seed Chemprop had OOF-LB gap of +0.031 (OOF 0.856 → LB 0.887).
3-seed Chemprop has OOF-LB gap of +0.022 (OOF 0.870 → LB 0.892) — the
gap is smaller because bagging already captures some of the refit-boost
variance that single-seed's OOF was missing. So the 3-seed's OOF is more
"honest" than single-seed's was.

LGB+Maxwell OOF-LB gap: -0.006 (aux-augmented CV OVERSTATES LB skill).

The Chemprop-bias mitigations (floor + bias) are still needed because
LGB still over-reports relative to Chemprop. Keep the same values as the
single-seed 2-way blend:
  - CHEMPROP_WEIGHT_FLOOR = 0.40
  - APPLY_CHEMPROP_BIAS   = 0.15

If the blend OOF comes back with excessive LGB weight (e.g., NNLS wants
0.7+ LGB on some targets), consider raising the floor to 0.5.

============================================================================
OUTPUTS  (under results/exp_blend_nnls_3seed/)
============================================================================

  run.log             — full log with decision guidance
  blend_summary.json  — per-target weights, R²s, deltas, config
  submission.csv      — blended predictions (Kaggle format id, target)
  blended_oof.csv     — per-row blended OOFs (for future stacking)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_nnls_3seed.py

Then submit results/exp_blend_nnls_3seed/submission.csv to Kaggle.

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
EXP_NAME = "exp_blend_nnls_3seed"
EXP_DIR = REPO / "results" / EXP_NAME

# Source directories — 3-seed Chemprop replaces single-seed
CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
LGB_DIR      = REPO / "results" / "exp_maxwell_prior_lgbm"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

# LB-bias mitigations (same as single-seed 2-way blend)
CHEMPROP_WEIGHT_FLOOR = 0.40
APPLY_CHEMPROP_BIAS   = 0.15


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
    for label, d in [("Chemprop 3-seed", CHEMPROP_DIR), ("LGB+Maxwell", LGB_DIR)]:
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
        log.error("  poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py")
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  Chemprop 3-seed dir: {CHEMPROP_DIR}")
    log.info(f"  LGB+Max dir        : {LGB_DIR}")


# ============================================================================
# LOAD + ALIGN
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading OOF from both sources...")
    oof_c = pd.read_csv(CHEMPROP_DIR / "oof.csv")
    oof_l = pd.read_csv(LGB_DIR / "oof.csv")
    log.info(f"  Chemprop 3-seed OOF: {oof_c.shape}")
    log.info(f"  LGB+Max         OOF: {oof_l.shape}")

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
    log.info(f"  LGB+Max         sub: {sub_l.shape}")

    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_c, on="id", how="left").merge(sub_l, on="id", how="left")
    log.info(f"  aligned sub: {sub.shape}")

    for col in ["target_chemprop", "target_lgb"]:
        n_nan = int(sub[col].isna().sum())
        if n_nan:
            log.warning(f"  sub {col} has {n_nan} NaN rows")

    return oof, sub


# ============================================================================
# PER-TARGET NNLS BLEND (identical logic to exp_blend_nnls.py)
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
    log.info("PER-TARGET NNLS BLEND  (Chemprop 3-seed + LGB+Max)")
    log.info("=" * 60)
    log.info(f"CONFIG: CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   "
             f"APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    # Single-seed 2-way reference (from exp_blend_nnls, LB 0.894)
    ref_2way = {"eea": 0.9010, "egb": 0.9325, "egc": 0.9115, "ei": 0.8092,
                "eps": 0.8295, "nc": 0.8834, "tg": 0.9126}
    ref_2way_w_chem = {"eea": 0.75, "egb": 0.77, "egc": 0.55, "ei": 0.58,
                       "eps": 0.45, "nc": 0.65, "tg": 0.54}

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
        log.info(f"  [{target}] individual OOF R²:  Chemprop_3seed={r2_c:.4f}   LGB+Max={r2_l:.4f}")

        w_c, w_l = fit_target_weights(y_true, y_c, y_l, log, target)
        y_blend = w_c * y_c + w_l * y_l
        r2_blend = float(r2_score(y_true, y_blend))

        better_solo = max(r2_c, r2_l)
        delta_vs_better = r2_blend - better_solo
        ref_r2 = ref_2way[target]
        delta_vs_ref = r2_blend - ref_r2
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   "
                 f"Δ vs better solo = {delta_vs_better:+.4f}   "
                 f"Δ vs single-seed 2-way blend ref = {delta_vs_ref:+.4f}")

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
            "delta_vs_2way_ref": delta_vs_ref,
            "better_solo":     "chemprop" if r2_c >= r2_l else "lgb",
            "prev_2way_w_chem": ref_2way_w_chem[target],
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

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY: per-target OOF R² (individual vs blend)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'chemp_3s':>10s}  {'lgb+max':>10s}  {'blend':>10s}  "
             f"{'w_c':>6s}  {'w_l':>6s}  {'w_c(prev)':>10s}  {'2way_ref':>10s}  {'Δ_ref':>8s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_chemprop']:>10.4f}  {info['r2_lgb']:>10.4f}  "
            f"{info['r2_blend']:>10.4f}  {info['w_chemprop']:>6.3f}  "
            f"{info['w_lgb']:>6.3f}  {info['prev_2way_w_chem']:>10.3f}  "
            f"{info['r2_blend'] - info['delta_vs_2way_ref']:>10.4f}  "
            f"{info['delta_vs_2way_ref']:>+8.4f}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_c = float(np.mean([per_target[t]["r2_chemprop"] for t in valid]))
    mean_l = float(np.mean([per_target[t]["r2_lgb"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_c:>10.4f}  {mean_l:>10.4f}  {mean_b:>10.4f}")

    # Reference means
    ref_2way_mean = 0.8828   # single-seed 2-way blend OOF
    log.info(f"  BLEND OOF vs Chemprop 3-seed solo: {mean_b - mean_c:+.4f}")
    log.info(f"  BLEND OOF vs LGB+Max solo         : {mean_b - mean_l:+.4f}")
    log.info(f"  BLEND OOF vs single-seed 2-way blend (0.8828): {mean_b - ref_2way_mean:+.4f}")

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
        "mean_r2_chemprop_3seed_oof": mean_c,
        "mean_r2_lgb_oof":            mean_l,
        "mean_r2_blend_oof":          mean_b,
        "blend_lift_vs_chemprop_3seed":     mean_b - mean_c,
        "blend_lift_vs_lgb":                mean_b - mean_l,
        "blend_lift_vs_single_seed_2way":   mean_b - ref_2way_mean,
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

    # Decision guidance
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Chemprop 3-seed solo LB (ref): 0.892")
    log.info(f"LGB+Max solo LB (ref):         0.860")
    log.info(f"Single-seed 2-way blend LB (ref): 0.894  (rank 5)")
    log.info(f"3-way blend LB (current best): 0.895  (rank ~4)")
    log.info(f"THIS blend OOF R² mean: {mean_b:.4f}  (single-seed 2-way was 0.8828)")
    log.info("")
    log.info("Interpretation rules:")
    log.info(f"  - Blend OOF > 0.8870:  strong signal, likely LB > 0.895. SUBMIT.")
    log.info(f"  - Blend OOF 0.8850–0.8870:  moderate, likely LB ~0.895-0.897. SUBMIT.")
    log.info(f"  - Blend OOF 0.8828–0.8850:  weak, LB likely 0.894–0.896. Marginal — submit if slot avail.")
    log.info(f"  - Blend OOF < 0.8828:  3-seed didn't help — investigate weights.")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
