"""
exp_blend_nnls_chainext_3way.py — 3-way NNLS blend:
  Chemprop 3-seed  +  LGB mono-only (Maxwell)  +  LGB chain-ext (Maxwell)

============================================================================
WHAT THIS IS
============================================================================

Same structure as `exp_blend_nnls_3way.py` (the prior 3-way, LB 0.895) but
with the third base swapped: instead of CatBoost, we use the chain-extended
LGB variant. This gives NNLS a per-target choice between the two LGB
variants: it can prefer chain-ext where it wins (eea, ei, egb, tg — 6 of 7
targets) and fall back to mono-only for `nc` (where chain-ext regressed
-0.013 in OOF).

Compare with `exp_blend_nnls_chainext.py` (the pure 2-way): the 2-way is
simpler but forces every target to use chain-ext LGB. If chain-ext's nc
regression drags down the blend's nc, the 3-way should recover it.

Expected LB: same 0.900–0.905 range as the 2-way, possibly +0.001 if nc
recovery from mono-LGB matters.

============================================================================
DEPENDENCIES (must exist BEFORE running)
============================================================================

Three prior experiments' outputs are required:

  1. results/exp_chemprop_multitask_cpu_3seed/     LB solo 0.892
  2. results/exp_maxwell_prior_lgbm/                LB solo 0.860
  3. results/exp_chain_ext_lgbm/                    LB solo 0.894

============================================================================
METHOD
============================================================================

Per target:
  1. Load OOF from all 3 sources. Align by (canon, target_type).
  2. NNLS on A = [y_chemprop_3seed, y_lgb_mono, y_lgb_chainext] vs y_true.
  3. Normalize weights to sum=1.
  4. Apply LB-bias mitigations (relaxed from prior 3-way — see below).
  5. Report per-target R² and blend R².
  6. Apply weights to test predictions.

============================================================================
OOF-LB BIAS CAVEAT — RELAXED
============================================================================

OOF-LB gaps for the 3 bases:
  Chemprop 3-seed  :  +0.022 (OOF 0.870 → LB 0.892)
  LGB mono-Maxwell :  -0.006 (OOF 0.866 → LB 0.860)  [aux-inflated, tiny miss]
  LGB chain-ext    :  +0.028 (OOF 0.866 → LB 0.894)  [surprising positive]

Chain-ext LGB actually UNDER-reports its true skill on OOF (opposite of prior
LGB behavior). So a strong Chemprop bias would over-penalize it.

New defaults vs prior 3-way (floor 0.40, bias +0.15):
  - CHEMPROP_WEIGHT_FLOOR = 0.30  (relaxed — still prevents 100% LGB blend)
  - APPLY_CHEMPROP_BIAS   = 0.00  (removed — chain-ext LGB is not OOF-inflated)

The blend logic: when Chemprop's floor is enforced, the remaining weight is
split between the two LGB variants proportional to their raw NNLS weights.

============================================================================
OUTPUTS  (under results/exp_blend_nnls_chainext_3way/)
============================================================================

  run.log              — full log with decision guidance
  blend_summary.json   — per-target weights, R²s, deltas, config
  submission.csv       — blended predictions (Kaggle format id, target)
  blended_oof.csv      — per-row blended OOFs

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_nnls_chainext_3way.py

Then submit results/exp_blend_nnls_chainext_3way/submission.csv to Kaggle.

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
EXP_NAME = "exp_blend_nnls_chainext_3way"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR   = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
LGB_MONO_DIR   = REPO / "results" / "exp_maxwell_prior_lgbm"
LGB_CHAIN_DIR  = REPO / "results" / "exp_chain_ext_lgbm"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

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
    for label, d in [("Chemprop 3-seed", CHEMPROP_DIR),
                     ("LGB mono-Maxwell", LGB_MONO_DIR),
                     ("LGB chain-ext",    LGB_CHAIN_DIR)]:
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
        log.error("  poly2-venv/bin/python experiments/exp_chain_ext_lgbm.py")
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  Chemprop 3-seed dir : {CHEMPROP_DIR}")
    log.info(f"  LGB mono-Max dir    : {LGB_MONO_DIR}")
    log.info(f"  LGB chain-ext dir   : {LGB_CHAIN_DIR}")


# ============================================================================
# LOAD + ALIGN
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading OOF from all three sources...")
    oof_c = pd.read_csv(CHEMPROP_DIR / "oof.csv")
    oof_m = pd.read_csv(LGB_MONO_DIR / "oof.csv")
    oof_x = pd.read_csv(LGB_CHAIN_DIR / "oof.csv")
    log.info(f"  Chemprop 3-seed OOF: {oof_c.shape}")
    log.info(f"  LGB mono-Max    OOF: {oof_m.shape}")
    log.info(f"  LGB chain-ext   OOF: {oof_x.shape}")

    oof = (
        oof_c.rename(columns={"y_pred": "y_pred_chemprop"})
        .merge(oof_m.rename(columns={"y_pred": "y_pred_lgb_mono"})[["canon", "target_type", "y_pred_lgb_mono"]],
               on=["canon", "target_type"], how="inner")
        .merge(oof_x.rename(columns={"y_pred": "y_pred_lgb_ext"})[["canon", "target_type", "y_pred_lgb_ext"]],
               on=["canon", "target_type"], how="inner")
    )
    log.info(f"  aligned OOF: {oof.shape}  (train rows with all 3 predictions)")

    for col in ["y_true", "y_pred_chemprop", "y_pred_lgb_mono", "y_pred_lgb_ext"]:
        n_nan = int(oof[col].isna().sum())
        if n_nan:
            log.warning(f"  {col} has {n_nan} NaN rows — dropped per-target")

    log.info("loading test submissions from all three sources...")
    sub_c = pd.read_csv(CHEMPROP_DIR / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_m = pd.read_csv(LGB_MONO_DIR / "submission.csv").rename(columns={"target": "target_lgb_mono"})
    sub_x = pd.read_csv(LGB_CHAIN_DIR / "submission.csv").rename(columns={"target": "target_lgb_ext"})

    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = (
        te.merge(sub_c, on="id", how="left")
          .merge(sub_m, on="id", how="left")
          .merge(sub_x, on="id", how="left")
    )
    log.info(f"  aligned sub: {sub.shape}")

    for col in ["target_chemprop", "target_lgb_mono", "target_lgb_ext"]:
        n_nan = int(sub[col].isna().sum())
        if n_nan:
            log.warning(f"  sub {col} has {n_nan} NaN rows")

    return oof, sub


# ============================================================================
# PER-TARGET 3-WAY NNLS
# ============================================================================

def fit_target_weights_3way(
    y_true: np.ndarray, y_c: np.ndarray, y_m: np.ndarray, y_x: np.ndarray,
    log: logging.Logger, target: str,
) -> tuple[float, float, float]:
    A = np.vstack([y_c, y_m, y_x]).T
    b = y_true
    x, _ = nnls(A, b)
    w_c_raw, w_m_raw, w_x_raw = float(x[0]), float(x[1]), float(x[2])

    s = w_c_raw + w_m_raw + w_x_raw
    if s < 1e-9:
        log.warning(f"[{target}] NNLS collapsed to zero; falling back to equal thirds")
        w_c_norm, w_m_norm, w_x_norm = 1/3, 1/3, 1/3
    else:
        w_c_norm = w_c_raw / s
        w_m_norm = w_m_raw / s
        w_x_norm = w_x_raw / s

    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_bias = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        rem = 1.0 - w_c_bias
        mx_sum = w_m_norm + w_x_norm
        if mx_sum < 1e-9:
            w_m_bias, w_x_bias = rem / 2, rem / 2
        else:
            w_m_bias = rem * (w_m_norm / mx_sum)
            w_x_bias = rem * (w_x_norm / mx_sum)
    else:
        w_c_bias, w_m_bias, w_x_bias = w_c_norm, w_m_norm, w_x_norm

    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        w_c_final = CHEMPROP_WEIGHT_FLOOR
        rem = 1.0 - CHEMPROP_WEIGHT_FLOOR
        mx_sum = w_m_bias + w_x_bias
        if mx_sum < 1e-9:
            w_m_final, w_x_final = rem / 2, rem / 2
        else:
            w_m_final = rem * (w_m_bias / mx_sum)
            w_x_final = rem * (w_x_bias / mx_sum)
    else:
        w_c_final, w_m_final, w_x_final = w_c_bias, w_m_bias, w_x_bias

    log.info(f"  [{target}] NNLS raw:   w_c={w_c_raw:.4f}  w_m={w_m_raw:.4f}  w_x={w_x_raw:.4f}  (sum={s:.4f})")
    log.info(f"  [{target}] normalized: w_c={w_c_norm:.4f}  w_m={w_m_norm:.4f}  w_x={w_x_norm:.4f}")
    if APPLY_CHEMPROP_BIAS != 0.0:
        log.info(f"  [{target}] + bias {APPLY_CHEMPROP_BIAS:+.3f}: "
                 f"w_c={w_c_bias:.4f}  w_m={w_m_bias:.4f}  w_x={w_x_bias:.4f}")
    if w_c_bias < CHEMPROP_WEIGHT_FLOOR:
        log.info(f"  [{target}] floor-clipped (Chemprop floor {CHEMPROP_WEIGHT_FLOOR}): "
                 f"w_c={w_c_final:.4f}  w_m={w_m_final:.4f}  w_x={w_x_final:.4f}")
    return w_c_final, w_m_final, w_x_final


def blend_all_targets(oof: pd.DataFrame, sub: pd.DataFrame, log: logging.Logger) -> dict:
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    log.info("=" * 60)
    log.info("PER-TARGET 3-WAY NNLS BLEND  (Chemprop 3-seed + LGB mono + LGB chain-ext)")
    log.info("=" * 60)
    log.info(f"CONFIG: CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}   "
             f"APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(
            subset=["y_true", "y_pred_chemprop", "y_pred_lgb_mono", "y_pred_lgb_ext"]
        )
        if len(g) < 10:
            log.warning(f"[{target}] only {len(g)} OOF rows — falling back to pure Chemprop")
            per_target[target] = {"n_oof": int(len(g)), "skipped": True}
            mask = sub["target_type"] == target
            sub.loc[mask, "target_blend"] = sub.loc[mask, "target_chemprop"]
            continue

        y_true = g["y_true"].values
        y_c = g["y_pred_chemprop"].values
        y_m = g["y_pred_lgb_mono"].values
        y_x = g["y_pred_lgb_ext"].values

        log.info(f"[{target}] n_oof={len(g)}")
        r2_c = float(r2_score(y_true, y_c))
        r2_m = float(r2_score(y_true, y_m))
        r2_x = float(r2_score(y_true, y_x))
        log.info(f"  [{target}] individual OOF R²:  "
                 f"Chemprop={r2_c:.4f}   LGB_mono={r2_m:.4f}   LGB_ext={r2_x:.4f}")

        w_c, w_m, w_x = fit_target_weights_3way(y_true, y_c, y_m, y_x, log, target)
        y_blend = w_c * y_c + w_m * y_m + w_x * y_x
        r2_blend = float(r2_score(y_true, y_blend))

        best_solo = max(r2_c, r2_m, r2_x)
        best_solo_name = ["chemprop", "lgb_mono", "lgb_ext"][int(np.argmax([r2_c, r2_m, r2_x]))]
        delta_vs_best = r2_blend - best_solo
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   "
                 f"Δ vs best solo ({best_solo_name}: {best_solo:.4f}) = {delta_vs_best:+.4f}")

        mask = sub["target_type"] == target
        sub.loc[mask, "target_blend"] = (
            w_c * sub.loc[mask, "target_chemprop"]
            + w_m * sub.loc[mask, "target_lgb_mono"]
            + w_x * sub.loc[mask, "target_lgb_ext"]
        )

        per_target[target] = {
            "n_oof":           int(len(g)),
            "r2_chemprop":     r2_c,
            "r2_lgb_mono":     r2_m,
            "r2_lgb_ext":      r2_x,
            "r2_blend":        r2_blend,
            "w_chemprop":      w_c,
            "w_lgb_mono":      w_m,
            "w_lgb_ext":       w_x,
            "best_solo":       best_solo_name,
            "delta_vs_best":   delta_vs_best,
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
            + info["w_lgb_mono"] * oof.loc[mask, "y_pred_lgb_mono"]
            + info["w_lgb_ext"]  * oof.loc[mask, "y_pred_lgb_ext"]
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
    log.info("SUMMARY: per-target OOF R² (individual vs 3-way blend)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'chemp':>10s}  {'lgb_mono':>10s}  {'lgb_ext':>10s}  "
             f"{'blend':>10s}  {'w_c':>6s}  {'w_m':>6s}  {'w_x':>6s}  {'best':>10s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_chemprop']:>10.4f}  {info['r2_lgb_mono']:>10.4f}  "
            f"{info['r2_lgb_ext']:>10.4f}  {info['r2_blend']:>10.4f}  "
            f"{info['w_chemprop']:>6.3f}  {info['w_lgb_mono']:>6.3f}  {info['w_lgb_ext']:>6.3f}  "
            f"{info['best_solo']:>10s}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_c = float(np.mean([per_target[t]["r2_chemprop"] for t in valid]))
    mean_m = float(np.mean([per_target[t]["r2_lgb_mono"] for t in valid]))
    mean_x = float(np.mean([per_target[t]["r2_lgb_ext"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_c:>10.4f}  {mean_m:>10.4f}  {mean_x:>10.4f}  {mean_b:>10.4f}")
    log.info(f"  BLEND OOF lift vs Chemprop solo : {mean_b - mean_c:+.4f}")
    log.info(f"  BLEND OOF lift vs LGB_mono solo : {mean_b - mean_m:+.4f}")
    log.info(f"  BLEND OOF lift vs LGB_ext solo  : {mean_b - mean_x:+.4f}")

    ref_prior_mean = 0.8873   # exp_blend_nnls_3seed OOF (LB 0.897)
    log.info(f"  Prior 2-way (mono-LGB) blend OOF: {ref_prior_mean:.4f}  (LB 0.897)")
    log.info(f"  3-way ΔOOF vs prior 2-way : {mean_b - ref_prior_mean:+.4f}")

    sub_out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    oof_path = EXP_DIR / "blended_oof.csv"
    oof_out[["canon", "target_type", "y_true",
             "y_pred_chemprop", "y_pred_lgb_mono", "y_pred_lgb_ext", "y_pred_blend"]] \
        .to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_out)}")

    summary = {
        "exp_name":  EXP_NAME,
        "mean_r2_chemprop_oof": mean_c,
        "mean_r2_lgb_mono_oof": mean_m,
        "mean_r2_lgb_ext_oof":  mean_x,
        "mean_r2_blend_oof":    mean_b,
        "blend_lift_vs_chemprop":       mean_b - mean_c,
        "blend_lift_vs_lgb_mono":       mean_b - mean_m,
        "blend_lift_vs_lgb_ext":        mean_b - mean_x,
        "blend_lift_vs_prior_2way":     mean_b - ref_prior_mean,
        "per_target": per_target,
        "config": {
            "chemprop_weight_floor": CHEMPROP_WEIGHT_FLOOR,
            "apply_chemprop_bias":   APPLY_CHEMPROP_BIAS,
            "chemprop_source":       str(CHEMPROP_DIR),
            "lgb_mono_source":       str(LGB_MONO_DIR),
            "lgb_chain_ext_source":  str(LGB_CHAIN_DIR),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(EXP_DIR / "blend_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'blend_summary.json'}")

    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"  Chemprop 3-seed solo LB (ref):    0.892")
    log.info(f"  LGB mono-Max solo LB (ref):       0.860")
    log.info(f"  LGB chain-ext solo LB (ref):      0.894")
    log.info(f"  Prior 2-way blend LB (mono-LGB):  0.897  (current best ensemble)")
    log.info(f"  This 3-way blend OOF R² mean:     {mean_b:.4f}")
    log.info("")
    log.info("Interpretation rules:")
    log.info("  - Blend OOF > pure 2-way chain-ext blend OOF:  submit the 3-way.")
    log.info("  - Blend OOF ≈ 2-way chain-ext blend OOF:  submit whichever is higher.")
    log.info("  - Watch nc column specifically: if 3-way's nc R² beats the 2-way's, the 3-way earns its keep.")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
