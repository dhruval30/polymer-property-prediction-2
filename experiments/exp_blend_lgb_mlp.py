"""
exp_blend_lgb_mlp.py — per-target NNLS blend of chain-ext LGB v1 + chain-ext MLP.

============================================================================
WHAT THIS IS
============================================================================

2-way blend of two chain-ext bases with STRUCTURALLY DIFFERENT model families:
  1. chain-ext LGB v1  (LB 0.894)  — axis-aligned tree splits + aux features
  2. chain-ext MLP     (OOF 0.856) — nonlinear neural, multitask shared trunk

Thesis: LGB and MLP have different error patterns (trees miss smooth nonlinear
interactions; MLPs miss sparse threshold effects). Even though MLP is weaker
alone, per-target NNLS should find real complementary signal — especially on
targets where MLP wins solo (nc) or ties (ei).

Recreates the pattern of mono-LGB (LB 0.860) + Chemprop (LB 0.892) → blend
LB 0.897, but with an MLP replacing Chemprop (Kaggle-runtime-compatible).

============================================================================
DEPENDENCIES (both bases must exist)
============================================================================

  results/exp_chain_ext_lgbm/{oof.csv, submission.csv}
  results/exp_chain_ext_mlp/{oof.csv, submission.csv}

Both use SPLIT_SEED=42 → OOFs are fold-aligned per canon.

============================================================================
METHOD
============================================================================

Per target:
  1. Load OOF and submission from both sources
  2. Align by (canon, target_type) for OOF, by id for submission
  3. NNLS on A = [y_lgb, y_mlp] vs y_true → non-negative weights
  4. Normalize to sum=1
  5. (Optional) Apply bias mitigations — DISABLED by default because
     MLP isn't Chemprop and doesn't need Chemprop-specific bias
  6. Compute blended OOF R²
  7. Apply weights to test predictions

============================================================================
BIAS MITIGATION — RELAXED FROM CHEMPROP BLEND
============================================================================

The previous 2-way blend (Chemprop + LGB) used:
  CHEMPROP_WEIGHT_FLOOR = 0.40  # Chemprop always keeps ≥40% weight
  APPLY_CHEMPROP_BIAS   = 0.15  # shift +0.15 to Chemprop after NNLS

Rationale was: Chemprop had honest OOF (+0.032 LB-OOF gap = UNDER-estimates),
while LGB had aux-inflated OOF (-0.006 gap = OVER-estimates). So NNLS
under-weighted Chemprop; we corrected.

For MLP: OOF-LB gap is UNKNOWN. MLP with heavy early stopping (median
best_epoch=16) might have small gap in either direction. Chain-ext LGB has
+0.028 gap. Without empirical LB for MLP, we DON'T know which base is
under-weighted by NNLS.

**Default: pure NNLS (no bias, no floor).** If this blend fails on LB, we
have data to calibrate bias for a follow-up.

============================================================================
OUTPUTS  (under results/exp_blend_lgb_mlp/)
============================================================================

  run.log             — per-target weights, R²s, deltas, config
  blend_summary.json  — machine-readable summary
  submission.csv      — blended predictions
  blended_oof.csv     — per-row blended OOFs (for future stacking)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_blend_lgb_mlp.py

============================================================================
EXPECTED
============================================================================

Chain-ext LGB v1 OOF: 0.8662, LB 0.894 (gap +0.028)
Chain-ext MLP OOF:    0.8559, LB unknown (guess 0.87-0.88)

Blend OOF: ~0.87-0.88 (should improve over both bases if errors decorrelate)
Blend LB: 0.894-0.898 realistic (best case 0.900)

MLP wins on nc, ties on ei — those are LGB's weak targets. NNLS should
give MLP meaningful weight there and near-zero elsewhere.

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
EXP_NAME = "exp_blend_lgb_mlp"
EXP_DIR = REPO / "results" / EXP_NAME

LGB_DIR = REPO / "results" / "exp_chain_ext_lgbm"
MLP_DIR = REPO / "results" / "exp_chain_ext_mlp"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")

# Bias mitigations — DISABLED (pure NNLS as default)
LGB_WEIGHT_FLOOR = 0.0
APPLY_LGB_BIAS = 0.0


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
    for label, d in [("LGB chain-ext", LGB_DIR), ("MLP chain-ext", MLP_DIR)]:
        for fname in ["oof.csv", "submission.csv"]:
            p = d / fname
            if not p.exists():
                missing.append(f"  - {p}   ({label})")
    if missing:
        log.error("MISSING INPUT FILES:")
        for m in missing:
            log.error(m)
        log.error("")
        log.error("Run these first:")
        log.error("  poly2-venv/bin/python experiments/exp_chain_ext_lgbm.py")
        log.error("  poly2-venv/bin/python experiments/exp_chain_ext_mlp.py")
        sys.exit(1)
    log.info("input files verified:")
    log.info(f"  LGB dir: {LGB_DIR}")
    log.info(f"  MLP dir: {MLP_DIR}")


# ============================================================================
# LOAD + ALIGN
# ============================================================================

def load_and_align(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading OOF from both sources...")
    oof_l = pd.read_csv(LGB_DIR / "oof.csv")
    oof_m = pd.read_csv(MLP_DIR / "oof.csv")
    log.info(f"  LGB OOF: {oof_l.shape}")
    log.info(f"  MLP OOF: {oof_m.shape}")

    oof = (
        oof_l.rename(columns={"y_pred": "y_pred_lgb"})
        .merge(oof_m.rename(columns={"y_pred": "y_pred_mlp"})[["canon", "target_type", "y_pred_mlp"]],
               on=["canon", "target_type"], how="inner")
    )
    log.info(f"  aligned OOF: {oof.shape}")

    for col in ["y_true", "y_pred_lgb", "y_pred_mlp"]:
        n_nan = int(oof[col].isna().sum())
        if n_nan:
            log.warning(f"  {col} has {n_nan} NaN rows — dropped per-target")

    log.info("loading test submissions...")
    sub_l = pd.read_csv(LGB_DIR / "submission.csv").rename(columns={"target": "target_lgb"})
    sub_m = pd.read_csv(MLP_DIR / "submission.csv").rename(columns={"target": "target_mlp"})
    log.info(f"  LGB sub: {sub_l.shape}")
    log.info(f"  MLP sub: {sub_m.shape}")

    te = pd.read_csv(REPO / "ppp-round-2" / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_l, on="id", how="left").merge(sub_m, on="id", how="left")
    log.info(f"  aligned sub: {sub.shape}")

    for col in ["target_lgb", "target_mlp"]:
        n_nan = int(sub[col].isna().sum())
        if n_nan:
            log.warning(f"  sub {col} has {n_nan} NaN rows")

    return oof, sub


# ============================================================================
# PER-TARGET NNLS BLEND
# ============================================================================

def fit_target_weights(y_true: np.ndarray, y_l: np.ndarray, y_m: np.ndarray,
                       log: logging.Logger, target: str) -> tuple[float, float]:
    A = np.vstack([y_l, y_m]).T
    b = y_true
    x, _ = nnls(A, b)
    w_l_raw, w_m_raw = float(x[0]), float(x[1])

    s = w_l_raw + w_m_raw
    if s < 1e-9:
        log.warning(f"[{target}] NNLS collapsed to zero; falling back to 100% LGB")
        w_l_norm, w_m_norm = 1.0, 0.0
    else:
        w_l_norm, w_m_norm = w_l_raw / s, w_m_raw / s

    if APPLY_LGB_BIAS != 0.0:
        w_l_bias = min(1.0, w_l_norm + APPLY_LGB_BIAS)
        w_m_bias = max(0.0, 1.0 - w_l_bias)
    else:
        w_l_bias, w_m_bias = w_l_norm, w_m_norm

    if w_l_bias < LGB_WEIGHT_FLOOR:
        w_l_final = LGB_WEIGHT_FLOOR
        w_m_final = 1.0 - LGB_WEIGHT_FLOOR
    else:
        w_l_final, w_m_final = w_l_bias, w_m_bias

    log.info(f"  [{target}] NNLS raw:   w_lgb={w_l_raw:.4f}  w_mlp={w_m_raw:.4f}  (sum={s:.4f})")
    log.info(f"  [{target}] normalized: w_lgb={w_l_norm:.4f}  w_mlp={w_m_norm:.4f}")
    if APPLY_LGB_BIAS != 0.0:
        log.info(f"  [{target}] + bias {APPLY_LGB_BIAS:+.3f}: w_lgb={w_l_bias:.4f}  w_mlp={w_m_bias:.4f}")
    if w_l_bias < LGB_WEIGHT_FLOOR:
        log.info(f"  [{target}] floor-clipped (LGB floor {LGB_WEIGHT_FLOOR}): "
                 f"w_lgb={w_l_final:.4f}  w_mlp={w_m_final:.4f}")
    return w_l_final, w_m_final


def blend_all_targets(oof: pd.DataFrame, sub: pd.DataFrame, log: logging.Logger) -> dict:
    per_target = {}
    sub = sub.copy()
    sub["target_blend"] = np.nan

    log.info("=" * 60)
    log.info("PER-TARGET NNLS BLEND  (LGB chain-ext + MLP chain-ext)")
    log.info("=" * 60)
    log.info(f"CONFIG: LGB_WEIGHT_FLOOR={LGB_WEIGHT_FLOOR}   APPLY_LGB_BIAS={APPLY_LGB_BIAS}")

    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(subset=["y_true", "y_pred_lgb", "y_pred_mlp"])
        if len(g) < 10:
            log.warning(f"[{target}] only {len(g)} OOF rows — falling back to LGB")
            per_target[target] = {"n_oof": int(len(g)), "skipped": True}
            mask = sub["target_type"] == target
            sub.loc[mask, "target_blend"] = sub.loc[mask, "target_lgb"]
            continue

        y_true = g["y_true"].values
        y_l    = g["y_pred_lgb"].values
        y_m    = g["y_pred_mlp"].values

        log.info(f"[{target}] n_oof={len(g)}")
        r2_l = float(r2_score(y_true, y_l))
        r2_m = float(r2_score(y_true, y_m))
        log.info(f"  [{target}] individual OOF R²:  LGB={r2_l:.4f}   MLP={r2_m:.4f}")

        w_l, w_m = fit_target_weights(y_true, y_l, y_m, log, target)
        y_blend = w_l * y_l + w_m * y_m
        r2_blend = float(r2_score(y_true, y_blend))

        better_solo = max(r2_l, r2_m)
        delta = r2_blend - better_solo
        log.info(f"  [{target}] BLEND OOF R² = {r2_blend:.4f}   Δ vs better solo = {delta:+.4f}")

        mask = sub["target_type"] == target
        sub.loc[mask, "target_blend"] = (
            w_l * sub.loc[mask, "target_lgb"] + w_m * sub.loc[mask, "target_mlp"]
        )

        per_target[target] = {
            "n_oof":         int(len(g)),
            "r2_lgb":        r2_l,
            "r2_mlp":        r2_m,
            "r2_blend":      r2_blend,
            "w_lgb":         w_l,
            "w_mlp":         w_m,
            "delta_vs_better": delta,
            "better_solo":   "lgb" if r2_l >= r2_m else "mlp",
        }

    oof = oof.copy()
    oof["y_pred_blend"] = np.nan
    for target in TARGETS:
        info = per_target[target]
        if info.get("skipped"):
            continue
        mask = oof["target_type"] == target
        oof.loc[mask, "y_pred_blend"] = (
            info["w_lgb"] * oof.loc[mask, "y_pred_lgb"]
            + info["w_mlp"] * oof.loc[mask, "y_pred_mlp"]
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
    log.info(f"  {'target':>6s}  {'LGB':>10s}  {'MLP':>10s}  {'blend':>10s}  "
             f"{'w_lgb':>6s}  {'w_mlp':>6s}  {'best':>6s}")
    for t in TARGETS:
        info = per_target[t]
        if info.get("skipped"):
            log.info(f"  {t:>6s}  {'—':>10s}  {'—':>10s}  {'skipped':>10s}")
            continue
        log.info(
            f"  {t:>6s}  {info['r2_lgb']:>10.4f}  {info['r2_mlp']:>10.4f}  "
            f"{info['r2_blend']:>10.4f}  {info['w_lgb']:>6.3f}  "
            f"{info['w_mlp']:>6.3f}  {info['better_solo']:>6s}"
        )

    valid = [t for t in TARGETS if not per_target[t].get("skipped")]
    mean_l = float(np.mean([per_target[t]["r2_lgb"] for t in valid]))
    mean_m = float(np.mean([per_target[t]["r2_mlp"] for t in valid]))
    mean_b = float(np.mean([per_target[t]["r2_blend"] for t in valid]))
    log.info(f"  {'MEAN':>6s}  {mean_l:>10.4f}  {mean_m:>10.4f}  {mean_b:>10.4f}")

    log.info(f"  BLEND OOF vs LGB solo: {mean_b - mean_l:+.4f}")
    log.info(f"  BLEND OOF vs MLP solo: {mean_b - mean_m:+.4f}")

    sub_out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"})
    sub_out = sub_out.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    oof_path = EXP_DIR / "blended_oof.csv"
    oof_out[["canon", "target_type", "y_true", "y_pred_lgb", "y_pred_mlp", "y_pred_blend"]] \
        .to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_out)}")

    summary = {
        "exp_name":  EXP_NAME,
        "mean_r2_lgb_oof":   mean_l,
        "mean_r2_mlp_oof":   mean_m,
        "mean_r2_blend_oof": mean_b,
        "blend_lift_vs_lgb": mean_b - mean_l,
        "blend_lift_vs_mlp": mean_b - mean_m,
        "per_target": per_target,
        "config": {
            "lgb_weight_floor":  LGB_WEIGHT_FLOOR,
            "apply_lgb_bias":    APPLY_LGB_BIAS,
            "lgb_source":        str(LGB_DIR),
            "mlp_source":        str(MLP_DIR),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(EXP_DIR / "blend_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'blend_summary.json'}")

    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"LGB chain-ext v1 solo LB (ref): 0.894")
    log.info(f"MLP chain-ext solo LB (guess):  0.87-0.88 (untested)")
    log.info(f"Prior best (blend_nnls_3seed):  0.897")
    log.info(f"THIS blend OOF: {mean_b:.4f}")
    log.info("")
    log.info("Interpretation:")
    log.info("  - If blend OOF > LGB solo OOF (0.8662):  real complementarity, LB likely ≥ 0.894")
    log.info("  - If blend OOF > 0.870:  strong signal, LB may exceed 0.897")
    log.info("  - If blend OOF ~ LGB OOF:  MLP added no diversity; don't submit")
    log.info("  - If blend OOF < LGB OOF:  MLP hurt blend; something's wrong")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
