"""
exp_isotonic_calibration_cv.py — HONEST nested-CV isotonic calibration of LB 0.902 sub.

Prior naive version (fit isotonic on OOF, evaluate on same OOF) reported +0.159
sum ΔR² which was pure in-sample overfitting (gain inversely proportional to N).

This version:
  1. Reconstructs the (blend + Koopmans) OOF for each of 7 targets.
  2. For each target: 5-fold KFold within its OOF rows. Fit isotonic on 4/5,
     transform the 1/5 val slice, record calibrated preds. Combine across
     folds → HONEST cross-fitted calibrated OOF. Compute R² vs raw R².
  3. Decision gates:
       - Sum ΔR² > 0.010 → SUBMIT.
       - Per-target: apply isotonic to that target's test rows only if
         its honest ΔR² > +0.003.
  4. Production isotonic is fit on FULL OOF (nested CV was for validation only);
     applied selectively per per-target guard rail.

Inputs (unchanged from prior version):
  results/exp_blend_nnls_3seed/blended_oof.csv
  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_*.pkl.gz
  results/exp_bandgap_koopmans_postfit/koopmans_summary.json
  results/exp_bandgap_koopmans_postfit/submission.csv        (LB 0.902 base)
  ppp-round-2/{train,test}.csv

Outputs:
  results/exp_isotonic_calibration_cv/
      run.log
      calibration_summary.json
      per_target_diagnostics.csv
      submission.csv               (only written if sum honest ΔR² > 0.010)
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR      = REPO / "ppp-round-2"
BLEND_OOF     = REPO / "results" / "exp_blend_nnls_3seed" / "blended_oof.csv"
CHEMPROP_DIR  = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
KOOPMANS_SUM  = REPO / "results" / "exp_bandgap_koopmans_postfit" / "koopmans_summary.json"
INPUT_SUB     = REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"
INPUT_SUB_LB  = 0.902

EXP_NAME = "exp_isotonic_calibration_cv"
EXP_DIR  = REPO / "results" / EXP_NAME

TARGETS    = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}

# Nested-CV settings
CV_N_SPLITS   = 5
CV_SEED       = 42

# Decision gates
SUBMIT_SUM_DELTA_R2      = 0.010   # sum ΔR² > this → submit
PER_TARGET_APPLY_DELTA   = 0.003   # per-target apply only if honest Δ > this


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
# DATA + OOF LOADING
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def build_wide_train(tr: pd.DataFrame) -> list[str]:
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    return wide[list(TARGETS)].index.tolist()


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    n = len(canons)
    oof = np.full((n, 7), np.nan, dtype=np.float32)
    for k in range(5):
        with gzip.open(CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            r = pickle.load(f)
        oof[r["val_idxs"]] = r["val_preds_avg"]
    log.info(f"chemprop OOF matrix: {oof.shape}   missing rows: "
             f"{int(np.isnan(oof).all(axis=1).sum())}")
    return oof


def reconstruct_koopmans_oof(blend_oof: pd.DataFrame, chem_oof: np.ndarray,
                              canons: list[str], alphas: dict, log: logging.Logger) -> pd.DataFrame:
    """Apply Koopmans α-blend to blend OOF for the 3 physics targets.
    Produces the (blend + Koopmans) OOF that mirrors the test surface."""
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    out = blend_oof.copy()
    out["y_pred_final"] = out["y_pred_blend"].values.copy()

    for tgt in ("egc", "ei", "eea"):
        alpha = alphas[tgt]
        src_a, src_b, combine = PHYSICS_RECIPES[tgt]
        sa_idx, sb_idx = TARGET_IDX[src_a], TARGET_IDX[src_b]

        mask = out["target_type"] == tgt
        rows = out[mask]
        canon_idx = np.array([canon_to_idx[c] for c in rows["canon"]])
        chem_a = chem_oof[canon_idx, sa_idx]
        chem_b = chem_oof[canon_idx, sb_idx]
        physics_oof = combine(chem_a, chem_b)
        own_oof = rows["y_pred_blend"].values

        blended = alpha * own_oof + (1 - alpha) * physics_oof
        nan_m = np.isnan(blended)
        blended[nan_m] = own_oof[nan_m]

        out.loc[mask, "y_pred_final"] = blended
        log.info(f"[koopmans OOF {tgt}] α={alpha:.3f}   n_rows={mask.sum()}   "
                 f"mean |Δ|={np.abs(blended - own_oof).mean():.4f}")

    return out


# ============================================================================
# HONEST NESTED-CV ISOTONIC EVALUATION
# ============================================================================

def nested_cv_isotonic(y_pred: np.ndarray, y_true: np.ndarray,
                        n_splits: int = CV_N_SPLITS, seed: int = CV_SEED
                        ) -> tuple[np.ndarray, float]:
    """Return (calibrated_oof, r2_calibrated) — honest cross-fitted isotonic.
    For each of n_splits sub-folds: fit isotonic on 4/5, transform 1/5 val.
    Concatenate val predictions → honest calibrated OOF."""
    n = len(y_pred)
    calibrated = np.zeros(n, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(np.arange(n)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_pred[train_idx], y_true[train_idx])
        calibrated[val_idx] = iso.transform(y_pred[val_idx])
    r2 = float(r2_score(y_true, calibrated))
    return calibrated, r2


def evaluate_calibration_per_target(koop_oof: pd.DataFrame, log: logging.Logger) -> dict:
    """For each target, compute raw OOF R², in-sample-fit R² (for reference),
    and honest cross-fitted R². Fit production isotonic on FULL OOF for later use."""
    per_target = {}
    for tgt in TARGETS:
        g = koop_oof[koop_oof["target_type"] == tgt].dropna(subset=["y_true", "y_pred_final"])
        y_true = g["y_true"].values.astype(np.float64)
        y_pred = g["y_pred_final"].values.astype(np.float64)

        r2_raw = float(r2_score(y_true, y_pred))

        # In-sample (for comparison to prior naive script)
        iso_full = IsotonicRegression(out_of_bounds="clip")
        iso_full.fit(y_pred, y_true)
        r2_insample = float(r2_score(y_true, iso_full.transform(y_pred)))

        # Honest cross-fitted
        _, r2_cv = nested_cv_isotonic(y_pred, y_true)

        delta_cv       = r2_cv - r2_raw
        delta_insample = r2_insample - r2_raw
        overfit_gap    = delta_insample - delta_cv

        log.info(f"[ISO {tgt:>4s}] n={len(g):>5d}  raw R²={r2_raw:.4f}  "
                 f"in-sample R²={r2_insample:.4f} (Δ={delta_insample:+.4f})  "
                 f"HONEST-CV R²={r2_cv:.4f} (Δ={delta_cv:+.4f})  "
                 f"overfit gap={overfit_gap:+.4f}")

        per_target[tgt] = {
            "n_rows":         int(len(g)),
            "r2_raw":         r2_raw,
            "r2_insample":    r2_insample,
            "r2_cv":          r2_cv,
            "delta_insample": delta_insample,
            "delta_cv":       delta_cv,
            "overfit_gap":    overfit_gap,
            "iso_full":       iso_full,      # for test application
            "y_pred_min":     float(y_pred.min()),
            "y_pred_max":     float(y_pred.max()),
        }
    return per_target


# ============================================================================
# APPLY TO SUBMISSION (per-target guard rail)
# ============================================================================

def apply_isotonic_to_submission(sub: pd.DataFrame, per_target: dict, te: pd.DataFrame,
                                   log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    sub_full = sub.merge(te[["id", "target_type"]], on="id", how="left")
    assert sub_full["target_type"].notna().all()

    diff_stats = {}
    for tgt in TARGETS:
        info = per_target[tgt]
        mask = sub_full["target_type"] == tgt
        n = int(mask.sum())

        if info["delta_cv"] <= PER_TARGET_APPLY_DELTA:
            log.warning(f"[APPLY {tgt}] SKIPPED — honest ΔR²={info['delta_cv']:+.4f} ≤ "
                        f"{PER_TARGET_APPLY_DELTA} threshold. Using raw predictions.")
            diff_stats[tgt] = {"n_rows": n, "applied": False,
                               "reason": f"honest ΔR² {info['delta_cv']:+.4f} ≤ threshold"}
            continue

        raw_preds = sub_full.loc[mask, "target"].values.astype(np.float64)
        cal_preds = info["iso_full"].transform(raw_preds)
        diffs = np.abs(cal_preds - raw_preds)

        # Extrapolation warning
        n_below = int((raw_preds < info["y_pred_min"]).sum())
        n_above = int((raw_preds > info["y_pred_max"]).sum())
        if n_below + n_above > 0:
            log.info(f"[APPLY {tgt}]   {n_below} test preds below OOF range, "
                     f"{n_above} above (isotonic clips at boundaries)")

        log.info(f"[APPLY {tgt}] n={n}  APPLIED (honest Δ={info['delta_cv']:+.4f})  "
                 f"mean |Δ|={diffs.mean():.4f}  max |Δ|={diffs.max():.4f}  "
                 f"raw range=[{raw_preds.min():.3f}, {raw_preds.max():.3f}]  "
                 f"cal range=[{cal_preds.min():.3f}, {cal_preds.max():.3f}]")

        sub_full.loc[mask, "target"] = cal_preds
        diff_stats[tgt] = {
            "n_rows": n, "applied": True,
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff":  float(diffs.max()),
            "n_test_below_oof_range": n_below,
            "n_test_above_oof_range": n_above,
        }

    out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    return out, diff_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"Input submission: {INPUT_SUB}  (LB {INPUT_SUB_LB})")
    log.info(f"Blend OOF:        {BLEND_OOF}")
    log.info(f"Chemprop source:  {CHEMPROP_DIR}")
    log.info(f"CV splits: {CV_N_SPLITS} (seed {CV_SEED})")
    log.info(f"Decision gates: sum ΔR² > {SUBMIT_SUM_DELTA_R2} → SUBMIT; "
             f"per-target apply if honest Δ > {PER_TARGET_APPLY_DELTA}")

    t0 = time.time()

    # ---- Load Koopmans alphas ----
    with open(KOOPMANS_SUM) as f:
        koop_summary = json.load(f)
    alphas = {t: koop_summary["alpha_tuning"][t]["best_alpha"] for t in ("egc", "ei", "eea")}
    log.info(f"Koopmans alphas: egc={alphas['egc']:.3f}  ei={alphas['ei']:.3f}  eea={alphas['eea']:.3f}")

    # ---- Load train (for canon list) ----
    log.info("loading train.csv...")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    all_smi = tr["smiles"].unique()
    cmap = {s: canonical(s) for s in tqdm(all_smi, desc="canon(tr)", ncols=100)}
    tr["canon"] = tr["smiles"].map(cmap)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean")))
    canons = build_wide_train(tr)
    log.info(f"train canons: {len(canons)}")

    # ---- Load OOFs and reconstruct Koopmans-adjusted OOF ----
    blend_oof = pd.read_csv(BLEND_OOF)
    log.info(f"blend OOF: {blend_oof.shape}")
    chem_oof = load_chemprop_oof_matrix(canons, log)

    log.info("=" * 60)
    log.info("RECONSTRUCT KOOPMANS OOF (blend + α·physics for egc/ei/eea)")
    log.info("=" * 60)
    koop_oof = reconstruct_koopmans_oof(blend_oof, chem_oof, canons, alphas, log)

    # ---- Honest nested-CV evaluation ----
    log.info("=" * 60)
    log.info("HONEST NESTED-CV ISOTONIC EVALUATION")
    log.info("  (5-fold KFold within each target's OOF rows)")
    log.info("=" * 60)
    per_target = evaluate_calibration_per_target(koop_oof, log)

    # ---- Decision ----
    log.info("=" * 60)
    log.info("DECISION")
    log.info("=" * 60)
    sum_delta_cv       = sum(per_target[t]["delta_cv"] for t in TARGETS)
    sum_delta_insample = sum(per_target[t]["delta_insample"] for t in TARGETS)
    log.info(f"sum in-sample ΔR² (overfit reference): {sum_delta_insample:+.4f}")
    log.info(f"sum HONEST-CV ΔR²:                     {sum_delta_cv:+.4f}")
    log.info(f"overfit inflation ratio: {sum_delta_insample / sum_delta_cv:.2f}x" if abs(sum_delta_cv) > 1e-9 else "overfit inflation: (cv=0)")

    per_target_apply = {t: per_target[t]["delta_cv"] > PER_TARGET_APPLY_DELTA for t in TARGETS}
    n_apply = sum(per_target_apply.values())
    log.info(f"targets that clear per-target apply threshold ({PER_TARGET_APPLY_DELTA}):")
    for t in TARGETS:
        marker = "✓" if per_target_apply[t] else "✗"
        log.info(f"  {marker} {t:>3s}: honest Δ={per_target[t]['delta_cv']:+.4f}")
    log.info(f"→ {n_apply}/7 targets will be calibrated")

    if sum_delta_cv > SUBMIT_SUM_DELTA_R2:
        decision = "SUBMIT"
        log.info(f"✅ SUBMIT: honest sum ΔR² = {sum_delta_cv:+.4f} > {SUBMIT_SUM_DELTA_R2}. "
                 f"Expected LB uplift: ~+{sum_delta_cv/7:.4f} → "
                 f"target LB ≈ {INPUT_SUB_LB + sum_delta_cv/7:.4f}")
    else:
        decision = "SKIP"
        log.info(f"❌ SKIP: honest sum ΔR² = {sum_delta_cv:+.4f} ≤ {SUBMIT_SUM_DELTA_R2}. "
                 f"Not worth a sub slot. Lock 0.902.")

    # ---- Apply to submission (only if decision is SUBMIT) ----
    if decision == "SUBMIT":
        log.info("=" * 60)
        log.info("APPLY ISOTONIC TO INPUT SUBMISSION (per-target guard rail)")
        log.info("=" * 60)
        sub = pd.read_csv(INPUT_SUB)
        te = pd.read_csv(DATA_DIR / "test.csv")
        new_sub, diff_stats = apply_isotonic_to_submission(sub, per_target, te, log)

        sub_path = EXP_DIR / "submission.csv"
        new_sub.to_csv(sub_path, index=False)
        log.info(f"wrote {sub_path}   rows={len(new_sub)}   "
                 f"NaNs={int(new_sub['target'].isna().sum())}")
    else:
        log.info("submission.csv NOT written (decision=SKIP)")
        diff_stats = {t: {"applied": False, "reason": "decision=SKIP"} for t in TARGETS}

    # ---- Per-target diagnostics CSV ----
    diag = pd.DataFrame([{
        "target":              t,
        "n_oof_rows":          per_target[t]["n_rows"],
        "r2_raw":              per_target[t]["r2_raw"],
        "r2_insample_iso":     per_target[t]["r2_insample"],
        "r2_honest_cv":        per_target[t]["r2_cv"],
        "delta_insample":      per_target[t]["delta_insample"],
        "delta_honest_cv":     per_target[t]["delta_cv"],
        "overfit_gap":         per_target[t]["overfit_gap"],
        "applied_to_test":     diff_stats[t].get("applied", False),
    } for t in TARGETS])
    diag_path = EXP_DIR / "per_target_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    log.info(f"wrote {diag_path}")

    # ---- Summary JSON ----
    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "input_submission":         str(INPUT_SUB),
            "input_lb_reference":       INPUT_SUB_LB,
            "cv_n_splits":              CV_N_SPLITS,
            "cv_seed":                  CV_SEED,
            "submit_sum_delta_r2":      SUBMIT_SUM_DELTA_R2,
            "per_target_apply_delta":   PER_TARGET_APPLY_DELTA,
            "koopmans_alphas":          alphas,
        },
        "per_target": {t: {k: v for k, v in per_target[t].items() if k != "iso_full"} for t in TARGETS},
        "sum_delta_insample": sum_delta_insample,
        "sum_delta_honest_cv": sum_delta_cv,
        "decision": decision,
        "expected_lb_if_submitted": INPUT_SUB_LB + sum_delta_cv / 7,
        "test_modification_stats": diff_stats,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(EXP_DIR / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'calibration_summary.json'}")

    log.info(f"wall time: {time.time() - t0:.1f}s")
    log.info("=" * 60)
    log.info(f"FINAL DECISION: {decision}   honest sum ΔR² = {sum_delta_cv:+.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
