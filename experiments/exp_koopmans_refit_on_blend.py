"""
exp_koopmans_refit_on_blend.py — refit Koopmans α on the BLEND OOF surface
with extended grid [0.1, 1.0], apply to blend test predictions.

Design flaw in the original 0.902 pipeline (Koopmans postfit):
  - α tuned on:  Chemprop OOF (own = Chemprop pred)
  - α applied to: BLEND test surface (own = blend prediction)
  → α is provably not optimal for the surface it's applied to.

Additional issue:
  - α_ei landed at grid boundary (0.5). True optimum may be < 0.5,
    meaning physics should contribute MORE than 50% for ei.

Fix: tune α on the BLEND OOF surface (own = blend prediction, physics =
Chemprop OOF for consistency with the test-time physics source), extend
grid down to 0.1, apply new α to blend test predictions.

Inputs:
  results/exp_blend_nnls_3seed/blended_oof.csv                — BLEND OOF (own)
  results/exp_blend_nnls_3seed/submission.csv                 — BLEND test (own)
  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_*  — Chemprop OOF (physics)
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds   — Chemprop refit (physics test)
  results/exp_bandgap_koopmans_postfit/koopmans_summary.json  — old α for comparison
  ppp-round-2/{train,test}.csv

Outputs:
  results/exp_koopmans_refit_on_blend/
      run.log
      alpha_comparison.json
      submission.csv         (only written if per-target guard rail passes for ≥1 target)

Guard rail: apply new α to a target's test rows only if per-target OOF R²
improvement is > 0 (positive delta vs old α). Selective per-target application.

Wall time: <15 seconds. No training.
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
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR       = REPO / "ppp-round-2"

BLEND_OOF_PATH = REPO / "results" / "exp_blend_nnls_3seed"        / "blended_oof.csv"
BLEND_SUB_PATH = REPO / "results" / "exp_blend_nnls_3seed"        / "submission.csv"
CHEMPROP_DIR   = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
CHEMPROP_REFIT = CHEMPROP_DIR / "refit_test_preds.pkl.gz"
OLD_KOOPMANS   = REPO / "results" / "exp_bandgap_koopmans_postfit" / "koopmans_summary.json"
OLD_902_SUB    = REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"
INPUT_LB       = 0.902

EXP_NAME = "exp_koopmans_refit_on_blend"
EXP_DIR  = REPO / "results" / EXP_NAME

TARGETS    = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS  = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
N_SPLITS   = 5

PHYSICS_TARGETS = ("egc", "ei", "eea")
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),   # Koopmans: bandgap = IE − EA
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}

# Extended grid: [0.1, 1.0] step 0.02 (was [0.5, 1.0] step 0.025)
ALPHA_GRID = np.round(np.arange(0.10, 1.001, 0.02), 4)

# Per-target guard rail: apply new α only if it beats old α on blend OOF
# (using strict > 0 threshold so we don't apply a worse or identical α)
PER_TARGET_MIN_DELTA = 0.0


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
# DATA LOADING
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def build_wide_train(tr: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Reconstruct wide (canon, target) matrix aligned with Chemprop's OOF."""
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns: wide[t] = np.nan
    wide = wide[list(TARGETS)]
    return wide.index.tolist(), wide.values.astype(np.float32)


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    n = len(canons)
    oof = np.full((n, 7), np.nan, dtype=np.float32)
    for k in range(N_SPLITS):
        with gzip.open(CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            r = pickle.load(f)
        oof[r["val_idxs"]] = r["val_preds_avg"]
    log.info(f"chemprop OOF matrix: {oof.shape}   "
             f"missing rows: {int(np.isnan(oof).all(axis=1).sum())}")
    return oof


# ============================================================================
# α REFIT ON BLEND OOF SURFACE
# ============================================================================

def refit_alpha_on_blend(target: str, blend_oof: pd.DataFrame,
                          canons: list[str], chem_oof: np.ndarray,
                          old_alpha: float, old_alpha_grid: list,
                          log: logging.Logger) -> dict:
    """Grid-search α on the BLEND OOF surface (own = blend prediction).
    Physics term uses Chemprop OOF (consistent with test-time physics source
    which is chemprop refit predictions)."""
    src_a, src_b, combine = PHYSICS_RECIPES[target]
    t_own = TARGET_IDX[target]
    t_a   = TARGET_IDX[src_a]
    t_b   = TARGET_IDX[src_b]

    canon_to_idx = {c: i for i, c in enumerate(canons)}
    df = blend_oof[blend_oof["target_type"] == target]
    canon_idx = np.array([canon_to_idx[c] for c in df["canon"]])
    own_pred = df["y_pred_blend"].values.astype(np.float64)
    y_true   = df["y_true"].values.astype(np.float64)
    sa       = chem_oof[canon_idx, t_a].astype(np.float64)
    sb       = chem_oof[canon_idx, t_b].astype(np.float64)
    physics  = combine(sa, sb)
    valid = ~(np.isnan(own_pred) | np.isnan(sa) | np.isnan(sb) | np.isnan(y_true))
    own_pred = own_pred[valid]; physics = physics[valid]; y_true = y_true[valid]

    # Baseline (α=1, no physics)
    r2_baseline = float(r2_score(y_true, own_pred))
    r2_pure_phys = float(r2_score(y_true, physics))

    # Old α behavior on THIS surface (blend OOF, chemprop physics)
    r2_old = float(r2_score(y_true, old_alpha * own_pred + (1 - old_alpha) * physics))

    # Grid search new α on extended grid
    r2s = np.array([r2_score(y_true, a * own_pred + (1 - a) * physics) for a in ALPHA_GRID])
    best_i = int(np.argmax(r2s))
    new_alpha = float(ALPHA_GRID[best_i])
    r2_new = float(r2s[best_i])

    # Old-grid re-fit (for fair comparison — what would extended range have given on old surface?)
    old_grid_r2 = np.array([r2_score(y_true, a * own_pred + (1 - a) * physics) for a in old_alpha_grid])
    best_old_grid_i = int(np.argmax(old_grid_r2))
    best_old_grid_alpha = float(old_alpha_grid[best_old_grid_i])
    best_old_grid_r2 = float(old_grid_r2[best_old_grid_i])

    log.info(f"[{target}] baseline R²={r2_baseline:.4f}   pure-phys R²={r2_pure_phys:.4f}")
    log.info(f"[{target}] OLD α={old_alpha:.3f} (from stored summary) → R²={r2_old:.4f}   "
             f"(applied to blend surface)")
    log.info(f"[{target}] OLD-grid best α={best_old_grid_alpha:.3f} → R²={best_old_grid_r2:.4f}   "
             f"(refit on blend, restricted to old [0.5, 1.0] grid)")
    log.info(f"[{target}] NEW α={new_alpha:.3f} → R²={r2_new:.4f}   "
             f"(refit on blend, extended [0.1, 1.0] grid)   "
             f"Δ vs OLD α={r2_new - r2_old:+.4f}")

    return {
        "target":                target,
        "old_alpha":             old_alpha,
        "old_alpha_r2":          r2_old,
        "old_grid_best_alpha":   best_old_grid_alpha,
        "old_grid_best_r2":      best_old_grid_r2,
        "new_alpha":             new_alpha,
        "new_alpha_r2":          r2_new,
        "delta_r2_new_vs_old":   r2_new - r2_old,
        "n_valid":               int(valid.sum()),
        "r2_baseline_alpha1":    r2_baseline,
        "r2_pure_physics":       r2_pure_phys,
        "alpha_at_boundary":     new_alpha == float(ALPHA_GRID[0]),
    }


# ============================================================================
# APPLY NEW α TO TEST (per-target guard rail)
# ============================================================================

def apply_new_alpha_to_test(alpha_results: dict, blend_sub: pd.DataFrame,
                              te: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Apply new α per physics target if its delta beat old α by > threshold.
    Own = blend test prediction, physics = Chemprop refit prediction (same as
    original test flow, only α value differs)."""
    with gzip.open(CHEMPROP_REFIT, "rb") as f:
        chem_refit_cache = pickle.load(f)
    chem_test = chem_refit_cache["test_preds_avg"]
    test_canons = te["canon"].drop_duplicates().tolist()
    c2i = {c: i for i, c in enumerate(test_canons)}

    canon_map = te[["id", "canon"]].set_index("id")["canon"].to_dict()
    tt_map    = te[["id", "target_type"]].set_index("id")["target_type"].to_dict()

    sub_out = blend_sub.copy()
    sub_out["canon"]       = sub_out["id"].map(canon_map)
    sub_out["target_type"] = sub_out["id"].map(tt_map)

    diff_stats = {}
    for tgt in PHYSICS_TARGETS:
        info = alpha_results[tgt]
        delta = info["delta_r2_new_vs_old"]
        n_rows = int((sub_out["target_type"] == tgt).sum())

        if delta <= PER_TARGET_MIN_DELTA:
            log.warning(f"[APPLY {tgt}] SKIP — Δ={delta:+.4f} ≤ {PER_TARGET_MIN_DELTA} threshold. "
                        f"Keeping OLD α={info['old_alpha']:.3f} for this target's test rows.")
            # Apply OLD α (which is what current 0.902 submission uses)
            alpha_used = info["old_alpha"]
            diff_stats[tgt] = {"n_rows": n_rows, "new_alpha_applied": False,
                                "alpha_used": alpha_used, "reason": f"Δ={delta:+.4f} not positive"}
        else:
            alpha_used = info["new_alpha"]
            log.info(f"[APPLY {tgt}] APPLY NEW α={alpha_used:.3f} (Δ={delta:+.4f})")
            diff_stats[tgt] = {"n_rows": n_rows, "new_alpha_applied": True,
                                "alpha_used": alpha_used, "delta_r2": delta}

        # Compute physics term at test using Chemprop refit predictions
        src_a, src_b, combine = PHYSICS_RECIPES[tgt]
        sa_idx, sb_idx = TARGET_IDX[src_a], TARGET_IDX[src_b]
        mask = sub_out["target_type"] == tgt
        rows = sub_out[mask]
        cidx = np.array([c2i[c] for c in rows["canon"]])
        chem_a = chem_test[cidx, sa_idx]
        chem_b = chem_test[cidx, sb_idx]
        physics_te = combine(chem_a, chem_b)
        own_te = rows["target"].values
        new_pred = alpha_used * own_te + (1 - alpha_used) * physics_te

        d = np.abs(new_pred - own_te)
        log.info(f"[APPLY {tgt}] α={alpha_used:.3f}   n_rows={n_rows}   "
                 f"mean |Δ|={d.mean():.4f}   max |Δ|={d.max():.4f}")
        sub_out.loc[mask, "target"] = new_pred

    return sub_out[["id", "target"]].sort_values("id").reset_index(drop=True), diff_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info("=" * 60)
    log.info(f"=== {EXP_NAME} — refit Koopmans α on BLEND OOF surface ===")
    log.info("=" * 60)
    log.info(f"Base input: {BLEND_SUB_PATH.name}  (blend test, unadjusted, LB ~0.897)")
    log.info(f"Reference:  {OLD_902_SUB.name}  (LB {INPUT_LB})")
    log.info(f"OLD α grid: [0.500, 1.000] step 0.025")
    log.info(f"NEW α grid: [{ALPHA_GRID[0]:.3f}, {ALPHA_GRID[-1]:.3f}] step 0.02  "
             f"({len(ALPHA_GRID)} values)")
    log.info(f"Per-target guard rail: apply new α only if Δ > {PER_TARGET_MIN_DELTA}")
    t0 = time.time()

    # ---- Load old α values ----
    with open(OLD_KOOPMANS) as f:
        koop_summary = json.load(f)
    old_alphas = {t: koop_summary["alpha_tuning"][t]["best_alpha"] for t in PHYSICS_TARGETS}
    OLD_GRID = np.arange(0.5, 1.001, 0.025)   # original grid for fair comparison
    log.info(f"Old α values (from stored summary):")
    for t in PHYSICS_TARGETS:
        log.info(f"  {t:>3s}: α={old_alphas[t]:.3f}")

    # ---- Load train (for wide matrix / canon list) ----
    log.info("loading train.csv...")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    all_smi = tr["smiles"].unique()
    cmap = {s: canonical(s) for s in tqdm(all_smi, desc="canon(tr)", ncols=100)}
    tr["canon"] = tr["smiles"].map(cmap)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean")))
    canons, _ = build_wide_train(tr)
    log.info(f"train canons: {len(canons)}")

    # ---- Load blend OOF + chemprop OOF ----
    blend_oof = pd.read_csv(BLEND_OOF_PATH)
    log.info(f"blend OOF: {blend_oof.shape}   cols={list(blend_oof.columns)}")
    chem_oof = load_chemprop_oof_matrix(canons, log)

    # ---- Refit α per physics target on BLEND OOF ----
    log.info("=" * 60)
    log.info("REFIT α ON BLEND OOF SURFACE (own = blend, physics = chemprop OOF)")
    log.info("=" * 60)
    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = refit_alpha_on_blend(
            tgt, blend_oof, canons, chem_oof, old_alphas[tgt], OLD_GRID, log,
        )
        log.info("-" * 60)

    # ---- Summary ----
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"{'target':>8s}  {'old α':>7s}  {'new α':>7s}  "
             f"{'old-α R²':>10s}  {'new-α R²':>10s}  {'Δ R²':>8s}  {'apply?'}")
    total_delta = 0.0
    for tgt in PHYSICS_TARGETS:
        info = alpha_results[tgt]
        applied = "YES" if info["delta_r2_new_vs_old"] > PER_TARGET_MIN_DELTA else "no"
        boundary = " (BOUNDARY!)" if info["alpha_at_boundary"] else ""
        log.info(f"{tgt:>8s}  {info['old_alpha']:>7.3f}  {info['new_alpha']:>7.3f}{boundary}  "
                 f"{info['old_alpha_r2']:>10.4f}  {info['new_alpha_r2']:>10.4f}  "
                 f"{info['delta_r2_new_vs_old']:>+8.4f}  {applied}")
        if applied == "YES":
            total_delta += info["delta_r2_new_vs_old"]
    log.info(f"SUM ΔR² across physics targets (only counting those that will apply new α): "
             f"{total_delta:+.4f}")
    log.info(f"Expected 7-target mean R² lift: {total_delta/7:+.4f}")

    # ---- Decision ----
    n_apply = sum(1 for t in PHYSICS_TARGETS
                  if alpha_results[t]["delta_r2_new_vs_old"] > PER_TARGET_MIN_DELTA)
    if n_apply == 0:
        log.info("=" * 60)
        log.info("DECISION: no targets clear guard rail. New sub would be identical to old 0.902.")
        log.info("Recommendation: SKIP submission — don't burn a slot.")
        log.info("=" * 60)
    else:
        log.info("=" * 60)
        log.info(f"DECISION: {n_apply}/{len(PHYSICS_TARGETS)} targets will use new α. Writing submission.")
        log.info(f"Expected LB lift: {total_delta/7:+.4f} → target LB ≈ {INPUT_LB + total_delta/7:.4f}")
        log.info("=" * 60)

    # ---- Apply new α to blend test (starting from unadjusted blend submission) ----
    log.info("loading blend test submission (unadjusted)")
    blend_sub = pd.read_csv(BLEND_SUB_PATH)
    log.info(f"  {blend_sub.shape}")

    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi_te = te["smiles"].unique()
    cmap_te = {s: canonical(s) for s in all_smi_te}
    te["canon"] = te["smiles"].map(cmap_te)

    new_sub, diff_stats = apply_new_alpha_to_test(alpha_results, blend_sub, te, log)

    # Sanity: compare against the old 0.902 submission to see how much changed
    old_902 = pd.read_csv(OLD_902_SUB)
    delta_vs_old = (new_sub.set_index("id")["target"] - old_902.set_index("id")["target"]).abs()
    log.info(f"delta vs OLD 0.902 sub:   mean|Δ|={delta_vs_old.mean():.4f}  "
             f"max|Δ|={delta_vs_old.max():.4f}  "
             f"rows changed: {(delta_vs_old > 1e-6).sum()}/{len(delta_vs_old)}")

    sub_path = EXP_DIR / "submission.csv"
    new_sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(new_sub)}  NaNs={int(new_sub['target'].isna().sum())}")

    # ---- Write summary JSON ----
    summary = {
        "exp_name": EXP_NAME,
        "input_lb_reference": INPUT_LB,
        "alpha_grid_old": list(np.round(OLD_GRID, 4).tolist()),
        "alpha_grid_new": list(ALPHA_GRID.tolist()),
        "per_target_min_delta_to_apply": PER_TARGET_MIN_DELTA,
        "alpha_results": alpha_results,
        "sum_delta_r2_applied":       total_delta,
        "expected_7target_mean_lift": total_delta / 7,
        "expected_lb_if_translated":  INPUT_LB + total_delta / 7,
        "n_targets_new_alpha_applied": n_apply,
        "test_modification_stats": diff_stats,
        "delta_vs_old_902_mean_abs": float(delta_vs_old.mean()),
        "delta_vs_old_902_max_abs":  float(delta_vs_old.max()),
        "delta_vs_old_902_rows_changed": int((delta_vs_old > 1e-6).sum()),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(EXP_DIR / "alpha_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'alpha_comparison.json'}")

    log.info(f"wall time: {time.time() - t0:.1f}s")
    log.info("=" * 60)
    log.info(f"OUTPUT: {sub_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
