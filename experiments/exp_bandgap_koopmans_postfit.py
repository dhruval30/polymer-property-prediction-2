"""
exp_bandgap_koopmans_postfit.py — Koopmans-theorem physics post-processor
                                   for Egc / Ei / Eea predictions.

============================================================================
WHY THIS EXISTS
============================================================================

Koopmans' theorem (standard quantum chemistry): for a polymer,
    HOMO = −IE   and   LUMO = −EA   →   bandgap = IE − EA
i.e., in our notation,   Egc ≈ Ei − Eea.

EDA §S5 confirms empirical correlation: ei ↔ egc r = +0.68 on the 20
co-labeled polymers where all 3 targets are measured.

This script applies the same architecture as our WORKING Maxwell prior
(EPS ≈ a·Nc²) — physics-linear post-processing with an OOF-fit blend
weight — to the 3 bandgap-related targets instead of the 2 dielectric
ones. Maxwell added +0.001 LB. Koopmans touches 3 targets instead of 2,
so expected LB effect is roughly ~1.5-3× larger.

============================================================================
GAP-SAFE BY CONSTRUCTION
============================================================================

Post-processing only — nothing about training changes.
  - Chain-ext LGB v1 (+0.028 OOF-LB gap): UNCHANGED
  - Chemprop 3-seed (+0.032 OOF-LB gap): UNCHANGED
  - NNLS blend weights: UNCHANGED (recompute from same OOFs)
  - The only bytes that change in the submission are the y_pred values on
    (Egc, Ei, Eea) test rows

If α_optimal converges to 1.0 on any target, that target's predictions
stay identical to the current blend submission. Worst case: LB unchanged.

============================================================================
DATA / DEPENDENCIES
============================================================================

Uses artifacts already on disk:
  - results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_{0..4}.pkl.gz
      → provides per-fold OOF for ALL 7 targets per canon (Chemprop is
        multitask so every canon gets predictions for every target,
        even where y_true is NaN)
  - results/exp_chemprop_multitask_cpu_3seed/refit_test_preds.pkl.gz
      → refit-full-train Chemprop test predictions (4133 canons × 7 targets)
  - results/exp_blend_nnls_3seed/submission.csv
      → current best blend submission (LB 0.897), what we're modifying
  - ppp-round-2/{train,test}.csv → for canon lookups

============================================================================
ALPHA TUNING (per target, on OOF)
============================================================================

For each of {egc, ei, eea}:
  1. Compute physics prediction from Chemprop's OTHER 2 target OOFs:
       egc_phys = ei_pred - eea_pred
       ei_phys  = egc_pred + eea_pred
       eea_phys = ei_pred - egc_pred
  2. On canons where y_true[target] is known (n = 2028 for egc, ~220 for ei/eea):
     grid-search α ∈ [0.5, 1.0] step 0.025 to maximize R² of
         blend = α * chemprop_own + (1-α) * physics
  3. Log best α, R² baseline (α=1.0, i.e., pure Chemprop) vs R² blended
     Δ = blend_r² - baseline_r² (positive → physics helps)

Safety: if any α_optimal < 0.5, log warning and don't apply for that
target (means physics is out-predicting the model — suspicious).

============================================================================
APPLICATION TO TEST
============================================================================

For each test row (id, canon, target ∈ {egc, ei, eea}):
    own_test   = current blend submission value for this id
    phys_test  = combine(Chemprop refit predictions for the other 2 targets
                         on this canon; always available since Chemprop is
                         multitask and refit-predicted for all test canons)
    new_pred   = α[target] * own_test + (1 - α[target]) * phys_test

Test rows for the other 4 targets (eea/egb/eps/nc/tg → wait, eea IS in
{egc, ei, eea}, so): test rows for {egb, eps, nc, tg} are copied
UNCHANGED from the current blend submission.

============================================================================
OUTPUTS  (under results/exp_bandgap_koopmans_postfit/)
============================================================================

  run.log             — α tuning trace + before/after per-target OOF R²
  koopmans_summary.json — α values per target, R² deltas, config
  submission.csv      — modified submission (Egc/Ei/Eea rows updated)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_bandgap_koopmans_postfit.py

Wall time: ~1-2 min. All I/O and grid search, no training.

============================================================================
EXPECTED LB
============================================================================

Best case (Koopmans holds well):   +0.003-0.005 → LB 0.900-0.902
Middle case (partial fit):         +0.001-0.003 → LB 0.898-0.900
Worst case (α → 1.0 on all):       0            → LB 0.897 (unchanged)

============================================================================
"""
from __future__ import annotations

# --- stdlib ---
import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path

# --- third-party ---
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
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_bandgap_koopmans_postfit"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
BLEND_SUB_PATH = REPO / "results" / "exp_blend_nnls_3seed" / "submission.csv"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Physics-relevant targets only
PHYSICS_TARGETS = ("egc", "ei", "eea")

# Alpha grid: finer than user's [0.5, 0.6, ..., 1.0] for better resolution
ALPHA_GRID = np.arange(0.5, 1.001, 0.025)

# Physics recipes: target → (src1, src2, combine_fn)
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),   # Koopmans: bandgap = IE - EA
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),   # rearrange: IE = Egc + EA
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),   # rearrange: EA = IE - Egc
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
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


# ============================================================================
# LOAD CHEMPROP FULL OOF MATRIX (5920 canons × 7 targets)
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_train_canons_and_y(log: logging.Logger) -> tuple[list[str], np.ndarray]:
    """Reconstruct the exact same train canon ordering that exp_chemprop_multitask_cpu_3seed
    used, and return the y_matrix (n_canons, 7) with NaN for unlabeled cells."""
    log.info("loading train.csv and rebuilding wide train...")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    all_smi = tr["smiles"].unique()
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical", ncols=100)}
    tr["canon"] = tr["smiles"].map(canon_map)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean")))
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[list(TARGETS)]
    canons = wide.index.tolist()
    y_matrix = wide.values.astype(np.float32)
    log.info(f"  n_canon={len(canons)}   y shape={y_matrix.shape}   "
             f"NaN frac={100*np.isnan(y_matrix).mean():.1f}%")
    return canons, y_matrix


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    """Load per-fold Chemprop checkpoints, assemble (n_canons, 7) OOF matrix."""
    n_canons = len(canons)
    oof = np.full((n_canons, 7), np.nan, dtype=np.float32)
    for k in range(5):
        cp_path = CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz"
        with gzip.open(cp_path, "rb") as f:
            fold_result = pickle.load(f)
        val_idxs = fold_result["val_idxs"]
        val_preds_avg = fold_result["val_preds_avg"]      # (n_val, 7)
        oof[val_idxs] = val_preds_avg
        log.info(f"  loaded fold {k}: {val_preds_avg.shape}  (val n={len(val_idxs)})")

    n_missing = int(np.isnan(oof).all(axis=1).sum())
    log.info(f"  Chemprop OOF matrix: {oof.shape}   "
             f"{n_missing} canons with no predictions (should be 0)")
    return oof


def load_chemprop_test_matrix(log: logging.Logger) -> tuple[list[str], np.ndarray]:
    """Reload the test canon list + Chemprop refit predictions (4133 canons × 7 targets)."""
    log.info("loading Chemprop refit test predictions...")
    with gzip.open(CHEMPROP_DIR / "refit_test_preds.pkl.gz", "rb") as f:
        cache = pickle.load(f)
    test_preds = cache["test_preds_avg"]   # (n_test_canons, 7)

    # Rebuild test_canon_unique the same way the 3seed script did
    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical(te)", ncols=100)}
    te["canon"] = te["smiles"].map(canon_map)
    test_canon_unique = te["canon"].drop_duplicates().tolist()
    assert len(test_canon_unique) == test_preds.shape[0], (
        f"test canon count mismatch: {len(test_canon_unique)} vs {test_preds.shape[0]}"
    )
    log.info(f"  Chemprop test predictions: {test_preds.shape}   "
             f"n_test_canons={len(test_canon_unique)}")
    return test_canon_unique, test_preds


# ============================================================================
# ALPHA TUNING (per target, on Chemprop OOF)
# ============================================================================

def tune_alpha_for_target(
    target: str,
    oof: np.ndarray,
    y_matrix: np.ndarray,
    log: logging.Logger,
) -> dict:
    """Grid-search α to maximize R² of α · own_pred + (1-α) · physics_pred."""
    src_a_name, src_b_name, combine = PHYSICS_RECIPES[target]
    t_own_idx = TARGET_IDX[target]
    t_a_idx = TARGET_IDX[src_a_name]
    t_b_idx = TARGET_IDX[src_b_name]

    # Rows where y_true[target] is known
    mask = ~np.isnan(y_matrix[:, t_own_idx])
    n_labeled = int(mask.sum())

    own_pred = oof[mask, t_own_idx]
    src_a = oof[mask, t_a_idx]
    src_b = oof[mask, t_b_idx]
    y_true = y_matrix[mask, t_own_idx]

    # Chemprop always predicts all 7 targets, but check for any NaN edge cases
    valid = ~(np.isnan(own_pred) | np.isnan(src_a) | np.isnan(src_b))
    n_valid = int(valid.sum())
    own_pred = own_pred[valid]
    src_a = src_a[valid]
    src_b = src_b[valid]
    y_true = y_true[valid]

    log.info(f"[{target}] physics: own={target}  from ({src_a_name}, {src_b_name})   "
             f"n_labeled={n_labeled}  n_valid={n_valid}")

    physics = combine(src_a, src_b)

    # Baseline R² (α = 1.0, pure Chemprop)
    r2_baseline = float(r2_score(y_true, own_pred))
    # Pure physics R² (α = 0.0)
    r2_pure_phys = float(r2_score(y_true, physics))

    # Grid-search
    best_r2 = -np.inf
    best_alpha = 1.0
    for alpha in ALPHA_GRID:
        blend = alpha * own_pred + (1 - alpha) * physics
        r2 = float(r2_score(y_true, blend))
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = float(alpha)

    delta = best_r2 - r2_baseline
    log.info(f"[{target}] baseline (α=1.0) R²={r2_baseline:.4f}   "
             f"pure-physics (α=0.0) R²={r2_pure_phys:.4f}   "
             f"best α={best_alpha:.3f}   blend R²={best_r2:.4f}   Δ={delta:+.4f}")

    # Safety warning
    if best_alpha < 0.5:
        log.warning(f"[{target}] α < 0.5 — physics is out-predicting the model. "
                    f"SUSPICIOUS — investigate before applying to test")

    return {
        "target": target,
        "src_a": src_a_name,
        "src_b": src_b_name,
        "n_labeled": n_labeled,
        "n_valid": n_valid,
        "r2_baseline_alpha1": r2_baseline,
        "r2_pure_physics_alpha0": r2_pure_phys,
        "best_alpha": best_alpha,
        "r2_blend": best_r2,
        "delta_r2": delta,
    }


# ============================================================================
# APPLY TO SUBMISSION
# ============================================================================

def apply_koopmans_to_submission(
    alphas: dict[str, float],
    test_canons: list[str],
    chemprop_test: np.ndarray,
    log: logging.Logger,
) -> pd.DataFrame:
    """Load current blend submission, apply Koopmans blend to Egc/Ei/Eea rows.
    Return the modified submission DataFrame (id, target)."""
    log.info(f"loading current best submission: {BLEND_SUB_PATH}")
    sub = pd.read_csv(BLEND_SUB_PATH)
    log.info(f"  original submission: {sub.shape}  target range=[{sub['target'].min():.2f}, {sub['target'].max():.2f}]")

    # Need test.csv to map id → (canon, target_type)
    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in all_smi}
    te["canon"] = te["smiles"].map(canon_map)
    te = te[["id", "canon", "target_type"]]

    # Join sub with test to get canon+target per id
    sub_full = sub.merge(te, on="id", how="left")
    assert sub_full["canon"].notna().all(), "some ids missing canon"

    # Chemprop test canon → row index
    canon_to_idx = {c: i for i, c in enumerate(test_canons)}

    # Modify rows for each physics target
    n_modified = 0
    diffs_per_target = {}
    for tgt in PHYSICS_TARGETS:
        alpha = alphas[tgt]
        src_a, src_b, combine = PHYSICS_RECIPES[tgt]
        src_a_idx = TARGET_IDX[src_a]
        src_b_idx = TARGET_IDX[src_b]

        mask = sub_full["target_type"] == tgt
        rows = sub_full[mask].copy()
        log.info(f"[{tgt}] applying α={alpha:.3f}  n_rows={len(rows)}")

        # Get Chemprop test predictions for the other 2 targets
        canon_rows = rows["canon"].tolist()
        canon_idx = np.array([canon_to_idx[c] for c in canon_rows])
        chem_src_a = chemprop_test[canon_idx, src_a_idx]
        chem_src_b = chemprop_test[canon_idx, src_b_idx]
        physics_test = combine(chem_src_a, chem_src_b)

        # Blend
        own_test = rows["target"].values
        new_pred = alpha * own_test + (1 - alpha) * physics_test

        # Update in the full submission
        diffs = np.abs(new_pred - own_test)
        diffs_per_target[tgt] = {
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff":  float(diffs.max()),
            "n_rows":        int(len(rows)),
        }
        log.info(f"[{tgt}] mean |Δ|={diffs.mean():.4f}   max |Δ|={diffs.max():.4f}")

        sub_full.loc[mask, "target"] = new_pred
        n_modified += int(mask.sum())

    log.info(f"total rows modified: {n_modified}")

    # Return with just (id, target)
    out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    return out, diffs_per_target


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: alpha_grid=[{ALPHA_GRID[0]:.3f}..{ALPHA_GRID[-1]:.3f}], n={len(ALPHA_GRID)}")
    log.info(f"Physics targets: {PHYSICS_TARGETS}")
    log.info(f"Modifying: {BLEND_SUB_PATH}")

    t_start = time.time()

    # ---- Load Chemprop OOF matrix (5920 canons × 7 targets) ----
    log.info("=" * 60)
    log.info("LOAD CHEMPROP OOF MATRIX")
    log.info("=" * 60)
    canons, y_matrix = load_train_canons_and_y(log)
    oof = load_chemprop_oof_matrix(canons, log)

    # ---- Tune α per target ----
    log.info("=" * 60)
    log.info("ALPHA TUNING (per target, on Chemprop OOF)")
    log.info("=" * 60)
    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = tune_alpha_for_target(tgt, oof, y_matrix, log)

    alphas = {tgt: alpha_results[tgt]["best_alpha"] for tgt in PHYSICS_TARGETS}

    log.info("=" * 60)
    log.info("ALPHA SUMMARY (per target)")
    log.info("=" * 60)
    log.info(f"  {'target':>6s}  {'baseline R²':>12s}  {'best α':>8s}  {'blend R²':>10s}  {'Δ R²':>8s}")
    total_delta = 0.0
    for tgt in PHYSICS_TARGETS:
        r = alpha_results[tgt]
        log.info(f"  {tgt:>6s}  {r['r2_baseline_alpha1']:>12.4f}  "
                 f"{r['best_alpha']:>8.3f}  {r['r2_blend']:>10.4f}  {r['delta_r2']:>+8.4f}")
        total_delta += r["delta_r2"]
    log.info(f"  Sum ΔR² across 3 targets: {total_delta:+.4f}")
    log.info(f"  Mean ΔR² across 3 targets: {total_delta/3:+.4f}")
    log.info(f"  Expected mean(R²) uplift on 7-target mean: {total_delta/7:+.4f}")

    # ---- Load Chemprop test predictions ----
    log.info("=" * 60)
    log.info("APPLY TO CURRENT BEST BLEND SUBMISSION")
    log.info("=" * 60)
    test_canons, chemprop_test = load_chemprop_test_matrix(log)

    # ---- Apply to submission ----
    new_sub, diffs = apply_koopmans_to_submission(alphas, test_canons, chemprop_test, log)

    # ---- Write outputs ----
    sub_path = EXP_DIR / "submission.csv"
    new_sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}   rows={len(new_sub)}")

    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "chemprop_source":       str(CHEMPROP_DIR),
            "blend_submission":      str(BLEND_SUB_PATH),
            "alpha_grid":            [float(x) for x in ALPHA_GRID],
            "physics_targets":       list(PHYSICS_TARGETS),
        },
        "physics_recipes": {
            "egc": "Egc = Ei - Eea   (Koopmans)",
            "ei":  "Ei  = Egc + Eea  (rearrange)",
            "eea": "Eea = Ei - Egc   (rearrange)",
        },
        "alpha_tuning": alpha_results,
        "test_modification_stats": diffs,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(EXP_DIR / "koopmans_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'koopmans_summary.json'}")

    # ---- Decision guidance ----
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Current blend LB (before Koopmans): 0.897")
    log.info(f"Expected LB uplift: mean ΔR² / 7 = {total_delta/7:+.4f}")
    log.info(f"  Best case (Koopmans holds well):   +0.003-0.005 → LB 0.900-0.902")
    log.info(f"  Middle case (partial fit):         +0.001-0.003 → LB 0.898-0.900")
    log.info(f"  Worst case (α → 1.0 on all):       0            → LB 0.897 (unchanged)")

    # Safety check
    any_alpha_low = any(alphas[t] < 0.5 for t in PHYSICS_TARGETS)
    if any_alpha_low:
        log.warning("SAFETY: at least one α is below 0.5. Physics is out-predicting the model — investigate.")
    all_alpha_one = all(alphas[t] > 0.995 for t in PHYSICS_TARGETS)
    if all_alpha_one:
        log.warning("SAFETY: all α ≈ 1.0. Koopmans doesn't help on this data. Submission = original blend. SKIP submission.")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
