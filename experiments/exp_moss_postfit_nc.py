"""
exp_moss_postfit_nc.py — Moss-rule physics post-fit for Nc (refractive index)
                        via bandgap Egc.

============================================================================
WHY THIS EXISTS
============================================================================

The Koopmans post-fit for bandgaps landed at LB 0.902 (+0.005). Egb 3-way
extension landed neutral (Chemprop already captured Egb↔Egc). This script
tries the OTHER canonical polymer-optics physics rule:

**Moss rule**: for many semiconductors and polymers,
    n² · Eg ≈ k          (roughly constant across materials)

i.e., refractive index and bandgap trade off: small-bandgap materials have
higher refractive indices. The classical Moss form is `n⁴·Eg = const` but
`n²·Eg = k` is a simpler linear-in-1/Eg approximation that works well over
a narrow bandgap range (polymers ~2-9 eV).

============================================================================
THE MATH
============================================================================

1. **Fit k** on the 26 train polymers labeled for BOTH Nc and Egc:
       k_estimated = median( Nc² · Egc )   over those 26 rows
   (using median instead of mean — robust to outlier fits)

2. **Physics prediction** per canon (needs Egc estimate):
       Nc_moss = sqrt( k / max(Egc_pred, 0.5) )
   (clip Egc to 0.5 eV floor to avoid divide-by-zero / negative sqrt)

3. **α tuning** on 229 Nc-labeled train rows using Chemprop OOF for Egc:
       Nc_blend = α · Nc_pred + (1-α) · Nc_moss
       grid: α ∈ [0.5, 1.0] step 0.025

4. **Apply α to submission's Nc rows** using Chemprop refit predictions for
   Egc (always available, Chemprop is multitask):
       Nc_new_test = α · Nc_own_test + (1-α) · sqrt( k / Egc_test )

Other 6 targets unchanged from input LB 0.902 submission.

============================================================================
GUARD RAILS
============================================================================

Same as Koopmans script:
  - If best α < 0.5 → physics dominating too much; warn
  - If best α > 0.99 → physics contributes < 1%; script says SKIP SUBMISSION

Additional Moss-specific guards:
  - If fitted k is negative or very large → warn (fit is broken)
  - If any Egc test prediction is ≤ 0.5 eV → clipped to 0.5, logged

============================================================================
INPUTS
============================================================================

  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_{0..4}.pkl.gz
      → Chemprop OOF matrix (5920 canons × 7 targets)
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds.pkl.gz
      → Chemprop test predictions (4133 canons × 7 targets)
  results/exp_bandgap_koopmans_postfit/submission.csv
      → CURRENT BEST (LB 0.902)
  ppp-round-2/{train,test}.csv

============================================================================
OUTPUTS  (under results/exp_moss_postfit_nc/)
============================================================================

  run.log              — k fit + α tuning + before/after Nc R² + LB estimate
  moss_summary.json    — fitted k, α, R² deltas, config
  submission.csv       — modified submission (Nc rows updated)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_moss_postfit_nc.py

Wall time: ~5 seconds.

============================================================================
EXPECTED
============================================================================

Chemprop Nc OOF R² is 0.8681 — middle-baseline (between ei 0.777 and egb 0.93).
This is roughly the "eea zone" where Koopmans got +0.008 R² lift.

  Best case (Moss holds well):     +0.001-0.003 → LB 0.903-0.905
  Middle case (partial fit):        +0.000-0.002 → LB 0.902-0.904
  Worst case (α → 1.0):             0            → LB 0.902 (unchanged)

Nc has only 153 test rows so mean-R² lift is bounded by (per-target Δ / 7).

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
EXP_NAME = "exp_moss_postfit_nc"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
INPUT_SUB_PATH = REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"
INPUT_SUB_LB = 0.902

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Moss rule: Nc² · Egc = k
EGC_FLOOR = 0.5   # eV — clip Egc predictions below this to prevent inf/nan

# Alpha grid (same as Koopmans)
ALPHA_GRID = np.arange(0.5, 1.001, 0.025)

# Guard rails
ALPHA_FLOOR = 0.5
ALPHA_CEILING = 0.99


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
# LOAD CHEMPROP OOF + TEST MATRICES
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_train_canons_and_y(log: logging.Logger) -> tuple[list[str], np.ndarray]:
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

    n_nc = int((~np.isnan(y_matrix[:, TARGET_IDX['nc']])).sum())
    n_egc = int((~np.isnan(y_matrix[:, TARGET_IDX['egc']])).sum())
    both = ~np.isnan(y_matrix[:, TARGET_IDX['nc']]) & ~np.isnan(y_matrix[:, TARGET_IDX['egc']])
    n_both = int(both.sum())
    log.info(f"  n_canon={len(canons)}   Nc labeled={n_nc}   Egc labeled={n_egc}   both={n_both}")
    return canons, y_matrix


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    n_canons = len(canons)
    oof = np.full((n_canons, 7), np.nan, dtype=np.float32)
    for k in range(5):
        with gzip.open(CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            fold_result = pickle.load(f)
        oof[fold_result["val_idxs"]] = fold_result["val_preds_avg"]
        log.info(f"  loaded fold {k}: {fold_result['val_preds_avg'].shape}  (val n={len(fold_result['val_idxs'])})")
    log.info(f"  Chemprop OOF matrix: {oof.shape}")
    return oof


def load_chemprop_test_matrix(log: logging.Logger) -> tuple[list[str], np.ndarray]:
    log.info("loading Chemprop refit test predictions...")
    with gzip.open(CHEMPROP_DIR / "refit_test_preds.pkl.gz", "rb") as f:
        cache = pickle.load(f)
    test_preds = cache["test_preds_avg"]
    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical(te)", ncols=100)}
    te["canon"] = te["smiles"].map(canon_map)
    test_canon_unique = te["canon"].drop_duplicates().tolist()
    assert len(test_canon_unique) == test_preds.shape[0]
    log.info(f"  Chemprop test predictions: {test_preds.shape}   n_test_canons={len(test_canon_unique)}")
    return test_canon_unique, test_preds


# ============================================================================
# FIT MOSS CONSTANT k ON CO-LABELED (Nc, Egc) TRAIN ROWS
# ============================================================================

def fit_moss_k(y_matrix: np.ndarray, log: logging.Logger) -> dict:
    """Fit k such that Nc² · Egc ≈ k on rows where both are labeled."""
    nc_idx  = TARGET_IDX["nc"]
    egc_idx = TARGET_IDX["egc"]
    mask = (~np.isnan(y_matrix[:, nc_idx])) & (~np.isnan(y_matrix[:, egc_idx]))
    nc = y_matrix[mask, nc_idx]
    egc = y_matrix[mask, egc_idx]
    n = int(mask.sum())

    k_products = nc ** 2 * egc
    k_median = float(np.median(k_products))
    k_mean   = float(np.mean(k_products))
    k_std    = float(np.std(k_products))
    k_min    = float(k_products.min())
    k_max    = float(k_products.max())

    log.info(f"[MOSS fit] n_co_labeled = {n}")
    log.info(f"[MOSS fit] Nc·Nc·Egc products: median={k_median:.4f}  mean={k_mean:.4f}  "
             f"std={k_std:.4f}  range=[{k_min:.3f}, {k_max:.3f}]")
    log.info(f"[MOSS fit] coefficient of variation (std/mean) = {k_std/k_mean:.3f}   "
             f"(smaller = tighter physics; >0.5 = weak law)")
    log.info(f"[MOSS fit] using k = median = {k_median:.4f}")

    # Sanity check: apply the fit back to the same rows and see how well Moss reproduces Nc
    nc_pred_moss = np.sqrt(k_median / np.clip(egc, EGC_FLOOR, None))
    r2_moss_on_fit_rows = float(r2_score(nc, nc_pred_moss))
    log.info(f"[MOSS fit] R² of Moss prediction on the 26 fit rows: {r2_moss_on_fit_rows:.4f}   "
             f"(sanity check — should be modest since k fits ONE global constant)")

    return {
        "n_co_labeled":       n,
        "k_median":           k_median,
        "k_mean":             k_mean,
        "k_std":              k_std,
        "k_range":            [k_min, k_max],
        "k_cov":              k_std / k_mean if k_mean > 0 else float("inf"),
        "k_used":             k_median,
        "r2_moss_on_fit":     r2_moss_on_fit_rows,
    }


# ============================================================================
# ALPHA TUNING (grid-search on Nc-labeled train rows using Chemprop OOF)
# ============================================================================

def tune_alpha_moss(k: float, oof: np.ndarray, y_matrix: np.ndarray,
                    log: logging.Logger) -> dict:
    nc_idx  = TARGET_IDX["nc"]
    egc_idx = TARGET_IDX["egc"]
    mask = ~np.isnan(y_matrix[:, nc_idx])
    n_labeled = int(mask.sum())

    y_true      = y_matrix[mask, nc_idx]
    own_pred    = oof[mask, nc_idx]
    egc_pred    = oof[mask, egc_idx]

    valid = ~(np.isnan(own_pred) | np.isnan(egc_pred))
    y_true = y_true[valid]; own_pred = own_pred[valid]; egc_pred = egc_pred[valid]

    # Clip Egc predictions to physical floor
    egc_clipped = np.clip(egc_pred, EGC_FLOOR, None)
    n_clipped = int((egc_pred < EGC_FLOOR).sum())
    if n_clipped > 0:
        log.info(f"[MOSS α tune] clipped {n_clipped} Egc OOF values to {EGC_FLOOR} eV floor "
                 f"(train-side, {100*n_clipped/len(egc_pred):.1f}%)")

    moss_pred = np.sqrt(k / egc_clipped)

    # Baseline (α=1) and pure-physics (α=0) R²
    r2_baseline  = float(r2_score(y_true, own_pred))
    r2_pure_moss = float(r2_score(y_true, moss_pred))
    log.info(f"[MOSS α tune] n_valid={len(y_true)}   "
             f"Chemprop Nc R²={r2_baseline:.4f}   pure-Moss R²={r2_pure_moss:.4f}")

    # Grid-search α
    best_r2, best_alpha = -np.inf, 1.0
    for alpha in ALPHA_GRID:
        blend = alpha * own_pred + (1 - alpha) * moss_pred
        r2 = float(r2_score(y_true, blend))
        if r2 > best_r2:
            best_r2, best_alpha = r2, float(alpha)

    delta = best_r2 - r2_baseline
    log.info(f"[MOSS α tune] best α = {best_alpha:.3f}   blend R² = {best_r2:.4f}   "
             f"Δ vs baseline = {delta:+.4f}")

    if best_alpha < ALPHA_FLOOR:
        log.warning(f"[MOSS α tune] α={best_alpha:.3f} < {ALPHA_FLOOR} — physics dominating, investigate")
    if best_alpha > ALPHA_CEILING:
        log.warning(f"[MOSS α tune] α={best_alpha:.3f} > {ALPHA_CEILING} — physics doesn't help, consider skip")

    return {
        "n_labeled": n_labeled,
        "n_valid":   int(len(y_true)),
        "n_egc_clipped": n_clipped,
        "r2_baseline":  r2_baseline,
        "r2_pure_moss": r2_pure_moss,
        "best_alpha":   best_alpha,
        "r2_blend":     best_r2,
        "delta_r2":     delta,
    }


# ============================================================================
# APPLY TO SUBMISSION (Nc rows only)
# ============================================================================

def apply_moss_to_submission(k: float, alpha: float,
                              test_canons: list[str], chemprop_test: np.ndarray,
                              log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    log.info(f"loading input submission: {INPUT_SUB_PATH}")
    sub = pd.read_csv(INPUT_SUB_PATH)
    log.info(f"  input submission: {sub.shape}  (LB reference: {INPUT_SUB_LB})")

    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in all_smi}
    te["canon"] = te["smiles"].map(canon_map)
    te = te[["id", "canon", "target_type"]]

    sub_full = sub.merge(te, on="id", how="left")
    assert sub_full["canon"].notna().all()

    canon_to_idx = {c: i for i, c in enumerate(test_canons)}
    egc_idx = TARGET_IDX["egc"]

    mask = sub_full["target_type"] == "nc"
    rows = sub_full[mask].copy()
    log.info(f"[MOSS apply] modifying {len(rows)} Nc test rows")

    canon_idx = np.array([canon_to_idx[c] for c in rows["canon"]])
    egc_test = chemprop_test[canon_idx, egc_idx]
    egc_test_clipped = np.clip(egc_test, EGC_FLOOR, None)
    n_clipped_test = int((egc_test < EGC_FLOOR).sum())
    if n_clipped_test > 0:
        log.info(f"[MOSS apply] clipped {n_clipped_test} test Egc values to {EGC_FLOOR} eV floor "
                 f"({100*n_clipped_test/len(egc_test):.1f}%)")

    nc_moss_test = np.sqrt(k / egc_test_clipped)
    own_test = rows["target"].values
    new_pred = alpha * own_test + (1 - alpha) * nc_moss_test

    diffs = np.abs(new_pred - own_test)
    log.info(f"[MOSS apply] mean |Δ|={diffs.mean():.4f}   max |Δ|={diffs.max():.4f}")
    log.info(f"[MOSS apply] new pred range: [{new_pred.min():.4f}, {new_pred.max():.4f}]   "
             f"(orig: [{own_test.min():.4f}, {own_test.max():.4f}])")

    sub_full.loc[mask, "target"] = new_pred
    out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    return out, {
        "n_rows":        int(len(rows)),
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff":  float(diffs.max()),
        "n_clipped_egc_test": n_clipped_test,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"Physics: Nc² · Egc ≈ k   →   Nc_moss = sqrt(k / Egc)")
    log.info(f"Input submission: {INPUT_SUB_PATH}  (LB {INPUT_SUB_LB})")
    log.info(f"Egc floor for clipping: {EGC_FLOOR} eV")
    log.info(f"Alpha grid: {ALPHA_GRID[0]:.3f} to {ALPHA_GRID[-1]:.3f} step {ALPHA_GRID[1]-ALPHA_GRID[0]:.3f}")

    t_start = time.time()

    # ---- Load Chemprop OOF matrix ----
    log.info("=" * 60)
    log.info("LOAD CHEMPROP OOF MATRIX")
    log.info("=" * 60)
    canons, y_matrix = load_train_canons_and_y(log)
    oof = load_chemprop_oof_matrix(canons, log)

    # ---- Fit Moss constant k ----
    log.info("=" * 60)
    log.info("FIT MOSS CONSTANT k")
    log.info("=" * 60)
    fit_result = fit_moss_k(y_matrix, log)
    k = fit_result["k_used"]

    # ---- Alpha tuning ----
    log.info("=" * 60)
    log.info("ALPHA TUNING FOR NC")
    log.info("=" * 60)
    alpha_result = tune_alpha_moss(k, oof, y_matrix, log)
    alpha = alpha_result["best_alpha"]

    # ---- Apply to submission ----
    log.info("=" * 60)
    log.info("APPLY TO INPUT SUBMISSION (Nc rows only)")
    log.info("=" * 60)
    test_canons, chemprop_test = load_chemprop_test_matrix(log)
    new_sub, diff_stats = apply_moss_to_submission(k, alpha, test_canons, chemprop_test, log)

    # ---- Write outputs ----
    sub_path = EXP_DIR / "submission.csv"
    new_sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}   rows={len(new_sub)}")

    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "input_submission":    str(INPUT_SUB_PATH),
            "input_lb_reference":  INPUT_SUB_LB,
            "chemprop_source":     str(CHEMPROP_DIR),
            "egc_floor_ev":        EGC_FLOOR,
            "alpha_grid":          [float(x) for x in ALPHA_GRID],
            "alpha_floor":         ALPHA_FLOOR,
            "alpha_ceiling":       ALPHA_CEILING,
        },
        "physics_recipe": "Nc² · Egc ≈ k    →    Nc_moss = sqrt(k / max(Egc, floor))",
        "moss_fit":       fit_result,
        "alpha_tuning":   alpha_result,
        "test_modification": diff_stats,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(EXP_DIR / "moss_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'moss_summary.json'}")

    # ---- Decision guidance ----
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Input submission LB (reference): {INPUT_SUB_LB}")
    log.info(f"Fitted k = {k:.4f}   α = {alpha:.3f}")
    log.info(f"Nc OOF R²: {alpha_result['r2_baseline']:.4f} → {alpha_result['r2_blend']:.4f}   "
             f"Δ = {alpha_result['delta_r2']:+.4f}")
    nc_lift_on_7 = alpha_result["delta_r2"] / 7
    log.info(f"Expected 7-target mean R² lift: {nc_lift_on_7:+.4f}   (one target out of seven)")

    if alpha > ALPHA_CEILING:
        log.warning("SKIP SUBMISSION: physics contributes < 1%. Save the sub slot.")
    elif alpha < ALPHA_FLOOR:
        log.warning("INVESTIGATE: physics dominates. Check k fit and pure-Moss R².")
    else:
        expected_lb = INPUT_SUB_LB + nc_lift_on_7
        log.info(f"SUBMIT: expected LB ≈ {expected_lb:.4f}   "
                 f"(range: {INPUT_SUB_LB:.3f} to {INPUT_SUB_LB + 0.003:.3f})")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
