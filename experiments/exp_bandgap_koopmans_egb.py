"""
exp_bandgap_koopmans_egb.py — 3-way physics blend for Egb, layered on the
                              LB 0.902 Koopmans submission.

============================================================================
WHY THIS EXISTS
============================================================================

The Koopmans post-processor (`exp_bandgap_koopmans_postfit.py`) modified 3
targets — Egc, Ei, Eea — via the Koopmans relation and got us LB 0.902.
Egb was untouched.

Egb is another bandgap target and has two natural physics tie-ins:
  1. **Koopmans** applied to Egb via Egb ≈ Egc ≈ Ei − Eea
     (transitively, since Egb ↔ Egc r=+0.93 is the tightest cross-target
      correlation in the training set)
  2. **Egb ≈ Egc** directly (empirical linear fit)

This script fits a 3-way NNLS blend on Chemprop OOF for Egb:
    y_egb ≈ w_own · Egb_pred + w_koop · (Ei_pred − Eea_pred) + w_egc · Egc_pred
    subject to w ≥ 0, sum(w) = 1

Then applies those weights to the current best submission's Egb rows,
using Chemprop's refit predictions for the physics terms (always
available since Chemprop is multitask).

**Important**: this modifies ONLY Egb rows in the submission. All other
targets (Eea, Ei, Eps, Egc, Nc, Tg) come through unchanged from the
Koopmans-applied submission (LB 0.902 baseline).

============================================================================
GAP-SAFE BY CONSTRUCTION
============================================================================

- No training happens. Chain-ext LGB, Chemprop 3-seed, NNLS blend weights,
  Koopmans α values — all UNCHANGED.
- NNLS with sum=1 constraint means we're finding a convex combination of
  three predictors. Worst case: w_own ≈ 1, no change to Egb.
- Guard rail: if w_own < 0.5 → warn (physics dominating too much, suspicious)
- Guard rail: if w_own > 0.99 → warn (physics doesn't help, skip submission)

============================================================================
INPUTS
============================================================================

  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_{0..4}.pkl.gz
      → 7-target Chemprop OOF matrix (5920 canons × 7 targets)
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds.pkl.gz
      → 7-target refit test predictions (4133 canons × 7 targets)
  results/exp_bandgap_koopmans_postfit/submission.csv
      → CURRENT BEST (LB 0.902) — modifies only Egb rows
  ppp-round-2/{train,test}.csv → canon lookups

============================================================================
OUTPUTS  (under results/exp_bandgap_koopmans_egb/)
============================================================================

  run.log              — NNLS weights + per-source R² + expected LB delta
  egb_summary.json     — machine-readable summary
  submission.csv       — modified submission (Egb rows updated, everything else identical)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_bandgap_koopmans_egb.py

Wall time: ~5 seconds.

============================================================================
EXPECTED
============================================================================

Chemprop Egb OOF R² is already strong (0.9305) so gain is bounded:
  - Best case (both physics terms contribute): +0.001-0.003 → LB 0.903-0.905
  - Middle case (only Egc-corr helps a bit):   +0.001       → LB 0.903
  - Worst case (physics doesn't help):         0            → LB 0.902 (unchanged)

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
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_bandgap_koopmans_egb"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
INPUT_SUB_PATH = REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"
INPUT_SUB_LB = 0.902  # for logging context

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Guard rails (in normalized-weight units, since w sum to 1 after NNLS-norm)
W_OWN_FLOOR = 0.5      # if best NNLS w_own < 0.5 → warn (physics dominating too much)
W_OWN_CEILING = 0.99   # if best w_own > 0.99 → physics didn't help; skip submission


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
# LOAD CHEMPROP OOF MATRIX (5920 canons × 7 targets)
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
    log.info(f"  n_canon={len(canons)}   y shape={y_matrix.shape}   "
             f"NaN frac={100*np.isnan(y_matrix).mean():.1f}%")
    log.info(f"  Egb labeled train rows: {int((~np.isnan(y_matrix[:, TARGET_IDX['egb']])).sum())}")
    return canons, y_matrix


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    n_canons = len(canons)
    oof = np.full((n_canons, 7), np.nan, dtype=np.float32)
    for k in range(5):
        cp_path = CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz"
        with gzip.open(cp_path, "rb") as f:
            fold_result = pickle.load(f)
        val_idxs = fold_result["val_idxs"]
        val_preds_avg = fold_result["val_preds_avg"]
        oof[val_idxs] = val_preds_avg
        log.info(f"  loaded fold {k}: {val_preds_avg.shape}  (val n={len(val_idxs)})")
    n_missing = int(np.isnan(oof).all(axis=1).sum())
    log.info(f"  Chemprop OOF matrix: {oof.shape}   {n_missing} canons with no predictions")
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
    assert len(test_canon_unique) == test_preds.shape[0], (
        f"test canon count mismatch: {len(test_canon_unique)} vs {test_preds.shape[0]}"
    )
    log.info(f"  Chemprop test predictions: {test_preds.shape}   n_test_canons={len(test_canon_unique)}")
    return test_canon_unique, test_preds


# ============================================================================
# 3-WAY NNLS FIT FOR EGB
# ============================================================================

def fit_egb_three_way(oof: np.ndarray, y_matrix: np.ndarray, log: logging.Logger) -> dict:
    """NNLS on 3 predictors: own_egb + (ei - eea) [Koopmans] + egc [correlation]."""
    egb_idx = TARGET_IDX["egb"]
    ei_idx  = TARGET_IDX["ei"]
    eea_idx = TARGET_IDX["eea"]
    egc_idx = TARGET_IDX["egc"]

    mask = ~np.isnan(y_matrix[:, egb_idx])
    n_labeled = int(mask.sum())

    y_true    = y_matrix[mask, egb_idx]
    own_pred  = oof[mask, egb_idx]
    phys_koop = oof[mask, ei_idx] - oof[mask, eea_idx]
    phys_egc  = oof[mask, egc_idx]

    valid = ~(np.isnan(own_pred) | np.isnan(phys_koop) | np.isnan(phys_egc))
    n_valid = int(valid.sum())
    y_true = y_true[valid]
    own_pred = own_pred[valid]
    phys_koop = phys_koop[valid]
    phys_egc = phys_egc[valid]

    log.info(f"[EGB 3-way] n_labeled={n_labeled}  n_valid={n_valid}")

    # Per-source R² for context
    r2_own       = float(r2_score(y_true, own_pred))
    r2_koop      = float(r2_score(y_true, phys_koop))
    r2_egc       = float(r2_score(y_true, phys_egc))
    log.info(f"[EGB 3-way] individual predictor R²:")
    log.info(f"  own (Chemprop Egb): {r2_own:.4f}")
    log.info(f"  phys Koopmans (Ei-Eea): {r2_koop:.4f}")
    log.info(f"  phys Egc-correlation:   {r2_egc:.4f}")

    # NNLS
    A = np.vstack([own_pred, phys_koop, phys_egc]).T
    x, _ = nnls(A, y_true)
    w_own_raw, w_koop_raw, w_egc_raw = float(x[0]), float(x[1]), float(x[2])
    s = w_own_raw + w_koop_raw + w_egc_raw
    log.info(f"[EGB 3-way] NNLS raw weights:")
    log.info(f"  w_own={w_own_raw:.4f}  w_koop={w_koop_raw:.4f}  w_egc={w_egc_raw:.4f}  (sum={s:.4f})")

    if s < 1e-9:
        log.warning("[EGB 3-way] NNLS collapsed to zero — falling back to w_own=1")
        w_own, w_koop, w_egc = 1.0, 0.0, 0.0
    else:
        w_own = w_own_raw / s
        w_koop = w_koop_raw / s
        w_egc = w_egc_raw / s

    log.info(f"[EGB 3-way] normalized weights (sum=1):")
    log.info(f"  w_own={w_own:.4f}  w_koop={w_koop:.4f}  w_egc={w_egc:.4f}")

    # Blend R²
    blend_pred = w_own * own_pred + w_koop * phys_koop + w_egc * phys_egc
    r2_blend = float(r2_score(y_true, blend_pred))
    delta = r2_blend - r2_own
    log.info(f"[EGB 3-way] blend R² = {r2_blend:.4f}   Δ vs own (Chemprop only) = {delta:+.4f}")

    # Guard rails
    if w_own < W_OWN_FLOOR:
        log.warning(f"[EGB 3-way] w_own={w_own:.3f} < {W_OWN_FLOOR} — physics dominating; investigate")
    if w_own > W_OWN_CEILING:
        log.warning(f"[EGB 3-way] w_own={w_own:.3f} > {W_OWN_CEILING} — physics contributes almost nothing; consider skipping submission")

    return {
        "n_labeled": n_labeled,
        "n_valid":   n_valid,
        "r2_own_alone":       r2_own,
        "r2_phys_koop_alone": r2_koop,
        "r2_phys_egc_alone":  r2_egc,
        "w_own":              w_own,
        "w_koop":             w_koop,
        "w_egc":              w_egc,
        "w_own_raw":          w_own_raw,
        "w_koop_raw":         w_koop_raw,
        "w_egc_raw":          w_egc_raw,
        "r2_blend":           r2_blend,
        "delta_r2":           delta,
    }


# ============================================================================
# APPLY TO SUBMISSION (Egb rows only)
# ============================================================================

def apply_to_submission(weights: dict, test_canons: list[str],
                        chemprop_test: np.ndarray, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Modify Egb rows in the input submission. Return new sub + diff stats."""
    log.info(f"loading input submission: {INPUT_SUB_PATH}")
    sub = pd.read_csv(INPUT_SUB_PATH)
    log.info(f"  input submission: {sub.shape}  (LB reference: {INPUT_SUB_LB})")

    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in all_smi}
    te["canon"] = te["smiles"].map(canon_map)
    te = te[["id", "canon", "target_type"]]

    sub_full = sub.merge(te, on="id", how="left")
    assert sub_full["canon"].notna().all(), "some ids missing canon"

    canon_to_idx = {c: i for i, c in enumerate(test_canons)}
    ei_idx  = TARGET_IDX["ei"]
    eea_idx = TARGET_IDX["eea"]
    egc_idx = TARGET_IDX["egc"]

    mask = sub_full["target_type"] == "egb"
    rows = sub_full[mask].copy()
    log.info(f"[EGB apply] modifying {len(rows)} Egb test rows")

    canon_idx = np.array([canon_to_idx[c] for c in rows["canon"]])
    chem_ei  = chemprop_test[canon_idx, ei_idx]
    chem_eea = chemprop_test[canon_idx, eea_idx]
    chem_egc = chemprop_test[canon_idx, egc_idx]
    phys_koop_test = chem_ei - chem_eea
    phys_egc_test  = chem_egc

    own_test = rows["target"].values
    new_pred = (weights["w_own"] * own_test
                + weights["w_koop"] * phys_koop_test
                + weights["w_egc"] * phys_egc_test)

    diffs = np.abs(new_pred - own_test)
    log.info(f"[EGB apply] mean |Δ|={diffs.mean():.4f}   max |Δ|={diffs.max():.4f}")
    log.info(f"[EGB apply] new pred range: [{new_pred.min():.3f}, {new_pred.max():.3f}]   "
             f"(orig: [{own_test.min():.3f}, {own_test.max():.3f}])")

    sub_full.loc[mask, "target"] = new_pred
    out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    diff_stats = {
        "n_rows":        int(len(rows)),
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff":  float(diffs.max()),
    }
    return out, diff_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"Input submission: {INPUT_SUB_PATH}  (LB {INPUT_SUB_LB})")
    log.info(f"Chemprop source:  {CHEMPROP_DIR}")
    log.info(f"Guard rails: w_own must be in [{W_OWN_FLOOR}, {W_OWN_CEILING}]")

    t_start = time.time()

    # ---- Load Chemprop OOF matrix ----
    log.info("=" * 60)
    log.info("LOAD CHEMPROP OOF MATRIX")
    log.info("=" * 60)
    canons, y_matrix = load_train_canons_and_y(log)
    oof = load_chemprop_oof_matrix(canons, log)

    # ---- Fit 3-way NNLS ----
    log.info("=" * 60)
    log.info("3-WAY NNLS FIT FOR EGB")
    log.info("=" * 60)
    weights = fit_egb_three_way(oof, y_matrix, log)

    # ---- Load Chemprop test preds ----
    log.info("=" * 60)
    log.info("APPLY TO INPUT SUBMISSION (Egb rows only)")
    log.info("=" * 60)
    test_canons, chemprop_test = load_chemprop_test_matrix(log)

    # ---- Apply to submission ----
    new_sub, diff_stats = apply_to_submission(weights, test_canons, chemprop_test, log)

    # ---- Write outputs ----
    sub_path = EXP_DIR / "submission.csv"
    new_sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}   rows={len(new_sub)}")

    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "input_submission": str(INPUT_SUB_PATH),
            "input_lb_reference": INPUT_SUB_LB,
            "chemprop_source": str(CHEMPROP_DIR),
            "w_own_floor": W_OWN_FLOOR,
            "w_own_ceiling": W_OWN_CEILING,
        },
        "physics_recipe": "Egb ≈ w_own·Egb_pred + w_koop·(Ei_pred - Eea_pred) + w_egc·Egc_pred",
        "nnls_fit": weights,
        "test_modification": diff_stats,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(EXP_DIR / "egb_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'egb_summary.json'}")

    # ---- Decision guidance ----
    log.info("=" * 60)
    log.info("DECISION GUIDANCE")
    log.info("=" * 60)
    log.info(f"Input submission LB (reference): {INPUT_SUB_LB}")
    log.info(f"NNLS weights (w_own, w_koop, w_egc): "
             f"({weights['w_own']:.3f}, {weights['w_koop']:.3f}, {weights['w_egc']:.3f})")
    log.info(f"OOF R² on Egb: {weights['r2_own_alone']:.4f} → {weights['r2_blend']:.4f}   "
             f"Δ = {weights['delta_r2']:+.4f}")
    egb_lift_on_7 = weights["delta_r2"] / 7
    log.info(f"Expected 7-target mean R² lift: {egb_lift_on_7:+.4f}   "
             f"(one target out of seven)")

    if weights["w_own"] > W_OWN_CEILING:
        log.warning("SKIP SUBMISSION: physics contributes < 1%. Save the sub slot.")
    elif weights["w_own"] < W_OWN_FLOOR:
        log.warning("INVESTIGATE: physics dominates. Verify OOF R² breakdown makes sense.")
    else:
        log.info(f"SUBMIT: expected LB {INPUT_SUB_LB + egb_lift_on_7:.4f} "
                 f"(best-case up to +0.003 → {INPUT_SUB_LB + 0.003:.3f})")
    log.info(f"wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
