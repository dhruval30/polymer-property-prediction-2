"""
exp_multisignal_nnls_post.py — multi-signal NNLS post-processor for Nc, EPS, Egc,
                              layered on the LB 0.902 Koopmans submission.

============================================================================
WHY THIS EXISTS
============================================================================

Koopmans (LB 0.902) applied α-tuned single-physics blends to egc/ei/eea.
Egb 3-way (LB 0.902 flat) and Moss Nc (LB 0.901) confirmed a pattern:
  * OOF ΔR² ≥ 0.015 (sum) tends to convert to positive LB
  * OOF ΔR² < 0.010 (sum) usually noise-dominated on LB

This script goes bigger: 3 targets simultaneously via NNLS over multiple
physics signals. Threshold-gated submission.

============================================================================
THE APPROACH
============================================================================

Per target, fit NNLS on Chemprop OOF against 3-4 signal predictors:

  **Nc** (3 signals — no Maxwell since LGB Maxwell base already has it):
    - own_Nc                  (Chemprop OOF for Nc)
    - linear(Egb → Nc)        (fitted a + b·Egb on co-labeled train)
    - linear(Egc → Nc)        (fitted a + b·Egc on co-labeled train)

  **EPS** (4 signals — Maxwell IS new here since applied to blend's Nc):
    - own_EPS                 (Chemprop OOF for EPS)
    - Maxwell(Nc)             (a·Nc² + b, fit on Nc/EPS co-labeled train)
    - linear(Egb → EPS)
    - linear(Egc → EPS)

  **Egc** (3 signals — Egb-from-Egc is the KEY untried lever with 82 rows r=+0.93):
    - own_Egc                 (Chemprop OOF for Egc)
    - Koopmans(Ei − Eea)      (already used in LB 0.902 sub with α=0.9;
                               NNLS re-decides its weight here)
    - linear(Egb → Egc)

NNLS with weights ≥ 0 and sum = 1 (normalized after). If a signal is weak
on OOF, NNLS drives its weight to zero.

============================================================================
THRESHOLD-GATED SUBMISSION
============================================================================

Based on the actual OOF→LB translation ratio from prior experiments:
  Koopmans (3 targets):  OOF ΔR² sum +0.028 → LB +0.005   (18% ratio)
  Egb 3-way:              +0.001         → 0             (0%)
  Moss Nc:                +0.0045        → -0.001        (-22%)

Rule for this script:
  - sum ΔR² > 0.015 → SUBMIT   (expect LB +0.002 to +0.004 → 0.904-0.906)
  - sum ΔR² ∈ [0.008, 0.015] → BORDERLINE (user decides)
  - sum ΔR² < 0.008 → SKIP     (will not reliably transfer to LB)

Guard rails per target (same as Koopmans):
  - w_own < 0.5 → warn (physics dominating; verify sanity)
  - w_own > 0.99 → warn (physics contributes < 1%; no LB effect from this target)

============================================================================
INPUTS
============================================================================

  results/exp_chemprop_multitask_cpu_3seed/checkpoint_fold_{0..4}.pkl.gz
  results/exp_chemprop_multitask_cpu_3seed/refit_test_preds.pkl.gz
  results/exp_bandgap_koopmans_postfit/submission.csv   (LB 0.902)
  ppp-round-2/{train,test}.csv

============================================================================
OUTPUTS  (under results/exp_multisignal_nnls_post/)
============================================================================

  run.log                  — linear fits + per-target NNLS + decision
  multisignal_summary.json — machine-readable weights, R²s, deltas
  submission.csv           — modified submission (Nc, EPS, Egc rows only if SUBMIT)

Written REGARDLESS of decision — user inspects log to decide whether to ship.

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_multisignal_nnls_post.py

Wall time: ~5-10 seconds.

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
EXP_NAME = "exp_multisignal_nnls_post"
EXP_DIR = REPO / "results" / EXP_NAME

CHEMPROP_DIR = REPO / "results" / "exp_chemprop_multitask_cpu_3seed"
INPUT_SUB_PATH = REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"
INPUT_SUB_LB = 0.902

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Submission decision thresholds
DELTA_SUBMIT_STRONG     = 0.015
DELTA_SUBMIT_BORDERLINE = 0.008

# Per-target guard rails
W_OWN_FLOOR   = 0.5
W_OWN_CEILING = 0.99


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
# LOAD CHEMPROP MATRICES + LABELS
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_train_canons_and_y(log: logging.Logger) -> tuple[list[str], np.ndarray]:
    log.info("loading train.csv...")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    all_smi = tr["smiles"].unique()
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical(tr)", ncols=100)}
    tr["canon"] = tr["smiles"].map(canon_map)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean")))
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[list(TARGETS)]
    canons = wide.index.tolist()
    y = wide.values.astype(np.float32)
    log.info(f"  n_canon={len(canons)}   y shape={y.shape}")
    for t in TARGETS:
        log.info(f"    {t:>4s}: {int((~np.isnan(y[:, TARGET_IDX[t]])).sum()):>5d} labeled")
    return canons, y


def load_chemprop_oof_matrix(canons: list[str], log: logging.Logger) -> np.ndarray:
    n_canons = len(canons)
    oof = np.full((n_canons, 7), np.nan, dtype=np.float32)
    for k in range(5):
        with gzip.open(CHEMPROP_DIR / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            r = pickle.load(f)
        oof[r["val_idxs"]] = r["val_preds_avg"]
        log.info(f"  loaded fold {k}: {r['val_preds_avg'].shape}")
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
    log.info(f"  Chemprop test predictions: {test_preds.shape}")
    return test_canon_unique, test_preds


# ============================================================================
# LINEAR FITS ON CO-LABELED TRAIN ROWS
# ============================================================================

def fit_linear(y: np.ndarray, src_target: str, dst_target: str,
               log: logging.Logger) -> tuple[float, float, int, float]:
    """Fit dst = a + b · src on train rows where both are labeled.
    Returns (a, b, n_rows, in_sample_R²)."""
    src_idx = TARGET_IDX[src_target]
    dst_idx = TARGET_IDX[dst_target]
    mask = (~np.isnan(y[:, src_idx])) & (~np.isnan(y[:, dst_idx]))
    n = int(mask.sum())
    x_vals = y[mask, src_idx].astype(np.float64)
    y_vals = y[mask, dst_idx].astype(np.float64)
    # Simple OLS: y = a + b·x
    if n < 3:
        log.warning(f"[linear {src_target}→{dst_target}] only n={n} rows — falling back to (a=y.mean, b=0)")
        return float(y_vals.mean()) if n > 0 else 0.0, 0.0, n, 0.0
    b, a = np.polyfit(x_vals, y_vals, deg=1)   # returns (slope, intercept)
    y_pred = a + b * x_vals
    r2 = float(r2_score(y_vals, y_pred))
    log.info(f"[linear {src_target:>3s}→{dst_target:>3s}] n={n:>4d}   a={a:+.4f}   b={b:+.4f}   "
             f"in-sample R²={r2:.4f}")
    return float(a), float(b), n, r2


def fit_maxwell_eps(y: np.ndarray, log: logging.Logger) -> tuple[float, float, int, float]:
    """Fit EPS = a·Nc² + b on co-labeled (Nc, EPS) train rows."""
    nc_idx  = TARGET_IDX["nc"]
    eps_idx = TARGET_IDX["eps"]
    mask = (~np.isnan(y[:, nc_idx])) & (~np.isnan(y[:, eps_idx]))
    n = int(mask.sum())
    nc = y[mask, nc_idx].astype(np.float64)
    eps = y[mask, eps_idx].astype(np.float64)
    x = nc ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, eps, rcond=None)
    a, b = float(a), float(b)
    eps_pred = a * x + b
    r2 = float(r2_score(eps, eps_pred))
    log.info(f"[maxwell EPS from Nc²] n={n:>4d}   a={a:+.4f}   b={b:+.4f}   in-sample R²={r2:.4f}")
    return a, b, n, r2


# ============================================================================
# PER-TARGET NNLS
# ============================================================================

def fit_nnls_target(target: str, signal_specs: list[tuple[str, np.ndarray]],
                    y_true: np.ndarray, log: logging.Logger) -> dict:
    """Fit NNLS on stack of signal arrays. Returns weights, per-signal R², blend R²."""
    names = [s[0] for s in signal_specs]
    arrays = [s[1] for s in signal_specs]

    # Restrict to rows where all signals + y_true are finite
    stacked = np.stack(arrays, axis=1)              # (n, k)
    valid = ~np.isnan(stacked).any(axis=1) & ~np.isnan(y_true)
    n_valid = int(valid.sum())
    log.info(f"[NNLS {target}] n_valid rows: {n_valid}   n_signals: {len(names)}")

    X = stacked[valid]
    y = y_true[valid]

    # Per-signal R²
    per_r2 = {}
    for i, name in enumerate(names):
        r2 = float(r2_score(y, X[:, i]))
        per_r2[name] = r2
        log.info(f"[NNLS {target}] signal '{name:>18s}':  R² alone = {r2:+.4f}")

    # NNLS
    weights_raw, _ = nnls(X, y)
    s = float(weights_raw.sum())
    if s < 1e-9:
        log.warning(f"[NNLS {target}] NNLS collapsed to zero — falling back to w_own=1.0")
        weights = np.zeros(len(names), dtype=np.float64)
        weights[0] = 1.0   # assume first signal is 'own'
    else:
        weights = weights_raw / s

    log.info(f"[NNLS {target}] NNLS raw weights (sum={s:.4f}):")
    for name, w in zip(names, weights_raw):
        log.info(f"    {name:>18s} = {w:.4f}")
    log.info(f"[NNLS {target}] normalized weights (sum=1):")
    for name, w in zip(names, weights):
        log.info(f"    {name:>18s} = {w:.4f}")

    # Blend R²
    blend = (X @ weights)
    r2_blend = float(r2_score(y, blend))
    r2_own = per_r2[names[0]]
    delta = r2_blend - r2_own
    log.info(f"[NNLS {target}] blend R² = {r2_blend:.4f}   Δ vs own = {delta:+.4f}")

    # Guard rail
    w_own = float(weights[0])
    if w_own < W_OWN_FLOOR:
        log.warning(f"[NNLS {target}] w_own={w_own:.3f} < {W_OWN_FLOOR} — physics dominating; verify sanity")
    if w_own > W_OWN_CEILING:
        log.warning(f"[NNLS {target}] w_own={w_own:.3f} > {W_OWN_CEILING} — physics contribution < 1%")

    return {
        "names":          names,
        "weights":        [float(w) for w in weights],
        "weights_raw":    [float(w) for w in weights_raw],
        "per_signal_r2":  per_r2,
        "r2_own":         r2_own,
        "r2_blend":       r2_blend,
        "delta_r2":       delta,
        "n_valid":        n_valid,
    }


# ============================================================================
# BUILD SIGNAL VECTORS (per target)
# ============================================================================

def build_nc_signals_oof(oof: np.ndarray, coefs: dict) -> list[tuple[str, np.ndarray]]:
    """3 signals for Nc: own + linear(Egb) + linear(Egc)."""
    a_eb, b_eb = coefs["nc_from_egb"]["a"], coefs["nc_from_egb"]["b"]
    a_ec, b_ec = coefs["nc_from_egc"]["a"], coefs["nc_from_egc"]["b"]
    egb = oof[:, TARGET_IDX["egb"]].astype(np.float64)
    egc = oof[:, TARGET_IDX["egc"]].astype(np.float64)
    return [
        ("own_Nc",           oof[:, TARGET_IDX["nc"]].astype(np.float64)),
        ("linear(Egb→Nc)",   a_eb + b_eb * egb),
        ("linear(Egc→Nc)",   a_ec + b_ec * egc),
    ]


def build_eps_signals_oof(oof: np.ndarray, coefs: dict) -> list[tuple[str, np.ndarray]]:
    """4 signals for EPS: own + Maxwell(Nc) + linear(Egb) + linear(Egc)."""
    a_max, b_max = coefs["eps_maxwell"]["a"], coefs["eps_maxwell"]["b"]
    a_eb, b_eb   = coefs["eps_from_egb"]["a"], coefs["eps_from_egb"]["b"]
    a_ec, b_ec   = coefs["eps_from_egc"]["a"], coefs["eps_from_egc"]["b"]
    nc  = oof[:, TARGET_IDX["nc"]].astype(np.float64)
    egb = oof[:, TARGET_IDX["egb"]].astype(np.float64)
    egc = oof[:, TARGET_IDX["egc"]].astype(np.float64)
    return [
        ("own_EPS",           oof[:, TARGET_IDX["eps"]].astype(np.float64)),
        ("maxwell(Nc²)",      a_max * (nc ** 2) + b_max),
        ("linear(Egb→EPS)",   a_eb + b_eb * egb),
        ("linear(Egc→EPS)",   a_ec + b_ec * egc),
    ]


def build_egc_signals_oof(oof: np.ndarray, coefs: dict) -> list[tuple[str, np.ndarray]]:
    """3 signals for Egc: own + Koopmans(Ei-Eea) + linear(Egb)."""
    a_eb, b_eb = coefs["egc_from_egb"]["a"], coefs["egc_from_egb"]["b"]
    ei  = oof[:, TARGET_IDX["ei"]].astype(np.float64)
    eea = oof[:, TARGET_IDX["eea"]].astype(np.float64)
    egb = oof[:, TARGET_IDX["egb"]].astype(np.float64)
    return [
        ("own_Egc",             oof[:, TARGET_IDX["egc"]].astype(np.float64)),
        ("koopmans(Ei-Eea)",    ei - eea),
        ("linear(Egb→Egc)",     a_eb + b_eb * egb),
    ]


# ============================================================================
# APPLY WEIGHTS TO SUBMISSION
# ============================================================================

def apply_weights_to_test(all_weights: dict, test_canons: list[str],
                           chemprop_test: np.ndarray, coefs: dict,
                           log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    log.info(f"loading input submission: {INPUT_SUB_PATH}")
    sub = pd.read_csv(INPUT_SUB_PATH)
    log.info(f"  input submission: {sub.shape}   (LB reference: {INPUT_SUB_LB})")

    te = pd.read_csv(DATA_DIR / "test.csv")
    all_smi = te["smiles"].unique()
    canon_map = {s: canonical(s) for s in all_smi}
    te["canon"] = te["smiles"].map(canon_map)
    te = te[["id", "canon", "target_type"]]
    sub_full = sub.merge(te, on="id", how="left")
    assert sub_full["canon"].notna().all()

    canon_to_idx = {c: i for i, c in enumerate(test_canons)}
    diff_stats = {}

    for target in ("nc", "eps", "egc"):
        weights_info = all_weights[target]
        weights = np.array(weights_info["weights"])
        mask = sub_full["target_type"] == target
        rows = sub_full[mask].copy()
        n = len(rows)
        canon_idx = np.array([canon_to_idx[c] for c in rows["canon"]])
        own_test = rows["target"].values.astype(np.float64)

        # Build test signals (parallel to OOF signal spec)
        if target == "nc":
            a_eb, b_eb = coefs["nc_from_egb"]["a"], coefs["nc_from_egb"]["b"]
            a_ec, b_ec = coefs["nc_from_egc"]["a"], coefs["nc_from_egc"]["b"]
            egb = chemprop_test[canon_idx, TARGET_IDX["egb"]].astype(np.float64)
            egc = chemprop_test[canon_idx, TARGET_IDX["egc"]].astype(np.float64)
            signals = [own_test, a_eb + b_eb * egb, a_ec + b_ec * egc]
        elif target == "eps":
            a_max, b_max = coefs["eps_maxwell"]["a"], coefs["eps_maxwell"]["b"]
            a_eb, b_eb   = coefs["eps_from_egb"]["a"], coefs["eps_from_egb"]["b"]
            a_ec, b_ec   = coefs["eps_from_egc"]["a"], coefs["eps_from_egc"]["b"]
            nc  = chemprop_test[canon_idx, TARGET_IDX["nc"]].astype(np.float64)
            egb = chemprop_test[canon_idx, TARGET_IDX["egb"]].astype(np.float64)
            egc = chemprop_test[canon_idx, TARGET_IDX["egc"]].astype(np.float64)
            signals = [own_test, a_max * (nc ** 2) + b_max,
                       a_eb + b_eb * egb, a_ec + b_ec * egc]
        elif target == "egc":
            a_eb, b_eb = coefs["egc_from_egb"]["a"], coefs["egc_from_egb"]["b"]
            ei  = chemprop_test[canon_idx, TARGET_IDX["ei"]].astype(np.float64)
            eea = chemprop_test[canon_idx, TARGET_IDX["eea"]].astype(np.float64)
            egb = chemprop_test[canon_idx, TARGET_IDX["egb"]].astype(np.float64)
            signals = [own_test, ei - eea, a_eb + b_eb * egb]

        stacked = np.stack(signals, axis=1)
        new_pred = stacked @ weights

        diffs = np.abs(new_pred - own_test)
        log.info(f"[APPLY {target}] n={n}  mean |Δ|={diffs.mean():.4f}  max |Δ|={diffs.max():.4f}   "
                 f"new range=[{new_pred.min():.3f}, {new_pred.max():.3f}]  "
                 f"(orig: [{own_test.min():.3f}, {own_test.max():.3f}])")

        sub_full.loc[mask, "target"] = new_pred
        diff_stats[target] = {
            "n_rows":        int(n),
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff":  float(diffs.max()),
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
    log.info(f"Input submission: {INPUT_SUB_PATH}  (LB {INPUT_SUB_LB})")
    log.info(f"Chemprop source:  {CHEMPROP_DIR}")
    log.info(f"Decision thresholds: sum ΔR² > {DELTA_SUBMIT_STRONG} → SUBMIT   "
             f"[{DELTA_SUBMIT_BORDERLINE}, {DELTA_SUBMIT_STRONG}] → BORDERLINE   "
             f"< {DELTA_SUBMIT_BORDERLINE} → SKIP")
    log.info(f"Per-target guard rails: w_own in [{W_OWN_FLOOR}, {W_OWN_CEILING}]")

    t_start = time.time()

    # ---- Load Chemprop OOF + y ----
    log.info("=" * 60)
    log.info("LOAD CHEMPROP OOF + LABELS")
    log.info("=" * 60)
    canons, y = load_train_canons_and_y(log)
    oof = load_chemprop_oof_matrix(canons, log)

    # ---- Fit all linear regressions + Maxwell ----
    log.info("=" * 60)
    log.info("FIT LINEAR REGRESSIONS ON CO-LABELED TRAIN ROWS")
    log.info("=" * 60)

    coefs = {}
    # For Nc
    a, b, n, r2 = fit_linear(y, "egb", "nc", log)
    coefs["nc_from_egb"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}
    a, b, n, r2 = fit_linear(y, "egc", "nc", log)
    coefs["nc_from_egc"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}
    # For EPS
    a, b, n, r2 = fit_maxwell_eps(y, log)
    coefs["eps_maxwell"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}
    a, b, n, r2 = fit_linear(y, "egb", "eps", log)
    coefs["eps_from_egb"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}
    a, b, n, r2 = fit_linear(y, "egc", "eps", log)
    coefs["eps_from_egc"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}
    # For Egc
    a, b, n, r2 = fit_linear(y, "egb", "egc", log)
    coefs["egc_from_egb"] = {"a": a, "b": b, "n_rows": n, "in_sample_r2": r2}

    # ---- Fit NNLS per target on Chemprop OOF ----
    log.info("=" * 60)
    log.info("NNLS PER TARGET (on Chemprop OOF signal matrices)")
    log.info("=" * 60)

    all_weights = {}
    nc_signals  = build_nc_signals_oof(oof, coefs)
    eps_signals = build_eps_signals_oof(oof, coefs)
    egc_signals = build_egc_signals_oof(oof, coefs)

    log.info("-" * 60)
    all_weights["nc"]  = fit_nnls_target("nc",  nc_signals,  y[:, TARGET_IDX["nc"]],  log)
    log.info("-" * 60)
    all_weights["eps"] = fit_nnls_target("eps", eps_signals, y[:, TARGET_IDX["eps"]], log)
    log.info("-" * 60)
    all_weights["egc"] = fit_nnls_target("egc", egc_signals, y[:, TARGET_IDX["egc"]], log)

    # ---- Decision ----
    log.info("=" * 60)
    log.info("DECISION")
    log.info("=" * 60)
    delta_sum = sum(all_weights[t]["delta_r2"] for t in ("nc", "eps", "egc"))
    log.info(f"per-target ΔR²:")
    for t in ("nc", "eps", "egc"):
        log.info(f"  {t:>3s}:  {all_weights[t]['delta_r2']:+.4f}   "
                 f"(w_own = {all_weights[t]['weights'][0]:.3f})")
    log.info(f"SUM ΔR² across 3 targets: {delta_sum:+.4f}")
    expected_lb_lift = delta_sum / 7
    log.info(f"expected 7-target mean R² lift: {expected_lb_lift:+.4f}")

    if delta_sum > DELTA_SUBMIT_STRONG:
        decision = "SUBMIT"
        log.info(f"✅ SUBMIT: sum ΔR² > {DELTA_SUBMIT_STRONG} threshold. "
                 f"Expected LB {INPUT_SUB_LB + max(0.001, expected_lb_lift):.4f}-{INPUT_SUB_LB + 0.004:.4f}")
    elif delta_sum > DELTA_SUBMIT_BORDERLINE:
        decision = "BORDERLINE"
        log.info(f"⚠️  BORDERLINE: sum ΔR² in [{DELTA_SUBMIT_BORDERLINE}, {DELTA_SUBMIT_STRONG}]. "
                 f"Expected LB {INPUT_SUB_LB - 0.001:.3f}-{INPUT_SUB_LB + 0.002:.3f}. "
                 f"User decides.")
    else:
        decision = "SKIP"
        log.info(f"❌ SKIP: sum ΔR² < {DELTA_SUBMIT_BORDERLINE}. Signal too weak to reliably transfer to LB. "
                 f"Recommendation: don't burn a sub slot.")

    # ---- Apply to submission (always write, user decides based on log) ----
    log.info("=" * 60)
    log.info("APPLY WEIGHTS TO SUBMISSION (writing regardless of decision)")
    log.info("=" * 60)
    test_canons, chemprop_test = load_chemprop_test_matrix(log)
    new_sub, diff_stats = apply_weights_to_test(all_weights, test_canons, chemprop_test, coefs, log)

    sub_path = EXP_DIR / "submission.csv"
    new_sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}   rows={len(new_sub)}")

    # ---- Write summary ----
    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "input_submission":    str(INPUT_SUB_PATH),
            "input_lb_reference":  INPUT_SUB_LB,
            "chemprop_source":     str(CHEMPROP_DIR),
            "delta_submit_strong": DELTA_SUBMIT_STRONG,
            "delta_submit_borderline": DELTA_SUBMIT_BORDERLINE,
            "w_own_floor":         W_OWN_FLOOR,
            "w_own_ceiling":       W_OWN_CEILING,
        },
        "linear_fits": coefs,
        "nnls_per_target": {t: all_weights[t] for t in ("nc", "eps", "egc")},
        "sum_delta_r2": delta_sum,
        "expected_lb_lift": expected_lb_lift,
        "decision": decision,
        "test_modification": diff_stats,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(EXP_DIR / "multisignal_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'multisignal_summary.json'}")

    log.info(f"wall time: {time.time()-t_start:.1f}s")
    log.info("=" * 60)
    log.info(f"FINAL DECISION: {decision}   |   submission ready at {sub_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
