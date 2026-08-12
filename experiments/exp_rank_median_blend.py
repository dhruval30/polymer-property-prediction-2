"""
exp_rank_median_blend.py — median blend of top LB submissions.

For each test row, compute the median prediction across N submission CSVs.
Rationale: for 3 subs, median = middle-ranked prediction (rank-median).
Robust to per-source outlier predictions on individual test rows.
Different from NNLS mean-blending which minimizes squared error.

Approach: try multiple sub combinations, pick the one with tightest
median-to-best-source deviation (proxy for stable consensus). Also emit
diagnostics per source so user can decide.

Inputs (configurable):
  Any list of submission.csv files (same 4940 rows, same id column)

Output:
  results/exp_rank_median_blend/
      run.log
      submission_median_top3.csv          (top-3 subs median)
      submission_median_top5.csv          (top-5 subs median)
      diagnostics.csv                     (per-source deltas vs median)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"

# All available "high-quality" submissions with their known LB scores.
# Ordered by known LB descending. LB=None if never submitted.
SUB_CANDIDATES = [
    {"name": "koopmans_postfit",     "lb": 0.902, "path": REPO / "results" / "exp_bandgap_koopmans_postfit" / "submission.csv"},
    {"name": "koopmans_egb",         "lb": 0.902, "path": REPO / "results" / "exp_bandgap_koopmans_egb"     / "submission.csv"},
    {"name": "moss_nc",              "lb": 0.901, "path": REPO / "results" / "exp_moss_postfit_nc"          / "submission.csv"},
    {"name": "catboost_add",         "lb": None,  "path": REPO / "results" / "exp_catboost_add_to_blend"    / "submission.csv"},  # unsubmitted
    {"name": "polymetrix_add",       "lb": 0.900, "path": REPO / "results" / "exp_polymetrix_add_to_blend"  / "submission.csv"},
    {"name": "multisignal_nnls",     "lb": None,  "path": REPO / "results" / "exp_multisignal_nnls_post"    / "submission.csv"},  # SKIP decision
]

EXP_NAME = "exp_rank_median_blend"
EXP_DIR  = REPO / "results" / EXP_NAME


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
# HELPERS
# ============================================================================

def load_submissions(candidates: list[dict], log: logging.Logger) -> list[dict]:
    """Load all available submissions from disk, tag with metadata."""
    loaded = []
    for c in candidates:
        p = c["path"]
        if not p.exists():
            log.warning(f"  MISSING: {c['name']} at {p}")
            continue
        df = pd.read_csv(p)
        assert list(df.columns) == ["id", "target"], f"bad cols in {p}: {list(df.columns)}"
        assert df["target"].notna().all(), f"NaNs in {p}"
        log.info(f"  loaded {c['name']:>18s}  LB={c['lb']}  rows={len(df)}  "
                 f"target range=[{df['target'].min():.3f}, {df['target'].max():.3f}]")
        loaded.append({**c, "df": df})
    return loaded


def median_blend(subs: list[dict], log: logging.Logger, label: str) -> pd.DataFrame:
    """Per-row median across submissions. Aligned on id."""
    log.info(f"[{label}] median blend of {len(subs)} sources: {[s['name'] for s in subs]}")
    dfs = [s["df"].rename(columns={"target": f"t_{s['name']}"}) for s in subs]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="id", how="inner")
    assert len(merged) == len(dfs[0]), f"row count mismatch after merge"

    pred_cols = [c for c in merged.columns if c.startswith("t_")]
    merged["target"] = merged[pred_cols].median(axis=1)

    # Diagnostics: per-source delta from median
    log.info(f"[{label}] per-source diagnostics (delta from median):")
    for c in pred_cols:
        diffs = np.abs(merged[c] - merged["target"])
        log.info(f"    {c:>25s}  mean|Δ|={diffs.mean():.4f}  max|Δ|={diffs.max():.4f}")

    return merged[["id", "target"]].sort_values("id").reset_index(drop=True)


def compare_with_test_types(final_sub: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Sanity check: per-target-type summary stats on the blended submission."""
    te = pd.read_csv(DATA_DIR / "test.csv")[["id", "target_type"]]
    joined = final_sub.merge(te, on="id", how="left")
    stats = joined.groupby("target_type")["target"].agg(["count", "mean", "std", "min", "max"])
    log.info("per-target-type stats on blended submission:")
    for t, row in stats.iterrows():
        log.info(f"  {t:>4s}  n={int(row['count']):>5d}  "
                 f"mean={row['mean']:>8.3f}  std={row['std']:>8.3f}  "
                 f"range=[{row['min']:>8.3f}, {row['max']:>8.3f}]")
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info("=" * 60)
    log.info(f"=== {EXP_NAME} — median blend of top LB submissions ===")
    log.info("=" * 60)
    t0 = time.time()

    log.info("loading candidate submissions:")
    loaded = load_submissions(SUB_CANDIDATES, log)
    log.info(f"loaded {len(loaded)}/{len(SUB_CANDIDATES)} candidates")

    # ---- Blend TOP-3: strongest LB signals (skip regressed subs) ----
    top3 = [s for s in loaded if s["lb"] is not None and s["lb"] >= 0.901][:3]
    if len(top3) < 3:
        # Fill with unsubmitted if needed
        top3 = loaded[:3]
    log.info("=" * 60)
    log.info("BLEND: top-3 by LB (median)")
    log.info("=" * 60)
    top3_sub = median_blend(top3, log, "top3")
    top3_path = EXP_DIR / "submission_median_top3.csv"
    top3_sub.to_csv(top3_path, index=False)
    log.info(f"wrote {top3_path}  rows={len(top3_sub)}")
    compare_with_test_types(top3_sub, log)

    # ---- Blend TOP-5: broader consensus (odd count for clean median) ----
    if len(loaded) >= 5:
        top5 = loaded[:5]
        log.info("=" * 60)
        log.info("BLEND: top-5 (broader consensus, median)")
        log.info("=" * 60)
        top5_sub = median_blend(top5, log, "top5")
        top5_path = EXP_DIR / "submission_median_top5.csv"
        top5_sub.to_csv(top5_path, index=False)
        log.info(f"wrote {top5_path}  rows={len(top5_sub)}")
        compare_with_test_types(top5_sub, log)

    # ---- Delta between top-3 median and the best individual sub (0.902) ----
    best = loaded[0]  # first candidate = best LB
    log.info("=" * 60)
    log.info(f"DELTA: median vs best individual sub ({best['name']}, LB {best['lb']})")
    log.info("=" * 60)
    merged = best["df"].rename(columns={"target": "target_best"}).merge(
        top3_sub.rename(columns={"target": "target_blend"}), on="id", how="inner")
    d = merged["target_blend"] - merged["target_best"]
    log.info(f"  median blend vs best-individual: mean|Δ|={d.abs().mean():.4f}  "
             f"max|Δ|={d.abs().max():.4f}  "
             f"rows with change: {(d.abs() > 1e-6).sum()}/{len(d)}")

    log.info(f"wall time: {time.time() - t0:.1f}s")
    log.info("=" * 60)
    log.info("PRIMARY OUTPUT: submission_median_top3.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
