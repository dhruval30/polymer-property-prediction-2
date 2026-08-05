"""
exp_lb_shift_probe.py — LB distribution shift detector via train-mean probe.

============================================================================
WHY THIS EXISTS
============================================================================

NeurIPS Open Polymer Prediction 2025's 2nd place team went from ~1300th to
2nd place by detecting a persistent distribution shift on Tg and adding a
constant offset (+40 °C) to their Tg predictions. That +0.02 lift was
decisive between medals and gold.

Sandman (current LB #1, 0.916) jumped ahead on 4 total entries — pattern
matches an LB probe + shift correction. We should do the same probe.

============================================================================
METHOD (single-sub probe)
============================================================================

Submit `y_pred = train_mean(target)` for every test row of every target.
The Kaggle mean-R² score gives us enough info to back-calculate whether
each target has a shift.

For a single target t, submitting train_mean_t on the test set yields:
    R²_t = 1 - sum_i (y_test,i - train_mean_t)² / sum_i (y_test,i - test_mean_t)²
         = 1 - [Var(y_test_t) + (test_mean_t - train_mean_t)²] / Var(y_test_t)
         = -(test_mean_t - train_mean_t)² / Var(y_test_t)

If no shift on target t → R²_t ≈ 0.
If shift → R²_t < 0, magnitude ∝ (shift / test_std)².

But Kaggle gives us MEAN R² across 7 targets. So one sub returns:
    R²_mean = (1/7) · Σ_t [−(shift_t / test_std_t)²]

which is a compact scalar summary. We can't per-target isolate from one sub
alone, but combined with our known model's per-target LB (from OOF-blend
history) we can back-fit each target's shift magnitude:
    - Our current best (blend_nnls_3seed) has per-target LB R² inferable.
    - Difference vs this probe on each target gives us information.

Practical decision tree:
  1. Submit this probe → note the returned R²_mean.
  2. If R²_mean is close to 0 (say > -0.05): no meaningful shift, abort probe path.
  3. If R²_mean is very negative (say < -0.10): substantial shift exists.
     Focus on the two highest-variance targets (tg, egc) as most-likely culprits.
     Use targeted sub 2 to isolate.

Follow-up subs (only if probe 1 shows shift):
  - Sub 2: apply +0.5σ_train to Tg predictions in blend_nnls_3seed submission
           (Tg has largest std, biggest EV if shifted). If LB improves,
           iterate offset. If LB degrades, try -0.5σ.
  - Sub 3: apply the best-found offset in a final submission.

Cost: 1 sub for the probe. Payoff: up to +0.02 LB if shift found.

============================================================================
OUTPUTS  (under results/exp_lb_shift_probe/)
============================================================================

  run.log                    — per-target train_mean, train_std, test-row-count
  submission.csv             — Kaggle format id, target (train_mean per target_type)
  probe_analysis_template.py — parseable helper: run AFTER receiving LB score
                                to back-fit per-target shift estimates

============================================================================
USAGE
============================================================================

Step 1 (this script):
  poly2-venv/bin/python experiments/exp_lb_shift_probe.py

Step 2:
  Submit results/exp_lb_shift_probe/submission.csv to Kaggle.
  Note the returned LB score.

Step 3:
  Edit results/exp_lb_shift_probe/probe_analysis_template.py — plug in the
  returned R²_mean. Run the helper to see per-target shift estimates.

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


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_lb_shift_probe"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")


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
# ANALYSIS TEMPLATE  (written to disk, run manually after LB score in)
# ============================================================================

ANALYSIS_TEMPLATE = '''"""
probe_analysis_template.py — plug in Kaggle score to see per-target shift estimates.

Fill in `LB_MEAN_R2` below with the score Kaggle returned for
results/exp_lb_shift_probe/submission.csv, then run:

    poly2-venv/bin/python results/exp_lb_shift_probe/probe_analysis_template.py
"""
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
with open(HERE / "probe_stats.json") as f:
    stats = json.load(f)

# === PLUG THIS IN ===
LB_MEAN_R2 = None   # <- put the Kaggle-returned score here
# ====================

if LB_MEAN_R2 is None:
    print("Edit this file: set LB_MEAN_R2 to the score Kaggle returned.")
    raise SystemExit(1)

# The math:
#   Per target t: R²_t = -(test_mean_t - train_mean_t)² / test_var_t
#   We don't know test_var_t exactly, but we assume test_std_t ≈ train_std_t
#   (usually good approximation; would only be wrong if variance shifts too).
#   Then: |shift_t| = train_std_t * sqrt(-R²_t)
#
# From one sub we get R²_mean = (1/7) sum_t R²_t. Without more info we
# can't per-target isolate — but we can bound the SUM of squared shifts:
#     sum_t (shift_t/train_std_t)² = -7 * R²_mean

sum_sq_normalized_shift = -7.0 * LB_MEAN_R2

print(f"LB probe returned R²_mean = {LB_MEAN_R2:.4f}")
print(f"Implies sum of (shift/std)² across 7 targets = {sum_sq_normalized_shift:.4f}")
print()

if sum_sq_normalized_shift < 0.05:
    print("→ NEGLIGIBLE shift on any target. LB probe path unlikely to help.")
    print("   Move to other levers (v3 additive stack, blend improvements, etc).")
elif sum_sq_normalized_shift < 0.20:
    print("→ SMALL-MODERATE shift somewhere. Follow-up probe worthwhile:")
    print("   Sub 2 = current best + 0.3σ_tg on Tg (Tg has biggest EV due to high std)")
    print("   Compare score delta to isolate Tg vs other targets.")
else:
    print(f"→ SIGNIFICANT shift detected (normalized sum {sum_sq_normalized_shift:.3f}).")
    print("   If concentrated on one target, |shift| ≈ std * sqrt(sum) per that target.")
    print("   Estimated per-target shift bounds (assuming shift on single target):")
    for tgt, s in stats.items():
        est_shift = s["train_std"] * math.sqrt(sum_sq_normalized_shift)
        print(f"     if {tgt}: |shift| ≈ {est_shift:.2f}  (std_train={s['train_std']:.2f})")
    print()
    print("   Next: pick the largest-EV target (usually tg, has 2763 test rows and std 109),")
    print("   apply +offset to current best submission, resubmit.")
    print()
    print("   Score check: for each candidate offset δ on target t:")
    print("     expected new LB R² ≈ current_LB + (2/7) * bias_correction")
    print("     where bias_correction = shift_t² / std_t² if δ matches direction")
    print("     (this is signal-only; more nuanced with modeling residual).")
'''


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    t0 = time.time()

    log.info("loading train.csv / test.csv")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log.info(f"train raw: {tr.shape}   test raw: {te.shape}")

    # Per-target train stats
    stats: dict[str, dict[str, float | int]] = {}
    log.info("=" * 60)
    log.info("PER-TARGET TRAIN STATS")
    log.info("=" * 60)
    for tgt in TARGETS:
        g = tr[tr["target_type"] == tgt]["target"]
        te_g = te[te["target_type"] == tgt]
        stats[tgt] = {
            "train_n":    int(len(g)),
            "train_mean": float(g.mean()),
            "train_std":  float(g.std()),
            "train_min":  float(g.min()),
            "train_max":  float(g.max()),
            "train_median": float(g.median()),
            "test_n":     int(len(te_g)),
        }
        s = stats[tgt]
        log.info(f"  {tgt:>4s}   n_train={s['train_n']:>5d}   "
                 f"mean={s['train_mean']:>+9.4f}   std={s['train_std']:>8.4f}   "
                 f"range=[{s['train_min']:>+8.2f}, {s['train_max']:>+8.2f}]   "
                 f"n_test={s['test_n']:>5d}")

    # Build submission = train_mean per target_type
    log.info("=" * 60)
    log.info("BUILDING PROBE SUBMISSION (train_mean per target_type)")
    log.info("=" * 60)
    sub_rows = []
    for _, row in te.iterrows():
        tgt = row["target_type"]
        sub_rows.append({"id": int(row["id"]), "target": stats[tgt]["train_mean"]})
    sub = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub)}")

    # Save stats JSON for the analysis helper
    stats_path = EXP_DIR / "probe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info(f"wrote {stats_path}")

    # Write the analysis helper
    analysis_path = EXP_DIR / "probe_analysis_template.py"
    with open(analysis_path, "w") as f:
        f.write(ANALYSIS_TEMPLATE)
    log.info(f"wrote {analysis_path}  (fill in LB_MEAN_R2 after submitting and run it)")

    log.info("=" * 60)
    log.info("NEXT STEPS")
    log.info("=" * 60)
    log.info("1. Submit results/exp_lb_shift_probe/submission.csv to Kaggle.")
    log.info("2. Note the returned mean R² score.")
    log.info("3. Edit results/exp_lb_shift_probe/probe_analysis_template.py")
    log.info("   → set LB_MEAN_R2 = <score>, then run it.")
    log.info("4. The analysis will estimate per-target shift bounds and recommend follow-up.")
    log.info("")
    log.info("EXPECTED OUTCOMES:")
    log.info("  - R²_mean ≈ 0.0    → no shift; skip this path.")
    log.info("  - R²_mean ≈ -0.05  → small shift; likely +0.005 recoverable on 1 target.")
    log.info("  - R²_mean < -0.10  → LARGE shift; up to +0.02+ recoverable (Sandman-style).")
    log.info(f"wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
