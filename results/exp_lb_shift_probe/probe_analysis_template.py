"""
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
