# Best Ensemble — Reproduction Tracker

Ensemble submissions depend on multiple base-model experiments. Without a tracker, reproduction becomes archaeology — you'd have to piece together which base models fed which blend, in what order, with what config. This doc keeps that chain explicit.

**Companion doc:** for single-model best-of-the-competition tracking, see [best-experiment.md](best-experiment.md). Every entry here is also cross-linked from the main tracker's submission history.

> **Rule:** update the "Current best ensemble" block only if a new ensemble scores higher on LB than the previous best ensemble. Every ensemble attempt (win or not) gets a history row with all dependencies documented.

---

## Current best ensemble

| field | value |
|---|---|
| **name** | `exp_blend_nnls_3way` |
| **LB score** | **0.895** |
| **LB rank** | ~4 / 154 |
| **OOF mean R²** | 0.8842 |
| **submission file** | `results/exp_blend_nnls_3way/submission.csv` |
| **blend script** | `experiments/exp_blend_nnls_3way.py` |
| **date submitted** | 2026-08-02 |
| **base models used** | 3 (Chemprop multitask + LGB+Maxwell + CatBoost+Maxwell) |
| **total reproduction wall time** | ~2h 47min |

> ⚠️ **The +0.001 LB gain over the 2-way blend cost +100 minutes of CatBoost training.** For practical reproduction the 2-way blend below is the better ROI. See [§ Preferred vs best-scoring ensemble](#preferred-vs-best-scoring-ensemble) for the decision framework.

---

## Preferred vs best-scoring ensemble

**Two ensembles you might want to reproduce, depending on your priority:**

| priority | ensemble | LB | wall time | when to use |
|----------|----------|:--:|:---------:|-------------|
| **max score** | `exp_blend_nnls_3way` | **0.895** | ~167 min | final submission run, ranking matters |
| **best ROI** | `exp_blend_nnls` (2-way) | **0.894** | ~67 min | iterating, adding new bases, debugging blend weights, fastest path to a strong submission |

The 2-way ensemble is 60% of the total wall time for 99.9% of the score. Use the 3-way only when the last 0.001 R² matters for ranking. When adding a new base model (e.g., Chemprop 3-seed bag, PI1M-pretrained model), extend the 2-way script first — that's the honest test of whether the new base adds signal, since CatBoost's redundancy with LGB means it dampens the marginal contribution of anything you add downstream.

---

### Base model dependencies (3-way, current best)

| # | source experiment | script | LB (solo) | wall time | contribution |
|---|-------------------|--------|:---------:|:---------:|--------------|
| 1 | `exp_chemprop_multitask_cpu` | `experiments/exp_chemprop_multitask_cpu.py` | 0.887 | ~52 min | Multitask D-MPNN. Strong on eea/egb (small-data cross-target overlap). |
| 2 | `exp_maxwell_prior_lgbm` | `experiments/exp_maxwell_prior_lgbm.py` | 0.860 | ~15 min | LightGBM per-target on full FP stack + Maxwell EPS↔Nc physics-prior. Strong on egb/eps/nc. |
| 3 | `exp_maxwell_prior_catboost` | `experiments/exp_maxwell_prior_catboost.py` | ~0.860 est. | ~100 min | CatBoost per-target on identical stack as LGB. Strong on egc/ei/tg (targets where CAT wins solo). |

Blend-time compute: **<1 second** (NNLS + weighted averaging in numpy).

### Reproduce the 3-way ensemble from scratch

```bash
# 1. Base — LightGBM with Maxwell prior (~15 min)
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py

# 2. Base — CatBoost with Maxwell prior (~100 min)  ← slowest step
poly2-venv/bin/pip install catboost   # if not already installed
poly2-venv/bin/python experiments/exp_maxwell_prior_catboost.py

# 3. Base — Multitask Chemprop D-MPNN (~52 min)
poly2-venv/bin/pip install chemprop   # if not already installed
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py

# 4. 3-way blend (<1 second)
poly2-venv/bin/python experiments/exp_blend_nnls_3way.py

# 5. Submit results/exp_blend_nnls_3way/submission.csv to Kaggle
```

Total wall time end-to-end from a clean repo: **~167 minutes**. Base scripts all have per-fold checkpointing — safe to Ctrl+C and resume.

### Reproduce the 2-way ensemble (preferred for practical iteration)

```bash
# 1. Base — LightGBM with Maxwell prior (~15 min)
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py

# 2. Base — Multitask Chemprop D-MPNN (~52 min)
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py

# 3. 2-way blend (<1 second)
poly2-venv/bin/python experiments/exp_blend_nnls.py

# 4. Submit results/exp_blend_nnls/submission.csv → LB 0.894
```

Total: **~67 minutes**. Skips CatBoost. LB 0.894 vs 0.895 for 3-way — you give up 0.001 LB to save 100 min.

### Blend config used

Both 2-way and 3-way scripts use the same mitigation constants:

```python
CHEMPROP_WEIGHT_FLOOR = 0.40   # Chemprop weight ≥ this per target
APPLY_CHEMPROP_BIAS   = 0.15   # post-NNLS shift toward Chemprop
```

Rationale: LGB and CAT both use aux-augmented CV which inflates OOF ~+0.006 above true LB skill. Chemprop uses honest OOF which underestimates true LB skill by ~+0.032. Pure OOF-optimal NNLS over-weights the trees. These mitigations push weights toward Chemprop to match its proven LB advantage.

In the 3-way blend, when the Chemprop bias is applied, the -0.15 loss to Chemprop is split between LGB and CAT proportionally to their normalized weights (so if LGB has 3× the weight of CAT after normalization, LGB absorbs 3× the reduction). Same logic for the floor clip.

Set both to `0.0` for pure OOF-optimal NNLS (both scripts document this alternative).

### Per-target weights (3-way, current best)

| target | chem OOF | lgb OOF | cat OOF | **blend OOF** | w_c | w_l | w_x | best solo |
|--------|:--------:|:-------:|:-------:|:-------------:|:---:|:---:|:---:|:---------:|
| eea | 0.888 | 0.871 | 0.865 | **0.902** | 0.72 | 0.15 | 0.13 | chemprop |
| egb | 0.925 | 0.911 | 0.900 | **0.933** | 0.77 | 0.23 | 0.00 | chemprop |
| egc | 0.883 | 0.900 | 0.903 | **0.913** | 0.50 | 0.22 | 0.27 | cat |
| ei  | 0.780 | 0.793 | 0.793 | **0.814** | 0.56 | 0.14 | 0.31 | cat |
| eps | 0.758 | 0.819 | 0.798 | **0.830** | 0.45 | 0.55 | 0.00 | lgb |
| nc  | 0.860 | 0.860 | 0.853 | **0.884** | 0.64 | 0.26 | 0.10 | lgb |
| tg  | 0.893 | 0.906 | 0.908 | **0.914** | 0.49 | 0.21 | 0.31 | cat |
| **mean** | 0.856 | 0.866 | 0.860 | **0.884** | 0.59 | 0.25 | 0.16 | — |

**What NNLS "learned" per target:**
- **egb, eps:** CAT gets ZERO weight. LGB dominates the tree slot for these targets — CAT's errors are correlated enough with LGB's that adding it is pure noise.
- **egc, ei, tg:** CAT gets meaningful weight (0.27-0.31). These are targets where CAT wins solo over LGB.
- **eea, nc:** CAT gets small weight (0.10-0.13). Marginal contribution.

### Why the 3-way barely beat the 2-way

CAT and LGB are both tree learners on identical features. Even where CAT wins solo per-target, its errors are structurally similar to LGB's. NNLS is smart enough to only use CAT where it genuinely differs (egc/ei/tg), but even there the marginal gain is ~+0.005-0.015 per target, weighted at ~0.30 → aggregate contribution of only ~+0.001-0.003 mean R².

The Chemprop → tree diversity contributed +0.007 LB. The LGB → CAT diversity contributed +0.001 LB. **Diversity between model families >> diversity within a family.**

---

## Ensemble history

Every ensemble attempt (blend, stack, meta-learner) tracked here. Base-model-only submissions go in [best-experiment.md](best-experiment.md).

| # | date | ensemble | LB | Δ | rank | OOF | bases | notes |
|--:|------|----------|:--:|:-:|:----:|:---:|-------|-------|
| 2 | 2026-08-02 | `exp_blend_nnls_3way` | **0.895** | ↑ +0.001 | ~4 | 0.8842 | Chemprop + LGB+Maxwell + **CatBoost+Maxwell** | Added CatBoost as 3rd base. CAT gets 0.27-0.31 weight on egc/ei/tg (its solo wins) but 0.0 weight on egb/eps (redundant with LGB). Marginal +0.001 LB gain for +100 min compute — poor ROI. Kept as current best because it does technically win, but 2-way is preferred for practical reproduction. |
| 1 | 2026-08-02 | `exp_blend_nnls` | 0.894 | — (1st ensemble) | 5 | 0.8828 | Chemprop + LGB+Maxwell | Per-target NNLS with Chemprop weight floor 0.40 + bias +0.15. First ensemble of the competition. +0.007 over pure Chemprop base. **Preferred reproduction target** — ~67 min wall time for 99.9% of the best-ever score. |

---

## Ensemble how-to guide

### Adding a new base model to the current ensemble

To include a third base (e.g., a Chemprop 3-seed bag, or CatBoost, or PI1M-pretrained model):

1. Run the new base model script. Confirm it produces `results/<new_exp>/oof.csv` and `submission.csv`.
2. Copy `experiments/exp_blend_nnls.py` to `experiments/exp_blend_nnls_v2.py` (fresh standalone).
3. Add the new source dir to the `LOAD` section. Extend the `oof` and `sub` merges to include a third y_pred column.
4. Extend `fit_target_weights` from 2 weights to 3. Use `scipy.optimize.nnls` on `A = [y_c, y_l, y_new]`.
5. Re-tune `CHEMPROP_WEIGHT_FLOOR` and `APPLY_CHEMPROP_BIAS` if needed. New model may need its own bias term.
6. Add a new row to the ensemble history table above with all 3 bases listed.

**Rule:** ensemble scripts are ALSO self-contained standalone (per CLAUDE.md project structure). Copy-paste, don't factor out — new blends might diverge in subtle ways from old ones and shared code becomes a footgun.

### Alternative ensemble strategies to try (research)

| strategy | expected lift over current 3-way NNLS | notes |
|----------|:--------------------------------------:|-------|
| **⭐ Re-blend with 3-seed Chemprop base** | +0.003 to +0.006 | Just swap the Chemprop OOF/submission source in the blend script. 3-seed solo LB 0.892 vs single-seed 0.887. **Highest-EV, do this next.** |
| **Rank-based blend** (rank predictions, blend ranks, then map back) | 0 to +0.005 | Robust to per-model scale bias. Good if models have different bias profiles. |
| ~~Add CatBoost as 3rd base~~ | ~~+0.002 to +0.008~~ | **Attempted 2026-08-02. Actual: +0.001 LB for +100 min compute.** LGB↔CAT correlation too high. |
| ~~Add bagged Chemprop as 4th base~~ | | Actually easier as a base *replacement* (see top row) than as a 4th base — the 3-seed OOFs strictly improve on single-seed OOFs, no reason to keep both. |
| **Add Chemprop with `--polymer` mode as 4th base** | +0.002 to +0.005 | Weighted repeat-unit bonds (Coley group fork). Different molecular representation. |
| **Ridge meta-stacker** on OOF (learned per-target linear combo without NNLS constraint) | 0 to +0.003 | Risky on 220-row small targets. May overfit. |
| **Per-target Bayesian model averaging** (weights proportional to `exp(-N * MSE_oof / 2)`) | +0.001 to +0.005 | Statistically principled but similar to NNLS in practice. |
| **Cross-target meta-features** (feed OOF of other 6 targets as features to a per-target Ridge on top of NNLS) | +0.003 to +0.008 | Second bite at matrix completion. Watch small-data overfit. |

**Key lesson from adding CatBoost:** diversity within a model family (tree ↔ tree) gives ~0.001 LB per new base. Diversity across families (tree ↔ graph like Chemprop) gave ~0.007 LB. **Future ensemble adds should be from new families**, not another tree.

### New base signals available (for future blend construction)

| base | LB solo | script | wall time | notes |
|------|:-------:|--------|:---------:|-------|
| Chemprop 1-seed | 0.887 | `exp_chemprop_multitask_cpu.py` | 52 min | Original — kept for historical alignment |
| **Chemprop 3-seed bag** | **0.892** | `exp_chemprop_multitask_cpu_3seed.py` | 225 min | **Best solo. Preferred Chemprop base for future blends.** |
| LGB + Maxwell | 0.860 | `exp_maxwell_prior_lgbm.py` | 15 min | Tree base with physics prior |
| CatBoost + Maxwell | ~0.860 | `exp_maxwell_prior_catboost.py` | 100 min | Marginal ensemble value |

### Ensemble reproduction gotchas

- **Base-model OOFs must use the same fold split.** All base experiments in this repo use `GroupKFold(5, seed=42)` on canonical SMILES. If a new base uses different folds, its OOFs won't align with the others for per-target NNLS.
- **Base-model OOF/test files must exist BEFORE running the blend.** `exp_blend_nnls.py` validates this and exits cleanly if either source is missing.
- **Blend weights are OOF-fit but LB-biased**. The bias-mitigation config (`CHEMPROP_WEIGHT_FLOOR`, `APPLY_CHEMPROP_BIAS`) is calibrated to the CURRENT LB-OOF gaps for Chemprop vs LGB. If a NEW base model has a different OOF-LB gap pattern, re-calibrate. Print the gap of any new base after its first submission.

---

## Quick reference: how the OOF-vs-LB gap informed the blend config

| base | OOF | LB | LB−OOF gap | interpretation |
|------|:---:|:--:|:----------:|----------------|
| `exp_chemprop_multitask_cpu` | 0.8555 | 0.887 | **+0.031** | honest OOF (no aux features) UNDERSTATES true skill — refit on full data adds ~+0.03 skill graphs can't measure in fold-CV |
| `exp_maxwell_prior_lgbm` | 0.8656 | 0.860 | **−0.006** | aux-augmented OOF (uses train labels of other targets on same molecule) OVERSTATES true skill by ~+0.006 |
| `exp_maxwell_prior_catboost` | 0.8602 | ~0.860 est | ~-0.000 | aux-augmented same as LGB, tied on LB |

Naive per-target NNLS on OOF would systematically over-weight the tree bases (LGB, CAT) by ~0.037 R² of "apparent" advantage that isn't real. The blend script applies `CHEMPROP_WEIGHT_FLOOR=0.40` (never trust trees more than 60% collectively for any target) and `APPLY_CHEMPROP_BIAS=+0.15` (add fixed +0.15 to Chemprop weight after NNLS normalization) to correct for this.

In the 3-way script, when the Chemprop bias/floor is applied, the -0.15 (or floor delta) is split between LGB and CAT proportionally to their pre-adjustment normalized weights. This means the "which tree wins per target" signal from NNLS is preserved — we only re-weight Chemprop vs the trees, not within the trees.
