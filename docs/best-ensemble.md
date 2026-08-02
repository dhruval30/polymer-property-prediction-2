# Best Ensemble — Reproduction Tracker

Ensemble submissions depend on multiple base-model experiments. Without a tracker, reproduction becomes archaeology — you'd have to piece together which base models fed which blend, in what order, with what config. This doc keeps that chain explicit.

**Companion doc:** for single-model best-of-the-competition tracking, see [best-experiment.md](best-experiment.md). Every entry here is also cross-linked from the main tracker's submission history.

> **Rule:** update the "Current best ensemble" block only if a new ensemble scores higher on LB than the previous best ensemble. Every ensemble attempt (win or not) gets a history row with all dependencies documented.

---

## Current best ensemble

| field | value |
|---|---|
| **name** | `exp_blend_nnls_3seed` |
| **LB score** | **0.897** |
| **LB rank** | **5 / 154** (tied with 1 other at 0.897; rank 3 is +0.001 away) |
| **OOF mean R²** | 0.8873 |
| **submission file** | `results/exp_blend_nnls_3seed/submission.csv` |
| **blend script** | `experiments/exp_blend_nnls_3seed.py` |
| **date submitted** | 2026-08-02 |
| **base models used** | 2 (Chemprop 3-seed bag + LGB+Maxwell) |
| **total reproduction wall time** | ~4h (Chemprop 3-seed 225min + LGB+Maxwell 15min + blend <1s) |

> **Key insight from evolution:** the 3-way blend (Chemprop single-seed + LGB + CAT, LB 0.895) was superseded by upgrading the Chemprop *base* itself to the 3-seed bag. **Stronger single Chemprop signal > adding a weaker third base (CatBoost).** Same LGB+Maxwell base as before; just swapped the Chemprop OOF/submission source.

---

## Preferred vs best-scoring ensemble

**Three ensembles worth knowing about:**

| priority | ensemble | LB | wall time | when to use |
|----------|----------|:--:|:---------:|-------------|
| **max score** | `exp_blend_nnls_3seed` | **0.897** | ~240 min | final submission, best-ever score |
| balance | `exp_blend_nnls_3way` | 0.895 | ~267 min | previous best; superseded, slower AND worse |
| **fast iteration** | `exp_blend_nnls` (2-way, single-seed) | **0.894** | **~67 min** | quick iteration on new blend recipes, debugging blend weights, fastest path to a strong submission |

The 2-way single-seed blend (67 min) hits 0.894 — only 0.003 below the best. If you're iterating on a new base model or blend strategy, run against this rather than the full 4h pipeline. Once your new base looks promising, promote it into the 3-seed pipeline for the final submission.

**Learned pattern:** upgrading the strongest base (Chemprop) gives more LB lift than adding a weaker fourth base. Prefer improving Chemprop over adding more tree models.

---

### Base model dependencies (current best 2-way with 3-seed Chemprop)

| # | source experiment | script | LB (solo) | wall time | contribution |
|---|-------------------|--------|:---------:|:---------:|--------------|
| 1 | `exp_chemprop_multitask_cpu_3seed` | `experiments/exp_chemprop_multitask_cpu_3seed.py` | **0.892** | ~225 min | 5-fold × 3-seed Chemprop D-MPNN bag. Best solo. Dominant across all 7 targets (mean w=0.70 in blend). |
| 2 | `exp_maxwell_prior_lgbm` | `experiments/exp_maxwell_prior_lgbm.py` | 0.860 | ~15 min | LightGBM per-target on full FP stack + Maxwell EPS↔Nc physics-prior. Complementary on egc/ei/eps/tg. |

Blend-time compute: **<1 second** (NNLS + weighted averaging in numpy).

### Reproduce the current best ensemble from scratch

```bash
# 1. Base — LightGBM with Maxwell prior (~15 min)
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py

# 2. Base — Multitask Chemprop D-MPNN, 5-fold × 3-seed bag (~225 min)   ← slowest step
poly2-venv/bin/pip install chemprop   # if not already installed
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu_3seed.py

# 3. 2-way blend (<1 second)
poly2-venv/bin/python experiments/exp_blend_nnls_3seed.py

# 4. Submit results/exp_blend_nnls_3seed/submission.csv → LB 0.897
```

Total wall time end-to-end: **~240 min (~4h)**. Chemprop 3-seed has per-fold checkpointing that bundles all 3 seeds per fold — safe to Ctrl+C and resume between folds. LGB has its own feature cache. Blend is instant.

### Alternative faster ensembles

**Reproduce the previous 3-way blend (LB 0.895, ~267 min)** — historical, no reason to use now:

```bash
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py           # 15 min
poly2-venv/bin/pip install catboost && poly2-venv/bin/python experiments/exp_maxwell_prior_catboost.py    # 100 min
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py       # 52 min (single-seed)
poly2-venv/bin/python experiments/exp_blend_nnls_3way.py
```

**Reproduce the single-seed 2-way blend (LB 0.894, ~67 min)** — fastest strong submission:

```bash
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py           # 15 min
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py       # 52 min (single-seed)
poly2-venv/bin/python experiments/exp_blend_nnls.py
```

Use this when iterating on new blend strategies — it's the honest test of whether a new addition helps.

### Blend config used

Both 2-way and 3-way scripts use the same mitigation constants:

```python
CHEMPROP_WEIGHT_FLOOR = 0.40   # Chemprop weight ≥ this per target
APPLY_CHEMPROP_BIAS   = 0.15   # post-NNLS shift toward Chemprop
```

Rationale: LGB and CAT both use aux-augmented CV which inflates OOF ~+0.006 above true LB skill. Chemprop uses honest OOF which underestimates true LB skill by ~+0.032. Pure OOF-optimal NNLS over-weights the trees. These mitigations push weights toward Chemprop to match its proven LB advantage.

In the 3-way blend, when the Chemprop bias is applied, the -0.15 loss to Chemprop is split between LGB and CAT proportionally to their normalized weights (so if LGB has 3× the weight of CAT after normalization, LGB absorbs 3× the reduction). Same logic for the floor clip.

Set both to `0.0` for pure OOF-optimal NNLS (both scripts document this alternative).

### Per-target weights (current best 2-way with 3-seed Chemprop)

| target | chem 3-seed OOF | lgb OOF | **blend OOF** | w_c | w_l | Δ vs 3-way blend |
|--------|:---------------:|:-------:|:-------------:|:---:|:---:|:----------------:|
| eea | 0.908 | 0.871 | **0.913** | 0.88 | 0.12 | +0.010 |
| egb | 0.931 | 0.911 | **0.935** | 0.83 | 0.17 | +0.003 |
| egc | 0.907 | 0.900 | **0.917** | 0.71 | 0.29 | +0.004 |
| ei  | 0.777 | 0.793 | **0.807** | 0.56 | 0.44 | -0.006 |
| eps | 0.792 | 0.819 | **0.836** | 0.54 | 0.46 | +0.007 |
| nc  | 0.868 | 0.860 | **0.886** | 0.69 | 0.31 | +0.002 |
| tg  | 0.908 | 0.906 | **0.917** | 0.68 | 0.32 | +0.003 |
| **mean** | 0.870 | 0.866 | **0.887** | 0.70 | 0.30 | +0.003 |

**What NNLS "learned" this time:**
- **Chemprop weight went UP across the board** (mean 0.61 → 0.70 vs single-seed 2-way blend). The 3-seed Chemprop is genuinely stronger per-target, and NNLS gives it more responsibility.
- Only target where LGB dominates: **ei** (LGB 0.793 > Chemprop 0.777) — same reason as before.
- The Chemprop bias floor (0.40) and additive bias (+0.15) still kick in on some targets, but the underlying NNLS weights are already Chemprop-heavy on 5 of 7 targets.

### Why upgrading Chemprop beat adding CatBoost

Two ways to improve the 2-way blend (Chemprop + LGB):
1. **Add a 3rd model (CatBoost):** +0.001 LB, +100 min compute → 3-way blend, LB 0.895
2. **Upgrade the Chemprop base (single-seed → 3-seed bag):** +0.002 LB, +173 min compute → new 2-way blend, LB 0.897

Option 2 delivered 2× the LB gain per minute of compute. Reason: **strengthening the strongest base gives more ensemble lift than adding a weaker third base**, because the weaker base's contribution is capped by both its own error rate AND the redundancy with other bases. Chemprop errors are structurally different from tree errors; more Chemprop signal ≈ more diversity in the blend even without adding a new family.

**Generalizable rule for our setup:** if a new base model has solo LB < 0.87, it's unlikely to add meaningful blend lift beyond bagging/improving the strongest existing base first.

---

## Ensemble history

Every ensemble attempt (blend, stack, meta-learner) tracked here. Base-model-only submissions go in [best-experiment.md](best-experiment.md).

| # | date | ensemble | LB | Δ | rank | OOF | bases | notes |
|--:|------|----------|:--:|:-:|:----:|:---:|-------|-------|
| 3 | 2026-08-02 | `exp_blend_nnls_3seed` | **0.897** 🎯 | ↑ +0.002 | **5** (tied w/ 4) | **0.8873** | **Chemprop 3-seed bag** + LGB+Maxwell | Same 2-way NNLS structure as #1 but with the Chemprop base upgraded to the 5-fold × 3-seed bag (LB 0.892 solo vs 0.887 single-seed). NNLS gave Chemprop more weight (mean 0.61 → 0.70). 6 of 7 target OOFs improved vs 3-way blend; only ei regressed -0.006. **Beats the 3-way blend at lower compute** because Chemprop base upgrade > adding a weak 3rd base (CatBoost). Current best. |
| 2 | 2026-08-02 | `exp_blend_nnls_3way` | 0.895 | ↑ +0.001 | ~4 | 0.8842 | Chemprop + LGB+Maxwell + **CatBoost+Maxwell** | Added CatBoost as 3rd base. CAT got 0.27-0.31 weight on egc/ei/tg (solo wins), 0.0 weight on egb/eps (redundant with LGB). +0.001 LB for +100 min compute — poor ROI. Superseded by #3. |
| 1 | 2026-08-02 | `exp_blend_nnls` | 0.894 | — (1st ensemble) | 5 | 0.8828 | Chemprop single-seed + LGB+Maxwell | Per-target NNLS with Chemprop weight floor 0.40 + bias +0.15. First ensemble of the competition. **Fastest reproduction target** — ~67 min wall time. Use when iterating on new base models or blend recipes. |

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
