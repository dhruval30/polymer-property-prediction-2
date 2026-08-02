# Best Ensemble — Reproduction Tracker

Ensemble submissions depend on multiple base-model experiments. Without a tracker, reproduction becomes archaeology — you'd have to piece together which base models fed which blend, in what order, with what config. This doc keeps that chain explicit.

**Companion doc:** for single-model best-of-the-competition tracking, see [best-experiment.md](best-experiment.md). Every entry here is also cross-linked from the main tracker's submission history.

> **Rule:** update the "Current best ensemble" block only if a new ensemble scores higher on LB than the previous best ensemble. Every ensemble attempt (win or not) gets a history row with all dependencies documented.

---

## Current best ensemble

| field | value |
|---|---|
| **name** | `exp_blend_nnls` |
| **LB score** | **0.894** |
| **LB rank** | **5 / 154** (tied with rank 4 `VOID` and rank 6 `ShiokParikh06`) |
| **OOF mean R²** | 0.8828 |
| **submission file** | `results/exp_blend_nnls/submission.csv` |
| **blend script** | `experiments/exp_blend_nnls.py` |
| **date submitted** | 2026-08-02 |
| **base models used** | 2 (Chemprop multitask + LGB with Maxwell prior) |
| **total reproduction wall time** | ~1h 7min (see breakdown below) |

### Base model dependencies

The blend depends on OOF and test predictions from these prior experiments:

| # | source experiment | script | LB (solo) | wall time | contribution |
|---|-------------------|--------|:---------:|:---------:|--------------|
| 1 | `exp_chemprop_multitask_cpu` | `experiments/exp_chemprop_multitask_cpu.py` | 0.887 | ~52 min | Multitask D-MPNN. Strong on eea/egb/nc (small-data with cross-target overlap). |
| 2 | `exp_maxwell_prior_lgbm` | `experiments/exp_maxwell_prior_lgbm.py` | 0.860 | ~15 min | LightGBM per-target on full FP stack + Maxwell EPS↔Nc physics-prior post-fit. Strong on egc/tg/ei/eps. |

Blend-time compute: **<1 second** (just NNLS solving + weighted averaging in numpy).

### Reproduce from scratch — one command chain

```bash
# 1. Base model — LightGBM with Maxwell physics prior (~15 min on Mac CPU)
poly2-venv/bin/python experiments/exp_maxwell_prior_lgbm.py

# 2. Base model — Multitask Chemprop D-MPNN (~52 min on Mac CPU)
poly2-venv/bin/python experiments/exp_chemprop_multitask_cpu.py

# 3. Blend (<1 second)
poly2-venv/bin/python experiments/exp_blend_nnls.py

# 4. Submit results/exp_blend_nnls/submission.csv to Kaggle
```

Both base scripts have per-fold checkpointing — if you Ctrl+C partway, they resume from where they left off on rerun. Total wall time end-to-end from a clean repo: ~67 minutes.

### Blend config used (defined at top of `exp_blend_nnls.py`)

```python
CHEMPROP_WEIGHT_FLOOR = 0.40   # each target's Chemprop weight ≥ this
APPLY_CHEMPROP_BIAS   = 0.15   # after NNLS, add this to w_chemprop, then renormalize
```

Both mitigations correct for the OOF-vs-LB bias asymmetry: LGB's OOF is aux-augmented and inflates ~+0.006 above LB; Chemprop's OOF is honest and underestimates ~+0.032 below LB. Pure OOF-optimal NNLS would over-weight LGB. These mitigations push weights back toward Chemprop to match its proven LB advantage.

Set both to `0.0` for pure OOF-optimal NNLS (documented as an alternative in the script).

### Per-target weights (this ensemble)

| target | chemprop OOF | lgb OOF | blend OOF | **w_chem** | **w_lgb** | Δ vs better solo |
|--------|:------------:|:-------:|:---------:|:----------:|:---------:|:----------------:|
| eea | 0.888 | 0.871 | **0.901** | 0.75 | 0.25 | +0.013 |
| egb | 0.925 | 0.911 | **0.933** | 0.77 | 0.23 | +0.007 |
| egc | 0.883 | 0.900 | **0.912** | 0.55 | 0.45 | +0.012 |
| ei  | 0.780 | 0.793 | **0.809** | 0.58 | 0.42 | +0.016 |
| eps | 0.758 | 0.819 | **0.830** | 0.45 | 0.55 | +0.011 |
| nc  | 0.860 | 0.860 | **0.883** | 0.65 | 0.35 | +0.023 |
| tg  | 0.893 | 0.906 | **0.913** | 0.54 | 0.46 | +0.007 |
| **mean** | 0.856 | 0.866 | **0.883** | 0.61 | 0.39 | +0.017 |

Every target's blend R² beat both base models. That's the signal ensemble is genuinely capturing complementary error modes, not just averaging redundant signal.

### Why this ensemble beat both bases

The two base models have **structurally different error patterns**:
- Chemprop: shared 300-dim graph representation, benefits from multitask sharing across all 7 target heads. Small-data targets (n<340) get massive lift from representation sharing.
- LGB+Maxwell: 9k explicit fingerprint/descriptor features + physics-prior post-fit on EPS/Nc. Handles large-data targets (tg, egc) via feature capacity; handles EPS/Nc via Maxwell physics.

These strengths don't overlap. Blending recovers both.

---

## Ensemble history

Every ensemble attempt (blend, stack, meta-learner) tracked here. Base-model-only submissions go in [best-experiment.md](best-experiment.md).

| # | date | ensemble | LB | Δ | rank | OOF | bases | notes |
|--:|------|----------|:--:|:-:|:----:|:---:|-------|-------|
| 1 | 2026-08-02 | `exp_blend_nnls` | **0.894** | — (1st ensemble) | **5** | 0.8828 | `exp_chemprop_multitask_cpu`, `exp_maxwell_prior_lgbm` | Per-target NNLS with Chemprop weight floor 0.40 + bias +0.15. First ensemble attempt of the competition. +0.007 over pure Chemprop base. |

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

| strategy | expected lift over current 2-way NNLS | notes |
|----------|:--------------------------------------:|-------|
| **Rank-based blend** (rank predictions, blend ranks, then map back) | 0 to +0.005 | Robust to per-model scale bias. Good if models have different bias profiles. |
| **3-way NNLS** (add CatBoost or bagged Chemprop) | +0.002 to +0.008 | Diminishing returns after 3 diverse bases. |
| **Ridge meta-stacker** on OOF (learned per-target linear combo without NNLS constraint) | 0 to +0.003 | Risky on 220-row small targets. May overfit. |
| **Per-target Bayesian model averaging** (weights proportional to `exp(-N * MSE_oof / 2)`) | +0.001 to +0.005 | Statistically principled but similar to NNLS in practice. |
| **Cross-target meta-features** (feed OOF of other 6 targets as features to a per-target Ridge on top of NNLS) | +0.003 to +0.008 | Second bite at matrix completion. Watch small-data overfit. |

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

Naive per-target NNLS on OOF would systematically over-weight LGB by ~0.037 R² of "apparent" advantage that isn't real. The blend script applies `CHEMPROP_WEIGHT_FLOOR=0.40` (never trust LGB more than 60% for any target) and `APPLY_CHEMPROP_BIAS=+0.15` (add fixed +0.15 to Chemprop weight after NNLS normalization) to correct for this.
