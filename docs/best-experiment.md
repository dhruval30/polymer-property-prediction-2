# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission that improved on the previous best. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update if the new LB > current best LB. If the new score matches or is lower, log it in the history table with a ↔ or ↓ arrow but don't overwrite the "current best" block.

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_baseline_lgbm` |
| **LB score (public)** | **0.843** |
| **LB rank** | **24 / 154** entrants |
| **CV OOF mean R²** | 0.8345 |
| **submission file** | `results/exp_baseline_lgbm/submission.csv` |
| **script** | `experiments/exp_baseline_lgbm.py` |
| **date submitted** | 2026-08-01 |
| **wall time (local)** | 9.5 min on Mac M-series CPU |
| **submission count used** | 1 of 3 today |

### Per-target OOF R² (this submission)

| target | n_train | n_test | OOF R² | fold R² range |
|--------|--:|--:|:--:|:--:|
| tg  | 4,139 | 2,763 | **0.9026** | 0.891–0.916 |
| egc | 2,028 | 1,352 | 0.8948 | 0.881–0.907 |
| egb | 337   | 224   | 0.8917 | 0.788–0.939 |
| eea | 221   | 147   | 0.8587 | 0.788–0.910 |
| nc  | 229   | 153   | 0.7814 | 0.737–0.828 |
| ei  | 222   | 148   | 0.7730 | **0.578**–0.839 |
| eps | 229   | 153   | 0.7392 | 0.655–0.824 |
| **mean** | | | **0.8345** | |

### Approach in one paragraph

Seven independent LightGBM regressors, one per target. Features per SMILES: RDKit 2D descriptors (207 after dropping constants, inf/NaN → median-imputed, clipped 0.5/99.5%) + Morgan-r2 count fingerprints (2048) + MACCS keys (167) = 2,422 features. Wildcards `*` replaced with `C` before featurization. Canonical SMILES used for both dedup (4 tg train duplicates averaged) and CV grouping. GroupKFold(5) on canonical SMILES with the same fold assignment across all targets. LightGBM hyperparameters lifted from Round 1 recipe (n_est=4000 with early stop 200, lr=0.03, num_leaves=63, min_child_samples=10, feature_fraction=0.5, bagging_fraction=0.85, reg_lambda=1.0). Identity target transform for everyone (tg has 370 negatives so log1p is off the table). Final test predictions from a refit on the full training set at 1.1× median-best-iter per target.

### What NOT in this submission (top future levers)

- ❌ **Matrix-completion Track B** for the 5-pack + egb — the other-target values as auxiliary features. Expected mean R² lift: **+0.03 to +0.05** → LB ~0.86–0.88.
- ❌ **Multitask Chemprop** (D-MPNN with 7 heads sharing a molecular representation). Requires Kaggle GPU. Expected additional +0.02–0.04.
- ❌ **PI1M SSL pretraining** on tg / egc (chemistry-relevant subset). Expected +0.005–0.015.
- ❌ **CatBoost + HistGradientBoosting cocktail** on top of LightGBM. Round-1 recipe used all three; here we're single-family. Expected +0.005–0.015.
- ❌ **Per-target hyperparameter tuning** (currently one-size-fits-all Round-1 defaults).
- ❌ **Target transforms tuned per target** (log1p on eps/nc/ei is worth trying).

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB score | Δ | rank | OOF | notes |
|--:|------|------------|:--------:|:-:|:----:|:---:|-------|
| 1 | 2026-08-01 | `exp_baseline_lgbm` | **0.843** | — | 24 | 0.8345 | First submission. Plumbing sanity check + LB probe rolled into one. LGB per target, no matrix completion, no Chemprop. |

---

## How to update this doc

When you submit a new experiment and it beats the current best:

1. **Move the current "Current best" block down to the history table** with today's date, experiment name, LB score, rank, OOF, and 1-line note.
2. **Overwrite the "Current best" block** with the new experiment's details.
3. **Update the "What NOT in this submission" section** to reflect what's still on the table.
4. **Add a new history row at the top of the table** with an ↑ delta.

When you submit and it does not beat the current best:

1. **Do not touch "Current best".**
2. **Add a new row to the history table** with ↔ or ↓ delta and a short note on why (overfit, feature bug, mistuned hyperparams, etc.). This is where we learn.

---

## LB landmarks (as of 2026-08-01, before this submission)

Reference points for what different scores buy us on rank:

| rank | team | score | gap to us (0.843) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch bhi Karna hai | 0.899 | +0.056 |
| 3  | MUGABROS           | 0.897 | +0.054 |
| 5  | ShiokParikh08      | 0.893 | +0.050 |
| 10 | Coding Brigades    | 0.876 | +0.033 |
| 15 | Bond               | 0.872 | +0.029 |
| 20 | The Polymaths      | 0.859 | +0.016 |
| **24** | **Dhruval Padia (us)** | **0.843** | **—** |

Score targets by our next planned experiments:
- After Track B matrix completion: ~0.86–0.88 → rank ~10–15.
- After Chemprop multitask (Kaggle GPU): ~0.89–0.91 → rank ~3–7.
- After PI1M SSL pretrain: ~0.90–0.92 → rank 1–3.
