# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_matrix_completion_lgbm` |
| **LB score (public)** | **0.857** |
| **LB rank** | **22 / 154** entrants |
| **CV OOF mean R²** | 0.8527 |
| **submission file** | `results/exp_matrix_completion_lgbm/submission.csv` |
| **script** | `experiments/exp_matrix_completion_lgbm.py` |
| **date submitted** | 2026-08-01 |
| **wall time (local)** | 9.2 min on Mac M-series CPU |
| **Δ vs previous best** | +0.014 (from 0.843) |

### Per-target OOF R² (this submission)

| target | n_train | n_test | OOF R² | fold R² range | Δ vs baseline | aux gain share |
|--------|--:|--:|:--:|:--:|:--:|:--:|
| tg  | 4,139 | 2,763 | 0.9026 | 0.891–0.917 | +0.0000 | 0.1% |
| egc | 2,028 | 1,352 | 0.8966 | 0.885–0.905 | +0.0018 | 0.8% |
| egb | 337   | 224   | **0.9050** | 0.808–0.946 | +0.0133 | 5.4% |
| eea | 221   | 147   | 0.8543 | 0.780–0.907 | **-0.0044** | 4.1% |
| ei  | 222   | 148   | 0.7944 | **0.607**–0.843 | +0.0214 | 3.4% |
| nc  | 229   | 153   | **0.8228** | 0.763–0.883 | **+0.0414** | 17.6% |
| eps | 229   | 153   | **0.7931** | 0.762–0.861 | **+0.0539** | 16.9% |
| **mean** | | | **0.8527** | | **+0.0182** | |

### Approach in one paragraph

Same as baseline (7 per-target LightGBM regressors, RDKit desc + Morgan-r2 count FP + MACCS = 2,422 SMILES features, GroupKFold(5) on canonical SMILES, Round-1 hyperparams, identity transforms, refit at 1.1× median-best-iter) — **plus** 14 auxiliary cross-target features per row: 7 mean-target values for each of the 7 target_types on the same canonical SMILES + 7 mask indicators. The target-being-predicted slot is always masked (NaN value + 0 mask) so a molecule's own T value never leaks into its own T prediction. CV mode is **aux-augmented**: the aux lookup is built once from the full training set, so both fold-train and fold-val rows draw aux from it. This mirrors LB conditions (test SMILES's train other-target values are legitimately available) while preserving zero label leakage. Aux coverage: 98%+ of test rows for eea/ei/eps/nc, 88% for egb, 37% for egc, 12% for tg.

### Where the gain came from

- **eps (+0.054)** and **nc (+0.041)** are the real winners. They have r = +0.92 with each other (dielectric constant ↔ refractive index is a Kramers–Kronig-related pair), so knowing one is nearly a direct predictor of the other. Their aux gain shares are 17% (highest by far).
- **egb (+0.013)** and **ei (+0.021)** got smaller lifts. The signal is real but Morgan fingerprints already encode most of the same-molecule identity, leaving less headroom for aux features to add.
- **egc (+0.002)**, **tg (0)** got nothing — as predicted (low cross-target coverage / weak inter-target correlations).
- **eea regressed by 0.004** — small-data noise + only 4% aux gain share = the 14 aux features basically added marginal overfitting noise for eea. Statistically indistinguishable from zero but a real signal that eea doesn't benefit here.

### What NOT in this submission (top future levers)

- ❌ **Multitask Chemprop** (single D-MPNN encoder + 7 target heads sharing molecular representation). Requires Kaggle GPU. Expected +0.02 to +0.04 mean R². **Now the highest-EV lever** since matrix completion under-delivered vs expectation (the Morgan-r2 fingerprint already implicitly encodes molecule identity, dampening the aux-feature benefit — multitask Chemprop explicitly shares representation and should extract more of the cross-target signal).
- ❌ **CatBoost + HistGradientBoosting cocktail** on top of LightGBM. Local, +0.005 to +0.015 without Kaggle GPU.
- ❌ **PI1M SSL pretraining** on tg / egc (chemistry-relevant subset). +0.005 to +0.015. Only makes sense once we have Chemprop.
- ❌ **Per-target hyperparameter tuning** (currently one-size-fits-all Round-1 defaults).
- ❌ **Target transforms tuned per target** (log1p on eps/nc/ei is worth trying).
- ❌ **Scaffold-balanced GroupKFold** — fold 4 consistently trails on small-data targets (ei fold 4 = 0.61, eps fold 4 = 0.76). A smarter split might smooth this.

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 2 | 2026-08-01 | `exp_matrix_completion_lgbm` | **0.857** | ↑ +0.014 | **22** | 0.8527 | Added 14 aux cross-target features per row (7 values + 7 masks), target slot masked. Aux-augmented CV. Biggest per-target lifts on eps (+0.054) and nc (+0.041); eea regressed -0.004. Half the expected mean lift because Morgan-r2 already implicitly encodes molecule identity — dampening aux-feature marginal utility. |
| 1 | 2026-08-01 | `exp_baseline_lgbm` | 0.843 | — | 24 | 0.8345 | First submission. Plumbing sanity check + LB probe rolled into one. LGB per target, no matrix completion, no Chemprop. |

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

## LB landmarks (as of 2026-08-01, before submission #2)

Reference points for what different scores buy us on rank:

| rank | team | score | gap to us (0.857) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch bhi Karna hai | 0.899 | +0.042 |
| 3  | MUGABROS           | 0.897 | +0.040 |
| 5  | ShiokParikh08      | 0.893 | +0.036 |
| 10 | Coding Brigades    | 0.876 | +0.019 |
| 15 | Bond               | 0.872 | +0.015 |
| 20 | The Polymaths      | 0.859 | +0.002 |
| **22** | **Dhruval Padia (us)** | **0.857** | **—** |
| 24 | (previous us: 0.843) | | |

Score targets by remaining planned experiments:
- **Local: LGB + CatBoost + HGB cocktail** (same features) → +0.005 to +0.015 → **0.86–0.87 → rank 15–20**.
- **Kaggle: multitask Chemprop** (single encoder, 7 heads) → +0.02 to +0.04 → **0.88–0.90 → rank 5–10**.
- **Kaggle: + PI1M SSL pretrain on tg/egc** → +0.005 to +0.015 → **0.89–0.91 → rank 1–5**.
