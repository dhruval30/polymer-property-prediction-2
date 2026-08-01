# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_maxwell_prior_lgbm` |
| **LB score (public)** | **0.860** |
| **LB rank** | ~19 / 154 |
| **CV OOF mean R²** | 0.8656 |
| **submission file** | `results/exp_maxwell_prior_lgbm/submission.csv` |
| **script** | `experiments/exp_maxwell_prior_lgbm.py` |
| **date submitted** | 2026-08-01 |
| **wall time (local)** | ~15 min on Mac M-series CPU |
| **Δ vs previous best** | +0.001 (from full_fp 0.859) |

### Per-target OOF R² (this submission)

| target | n_train | n_test | OOF R² | fold R² range |
|--------|--:|--:|:--:|:--:|
| tg  | 4,139 | 2,763 | 0.9057 | 0.890–0.924 |
| egc | 2,028 | 1,352 | 0.9000 | 0.882–0.913 |
| egb | 337   | 224   | **0.9105** | 0.814–0.946 |
| eea | 221   | 147   | 0.8708 | 0.806–0.910 |
| ei  | 222   | 148   | 0.7933 | **0.634**–0.830 |
| nc  | 229   | 153   | 0.8367 | 0.793–0.888 |
| eps | 229   | 153   | 0.7854 | 0.756–0.851 |
| **mean** | | | **0.8575** | |

### Approach in one paragraph

Matrix-completion pipeline + full Round-1 fingerprint stack. 7 per-target LightGBM regressors with the SMILES feature stack expanded to include RDKit 2D descriptors (207), Morgan-r2 count FP (2048), Morgan-r3 count FP (2048), MACCS (167), Atom-Pair count FP (2048), Topological-Torsion count FP (2048), and Avalon FP (512) — plus 14 aux cross-target features (target-being-predicted slot masked). Total 9,052 features. GroupKFold(5) on canonical SMILES, aux-augmented CV, Round-1 hyperparams, identity transforms, refit at 1.1× median-best-iter.

### Feature family gain diagnostics (from this run)

- **RDKit desc: 52-86% of gain** across targets — the workhorse.
- **atom-pair count: 8-23%** — standout new FP, big on eea/nc/tg (~20% each).
- **avalon: 3-11%** — consistent modest value.
- **maccs: 1-13%** — useful for egb (13%), marginal elsewhere.
- **morgan-r2 count: 0.4-12%** — matters mainly for egc.
- **morgan-r3 count: 0.2-5%** — weak, adds noise on small-data.
- **topological-torsion count: 0.1-2%** — basically useless.
- **aux (matrix completion): 0.1-17.7%** — dominates for eps (17.7%), meaningful for nc (11%).

### What NOT in this submission (top future levers)

- ❌ **Multitask Chemprop** (single D-MPNN encoder + 7 target heads). Requires Kaggle GPU. **Now the dominant remaining lever** (+0.02 to +0.04 expected mean R²).
- ❌ **CatBoost + HGB cocktail** on top of LGB. Local, +0.005 to +0.015. Modest gain, may or may not survive OOF→LB translation.
- ❌ **PI1M SSL pretraining** on tg / egc. +0.005 to +0.015. Only makes sense after Chemprop.
- ❌ **Per-target hyperparameter tuning** (currently one-size-fits-all Round-1 defaults).
- ❌ **Target transforms tuned per target** (log1p on eps/nc/ei worth trying).
- ❌ **Scaffold-balanced GroupKFold** — fold 4 consistently trails on small-data targets.

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 5 | 2026-08-01 | `exp_maxwell_prior_lgbm` | **0.860** | ↑ +0.001 | ~19 | **0.8656** | Full_fp pipeline + Maxwell relation `EPS = a·Nc² + b` post-fit on 134 co-labeled train molecules. Maxwell forward fit R²=0.855 on 134 points. Optimal blend weights: eps w=0.405 (60% Maxwell), nc w=0.605 (40% Maxwell). **OOF Δ per target: eps +0.033, nc +0.024, mean +0.008.** LB Δ +0.001 only — physics is real (verified by fit R²) but LGB features already captured most of it implicitly. OOF-LB gap widened to -0.007 (biggest yet). At test time, 62% coverage on the aux path (95/153 rows) dampened the real gain. |
| 4 | 2026-08-01 | `exp_trimmed_smarts_lgbm` | 0.858 | ↓ -0.001 | — | 0.8610 | Path A: dropped morgan-r3 (2048) + topological-torsion (2048), added 25 SMARTS polymer-class flags + backbone-atom-count. ~5k features vs 9k. **OOF gained +0.0035** (eps recovered strongly: 0.785→0.805; eea +0.003; egc +0.003) but **LB lost 0.001**. Backbone feature useless (0.0-0.1% gain). SMARTS marginal (0.1-9% gain, mostly under 3%); only `vinyl_polymer` (eps) and `ester`/`amide` (tg) pulled real weight. OOF-LB gap now negative — CV starting to overfit fold structure. |
| 3 | 2026-08-01 | `exp_full_fp_lgbm` | 0.859 | ↑ +0.002 | ~20 | 0.8575 | Added full Round-1 fingerprint stack (Morgan-r3 count, Atom-Pair count, Topological-Torsion count, Avalon) on top of matcomp. Modest LB lift. Family gain diagnostics: atom-pair (8-23%) and avalon (3-11%) earned their spots; morgan-r3 and topological-torsion are weak. eps regressed on OOF (-0.008) but egb/eea/nc gains carried the mean up. |
| 2 | 2026-08-01 | `exp_matrix_completion_lgbm` | 0.857 | ↑ +0.014 | 22 | 0.8527 | Added 14 aux cross-target features (7 values + 7 masks), target slot masked. Aux-augmented CV. Biggest per-target lifts on eps (+0.054 OOF) and nc (+0.041 OOF). eea regressed -0.004. Half the expected mean OOF lift because Morgan-r2 already implicitly encodes molecule identity. |
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

## LB landmarks (approx, as of 2026-08-01)

Reference points for what different scores buy us on rank:

| rank | team | score | gap to us (0.859) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch bhi Karna hai | 0.899 | +0.040 |
| 3  | MUGABROS           | 0.897 | +0.038 |
| 5  | ShiokParikh08      | 0.893 | +0.034 |
| 10 | Coding Brigades    | 0.876 | +0.017 |
| 15 | Bond               | 0.872 | +0.013 |
| 20 | The Polymaths      | 0.859 | ~tied |
| **~20** | **Dhruval Padia (us)** | **0.859** | **—** |

Score targets by remaining planned experiments:
- **Local: Path A** (trim dead FPs, add SMARTS + backbone) → **narrow-lever, ~+0.005 to +0.010**, expected LB 0.86–0.87 → rank 15–18.
- **Kaggle: multitask Chemprop** (single encoder, 7 heads) → +0.02 to +0.04 → **0.88–0.90 → rank 5–10**. This is now the dominant remaining lever.
- **Kaggle: + PI1M SSL pretrain on tg/egc** → +0.005 to +0.015 → **0.89–0.91 → rank 1–5**.

## OOF-vs-LB tracking

| exp | OOF | LB | LB−OOF | LB Δ | note |
|-----|:---:|:--:|:------:|:----:|------|
| baseline | 0.8345 | 0.843 | +0.009 | — | refit-on-full-train boost, first sub |
| matcomp  | 0.8527 | 0.857 | +0.004 | +0.014 | consistent boost, matrix-completion pays off big |
| full_fp  | 0.8575 | 0.859 | +0.002 | +0.002 | tiny lift; OOF-LB gap narrowing |
| trimmed  | 0.8610 | 0.858 | **-0.003** | -0.001 | OOF up but LB flat |
| maxwell  | 0.8656 | 0.860 | **-0.007** | +0.001 | **worst OOF-LB gap yet.** Maxwell physics is real (fit R²=0.85) but LGB already captures most of it implicitly. 62% test-row aux coverage dampens gain vs OOF |

**Read of the trend.** Since matcomp we've done 3 more experiments; LB gains are +0.002, -0.001, +0.001. The OOF-LB gap keeps widening in the wrong direction (LB now 0.007 UNDER what OOF suggests). Our aux-augmented CV is systematically inflating OOF measurements. **We cannot trust OOF as a submission decision-maker anymore.** Every LGB-based tweak within this pipeline will show OOF gains that don't translate. Only fundamental changes (new model family = Chemprop, or LB-probe-based corrections) will move the needle meaningfully from here.
