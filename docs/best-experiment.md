# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_chemprop_multitask_cpu` |
| **LB score (public)** | **0.887** 🎯 |
| **LB rank** | **9 / 154** — medal territory |
| **CV OOF mean R²** | 0.8555 |
| **submission file** | `results/exp_chemprop_multitask_cpu/submission.csv` |
| **script** | `experiments/exp_chemprop_multitask_cpu.py` |
| **date submitted** | 2026-08-02 |
| **wall time (local)** | 51.6 min on Mac M-series CPU |
| **Δ vs previous best** | **+0.027** (from maxwell_prior 0.860) |

### Per-target OOF R² (this submission)

| target | n_train | OOF R² | fold R² range |
|--------|--:|:--:|:--:|
| eea | 221   | 0.8883 | 0.870–0.914 |
| egb | 337   | **0.9251** | 0.873–0.936 |
| egc | 2,028 | 0.8830 | 0.797–0.916 |
| ei  | 222   | 0.7804 | 0.682–0.842 |
| eps | 229   | 0.7584 | 0.649–0.806 |
| nc  | 229   | 0.8599 | 0.833–0.888 |
| tg  | 4,139 | 0.8934 | 0.873–0.912 |
| **mean OOF** | | **0.8555** | |
| **LB (actual)** | | **0.887** | LB−OOF gap = **+0.032** 🎯 |

### Approach in one paragraph

Multitask D-MPNN. Single shared `BondMessagePassing` encoder (d_h=300, depth=4, dropout=0.05) → `MeanAggregation` → `RegressionFFN(n_tasks=7, hidden=300, n_layers=2, dropout=0.05)`, `batch_norm=True`. Trained on all 7 targets jointly with NaN targets masked from loss (each row contributes only to its labeled targets). Per-target standardization on train fold. 5-fold GroupKFold on canonical SMILES. Chemprop 2.x, CPU only (Mac M-series, 8 threads). Batch 64, max_epochs=40, patience=8, LR 1e-3→1e-4 with 2-epoch warmup, grad_clip=1.0. Refit on full train for 44 epochs (1.1× median best-epoch), then predicted all 7 targets for each unique test SMILES and indexed by test row's `target_type`. Per-fold checkpointing + per-epoch logging built in.

### Why the OOF-LB gap flipped strongly positive (+0.032)

Every prior LGB experiment had LB ≤ OOF (aux-augmented CV was systematically inflating OOF). Chemprop went the opposite way — LB WAY higher than OOF. Two mechanisms:

1. **Chemprop's shared encoder benefits massively from more training data.** OOF trains on 80% of molecules (~4,736); the refit uses all 100% (~5,920). +25% more data → substantially better learned molecular representation. Trees don't gain nearly this much from 25% more rows; graph encoders do.
2. **Chemprop uses HONEST OOF** (no aux features). LGB's OOF was aux-augmented (leaked train-label info for the same molecule via other-target columns). So LGB's OOF was overestimating, Chemprop's OOF was underestimating.

Net: OOF-based comparisons across model families are misleading. LB is the honest truth.

### What NOT in this submission (top future levers)

- ❌ **NNLS per-target blend of Chemprop + LGB (Maxwell-corrected).** Highest-EV lever now. Chemprop and LGB win on different targets (Chemprop: eea/egb/nc; LGB: egc/ei/eps/tg per OOF ordering). Blend should push higher — but need to be careful: LGB OOF is aux-inflated relative to Chemprop, so simple per-target NNLS on OOF may over-weight LGB. Consider using held-out (non-aux) OOF for blend weight fit.
- ❌ **Chemprop with 3-seed bagging** (5-fold × 3 seeds = 15 models, per Round-1 recipe). ~2.5h more compute. Expected +0.003 to +0.008.
- ❌ **Longer Chemprop training** — folds 0, 1, 4 hit max_epochs=40 without early stopping. Extending to 60 epochs might help.
- ❌ **Chemprop `--polymer` mode** with weighted repeat-unit bonds (Coley group fork).
- ❌ **PI1M SSL pretraining** on tg / egc chemistry (research doc §6).
- ❌ **CatBoost + XGBoost** added to the tree cocktail before blending with Chemprop.
- ❌ **LB distribution shift probe** (research doc §9) — 3 subs could unlock up to +0.03 hidden shift correction.

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 6 | 2026-08-02 | `exp_chemprop_multitask_cpu` | **0.887** 🎯 | ↑ **+0.027** | **9** | 0.8555 | Multitask D-MPNN (shared BondMessagePassing + 7 regression heads), Chemprop 2.x on Mac CPU, 51.6 min. 5-fold GroupKFold, honest OOF (no aux features), refit on full train for 44 epochs. **OOF 0.856 but LB 0.887 (+0.032 LB-OOF gap!!)** because (a) graph encoder benefits massively from +25% training data at refit, (b) prior LGB OOFs were aux-inflated so relative comparison was misleading. Biggest single-experiment jump of the competition. Cracks top 10 / medal territory. |
| 5 | 2026-08-01 | `exp_maxwell_prior_lgbm` | 0.860 | ↑ +0.001 | ~19 | 0.8656 | Full_fp pipeline + Maxwell relation `EPS = a·Nc² + b` post-fit on 134 co-labeled train molecules. Maxwell forward fit R²=0.855. Optimal blend weights: eps w=0.405, nc w=0.605. OOF Δ +0.008 but LB Δ only +0.001 — physics real but LGB features implicitly captured most of it; also 62% test aux coverage limited gain. |
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

## LB landmarks (approx, as of 2026-08-02)

Reference points for what different scores buy us on rank:

| rank | approx score | gap to us (0.887) |
|------|:-----:|:-----------------:|
| 1  | ~0.899 | +0.012 |
| 3  | ~0.897 | +0.010 |
| 5  | ~0.893 | +0.006 |
| **9 (us)** | **0.887** | **—** |
| 10 | ~0.876 | -0.011 |
| 15 | ~0.872 | -0.015 |

Score targets by remaining experiments:
- **NNLS blend of Chemprop + LGB+Maxwell** → +0.003 to +0.010 → **0.89–0.897 → rank 4–7** (if well-tuned).
- **Chemprop 3-seed bag** (5-fold × 3 seeds, ~2.5h more CPU) → +0.003 to +0.008 → **0.89–0.895 → rank 4–7**.
- **PI1M SSL pretrain + Chemprop fine-tune** (Kaggle-only for GPU) → +0.005 to +0.015 → **0.89–0.90 → rank 1–5**.
- **LB distribution shift probe** (research doc §9, 3 subs) → up to +0.03 if shift found → **potentially #1**.

## OOF-vs-LB tracking

| exp | OOF | LB | LB−OOF | LB Δ | note |
|-----|:---:|:--:|:------:|:----:|------|
| baseline | 0.8345 | 0.843 | +0.009 | — | refit-on-full-train boost, first sub |
| matcomp  | 0.8527 | 0.857 | +0.004 | +0.014 | consistent boost, matrix-completion pays off big |
| full_fp  | 0.8575 | 0.859 | +0.002 | +0.002 | tiny lift; OOF-LB gap narrowing |
| trimmed  | 0.8610 | 0.858 | -0.003 | -0.001 | OOF up but LB flat |
| maxwell  | 0.8656 | 0.860 | -0.007 | +0.001 | worst OOF-LB gap for LGB — aux-augmented CV inflating OOF |
| **chemprop** | **0.8555** | **0.887** | **+0.032** | **+0.027** | **honest OOF (no aux) + graph encoder benefits from full-data refit → LB WAY above OOF. New paradigm.** |

**Read of the trend.** The LGB experiments (baseline through maxwell) had a shrinking / negative OOF-LB gap because aux-augmented CV was inflating OOF. Chemprop broke the pattern: honest OOF (no aux) + a model family that benefits substantially from more training data → OOF underestimated LB by 0.032. Going forward, when comparing across model families we should trust LB, not OOF. Within a single model family, OOF trends remain informative.
