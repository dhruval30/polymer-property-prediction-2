# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

**Companion doc:** for reproducing ensembles (which depend on multiple base-model experiments), see [best-ensemble.md](best-ensemble.md).

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_blend_nnls` (ENSEMBLE) |
| **LB score (public)** | **0.894** 🎯 |
| **LB rank** | **5 / 154** — tied with rank 4 (VOID) and rank 6 (ShiokParikh06) |
| **CV OOF mean R²** | 0.8828 |
| **submission file** | `results/exp_blend_nnls/submission.csv` |
| **script** | `experiments/exp_blend_nnls.py` (see [best-ensemble.md](best-ensemble.md) for full reproduction) |
| **date submitted** | 2026-08-02 |
| **wall time (local)** | 0.1s blend + prerequisite base models (~1h total for Chemprop + ~15min for LGB+Maxwell) |
| **Δ vs previous best** | **+0.007** (from chemprop-only 0.887) |

### Per-target OOF R² (this submission)

| target | chemprop | lgb+max | **blend** | w_chem | w_lgb |
|--------|:--------:|:-------:|:---------:|:------:|:-----:|
| eea | 0.8883 | 0.8708 | **0.9010** | 0.75 | 0.25 |
| egb | 0.9251 | 0.9105 | **0.9325** | 0.77 | 0.23 |
| egc | 0.8830 | 0.9000 | **0.9115** | 0.55 | 0.45 |
| ei  | 0.7804 | 0.7933 | **0.8092** | 0.58 | 0.42 |
| eps | 0.7584 | 0.8186 | **0.8295** | 0.45 | 0.55 |
| nc  | 0.8599 | 0.8603 | **0.8834** | 0.65 | 0.35 |
| tg  | 0.8934 | 0.9057 | **0.9126** | 0.54 | 0.46 |
| **mean OOF** | 0.8555 | 0.8656 | **0.8828** | 0.61 | 0.39 |
| **LB actual** | 0.887 | 0.860 | **0.894** | — | — |

Every single target improved in the blend vs either base model alone.

### Approach in one paragraph

Per-target NNLS blend of two base models: multitask Chemprop D-MPNN (`exp_chemprop_multitask_cpu`, LB 0.887) and LightGBM+Maxwell physics-prior (`exp_maxwell_prior_lgbm`, LB 0.860). Weights fit per target on the aligned OOFs via `scipy.optimize.nnls`, normalized to sum=1, then adjusted with two LB-bias-aware mitigations: (a) a Chemprop weight floor of 0.40 to prevent over-trusting the aux-inflated LGB OOF, and (b) a +0.15 additive Chemprop bias to reflect its proven LB advantage. Blend weights applied identically to test predictions. Whole blend script runs in <1 second — the compute is all in the base models. Full reproduction chain in [best-ensemble.md](best-ensemble.md).

### Why the blend worked

Chemprop and LGB have complementary per-target strengths (per OOF):
- **Chemprop wins** on small-data + heavy cross-target-overlap targets (eea, egb, nc)
- **LGB wins** on larger-data or physics-benefiting targets (egc, ei, eps, tg)

Blending captured **both** advantages. Every per-target OOF improved (+0.007 to +0.023). Mean blend OOF 0.883 vs Chemprop 0.856 vs LGB 0.866. The LB-bias mitigations (weight floor + bias) kept the blend from over-weighting LGB where its OOF was inflated.

### What NOT in this submission (top future levers, ordered by EV)

- ❌ **Chemprop 3-seed bagging** (5-fold × 3 seeds = 15 models, per Round-1 recipe). ~2.5h more Mac CPU. Expected +0.003 to +0.008. Then re-blend.
- ❌ **Longer Chemprop training** — folds 0, 1, 4 hit max_epochs=40 without early stopping. Extend to 60 epochs, +30 min per fold.
- ❌ **Add CatBoost or XGBoost** as a third base model to the blend. Diverse tree family, complements LGB.
- ❌ **Chemprop `--polymer` mode** (Coley group fork) with weighted repeat-unit bonds.
- ❌ **PI1M SSL pretraining** on tg / egc chemistry (research doc §6). Kaggle GPU only.
- ❌ **LB distribution shift probe** (research doc §9) — 3 subs could unlock up to +0.03 hidden shift correction. **HIGHEST-EV single lever** if there's a shift.
- ❌ **Ridge meta-stacker with cross-target OOF as features** (research doc §8.5).

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 7 | 2026-08-02 | `exp_blend_nnls` **(ensemble)** | **0.894** 🎯 | ↑ **+0.007** | **5** | 0.8828 | Per-target NNLS blend of Chemprop + LGB+Maxwell OOFs, normalized to sum=1, with Chemprop weight floor=0.40 + additive bias=0.15 as LB-vs-OOF-bias mitigations. Every target's blend OOF improved over either base (+0.007 to +0.023 per target). Weights lean Chemprop on small-data/multitask targets (eea 0.75, egb 0.77, nc 0.65), lean LGB on physics/larger-data (eps 0.45, tg 0.54). LB +0.007 over pure Chemprop → rank 5, tied with 2 others. Reproduction requires both base experiments — see [best-ensemble.md](best-ensemble.md). |
| 6 | 2026-08-02 | `exp_chemprop_multitask_cpu` | 0.887 | ↑ +0.027 | 9 | 0.8555 | Multitask D-MPNN (shared BondMessagePassing + 7 regression heads), Chemprop 2.x on Mac CPU, 51.6 min. 5-fold GroupKFold, honest OOF (no aux features), refit on full train for 44 epochs. **OOF 0.856 but LB 0.887 (+0.032 LB-OOF gap)** because (a) graph encoder benefits massively from +25% training data at refit, (b) prior LGB OOFs were aux-inflated so relative comparison was misleading. Biggest single-experiment jump of the competition. |
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

## LB landmarks (as of 2026-08-02, after blend submission)

| rank | team | score | gap to us (0.894) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch toh Karna hai | 0.902 | +0.008 |
| 2  | MUGABROS           | 0.900 | +0.006 |
| 3  | Opus 6.7           | 0.898 | +0.004 |
| 4  | 『VOID』           | 0.894 | tie |
| **5** | **Dhruval Padia (us)** | **0.894** | **—** |
| 6  | ShiokParikh06      | 0.894 | tie |
| 7  | The Invincibles    | 0.891 | -0.003 |

Score targets by remaining experiments (ordered by EV):
- **Chemprop 3-seed bag → re-blend** → +0.003 to +0.008 → **~0.90 → rank 1-3**. Highest-EV local lever.
- **Longer Chemprop training (60 epochs) → re-blend** → +0.002 to +0.005.
- **Add CatBoost as third base model → 3-way blend** → +0.002 to +0.006. Local.
- **PI1M SSL pretrain + Chemprop fine-tune** (Kaggle GPU only) → +0.005 to +0.015 → **could hit 0.90–0.91**.
- **LB distribution shift probe** (research doc §9, 3 subs) → 0 or **+0.03 hidden lift** if shift exists.

## OOF-vs-LB tracking

| exp | OOF | LB | LB−OOF | LB Δ | note |
|-----|:---:|:--:|:------:|:----:|------|
| baseline | 0.8345 | 0.843 | +0.009 | — | refit-on-full-train boost, first sub |
| matcomp  | 0.8527 | 0.857 | +0.004 | +0.014 | consistent boost, matrix-completion pays off big |
| full_fp  | 0.8575 | 0.859 | +0.002 | +0.002 | tiny lift; OOF-LB gap narrowing |
| trimmed  | 0.8610 | 0.858 | -0.003 | -0.001 | OOF up but LB flat |
| maxwell  | 0.8656 | 0.860 | -0.007 | +0.001 | worst OOF-LB gap for LGB — aux-augmented CV inflating OOF |
| chemprop | 0.8555 | 0.887 | +0.032 | +0.027 | honest OOF (no aux) + graph encoder benefits from full-data refit → LB WAY above OOF |
| **blend_nnls** | **0.8828** | **0.894** | **+0.011** | **+0.007** | **ensemble of chemprop+lgb, per-target NNLS with Chemprop bias. OOF is average of one honest + one aux-inflated → gap smaller than pure chemprop.** |

**Read of the trend.** The LGB experiments (baseline through maxwell) had a shrinking / negative OOF-LB gap because aux-augmented CV was inflating OOF. Chemprop broke the pattern: honest OOF (no aux) + a model family that benefits substantially from more training data → OOF underestimated LB by 0.032. Going forward, when comparing across model families we should trust LB, not OOF. Within a single model family, OOF trends remain informative.
