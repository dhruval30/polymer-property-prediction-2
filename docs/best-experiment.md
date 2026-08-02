# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

**Companion doc:** for reproducing ensembles (which depend on multiple base-model experiments), see [best-ensemble.md](best-ensemble.md).

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_blend_nnls_3seed` (ENSEMBLE) |
| **LB score (public)** | **0.897** 🎯 |
| **LB rank** | **5 / 154** (tied with 1 other at 0.897; rank 3 Opus 6.7 at 0.898 is +0.001 away) |
| **CV OOF mean R²** | 0.8873 |
| **submission file** | `results/exp_blend_nnls_3seed/submission.csv` |
| **script** | `experiments/exp_blend_nnls_3seed.py` (see [best-ensemble.md](best-ensemble.md) for full reproduction) |
| **date submitted** | 2026-08-02 |
| **wall time (local)** | 0.2s blend + 2 base models (~240 min total: Chemprop 3-seed 225min + LGB+Maxwell 15min) |
| **Δ vs previous best** | **+0.002** (from 3-way blend 0.895) |

> **What changed:** swapped single-seed Chemprop for the 5-fold × 3-seed bagged version. Same LGB+Maxwell base, same NNLS blend logic. Chemprop base is now stronger → NNLS gives it more weight per-target (mean w_chem 0.61 → 0.70). 6 of 7 target OOFs improved (only ei slightly regressed).

## Best solo (non-ensemble) submission

| field | value |
|---|---|
| **experiment** | `exp_chemprop_multitask_cpu_3seed` |
| **LB score** | **0.892** |
| **CV OOF mean R²** | 0.8701 |
| **submission file** | `results/exp_chemprop_multitask_cpu_3seed/submission.csv` |
| **script** | `experiments/exp_chemprop_multitask_cpu_3seed.py` |
| **wall time** | 224.7 min on Mac CPU (5-fold × 3-seed × 60 epochs + 3 refits × 50 epochs = 18 model trainings total) |
| **Δ vs prior solo best** | +0.005 (single-seed Chemprop at 0.887) |

The 3-seed bag improves OOF by +0.015 mean R² over single-seed (biggest wins: eps +0.033, egc +0.024, eea +0.020, tg +0.015). LB gain is smaller (+0.005) because the bag already averages some of the variance that single-seed's refit-on-full-data captured. This is the best-performing SOLO base model in the pipeline — makes it the strongest candidate to replace single-seed Chemprop in the ensemble blend.

### Per-target OOF R² (this submission — 2-way with 3-seed Chemprop)

| target | chemprop 3-seed | lgb+max | **blend** | w_c | w_l | Δ vs 3-way blend ref |
|--------|:---------------:|:-------:|:---------:|:---:|:---:|:--------------------:|
| eea | 0.9082 | 0.8708 | **0.9125** | 0.88 | 0.12 | +0.010 |
| egb | 0.9305 | 0.9105 | **0.9351** | 0.83 | 0.17 | +0.003 |
| egc | 0.9070 | 0.9000 | **0.9170** | 0.71 | 0.29 | +0.004 |
| ei  | 0.7766 | 0.7933 | **0.8073** | 0.56 | 0.44 | -0.006 |
| eps | 0.7916 | 0.8186 | **0.8362** | 0.54 | 0.46 | +0.007 |
| nc  | 0.8681 | 0.8603 | **0.8858** | 0.69 | 0.31 | +0.002 |
| tg  | 0.9083 | 0.9057 | **0.9170** | 0.68 | 0.32 | +0.003 |
| **mean OOF** | **0.8701** | **0.8656** | **0.8873** | 0.70 | 0.30 | +0.003 |
| **LB actual** | 0.892 | 0.860 | **0.897** | — | — | +0.002 |

6 of 7 targets improved over the 3-way blend (which included CatBoost). The stronger 3-seed Chemprop base carries more weight than the CAT contribution ever did — and does so at *lower total wall time* than the 3-way blend (240 min vs 267 min).

### Approach in one paragraph

Per-target NNLS blend of two base models: 5-fold × 3-seed multitask Chemprop D-MPNN (`exp_chemprop_multitask_cpu_3seed`, LB 0.892) and LightGBM+Maxwell physics-prior (`exp_maxwell_prior_lgbm`, LB 0.860). Weights fit per target on aligned OOFs via `scipy.optimize.nnls`, normalized to sum=1, then adjusted with LB-bias mitigations (Chemprop weight floor 0.40, +0.15 additive bias). Blend script runs in <1 second — all compute is in the base models.

### Why this beat the 3-way blend

The 3-way blend (Chemprop single-seed + LGB + CAT) added CatBoost's marginal per-target skill on egc/ei/tg but cost 100 extra minutes for +0.001 LB. This 2-way blend instead **upgrades the Chemprop base itself** (3-seed bag instead of single-seed) — a stronger single signal beats adding a weaker third signal. Chemprop 3-seed solo (LB 0.892) is worth more in a blend than adding a redundant tree model (CAT LB ~0.860).

**Per-target intuition:** on each target, the blend gets stronger raw Chemprop signal (+0.015 OOF gain from 3-seed vs single-seed), which propagates through NNLS. The blend weights show it: mean w_chemprop went from 0.61 (single-seed 2-way) to 0.70 (3-seed 2-way).

### Runtime

Total ~240 min from clean repo (~225 min Chemprop 3-seed + ~15 min LGB+Maxwell + <1 sec blend). Cheaper than the 3-way blend (~267 min) AND scores higher. **Best ROI ensemble.**

### What NOT in this submission (top future levers, ordered by EV)

- ~~Re-blend with 3-seed Chemprop~~ ✅ **DONE** — this submission. +0.002 LB.
- ~~Chemprop 3-seed bagging~~ ✅ **DONE**.
- ~~Longer Chemprop training (60 epochs)~~ ✅ **DONE**.
- ~~Add CatBoost as third base~~ ✅ **DONE** — marginal, and now superseded by the 2-way 3-seed variant.
- ❌ **3-way blend with 3-seed Chemprop + LGB + CAT.** Likely +0.001 LB based on prior 3-way vs 2-way pattern. ~+100 min if CAT cache already exists. Marginal but easy.
- ❌ **LB distribution shift probe** (research doc §9) — 3 subs could unlock up to +0.03 hidden shift correction. **HIGHEST-EV single lever** if there's a shift. Getting more attractive as we approach the ceiling.
- ❌ **Chemprop `--polymer` mode** (Coley group fork) with weighted repeat-unit bonds. ~4h more.
- ❌ **PI1M SSL pretraining** on tg / egc chemistry (research doc §6). Kaggle GPU only.
- ❌ **5-seed Chemprop bag instead of 3-seed** — diminishing returns, another ~3.5h.
- ❌ **Ridge meta-stacker with cross-target OOF as features** (research doc §8.5).

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 10 | 2026-08-02 | `exp_blend_nnls_3seed` **(ensemble)** | **0.897** 🎯 | ↑ +0.002 | **5** (tied with 4) | **0.8873** | 2-way per-target NNLS blend of **3-seed** Chemprop + LGB+Maxwell. Same NNLS + bias mitigations as single-seed 2-way (floor 0.40, bias +0.15). Chemprop base upgrade from 0.887 → 0.892 solo propagated as +0.002 LB blend lift. NNLS gave Chemprop more weight per-target (mean 0.61 → 0.70) because 3-seed base is genuinely stronger. **Beats the 3-way blend at lower wall time** — Chemprop base upgrade > adding CatBoost. Rank 3 (Opus 6.7, 0.898) only +0.001 away. |
| 9 | 2026-08-02 | `exp_chemprop_multitask_cpu_3seed` | 0.892 (best solo) | — vs blend | ~6 | 0.8701 | 5-fold × 3-seed Chemprop bag, max_epochs 60, patience 10. 224 min wall time. OOF beats single-seed by +0.015. LB +0.005 vs single-seed. Best solo model in the pipeline — feeds into blend #10. |
| 8 | 2026-08-02 | `exp_blend_nnls_3way` **(ensemble)** | **0.895** 🎯 | ↑ +0.001 | ~4 | 0.8842 | 3-way per-target NNLS blend of Chemprop + LGB+Maxwell + CatBoost+Maxwell. Same bias-mitigation config as 2-way (Chemprop floor 0.40, bias +0.15). CAT gets meaningful weight (0.27-0.31) on egc/ei/tg where it wins solo; zero weight on egb/eps where redundant with LGB. Blend OOF +0.001 over 2-way. LB +0.001. Marginal gain for +100 min CAT compute — poor ROI. |
| 7 | 2026-08-02 | `exp_blend_nnls` **(ensemble)** | 0.894 | ↑ +0.007 | 5 | 0.8828 | Per-target NNLS blend of Chemprop + LGB+Maxwell. Chemprop weight floor 0.40 + bias +0.15. Every target's blend OOF improved over either base (+0.007 to +0.023). Weights lean Chemprop on small-data/multitask targets (eea 0.75, egb 0.77, nc 0.65), lean LGB on physics/larger-data (eps 0.45, tg 0.54). LB +0.007 over pure Chemprop. **Preferred ensemble for reproduction** (~67 min vs 3-way's ~167 min for only +0.001 LB gain). |
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

## LB landmarks (as of 2026-08-02, after 3-seed blend submission)

| rank | team | score | gap to us (0.897) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch toh Karna hai | 0.902 | +0.005 |
| 2  | MUGABROS           | 0.900 | +0.003 |
| 3  | Opus 6.7           | 0.898 | +0.001 |
| 4  | (tied at 0.897)    | 0.897 | tie |
| **5** | **Dhruval Padia (us)** | **0.897** | **—** |

Score targets by remaining experiments (ordered by EV):
- **LB distribution shift probe** (research doc §9, 3 subs) → 0 or **+0.03 hidden lift** if shift exists. **HIGHEST-EV single lever now.** At our current score, +0.005 puts us at #1.
- **3-way blend with 3-seed Chemprop + LGB + CAT** → probably +0.001, easy since bases already exist.
- **Chemprop `--polymer` mode** (Coley group fork) with weighted repeat-unit bonds → +0.002 to +0.005. ~4h more compute.
- **5-seed instead of 3-seed Chemprop bag** → diminishing returns, +0.001 to +0.003 for +3.5h.
- **PI1M SSL pretrain + Chemprop fine-tune** (Kaggle GPU only) → +0.005 to +0.015.

## OOF-vs-LB tracking

| exp | OOF | LB | LB−OOF | LB Δ | note |
|-----|:---:|:--:|:------:|:----:|------|
| baseline | 0.8345 | 0.843 | +0.009 | — | refit-on-full-train boost, first sub |
| matcomp  | 0.8527 | 0.857 | +0.004 | +0.014 | consistent boost, matrix-completion pays off big |
| full_fp  | 0.8575 | 0.859 | +0.002 | +0.002 | tiny lift; OOF-LB gap narrowing |
| trimmed  | 0.8610 | 0.858 | -0.003 | -0.001 | OOF up but LB flat |
| maxwell  | 0.8656 | 0.860 | -0.007 | +0.001 | worst OOF-LB gap for LGB — aux-augmented CV inflating OOF |
| chemprop | 0.8555 | 0.887 | +0.032 | +0.027 | honest OOF (no aux) + graph encoder benefits from full-data refit → LB WAY above OOF |
| catboost | 0.8602 | ~0.860 est | ~-0.000 | — (unsubmitted) | tied with LGB solo, wall time 100 min |
| blend_nnls (2-way) | 0.8828 | 0.894 | +0.011 | +0.007 | ensemble of chemprop+lgb, per-target NNLS with Chemprop bias. Preferred for reproduction. |
| blend_nnls_3way | 0.8842 | 0.895 | +0.011 | +0.001 | 3-way (adds CatBoost). Marginal +0.001 for +100 min compute. |
| chemprop_3seed | 0.8701 | 0.892 | +0.022 | +0.005 (vs single-seed) | Best solo model. Smaller gap than single-seed because bagging already captures some refit variance. |
| **blend_nnls_3seed** | **0.8873** | **0.897** | **+0.010** | **+0.002** | **2-way blend with 3-seed Chemprop replacing single-seed. Best overall — Chemprop base upgrade > adding CatBoost.** |

**Read of the trend.** The LGB experiments (baseline through maxwell) had a shrinking / negative OOF-LB gap because aux-augmented CV was inflating OOF. Chemprop broke the pattern: honest OOF (no aux) + a model family that benefits substantially from more training data → OOF underestimated LB by 0.032. Going forward, when comparing across model families we should trust LB, not OOF. Within a single model family, OOF trends remain informative.
