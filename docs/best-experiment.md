# Best Experiment — Living Tracker

This doc always describes **the single best submission** we've made on the public LB, plus a history of every submission. Update this doc every time an experiment gives a higher LB score.

> **Rule:** only update the "Current best" block if the new LB > current best LB. Every submission (win or not) gets a history row.

**Companion doc:** for reproducing ensembles (which depend on multiple base-model experiments), see [best-ensemble.md](best-ensemble.md).

---

## Current best

| field | value |
|---|---|
| **experiment** | `exp_blend_nnls_3way` (ENSEMBLE) |
| **LB score (public)** | **0.895** 🎯 |
| **LB rank** | ~4 / 154 |
| **CV OOF mean R²** | 0.8842 |
| **submission file** | `results/exp_blend_nnls_3way/submission.csv` |
| **script** | `experiments/exp_blend_nnls_3way.py` (see [best-ensemble.md](best-ensemble.md) for full reproduction) |
| **date submitted** | 2026-08-02 |
| **wall time (local)** | 0.2s blend + 3 base models (~167min total: Chemprop 52min + LGB 15min + CatBoost 100min) |
| **Δ vs previous best** | **+0.001** (from 2-way blend 0.894) |

> **Reproduction note:** Δ +0.001 LB comes at the cost of +100 min of CatBoost compute. For practical reproduction the **2-way blend (Chemprop + LGB, LB 0.894, ~67 min)** is a much better ROI. See [best-ensemble.md § "Preferred vs best-scoring ensemble"](best-ensemble.md) for the tradeoff.

### Per-target OOF R² (3-way blend)

| target | chemprop | lgb+max | cat+max | **3-way blend** | w_c | w_l | w_x |
|--------|:--------:|:-------:|:-------:|:---------------:|:---:|:---:|:---:|
| eea | 0.8883 | 0.8708 | 0.8650 | **0.9022** | 0.72 | 0.15 | 0.13 |
| egb | 0.9251 | 0.9105 | 0.9002 | **0.9325** | 0.77 | 0.23 | 0.00 |
| egc | 0.8830 | 0.9000 | 0.9034 | **0.9133** | 0.50 | 0.22 | 0.27 |
| ei  | 0.7804 | 0.7933 | 0.7934 | **0.8137** | 0.56 | 0.14 | 0.31 |
| eps | 0.7584 | 0.8186 | 0.7978 | **0.8295** | 0.45 | 0.55 | 0.00 |
| nc  | 0.8599 | 0.8603 | 0.8530 | **0.8838** | 0.64 | 0.26 | 0.10 |
| tg  | 0.8934 | 0.9057 | 0.9083 | **0.9142** | 0.49 | 0.21 | 0.31 |
| **mean OOF** | 0.8555 | 0.8656 | 0.8602 | **0.8842** | 0.59 | 0.25 | 0.16 |
| **LB actual** | 0.887 | 0.860 | ~0.860 est. | **0.895** | — | — | — |

Every single target OOF improved in the 3-way blend vs any single base. CAT gets zero weight on egb and eps (fully redundant with LGB there); meaningful weight (0.27-0.31) on egc, ei, tg where CAT wins solo.

### Approach in one paragraph

Per-target NNLS blend of three base models: multitask Chemprop D-MPNN (`exp_chemprop_multitask_cpu`, LB 0.887), LightGBM+Maxwell physics-prior (`exp_maxwell_prior_lgbm`, LB 0.860), and CatBoost+Maxwell (`exp_maxwell_prior_catboost`, LB ~0.860). Weights fit per target on the aligned OOFs via `scipy.optimize.nnls`, normalized to sum=1, then adjusted with two LB-bias-aware mitigations: (a) Chemprop weight floor of 0.40 to prevent over-trusting the aux-inflated tree OOFs, and (b) +0.15 additive Chemprop bias, with the -0.15 redistribution split between LGB and CAT proportional to their normalized weights. Blend script runs in <1 second — all compute is in the base models. Full reproduction chain in [best-ensemble.md](best-ensemble.md).

### Why the 3-way blend beat the 2-way (barely)

CAT and LGB have per-target complementarity even though their mean OOFs are equal (0.860 vs 0.866):
- **CAT wins solo** on eea (+0.011), egc (+0.007), tg (+0.006) vs LGB
- **LGB wins solo** on egb (-0.005), eps (-0.021), nc (-0.007) vs CAT

NNLS confirmed the pattern per target: CAT gets meaningful weight (0.27-0.31) on the 3 targets where it wins solo, zero weight on egb and eps where it's redundant with LGB. Blend OOF 0.884 vs 2-way 0.883 (+0.001). LB gain +0.001 (from 0.894 to 0.895).

### Runtime vs score tradeoff

The +0.001 LB improvement over the 2-way blend costs +100 minutes of CatBoost training. **For practical reproduction the 2-way blend (Chemprop + LGB, LB 0.894, ~67 min total) is the preferred target.** Only rebuild the 3-way if the last 0.001 R² matters for rank. See [best-ensemble.md § "Preferred vs best-scoring ensemble"](best-ensemble.md).

### What NOT in this submission (top future levers, ordered by EV)

- ❌ **Chemprop 3-seed bagging** (5-fold × 3 seeds = 15 models, per Round-1 recipe). ~2.5h more Mac CPU. Expected +0.003 to +0.008. Then re-blend. **Highest-EV local lever now.**
- ❌ **Longer Chemprop training** — folds 0, 1, 4 hit max_epochs=40 without early stopping. Extend to 60 epochs, +30 min per fold.
- ~~Add CatBoost as a third base model~~ ✅ **DONE** — added; +0.001 LB for +100 min compute. Poor ROI but adopted.
- ❌ **Chemprop `--polymer` mode** (Coley group fork) with weighted repeat-unit bonds.
- ❌ **PI1M SSL pretraining** on tg / egc chemistry (research doc §6). Kaggle GPU only.
- ❌ **LB distribution shift probe** (research doc §9) — 3 subs could unlock up to +0.03 hidden shift correction. **HIGHEST-EV single lever** if there's a shift.
- ❌ **Ridge meta-stacker with cross-target OOF as features** (research doc §8.5).

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 8 | 2026-08-02 | `exp_blend_nnls_3way` **(ensemble)** | **0.895** 🎯 | ↑ +0.001 | ~4 | 0.8842 | 3-way per-target NNLS blend of Chemprop + LGB+Maxwell + CatBoost+Maxwell. Same bias-mitigation config as 2-way (Chemprop floor 0.40, bias +0.15). CAT gets meaningful weight (0.27-0.31) on egc/ei/tg where it wins solo; zero weight on egb/eps where redundant with LGB. Blend OOF +0.001 over 2-way. LB +0.001. **Marginal gain for +100 min CAT compute — poor ROI. 2-way remains the preferred reproduction target.** |
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

## LB landmarks (as of 2026-08-02, after 3-way blend submission)

| rank | team | score | gap to us (0.895) |
|------|------|:-----:|:-----------------:|
| 1  | Kuch toh Karna hai | 0.902 | +0.007 |
| 2  | MUGABROS           | 0.900 | +0.005 |
| 3  | Opus 6.7           | 0.898 | +0.003 |
| **~4** | **Dhruval Padia (us)** | **0.895** | **—** |

Score targets by remaining experiments (ordered by EV):
- **Chemprop 3-seed bag → re-blend** → +0.003 to +0.008 → **~0.90 → rank 1-3**. Highest-EV local lever now.
- **Longer Chemprop training (60 epochs) → re-blend** → +0.002 to +0.005.
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
| catboost | 0.8602 | ~0.860 est | ~-0.000 | — (unsubmitted) | tied with LGB solo, wall time 100 min |
| blend_nnls (2-way) | 0.8828 | 0.894 | +0.011 | +0.007 | ensemble of chemprop+lgb, per-target NNLS with Chemprop bias. Preferred for reproduction. |
| **blend_nnls_3way** | **0.8842** | **0.895** | **+0.011** | **+0.001** | **3-way (adds CatBoost). Marginal +0.001 LB for +100 min compute. Best score but 2-way is the practical target.** |

**Read of the trend.** The LGB experiments (baseline through maxwell) had a shrinking / negative OOF-LB gap because aux-augmented CV was inflating OOF. Chemprop broke the pattern: honest OOF (no aux) + a model family that benefits substantially from more training data → OOF underestimated LB by 0.032. Going forward, when comparing across model families we should trust LB, not OOF. Within a single model family, OOF trends remain informative.
