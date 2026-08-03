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
| **experiment** | `exp_chain_ext_lgbm` |
| **LB score** | **0.894** 🏆 (best solo of the competition) |
| **CV OOF mean R²** | 0.8662 |
| **submission file** | `results/exp_chain_ext_lgbm/submission.csv` |
| **script** | `experiments/exp_chain_ext_lgbm.py` |
| **wall time** | 36.5 min on Mac CPU (featurize monomer + trimer + LGB × 5-fold × 7 targets + Maxwell blend) |
| **Δ vs prior solo best** | +0.002 (3-seed Chemprop at 0.892) |

**A single LightGBM model matched the 2-way blend (single-seed) at LB 0.894.** Extended every polymer `*A*` SMILES to a trimer `*AAA*` (head-to-tail via RDKit RWMol), computed a streamlined feature stack on the trimer (RDKit descriptors, Morgan-r2, MACCS, atom-pair, Avalon), and stacked it alongside the full monomer stack (~14k features total + 14 aux). Trimer features carried 38–72% of the per-target gain share — LGB actively prefers trimer features on tg (72%), ei (69%), egb (63%). Maxwell EPS↔Nc physics prior applied on top. **LB gap +0.034 vs prior LGB best (0.860 Maxwell mono-only)** — chain extension unlocked way more LB skill than the +0.005 OOF gain suggested, likely because trimer features generalize to test-set polymers that don't share monomer patterns with train.

### Per-target OOF R² (best-solo — chain-extended LGB)

| target | mono-only LGB (LB 0.860) | **chain-ext LGB (LB 0.894)** | Δ OOF | trimer gain share |
|--------|:------------------------:|:----------------------------:|:-----:|:-----------------:|
| eea | 0.8543 | **0.8734** | +0.019 | 43% |
| egb | 0.9050 | **0.9087** | +0.004 | 63% |
| egc | 0.8966 | **0.9023** | +0.006 | 38% |
| ei  | 0.7944 | **0.8041** | +0.010 | 69% |
| eps | 0.8186 | **0.8218** | +0.003 | 56% |
| nc  | 0.8603 | **0.8471** | **-0.013** ⚠️ | 63% (dropped) |
| tg  | 0.9026 | **0.9063** | +0.004 | 72% |
| **mean OOF** | **0.8617** | **0.8662** | **+0.005** | avg 57% |
| **LB actual** | 0.860 | **0.894** | **+0.034** | — |

6 of 7 targets improved on OOF; only **nc regressed -0.013** (probably because nc's tight value range 1.5–2.7 amplifies R² sensitivity to prediction noise, and trimer features add structural context that blurs subtle refractive-index signal). Despite the nc regression, LB shot up +0.034 — trimer features clearly generalize to test-set polymers better than the OOF suggested.

### Ensemble (best overall) — 2-way NNLS blend, still current best on LB

| field | value |
|---|---|
| **experiment** | `exp_blend_nnls_3seed` (ENSEMBLE) |
| **LB score** | **0.897** 🎯 |
| **CV OOF mean R²** | 0.8873 |

The chain-ext LGB solo at 0.894 is **only 0.003 below the current best ensemble** — meaning a new blend of chain-ext LGB + 3-seed Chemprop should push us well above 0.897. See [best-ensemble.md](best-ensemble.md).

### Approach in one paragraph

Per-target NNLS blend of two base models: 5-fold × 3-seed multitask Chemprop D-MPNN (`exp_chemprop_multitask_cpu_3seed`, LB 0.892) and LightGBM+Maxwell physics-prior (`exp_maxwell_prior_lgbm`, LB 0.860). Weights fit per target on aligned OOFs via `scipy.optimize.nnls`, normalized to sum=1, then adjusted with LB-bias mitigations (Chemprop weight floor 0.40, +0.15 additive bias). Blend script runs in <1 second — all compute is in the base models.

### Why this beat the 3-way blend

The 3-way blend (Chemprop single-seed + LGB + CAT) added CatBoost's marginal per-target skill on egc/ei/tg but cost 100 extra minutes for +0.001 LB. This 2-way blend instead **upgrades the Chemprop base itself** (3-seed bag instead of single-seed) — a stronger single signal beats adding a weaker third signal. Chemprop 3-seed solo (LB 0.892) is worth more in a blend than adding a redundant tree model (CAT LB ~0.860).

**Per-target intuition:** on each target, the blend gets stronger raw Chemprop signal (+0.015 OOF gain from 3-seed vs single-seed), which propagates through NNLS. The blend weights show it: mean w_chemprop went from 0.61 (single-seed 2-way) to 0.70 (3-seed 2-way).

### Runtime

Total ~240 min from clean repo (~225 min Chemprop 3-seed + ~15 min LGB+Maxwell + <1 sec blend). Cheaper than the 3-way blend (~267 min) AND scores higher. **Best ROI ensemble.**

### What NOT in this submission (top future levers, ordered by EV)

- ~~Re-blend with 3-seed Chemprop~~ ✅ **DONE** — LB 0.897, current best ensemble.
- ~~Chemprop 3-seed bagging~~ ✅ **DONE** — LB 0.892 solo.
- ~~Longer Chemprop training (60 epochs)~~ ✅ **DONE**.
- ~~Add CatBoost as third base~~ ✅ **DONE** — marginal, now superseded.
- ~~Chain extension (polymer → trimer features)~~ ✅ **DONE** — LB 0.894 solo, biggest single-experiment LB jump for LGB (+0.034 vs mono-only). New best solo.
- ⭐ **Re-blend with chain-ext LGB + 3-seed Chemprop (2-way NNLS).** Chain-ext LGB solo (LB 0.894) is now nearly tied with Chemprop 3-seed (LB 0.892) — NNLS will find a more balanced weighting than the previous 0.70/0.30 split. **Expected LB: 0.900–0.905**, +0.003 to +0.008 lift. ~1 min to write, blend runs in <1 sec (bases already exist). **HIGHEST-EV next lever.**
- ❌ **3-way blend with chain-ext LGB + mono-LGB + 3-seed Chemprop.** NNLS could pick mono-LGB for nc (where chain-ext regressed) and chain-ext elsewhere. Safer than 2-way — but if 2-way already breaks 0.900 may be unnecessary. ~1 min to write.
- ❌ **LB distribution shift probe** (research doc §9) — 3 subs could unlock up to +0.03 hidden shift correction. Still on the table but chain extension may have already captured most of the shift signal.
- ❌ **Chemprop `--polymer` mode** (Coley group fork) with weighted repeat-unit bonds. ~4h more.
- ❌ **Chemprop on trimer SMILES instead of monomer.** Trimer graphs would be 3× larger — big compute hit. Worth trying if chain-ext LGB re-blend saturates.
- ❌ **PI1M SSL pretraining** on tg / egc chemistry (research doc §6). Kaggle GPU only.
- ❌ **5-seed Chemprop bag instead of 3-seed** — diminishing returns, another ~3.5h.
- ❌ **Fix nc regression** — could try chain-ext with only monomer features for nc, or increase chain length to 5-mer only for nc.

---

## Submission history

Every submission ever made, most-recent first. Arrows show delta vs previous entry: ↑ improvement, ↔ tie, ↓ regression.

| # | date | experiment | LB | Δ | rank | OOF | notes |
|--:|------|------------|:--:|:-:|:----:|:---:|-------|
| 11 | 2026-08-03 | `exp_chain_ext_lgbm` | **0.894** 🏆 (best solo) | — vs blend | ~5 | 0.8662 | **Polymer → trimer chain extension.** Extended each `*A*` monomer SMILES to `*AAA*` trimer via RDKit RWMol, computed streamlined feature stack on trimer, stacked with monomer full stack (~14k features + 14 aux). Maxwell prior on top. Trimer features earned 38–72% per-target gain share (biggest: tg 72%, ei 69%). 6 of 7 OOF wins; only nc regressed -0.013. **LB +0.034 vs mono-only Maxwell LGB (0.860) — massive jump.** New best solo, ties single-seed 2-way blend. Next: re-blend with 3-seed Chemprop → expect 0.900+. |
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
| chemprop_3seed | 0.8701 | 0.892 | +0.022 | +0.005 (vs single-seed) | Solo Chemprop bag. Smaller gap than single-seed because bagging already captures some refit variance. |
| **blend_nnls_3seed** | **0.8873** | **0.897** | **+0.010** | **+0.002** | **2-way blend with 3-seed Chemprop. Best overall.** |
| **chain_ext_lgbm** | **0.8662** | **0.894** | **+0.028** | **+0.002** (vs 3-seed Chemprop solo) | **NEW BEST SOLO.** Polymer → trimer chain extension applied to LGB. LB-OOF gap +0.028 huge — chain-ext features generalize to test-set polymers much better than fold-CV suggests. Trimer captures signal that survives distribution shift. |

**Read of the trend.** The LGB experiments (baseline through maxwell) had a shrinking / negative OOF-LB gap because aux-augmented CV was inflating OOF. Chemprop broke the pattern: honest OOF (no aux) + a model family that benefits substantially from more training data → OOF underestimated LB by 0.032. Going forward, when comparing across model families we should trust LB, not OOF. Within a single model family, OOF trends remain informative.
