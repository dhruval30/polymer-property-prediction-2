# Session Handoff — Polymer Property Prediction Round 2

**Purpose:** self-contained brief for a research agent (Fable) or another engineer. Everything we've tried, why it failed or worked, and what we haven't touched. Use this as a prompt to find techniques we haven't considered.

---

## 1. Competition context

- **Kaggle-hosted "ANRF AISEHack 2.0 Theme 2 — Polymer Property Prediction Round 2".**
- **Task:** given a polymer SMILES (with two `*` wildcards marking repeat-unit endpoints), predict 7 physical properties: `eea, egb, egc, ei, eps, nc, tg`.
- **Metric:** mean R² across the 7 targets.
- **Data:**
  - Train: 7,405 (canon, target_type, value) rows across 5,920 unique canonical SMILES. Per-target counts: tg=4139, egc=2028, egb=337, eps=229, nc=229, ei=222, eea=221. **82% sparse** — most polymers have only 1-2 target labels.
  - Test: 4,940 (id, canon, target_type) rows across 4,133 unique canonical SMILES.
  - PI1M: 995,800 unlabeled polymer SMILES available as external data.
- **Constraints:** submissions must run within Kaggle notebook runtime (~12 h wall time, CPU or single-GPU). Some Kaggle sessions have CUDA kernel mismatch with recent PyTorch → CPU fallback needed.
- **LB is 37% of test.** Private LB is the other 63%.

## 2. Current standing (2026-08-06)

- **Our best submission: LB 0.897** (`exp_blend_nnls_3seed` — 2-way per-target NNLS blend of 3-seed Chemprop mono + LGB+Maxwell mono).
- **Best solo: chain-ext LGB v1, LB 0.894.**
- **Public rank: 8 / 154+** (fell from 5 as other teams pushed to 0.898-0.902).
- **Top 3 requires LB ≥ 0.903 (+0.006 from us).**
- **LB leaders:**
  - 1. MUGABROS 0.916 (15 entries)
  - 2. Sandman 0.916 (6 entries) — with only 6 subs, this is either a lucky private-model or PI1M/RankUp-style external data
  - 3. Kuch toh Karna hai 0.903 (24 entries)
  - 4. who-knows 0.902
  - 5. Cross Linkers 0.900
  - 6-8: cluster at 0.897-0.899 (including us)

## 3. Pipeline architecture (best-known-good)

```
canonical SMILES ─┬─► RDKit descriptors (~207)
                  ├─► Morgan-r2 count FP (2048)
                  ├─► Morgan-r3 count FP (2048)
                  ├─► MACCS keys (167)
                  ├─► Atom-pair count FP (2048)
                  ├─► Topological-torsion count FP (2048)
                  ├─► Avalon FP (512)                                → LGB per-target × 7 → LB 0.860
                  │                                                      │
polymer *A* ─► *AAA* (RWMol chain extension, trimer) ─┬─► same 5     │
                                                       │  fingerprint  │
                                                       └─► + trimer    ├─► LGB per-target × 7 → LB 0.894
                                                          features     │      (chain-ext LGB v1, our best solo)
                                                          (~4967)      │
                                                                       │
+ Maxwell EPS↔Nc physics blend (post-fit, w grid-searched on OOF)      │
+ 14 aux features per row (7 other-target-values + 7 presence masks)    │
                                                                        │
Chemprop 2.x multitask D-MPNN (5-fold × 3-seed bag, 60 epochs, mono SMILES) → LB 0.892
                                                                        │
Per-target NNLS blend (Chemprop 3-seed + LGB Maxwell mono) ────────────┘  → LB 0.897 (current best)
+ Chemprop weight floor 0.40 + additive bias +0.15 (calibrated to their OOF-LB gaps)
```

**Key config:**
- **5-fold GroupKFold on canonical SMILES, split_seed=42** — all experiments use this so OOFs are blend-alignable.
- LGB: `lr=0.03, num_leaves=63, min_child_samples=10, feature_fraction=0.5, bagging_fraction=0.85, reg_lambda=1.0, n_est=4000, early_stop=200`. Refit at `median(best_iter)*1.10`.
- Chemprop 2.x: `BondMessagePassing(d_h=300, depth=4, dropout=0.05)`, `MeanAggregation`, `RegressionFFN(hidden_dim=300, n_layers=2, dropout=0.05)`, `batch_norm=True, max_epochs=60, patience=10, batch_size=64`.

## 4. Complete experiment log (in order)

Every submission with LB, OOF, and 1-line note. Prior to chain-ext v1 was Round-1-like linear improvement; after chain-ext v1 was mostly failures.

| # | experiment | LB | OOF | Δ vs prev best | 1-line note |
|--:|------------|:--:|:---:|:--:|-------|
| 1 | `exp_baseline_lgbm` | 0.843 | 0.8345 | — | LGB per-target on Round-1 feature stack |
| 2 | `exp_matrix_completion_lgbm` | 0.857 | 0.8527 | +0.014 | added 14 aux features (7 other-target + 7 masks) |
| 3 | `exp_full_fp_lgbm` | 0.859 | 0.8575 | +0.002 | added morgan-r3, atom-pair, top-torsion, avalon |
| 4 | `exp_trimmed_smarts_lgbm` | 0.858 | 0.8610 | -0.001 | dropped weak FPs, added 25 SMARTS class flags |
| 5 | `exp_maxwell_prior_lgbm` | 0.860 | 0.8656 | +0.001 | added Maxwell EPS↔Nc physics blend |
| 6 | `exp_chemprop_multitask_cpu` | 0.887 | 0.8555 | +0.027 | multitask Chemprop D-MPNN, 5-fold, CPU |
| 7 | `exp_blend_nnls` | 0.894 | 0.8828 | +0.007 | 2-way NNLS blend (Chemprop + LGB+Maxwell) |
| 8 | `exp_blend_nnls_3way` | 0.895 | 0.8842 | +0.001 | 3-way add CatBoost+Maxwell — marginal for 100min |
| 9 | `exp_chemprop_multitask_cpu_3seed` | 0.892 | 0.8701 | — | 5-fold × 3-seed Chemprop bag, 60 epochs |
| **10** | **`exp_blend_nnls_3seed`** | **0.897** | **0.8873** | **+0.002** | **2-way with 3-seed Chemprop → current best** |
| 11 | `exp_chain_ext_lgbm` | 0.894 | 0.8662 | — solo | **chain-ext (polymer → trimer via RWMol)** — best solo, +0.034 solo LB, +0.028 OOF-LB gap |
| 12 | `exp_blend_nnls_chainext` (2-way) | 0.893 | 0.8893 | -0.004 | Chemprop 3-seed + chain-ext LGB — TOO CORRELATED |
| 13 | `exp_blend_nnls_chainext_3way` | 0.894 | 0.8903 | -0.003 | Add mono LGB back — same correlation problem |
| 14 | `exp_chemprop_multitask_chainext_cpu` | 0.891 | 0.8793 | -0.001 vs mono Chemprop | Chemprop on trimer SMILES — 27.5h wall time, KAGGLE-INCOMPATIBLE |
| 15 | `exp_chain_ext_lgbm_v2` | **0.868** | 0.8776 | **-0.026** | Per-target Optuna + target transforms + nc-fix + bandgap — OOF-LB gap collapsed |
| 16 | `exp_lb_shift_probe` | -0.007 | — | (probe) | Train_mean per target — no LB shift detected (`sum_t shift²/std² ≤ 0.049`) |
| 17 | `exp_chain_ext_lgbm_5mer` | (not submitted) | 0.8604 | -0.006 OOF | Pentamer chain extension — pentamer too big for small-data targets |
| 18 | `exp_chain_ext_lgbm_v3` | **0.601** | 0.8954 | **-0.293** | IterImputer for aux with fold-alignment bug — leaked val fold targets, aux gain share hit 91% |
| 19 | `exp_chain_ext_lgbm_v3fixed` | **0.857** | 0.8696 | **-0.037** | No IterImputer, only nc-fix + bandgap + 15 domain features — domain features overfit (30-39% gain share) |
| 20 | `exp_blend_lgb_mlp` | **0.867** | 0.8796 | **-0.027** | LGB + multitask MLP blend — OOF perfect (+0.013 all-target improvement) but MLP's negative OOF-LB gap wrecked LB |
| 21 | `exp_chain_ext_catboost` | (not submitted) | 0.8605 | -0.006 OOF | CatBoost on chain-ext features — slightly worse LGB clone |

Also aborted: PI1M pseudo-labeling (leak + 10h+ wall time; diagnostic showed 91% test coverage at Tanimoto ≥ 0.5, so worth revisiting with correct per-fold pilots).

## 5. Recurring failure pattern

**Every single "improvement" past chain-ext v1 has failed on LB, always with the same signature:**
- OOF R² up (sometimes dramatically — v3 was +0.03)
- **LB R² down** (sometimes catastrophically — v3 was -0.29)
- OOF-LB gap flips from LGB v1's healthy **+0.028** to negative

**Why (mechanistic):** chain-ext LGB v1 works because chain-extension features (RDKit descriptors + fingerprints computed on the trimer) genuinely generalize from train chemistry to test chemistry. This gives a "generalization slack" of +0.028 that shows up as LB > OOF. Any modification that tightens fit to OOF (Optuna, transform search, adding features with high gain-per-feature ratio, blend partners with worse OOF-LB gaps) consumes that slack and pulls LB down.

**Six variations of this failure:**
1. **v2 Optuna:** per-target hparam tune maximized fold-CV variance. LB -0.026.
2. **v3 IterImputer:** fold-alignment bug via different-length permutations. Val fold's target values leaked into aux matrix via imputer. Aux gain share reached 91% → LB collapsed.
3. **v3fixed domain features:** 15 hand-engineered features (fluorine counts, backbone rigidity, Kier-Hall, SMARTS flags) captured 30-39% of gain on some targets. Overfit train's structural distribution. LB -0.037.
4. **5-mer chain:** pentamer molecules too big for small-data targets (220 rows). More features, worse signal-per-feature ratio. OOF regressed.
5. **Chain-ext blends:** chain-ext LGB errors correlate with Chemprop errors (both learn polymer context via different mechanisms). Blend LB below LGB solo.
6. **MLP blend:** MLP OOF-LB gap is negative (aggressive early stopping = zero generalization slack). Blend LB pulled toward MLP's true LB (~0.84). Textbook OOF improvement doesn't imply LB improvement when bases have divergent gaps.

## 6. Distilled lessons (verified across 8+ failures)

1. **Trust LB, not OOF, when adding new ingredients to chain-ext v1.** OOF is a fold-fit metric; our chain-ext win comes from a property (generalization slack) that only shows up on LB.
2. **A small feature block getting >20% gain share is a WARNING, not a triumph.** If 15 domain features earn 30% of gain when Morgan bits (2048 features) earn 5%, those 15 features are 50× more informative per feature — usually overfit, not miraculous.
3. **Blend diversity requires SIMILAR OOF-LB gaps.** Chemprop (+0.032 gap) + LGB (-0.006 gap after aux inflation) blended cleanly to 0.897. MLP (probably -0.02 gap) + LGB (+0.028 gap) blended to 0.867 despite great OOF. Match the gaps or don't blend.
4. **Per-target Optuna / transform search on 220-row targets always overfits.** The compound selection bias across 5 transforms × 30 trials is too much for the small-data targets.
5. **LB shift probe is legit but expensive.** Our probe returned R² = -0.007, ruling out major distribution shifts (unlike NeurIPS 2025 2nd place's Tg trick). Cost us 1 sub slot but eliminated a whole hypothesis class.
6. **PI1M distribution IS similar enough** (91% of test at Tanimoto ≥ 0.5, adv AUC 0.876). The theoretical basis for pseudo-labeling to work is there — the pipeline that tries it just needs per-fold pilots to avoid leakage.
7. **Chemprop is Kaggle-runtime-borderline.** Mono 3-seed × 60 epochs was 3.75h on Mac CPU. Trimer 3-seed × 60 epochs was 27.5h — impossible to run in a Kaggle notebook.

## 7. Everything I HAVE tried (compact list)

- ✅ Chain extension trimer (via RDKit RWMol)
- ✅ Full fingerprint stack (Morgan-r2/r3, MACCS, AtomPair, TopTorsion, Avalon, RDKit desc)
- ✅ Aux matrix completion via single-pass target-lookup + 7 presence masks
- ✅ Maxwell EPS↔Nc physics prior with grid-searched blend weight
- ✅ Chemprop 2.x multitask D-MPNN, 5-fold × 3-seed bag
- ✅ Per-target NNLS blend with Chemprop weight floor/bias corrections
- ✅ 3-way NNLS blend with CatBoost as third base
- ✅ Chemprop on trimer SMILES (worked but too slow for Kaggle)
- ✅ Per-target Optuna hyperparameter tune
- ✅ Per-target target-transform search (identity/log1p/sqrt/yeo-johnson/rank-Gauss)
- ✅ Nc-fix (drop trimer features for nc target only)
- ✅ Bandgap consistency post-processor (Egc, Egb, Ei, Eea cross-target physics)
- ✅ 15 hand-engineered domain features (F count, backbone rigidity, Kier-Hall, SMARTS flags)
- ✅ Pentamer (5-mer) chain extension
- ✅ IterativeImputer for aux matrix (with fatal fold-alignment bug)
- ✅ LB distribution shift probe (single-sub train_mean probe)
- ✅ PI1M diagnostic (Tanimoto NN + adversarial validation)
- ✅ Multitask MLP with shared trunk + 7 heads (chain-ext features, no aux)
- ✅ CatBoost on chain-ext feature stack (Round 1 hparams)
- ✅ Per-target NNLS blend of LGB + MLP (pure NNLS, no bias)
- ❌ Aborted: PI1M pseudo-label augmentation (leak + 10h+ wall time)

## 8. What we HAVEN'T tried (research priority list)

**High-EV, unexplored:**
1. **RankUp pseudo-label pretraining** (NeurIPS 2024) — 1st-place NeurIPS 2025 recipe. Train student MLP with pairwise ranking loss on teacher-pseudo-labels of PI1M. Might be the reason Sandman/MUGABROS hit 0.916+.
2. **PI1M pseudo-labeling done right** (per-fold pilots + smaller PI1M sample, e.g. 10K) — the theoretical basis is there (91% test coverage), execution failed.
3. **Bicerano-style group additivity for Tg** — classical polymer property estimator, would provide error-diverse base for Tg blend.
4. **Van Krevelen group contributions for dielectric ε (eps)** — same idea for eps.
5. **Mordred descriptor library** — ~1600 additional descriptors beyond RDKit. All top-5 NeurIPS 2025 solutions used Mordred.
6. **Uni-Mol / MMPolymer 3D features** — expensive (needs 3D conformer generation, ~20h+ for 1M PI1M molecules), but incorporates 3D shape.
7. **Chemprop with `--polymer` mode** (coleygroup/polymer-chemprop) — native weighted repeat-unit bonds instead of `*` capping. Different molecular representation.
8. **Rank-based ensemble** instead of mean/NNLS — rank the base predictions per target, average ranks, map back. Robust to per-model scale bias.
9. **Cross-target meta-features** — feed OOF of other 6 targets as features to a per-target Ridge meta-learner on top of NNLS.
10. **Scaffold-balanced folds** (Bemis-Murcko clustering + stratified GroupKFold) — smoother OOF variance. Fold 4 has consistently trailed on our small targets.
11. **Adversarial validation on features** — train train-vs-test classifier on our 14K features, drop top-discriminating ones. If train-test AUC > 0.6 (we saw 0.488 — no shift), might not help.

**Medium-EV, worth considering:**
12. **Chemprop with SMILES enumeration TTA** — predict on 5 randomized SMILES per test row, average. Chemprop is theoretically permutation-invariant but empirically shows small variance.
13. **Multiple random seeds for chain-ext LGB v1 alone** — variance reduction on the strongest solo. Might give +0.001-0.003 LB stably.
14. **NGBoost** — probabilistic GBM, gives per-prediction uncertainty. Small-data specialty.
15. **Gaussian Process per target** — genuinely different mathematical framework. Historically strong on small-data chemistry. O(n³) cost is a concern for tg (4139 rows).
16. **TabPFN** — priors-fit-in-context transformer, zero training. But hard cap of ~500 features and 1000 samples — would require aggressive feature selection.
17. **Auto-sklearn / AutoGluon TabularPredictor** — automatic ensemble frameworks. Slow.

**Skip these (tried in Round 1 or ruled out by rules/compute):**
- Gasteiger charges (Round 1 negative result even after `*` capping)
- 3D conformer descriptors (Round 1 negative result)
- Coulomb matrix eigenvalues (Round 1 negative result)
- polyBERT / TransPolymer / MMPolymer pretrained weights (disallowed by rules)
- CatBoost meta-stacker (Round 1 overfit vs simple NNLS)
- MMoE / cross-stitch multi-task architectures (need >10K per task, we have 220)
- Aggressive tautomer/stereoisomer enumeration (OPC post-comp report says overfits)

## 9. Specific research questions

Please investigate:

1. **RankUp pretraining details** — for the 1st-place NeurIPS Open Polymer Prediction 2025 solution (jday96314):
   - Exact pairwise ranking loss formulation with margin calibration
   - What model architecture and pretrain schedule for a from-scratch encoder?
   - How they combined RankUp-pretrained encoder with tabular ensemble
   - Practical Kaggle-notebook-runtime feasibility of their approach

2. **PI1M pseudo-labeling recipes that worked** — Round 1 pseudo-labeling failed. What made it fail (Round-1 was without matrix completion structure)? What are the known best practices for pseudo-label weighting, filtering, and per-fold pilot generation to avoid fold-alignment leaks?

3. **Bicerano group additivity for Tg** — precise SMARTS patterns and coefficients from Bicerano's book (or equivalent papers). Is there a Python implementation? What's typical accuracy on independent polymer test sets?

4. **Van Krevelen group contributions for dielectric constant** — same as above for epsilon.

5. **Mordred descriptor library** — feature families that specifically help polymer prediction. Which subsets are most informative? Are there known descriptors that correlate with our target physics (n, ε, ionization/affinity, Tg)?

6. **The 2nd place NeurIPS 2025 team's "Tg shift" trick** — our LB probe found NO shift. Did their competition have a shift specific to Tg only, or is there something in the metric definition we're missing?

7. **Are there polymer-specific model architectures** (beyond Chemprop, GATv2, polyBERT) that fit in Kaggle-notebook runtime for 5920 train samples + 4133 test?

8. **How top teams handle small-data targets (200-400 rows)** — hyperparameter tuning that doesn't overfit, feature selection that doesn't leak, external data augmentation.

9. **Is there any known way to preserve a positive OOF-LB gap while adding features?** — our chain-ext v1's +0.028 gap is what wins us LB 0.894. Every attempt to add on top has destroyed it.

10. **Ensemble strategies for models with divergent OOF-LB gaps** — how do you calibrate blend weights when base models over/under-report their OOF R²?

## 10. Files and directories relevant to handoff

```
polymer-property-prediction-2/
├── CLAUDE.md                        # project conventions
├── docs/
│   ├── best-experiment.md          # living tracker of all submissions
│   ├── best-ensemble.md            # ensemble reproduction chain
│   ├── research.md                 # Round 2 idea list (partially executed)
│   └── SESSION_HANDOFF.md          # THIS FILE
├── experiments/
│   ├── exp_baseline_lgbm.py
│   ├── exp_matrix_completion_lgbm.py
│   ├── exp_full_fp_lgbm.py
│   ├── exp_trimmed_smarts_lgbm.py
│   ├── exp_maxwell_prior_lgbm.py
│   ├── exp_maxwell_prior_catboost.py
│   ├── exp_chemprop_multitask_cpu.py
│   ├── exp_chemprop_multitask_cpu_3seed.py
│   ├── exp_chemprop_multitask_chainext_cpu.py
│   ├── exp_chain_ext_lgbm.py                 # BEST SOLO
│   ├── exp_chain_ext_lgbm_v2.py              # Optuna disaster
│   ├── exp_chain_ext_lgbm_v3.py              # IterImputer disaster
│   ├── exp_chain_ext_lgbm_v3fixed.py         # domain features disaster
│   ├── exp_chain_ext_lgbm_5mer.py            # 5-mer, unsubmitted
│   ├── exp_chain_ext_catboost.py             # CatBoost, unsubmitted
│   ├── exp_chain_ext_mlp.py                  # multitask MLP
│   ├── exp_blend_nnls.py                     # 2-way single-seed blend
│   ├── exp_blend_nnls_3way.py                # 3-way with CAT
│   ├── exp_blend_nnls_3seed.py               # BEST BLEND (LB 0.897)
│   ├── exp_blend_nnls_chainext.py            # failed chain-ext blend
│   ├── exp_blend_nnls_chainext_3way.py       # failed 3-way chain-ext blend
│   ├── exp_blend_lgb_mlp.py                  # failed LGB+MLP blend
│   ├── exp_lb_shift_probe.py                 # single-sub probe
│   └── exp_pi1m_diagnostic.py                # PI1M similarity check
└── results/                                   # per-experiment outputs
    └── <exp_name>/{run.log, oof.csv, submission.csv, cv_summary.json}
```

Every experiment writes a `run.log` (Python `logging`, stdout+file) and per-target OOF/submission CSVs in the same format for cross-experiment blending. Split seed 42 is used across all experiments so OOFs are canon-aligned for NNLS/rank blending.

## 11. Constraints for any proposed new approach

- **Kaggle notebook runtime ≤ 12 h** (recommend ≤ 8h with buffer)
- **CPU-only preferred** (Mac M-series for local iteration; Kaggle GPU sometimes has CUDA kernel mismatch)
- **No pretrained model weights** (polyBERT, TransPolymer, MMPolymer, ChemBERTa all DISALLOWED by rules)
- **No uploaded feature caches** (all artifacts must be generated inside the notebook run)
- **PI1M IS allowed** as auxiliary data
- **Only 3 submissions per day** (as of this handoff)
- **Time remaining: ~2-3 days** to competition close

## 12. The ONE-LINE ask

**Given the pattern of failures documented above, what specific techniques (not yet tried) are most likely to add +0.006 LB to a robust chain-ext LGB + Chemprop blend pipeline within Kaggle-notebook runtime constraints and no pretrained weights?**

Prioritize concrete, implementable ideas over general concepts. Cite papers/writeups where possible.
