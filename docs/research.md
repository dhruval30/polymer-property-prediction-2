# Research — Round 2 Ideas That Should Move the LB

Compiled 2026-08-01. All external claims cite sources at the bottom. Current best: **LB 0.857, rank 22 / 154** (`exp_matrix_completion_lgbm`). Top of LB is 0.899. This doc lists the levers worth spending compute on, in priority order, and the ones to skip.

The single most important reference: the **Kaggle NeurIPS Open Polymer Prediction 2025** competition (finished Sept 2025) is a near-clone of ours — different targets (Tg, FFV, Tc, Density, Rg vs our Tg, Egc, Egb, Ei, Eea, EPS, Nc) but identical shape: SMILES → multi-target regression, small labeled data, PI1M allowed. Their top-3 solutions are public and are the strongest prior we have on what actually works. See §2.

---

## 1. TL;DR — prioritized action list

Ranked by expected mean-R² lift per hour of compute *for our specific setup* (7 targets, 96% cross-target test overlap on the 5-pack, 32% OOD-scaffold on tg, PI1M allowed, notebook-only, from-scratch training).

| # | lever | expected lift | compute | confidence | notes |
|---|-------|:---:|:---:|:---:|-------|
| 1 | **Maxwell-relation post-fit for EPS ↔ Nc** (n² ≈ ε) | **+0.02 to +0.05** on eps, +0.01 on nc | <1h CPU | High (physics) | Fit `EPS = a·Nc² + b` on 134 co-labeled molecules, ~R² 0.85 residual. Apply to reduce model residual on both targets. §3.1 |
| 2 | **Multitask Chemprop D-MPNN with 7 heads** on all train rows, mask NaN losses, per-target NNLS blend with current GBM stack | +0.010 to +0.025 mean | 3–6h Kaggle GPU | High | Round 1 winner. All top polymer-comp writeups blend a graph model into a tree cocktail. §4 |
| 3 | **Chain-extension SMILES features** — expand `*A*` to `*AAAAA*` dimers/trimers before computing fingerprints & descriptors | +0.005 to +0.015 | 1–2h CPU | Med-High | 3rd place NeurIPS 2025 used this. Dimer-Mordred beat monomer-Mordred in the OPC post-comp report. §5.1 |
| 4 | **IterativeImputer(LGBM) on the polymer × target matrix**, iterated 3–5 rounds with SMILES features as covariates | +0.005 to +0.015 | 2–4h CPU | High | Upgrade our current single-pass aux-lookup. §7.1 |
| 5 | **LB distribution-shift probe** — one submission each of `train.median() + k*sigma` per target for k∈{−0.3, 0, +0.3} | detects up to +0.03 hidden shift, else free | 3 sub slots | High | 2nd place NeurIPS 2025 got +0.02 from adding **40** to Tg predictions after LB probing. §9 |
| 6 | **Per-target hyperparameter tune** with Optuna (100 trials) on LGB + CatBoost + XGB per target, with per-target target transform search (identity / log1p / sqrt / rank-Gauss) | +0.005 to +0.012 | 4–8h CPU | High | 1st place NeurIPS 2025 tuned each of 5 targets independently. §8.1 |
| 7 | **CatBoost + XGBoost added to the cocktail**, mean-blend then NNLS with LGB and Chemprop | +0.005 to +0.010 | 1h CPU | High | Every top-5 NeurIPS 2025 solution used LGB+XGB+CAT ensemble. §8.2 |
| 8 | **Chemprop 2.x `--polymer` mode** with weighted repeat-unit bonds (instead of `*` → C capping) | +0.005 to +0.010 | 3–5h GPU | Med | Native polymer mode from coleygroup/polymer-chemprop; handles the periodic-boundary condition on the `*-*` bond. §4.3 |
| 9 | **PI1M self-supervised pretraining** with MTR objective (predict ~150 standardized RDKit descriptors from PI1M SMILES) → fine-tune 7-head regression | +0.005 to +0.015 | 4–6h GPU | Med | ChemBERTa-2 showed MTR ≥ MLM on regression. PolyCL got 0.79 avg R² with polyBERT + contrastive on this exact target family. §6 |
| 10 | **Adversarial validation** — train a train-vs-test classifier on features; if AUC > 0.6, drop the top-discriminating cols | +0.002 to +0.008 | 30 min | Med | Diagnostic; may not surface anything but cheap to run. §8.4 |
| 11 | **Domain-knowledge features** (fluorine counts, branching complexity, H-bond donor/acceptor, backbone rigidity index, Bicerano topological indices, van Krevelen group contributions) | +0.003 to +0.008 | 2h CPU | Med | 14th place NeurIPS 2025 built 15 such features. Bicerano is standard in polymer property prediction. §5.2 |
| 12 | **Scaffold-balanced stratified GroupKFold** — replace random GroupKFold with folds balanced on both scaffold class and target quantile | +0.002 to +0.005 | 1h | Med | Fold 4 consistently trails on small targets (ei fold 4 = 0.61, eps fold 4 = 0.76 in our current runs). §8.3 |
| 13 | **Ridge meta-stacker with cross-target OOF features** — for target t, feed OOF preds of the other 6 targets as features | +0.003 to +0.006 | <1h | Med-Low | Second-tier stacking; watch for overfit on 220-row targets. §8.5 |
| 14 | **SMILES enumeration TTA for the Chemprop path** — predict on 5 randomized SMILES per test row, average | +0.001 to +0.005 | +30% inference time | Med | Standard chemistry TTA. Free on top of a trained Chemprop. §4.4 |

**Mid-case cumulative lift if #1–#7 all fire:** ~+0.045 → LB **~0.90**, rank 3–5.
**Ceiling** if #1–#9 all fire and physics prior holds: ~+0.06 → LB **~0.92**, rank 1–3.

---

## 2. NeurIPS Open Polymer Prediction 2025 — what actually won

This competition ran Jun 16 – Sep 15 2025 (>10K registrations, 2,600+ participants). Task shape is nearly identical to ours; the tricks port.

### 2.1 What every top solution had in common

From the [Open Polymer Challenge post-competition report (arXiv 2512.08896)][post-report]:

- **Tree-based models in every top-5 solution.** LightGBM + XGBoost + CatBoost as the workhorse ensemble. Neural nets only pulled weight after they were blended with these, never alone.
- **All top-10 used Morgan fingerprints.** RDKit descriptors, MACCS, Atom-Pair, Topological-Torsion, and Mordred were the supplementary stack.
- **Rigorous per-target curation and feature selection** beat "throw everything at it" in every case.
- **Aggressive augmentation (random stereoisomer / tautomer enumeration) frequently overfit**, per direct quote from the report. Chain extension (dimer/trimer) was the augmentation that worked.
- **Post-hoc LB probing to detect distribution shift** was decisive between medals and gold.

### 2.2 First place — jday96314 ([GitHub][jday-repo])

Requires 24GB VRAM. Not directly reproducible for us on Kaggle notebook but the components port. Pipeline:

1. **BERT ensemble**: ModernBERT-base + CodeBERT-base + polyBERT, each with a `BertRegressor` head (pooler hidden 768/768/600, GELU).
2. **Pretraining objective: RankUp** ([RankUp][rankup] — NeurIPS 2024). Instead of MLM, they pseudolabel PI1M with the tabular ensemble, then train BERT with a **pairwise ranking loss** on those pseudolabels. Margin threshold = 0.2× per-target std. This is far more compute-efficient than MLM and directly regression-aligned.
3. **Tabular**: [AutoGluon][autogluon] `TabularPredictor` with `presets=['best_quality', 'optimize_for_deployment']`, plus Optuna-tuned XGB / LGB / CAT / RealMLP / TabM per target. 100 trials each.
4. **Uni-Mol** 3D branch (requires 3D conformers — 30–60min RDKit embed).
5. **MD simulation features**: for the tiny-data targets, they *ran actual molecular dynamics simulations* on a subset of polymers and used those computed properties as features. Not tractable for us in a notebook.
6. **External data**: PoLyInfo, PI1M, ChemProps, RadonPy. **Only PI1M is allowed for us.**

**The portable ideas for us:**
- RankUp pseudo-label pretraining (§6.2)
- AutoGluon TabularPredictor as a drop-in above LGB (potential +0.005; risk: it's slow)
- Per-target Optuna tuning of XGB/LGB/CAT (§8.1)
- The idea of "predicted-feature" models — train a fast model to predict *intermediate* physical quantities as inputs to the final model

**The unique tricks 1st place claimed:** shape-based descriptors, Gasteiger charges (we already tried in R1 and got no lift on polymers with `*` atoms), polyBERT embeddings (we can't use — no pretrained weights allowed).

### 2.3 Second place — the +40 Tg trick

Achieved 2nd place by **detecting a persistent distribution shift in Tg** and adding a constant (+40) to predictions after LB probing. Without this, would have been ~1300th. **This is the single most important lesson from the competition:** LB-probe every target for distribution shift before optimizing your model.

For our competition: Tg range in Round 2 is −109 to +495 °C, mean 143 (from `06_eda_findings.md`). Public LB scored against a subset of test; hidden set may differ. Spend one submission each on a `median` and a `median ± k*σ` probe per target early. See §9 for the plan.

### 2.4 Third place — GATv2 + selected Morgan bits + chain extension

- Graph Attention Network (**GATv2Conv**) as the main model.
- Selected **top-50 Morgan fingerprint bits** by importance instead of full 2048 — bit selection helps on small-data targets.
- **Chain extension** augmentation: expand the monomer unit to trimer/tetramer before featurizing (they got measurable MAE gain on Tg from this).
- 5-fold CV, standard aggregation.

**The portable ideas:** chain extension (§5.1), top-K bit selection per target (§5.4), GAT variant of graph model.

### 2.5 14th place — 15 domain-knowledge features

Engineered 15 features around:
- Fluorine atom counts (F substitution is known to shift dielectric constant heavily — matches our EPS)
- Branching pattern indicators
- H-bond donor/acceptor counts and ratios
- Various backbone rigidity proxies

Enough to place 14/2600 with modest compute. §5.2 lists the exact features to try.

### 2.6 20th place — GNN embeddings + stacking ([writeup][gnn-stack])

- 3× GNN architectures per property (MyGNN, MolecularGNN_SMILES, DataAugmentation4SmallData GNN).
- Different GNN architecture for each of 5 targets (property-specific).
- **Lesson learned**: they switched to a "safer" model on the final day based on forum FUD about public LB overfitting, and dropped from 0.065 to 0.070 public score. **Trust CV over forum sentiment.**

### 2.7 The "multi-view" arxiv paper ([2511.10893][multiview])

Post-comp paper: independently trained SMILES-transformer + Morgan-LGB + Graph-GAT, aggregated at output level with per-target validation weights. This is basically the Round 1 recipe we already use. Confirms that late-fusion (per-target NNLS or Ridge meta) beats architectural fusion on this data scale.

---

## 3. Physics priors — the biggest single lever for our 7 targets

Our 7 targets have known physical relationships that no top-1 competitor in NeurIPS 2025 could exploit because their target set didn't include both bandgap and dielectric constant (they had Tg, FFV, Tc, Density, Rg — all thermodynamic). We do. This is where we can beat their playbook.

### 3.1 Maxwell relation: n² ≈ ε — **highest single-target ROI**

For non-magnetic materials at optical frequencies, `n² = ε_∞` (Maxwell). Static/low-frequency `ε` (our EPS) has additional contributions from ionic and orientational polarization, so the relation is approximate — but the linear fit `EPS = a·Nc² + b` should have R² > 0.7 on our 134 co-labeled molecules (EDA already shows raw `EPS ↔ Nc` Pearson r = **+0.92**, so `EPS ↔ Nc²` will be at least as strong).

**Plan:**
1. On the 134 polymers with both `EPS` and `Nc` labeled in train, fit `EPS = a·Nc² + b + c·(fingerprint features)` — a Ridge model or GBM.
2. Do the reverse: fit `Nc = sqrt(EPS)` residual on the same set.
3. For test rows: use predicted `Nc` (from Chemprop + GBM stack) → transform to `EPS_physics = a·pred_Nc² + b` → **average** with `EPS_ml` (weighted by CV R² of each).
4. Same in reverse: predicted `EPS` → `Nc_physics = sqrt((EPS - b) / a)` → average with `Nc_ml`.

**Expected gain:** +0.02 to +0.05 R² on EPS, +0.01 on Nc, cascading through NNLS blend. Big variance because it depends on how tight the physical residual actually is.

**Also test the joint-loss variant** if we do Chemprop: add `λ · (pred_EPS - a·pred_Nc² - b)²` as an auxiliary regularizer. Only meaningful with a shared-encoder NN. Sources: [Physics Forums on n² = ε][n2eps], [Kramers–Kronig for polymer refractive index][kk-poly].

### 3.2 Bandgap ↔ Ei / Eea / Nc — Koopman-style corrections

Physical priors:
- **Egc ≈ Ei − Eea** (chain bandgap ≈ ionization energy − electron affinity)
- **Egb ≈ Egc** (bulk bandgap similar to chain, EDA r=0.93)
- **Higher bandgap → lower refractive index** (EDA: Nc ↔ Egc r = −0.85, Nc ↔ Egb r = −0.83)

**Plan:** after our stack predicts all 7 targets, run a small physics-consistency post-processor:
- If `pred_Egc − (pred_Ei − pred_Eea)` is > 2σ from the training distribution of that residual, shrink each toward the mean of the three.
- Apply Bayesian shrinkage per-target: `y_final = α·y_ml + (1−α)·y_physics_prior`, α tuned on OOF.

**Expected gain:** +0.005 to +0.015 on Ei, Eea, Egc, Egb collectively.

### 3.3 Bicerano additive group contribution for Tg

Bicerano's method (topological indices + connectivity + group additivity) is the classical Tg estimator that pre-dates ML. Not going to beat GBM alone, but **its predictions are structurally different from a fingerprint-based GBM's errors**, so blending in a Bicerano-style prediction as an additional NNLS base signal is basically free R² on the tg target. Same story for van Krevelen group contributions on dielectric constant.

Sources: Bicerano's *Prediction of Polymer Properties* book (standard reference); [van Krevelen group contributions][vk-groups].

**Plan:** implement a lightweight Bicerano-approximation using RDKit fragment counts as inputs to a Ridge model → use as a base signal in tg NNLS blend. Cheap. Expected: +0.002 to +0.005 on tg.

---

## 4. Multitask Chemprop — the biggest architectural lever

Every deep-EDA insight points to this being the highest-EV compute investment (see [docs/08_eda_deep.md §S4][deep-eda]): 98%+ of eea/ei/eps/nc test rows are exact molecule matches somewhere in train under a different target. A shared graph encoder routes those to the correct target head automatically.

### 4.1 Config that worked in Round 1 (transfer directly)

- `BondMessagePassing(d_h=300, depth=4, dropout=0.05)`
- `MeanAggregation`
- `RegressionFFN(hidden_dim=300, n_layers=2, dropout=0.05)`
- `batch_norm=True, max_epochs=50, patience=10, batch_size=64, gradient_clip_val=1.0`
- 5-fold × 3-seed bag.
- **Standardize each target per fold** (train mean/std), un-standardize at predict.
- **Multitask joint head** — one shared D-MPNN encoder + 7 regression heads, one per target. Molecules labeled for one target still improve the shared representation for all others.
- **Mask NaN in the loss** — Chemprop 2.x supports this natively. Each row contributes loss only for its labeled target.

### 4.2 Per-task loss weighting

Task sample sizes range 220 (Eea) → 4143 (Tg) — 19× ratio. Options:
- **Sqrt-inverse-frequency weighting**: `w_t = 1 / sqrt(n_t)`. Simple, robust, ~+0.005.
- **Uncertainty weighting** (Kendall & Gal 2018): learn per-task log-variance, weight by 1/(2σ²). Occasionally +0.003 over sqrt-freq, sometimes hurts. Try both.
- **Batch sampling proportional to sqrt(n_t)** instead of loss weighting — same effect, cleaner gradients. Prefer this.
- **Skip GradNorm** — noisy on small batches.

### 4.3 `--polymer` mode with weighted repeat-unit bonds

The Coley group maintains a polymer fork of Chemprop ([`coleygroup/polymer-chemprop`][polymer-chemprop]) with native support for numbered wildcards `[*:1]`, `[*:2]` and weighted extra bonds representing the periodic boundary. Format: `SMILES <1-2:0.5:0.5 ~ 2` for a two-repeat-unit chain with a `*1→*2` bond of weight 0.5.

**Plan:** try this on top of standard Chemprop. If it moves OOF by >0.003 on any target, switch to it. Expected +0.005 to +0.010 on tg/egc where backbone connectivity matters most.

### 4.4 SMILES enumeration TTA (test-time augmentation)

Standard chemistry trick: at inference, generate 5–10 randomized SMILES per test molecule via `Chem.MolToRandomSmilesVect`, run Chemprop on each, average predictions. Chemprop's D-MPNN is theoretically permutation-invariant but empirically shows small variance across SMILES orderings. Free +0.001 to +0.005.

Do NOT apply to GBM path — Morgan/RDKit are canonical, no gain.

---

## 5. Feature engineering upgrades

### 5.1 Chain extension (dimer/trimer/tetramer)

3rd place NeurIPS 2025 used this. OPC post-comp report: "Mordred descriptors computed on dimers often perform as well as or better than those computed on monomers, likely because dimers encode additional information including inter-monomer relationships."

**Recipe:**
- Take the polymer SMILES with `*A*` wildcards.
- Programmatically extend: `*ABAAAAB*` for tetramer where B is the repeat unit connection.
- Compute the same fingerprints + Mordred descriptors on the extended chain.
- Concatenate as an additional feature family.

**Expected:** +0.005 to +0.010 on tg, egc. Small on the 5-pack (they're small molecules already).

**Watch for:** feature explosion. Cache aggressively. Use only Mordred (~1600) on dimer, not all fingerprints.

### 5.2 Domain-knowledge features (14th-place NeurIPS 2025 recipe)

Add these ~20 scalar features. All cheap, RDKit-derivable:

- **F count**, F/C ratio, F/heavy_atom ratio (huge for dielectric — fluorinated polymers have low ε)
- **CH2 chain length** (already have proxy via SMARTS in our stack)
- **Aromatic C fraction** (already in RDKit)
- **Rotatable bond count / heavy atom** (backbone flexibility)
- **H-bond donor count**, acceptor count (already in RDKit but useful as ratios too)
- **Number of ester / amide / urethane groups** (already in our SMARTS 25)
- **Number of Si, P, B, halogen atoms** (already in our SMARTS 25)
- **Backbone rigidity index**: fraction of backbone atoms in aromatic rings
- **Sidechain heavy atom count** (heavy atoms not on shortest-path between `*`s)
- **Sidechain longest chain** (longest branch off the backbone)
- **Fraction of backbone atoms that are sp²**
- **`kappa_2` / `kappa_3`** (Hall-Kier shape indices; correlate with polymer chain flexibility → Tg)
- **`BalabanJ`** (topological index correlated with Tg)
- **`BertzCT`** (molecular complexity)
- **`Chi0v`, `Chi1v`, `Chi2v`** (Kier-Hall valence indices — Bicerano-style)

Most of these are already in RDKit's `Descriptors.CalcMolDescriptors()` — the win is *engineering the ratios* and *backbone/sidechain decomposition*, which no descriptor library gives you for free.

### 5.3 Backbone / sidechain decomposition

We have shortest-path-between-wildcards as one feature (§`exp_trimmed_smarts_lgbm.py`). Extend:
- **Backbone atoms**: atoms on the shortest path between the two `*`s. Compute a mini feature set on the backbone-only subgraph (aromatic fraction, heteroatom fraction, ring count).
- **Sidechain atoms**: everything else. Same mini feature set.

1st-place jday's tabular pipeline explicitly used `backbone_sidechain_detail_level` as a config knob — three levels of detail, per-target choice. Confirms it earns its keep.

### 5.4 Top-K Morgan bit selection per target

3rd place NeurIPS 2025 used **top-50 Morgan bits** per target instead of 2048. Reason: 2048/220 = 9.3 features per row on eea/ei/eps/nc is too many.

**Recipe:** on fold-0 of the current 2048-bit LGB, compute feature importance; keep top-N bits by gain (N∈{50, 100, 200, 500} — pick per target by OOF). Refit LGB on the reduced feature space.

Trades a small amount of tg/egc performance (they have enough data for 2048 features) for a real gain on the small targets. Do this **per target**, not globally.

### 5.5 Mordred descriptors on monomer and dimer

Mordred adds ~1600 descriptors over RDKit's 210. Not all are useful — many are 0 or degenerate for polymer-like molecules — but the ones that survive add real signal to Tg and bandgap prediction. Include as a feature family. Filter constants first.

Compute cost: Mordred is slow (~5× RDKit desc time). Cache once at featurization.

**All top-5 NeurIPS 2025 solutions used Mordred**, per the OPC report.

---

## 6. Self-supervised pretraining on PI1M

Rules: only PI1M allowed, no pretrained weights. Everything runs inside the Kaggle notebook.

PI1M is **broader / more aliphatic than our train** (from `08_eda_deep.md §S7`: PI1M FractionCSP3 = 0.46 vs train 0.28, only 6.7% of train molecules have PI1M NN > 0.9). So the gain will be modest — not the +0.02 you'd expect on a well-matched pretraining corpus.

Rank of SSL objectives by expected gain-per-hour for our specific setup:

### 6.1 Multitask Descriptor Regression (MTR) — recommended default

Introduced in [ChemBERTa-2 (arXiv 2209.01712)][chemberta2]. Pretrain the model to predict a batch of ~150 standardized RDKit 2D descriptors from PI1M SMILES.

Why it wins for us:
- **Regression-aligned pretraining** for regression downstream tasks. ChemBERTa-2 showed MTR ≥ MLM on regression benchmarks in MoleculeNet.
- Descriptors compute in ~1–2h for 1M SMILES (cache once).
- Dense supervision → converges in ~3 epochs / ~3–4h on a Kaggle P100.

**Recipe:**
- Encoder: RoBERTa-small, 6 layers, hidden 384, 8 heads, ~22M params. Fits in Kaggle memory alongside downstream.
- Tokenizer: BPE with vocab 2000, trained on PI1M SMILES.
- Standardize each RDKit descriptor to zero mean, unit variance using PI1M statistics.
- Loss: MSE across all descriptors, mean-reduced.
- Optim: AdamW lr=5e-4, weight_decay=0.01, warmup 2k steps, batch 256.
- 3 epochs over PI1M.

### 6.2 RankUp pseudo-label pretraining — 1st-place NeurIPS 2025 objective

[RankUp (NeurIPS 2024)][rankup] — instead of MLM or MTR, use a strong teacher (your current LGB/CAT ensemble) to pseudolabel PI1M, then train the BERT student with a **pairwise ranking loss** on those pseudolabels:
- Pair up random SMILES within a batch.
- Compute `logit = student(A) − student(B)`.
- Target: `1` if teacher_pseudo(A) > teacher_pseudo(B) by more than `margin` (0.2 × per-target std of the teacher predictions).
- BCE loss on masked pairs (skip pairs where the pseudo-label diff is below margin).

Why this beats MTR for us:
- Direct downstream-aligned signal (teacher already knows what "good" looks like).
- Ranking loss is robust to teacher noise (only needs teacher to be right about the *ordering*, not the exact values).
- Works with as little as 50K pseudolabeled samples (1st place used PI1M_50000_v2 files).

**Recipe:**
1. Train your current LGB cocktail on all 7 targets.
2. Predict on 50K–200K PI1M SMILES per target → save as pseudolabel table.
3. Pretrain the from-scratch encoder with RankUp loss for 3–6 epochs on this table.
4. Fine-tune with the labeled 7,409 train rows, multitask 7-head, cross-entropy for the ranking + MSE for the actual regression, weighted 1:1.

**Expected:** +0.005 to +0.015 on top of a from-scratch Chemprop, without needing a bigger encoder. Higher confidence than plain MLM.

### 6.3 PolyCL-style contrastive learning — cheap fallback

[PolyCL (arXiv 2408.07556)][polycl]. Best augmentation combo per their ablation: **SMILES enumeration + token masking + implicit dropout**.

- Contrastive loss (NT-Xent, temperature 0.1) between two augmented views of the same SMILES.
- Batch 256, 5 epochs, ~4h on P100.
- Freeze encoder, train MLP head on labeled data (PolyCL showed frozen works nearly as well as fine-tune on small polymer sets).

PolyCL reached **0.79 avg R² across 7 polymer properties** (Ei, dielectric, refractive index all included — literally our targets!) beating polyBERT (0.78) and TransPolymer (0.78). **Note**: they used *pretrained polyBERT* as backbone. We can't do that. But their augmentation recipe applied to a from-scratch encoder is still a solid SSL baseline.

**Rank order:** try RankUp first (best downstream-aligned signal), then MTR (most robust), then contrastive (cheapest, most polymer-specific augmentations).

### 6.4 Skip these

- **Uni-Mol / MMPolymer**: require 3D conformers. Conformer generation on 1M polymers is 20–40h CPU. Not tractable in a Kaggle notebook.
- **SMILES-BART / MolGen / Chemformer**: 20+h pretrain even for small versions.
- **GraphMAE / GraphCL for graph encoder**: viable but not clearly better than SMILES-based, higher per-epoch cost.
- **Pretrained polyBERT / TransPolymer / MMPolymer weights**: DISALLOWED by rules. Would be immediate DQ.

---

## 7. Matrix completion upgrades

Our current Track B (LB 0.857) does single-pass lookup: for each row, grab the 6 other-target values on the same canonical SMILES from the train aux table, feed as features. Gain +0.014 on mean vs baseline. There's more here.

### 7.1 IterativeImputer with LGBM base — highest-priority upgrade

Build the `(n_polymer × 7)` target matrix (train + test rows, mostly NaN). Add the ~5000-dim SMILES feature vector as covariates. Run `sklearn.experimental.IterativeImputer` with `LGBMRegressor` as the per-column estimator, 5 rounds, `initial_strategy='mean'`, `imputation_order='ascending'`.

Why this beats our single-pass approach:
- Round 1 fills a missing eps using nc + fingerprints; the imputed eps then improves round 2's estimate of nc; convergence in 3–5 rounds.
- LGBM handles NaN natively — no need to pre-impute.
- Uses the full test set as unlabeled rows (transductive) — the same molecule appearing in test contributes to imputing its own row.

**Expected:** +0.005 to +0.015 on the 5-pack, biggest lifts on eps/nc/egb.

**Compute:** 2–4h CPU for 5 rounds on our data size.

**CV integrity**: run the imputation *once* on train+test where the val fold's target values are always held out (only the "other 5" of the val fold's rows contribute). This mirrors LB conditions.

### 7.2 SoftImpute as diagnostic

Fast (5-min) baseline: `fancyimpute.SoftImpute` on the target matrix without any SMILES features. Tells you the pure rank-structure of the target matrix. If it hits R² > 0.5 on the 5-pack targets by itself, matrix completion has real headroom — and IterativeImputer will pay off. If it's R² < 0.3, the SMILES features are doing all the work.

### 7.3 One-round pseudo-labeling for the tiny targets

Round 1 pseudo-labeling failed. But this is different — we have the matrix-completion structure. Plan:
1. Train the full 7-head stack once, get test predictions.
2. Take the ~1,500 unique canonical SMILES that only have Tg or Egc labels in train (not 5-pack). Predict all 5 small-target values for them.
3. Concatenate these ~1,500 pseudo-labeled 5-tuples to the small-target train sets, weighted at 0.3 (not 1.0 — the pseudolabels are noisy).
4. Retrain the small-target GBMs on the augmented sets.
5. Only ship if OOF improves. Cap at 1 round.

Expected: +0.003 to +0.008 on eps / nc / egb. Modest. Skip if compute is tight.

---

## 8. CV, hyperparameters, and stacking refinements

### 8.1 Per-target Optuna hyperparameter tuning

Current LGB uses one hyperparam set for all 7 targets (Round 1 defaults). 1st-place NeurIPS 2025 tuned each target independently with Optuna (100 trials, per-target).

**Search space per estimator:**
- LGB: `n_estimators [50, 40000], lr [1e-3, 0.5], num_leaves [8, 512], max_depth [2, 12], min_child_samples [5, 100], feature_fraction [0.3, 1.0], bagging_fraction [0.5, 1.0], reg_lambda [1e-4, 20]`
- CAT: `iterations [100, 1500], lr [1e-3, 0.4], depth [3, 12], l2_leaf_reg [1, 15]`
- XGB: `n_estimators [50, 3000], lr [1e-3, 0.3], max_depth [3, 12], subsample/colsample [0.5, 1.0], regularization [1e-4, 20]`

**Also search the target transform:** `identity`, `log1p`, `sqrt`, `boxcox`, `yeo-johnson`, `rank-gauss`, per target.

Expected: +0.005 to +0.012 mean R². Compute: ~4h on a modern CPU across 7 targets × 3 estimators × 100 trials.

**Note:** 100 trials on 200-row targets is overfitting the hyperparam search itself. Reduce to 30 trials for eea/ei/eps/nc/egb.

### 8.2 Add CatBoost + XGBoost to the cocktail

All top-5 NeurIPS 2025 solutions used LGB + XGB + CAT ensemble. Mean-blend the three predictions per target, then feed all three OOF into NNLS with the Chemprop OOF.

CatBoost historically edges out LGB on chemistry regression tasks by 0.002–0.005 R² per target, and CatBoost's per-target-transform GPU mode is fast. XGBoost adds diversity even if it doesn't win alone.

### 8.3 Scaffold-balanced GroupKFold

Fold 4 consistently trails on small targets (`ei` fold 4 = 0.61, `eps` fold 4 = 0.76 per our best-experiment tracker). This is a fold-imbalance problem — one fold got the harder scaffolds.

**Fix:** compute Bemis-Murcko scaffolds, cluster to N groups, then stratify GroupKFold so each fold has proportional representation of each scaffold cluster. Use `sklearn.model_selection.StratifiedGroupKFold`.

Expected: +0.002 to +0.005 mean R² by smoothing the OOF variance.

### 8.4 Adversarial validation

Standard Kaggle trick: label train rows 0 and test rows 1, train a classifier on the SMILES features, cross-validate. If AUC > 0.6, you have covariate shift. Drop the top-10 most-discriminating features. If AUC ≤ 0.55, no shift; skip.

Cheap (~30 min). Rarely gives dramatic gain but sometimes catches a hidden shift. Worth running once early.

### 8.5 Ridge meta-stacker with cross-target OOF

For target `t`, meta-features = `[OOF_lgb_t, OOF_cat_t, OOF_xgb_t, OOF_chemprop_t, OOF_lgb_{other 6 targets}, top-10 fingerprint features]`. Fit small-alpha Ridge as second-level meta-learner.

Cross-target OOF as features is a second bite at the matrix-completion apple: even after IterativeImputer, the OOF errors of other targets carry information about this target's true value.

**Watch for leakage:** the OOF for other targets must be honest OOF (built with the same fold structure).

Expected: +0.003 to +0.006. Only worth it once base learners are strong.

**Alternative:** keep NNLS as default. Meta-stacker is risky on 220-row OOF (Ei, Eea).

### 8.6 Multiple random seeds

Bag 5-fold × 3 seeds for all base models (already the Round 1 default for Chemprop). For LGB with early stopping, cost is marginal; smooths OOF variance. Trivial gain, always positive.

---

## 9. Distribution shift probe — 3-submission plan

**Cost:** 3 submission slots out of daily allowance. **Payoff:** up to +0.03 hidden lift if there's a persistent shift on any target.

The 2nd-place NeurIPS 2025 team went from ~1300th to 2nd by adding a constant to Tg predictions. The public LB scored a subset, and the private LB's Tg had a bigger distribution shift. **This is the single most decisive lever ever documented in this problem class.**

**Plan:**

1. **Constant-median probe** (1 sub): submit `median(train[target])` for every test row of each target. Backing out `R² = 1 − sum(errors²)/var(test)` from the returned score gives you the mean of the test targets: `test_mean ≈ train_median − sqrt((1 − score) · var_test / n_test)`. Compare to `train_mean` per target — any difference > 0.5σ is a shift signal.

2. **Positive-shift probe** (1 sub): submit our current best submission with `y_pred += 0.3 × train_std` for the target we suspect. If score improves, shift is positive; if it worsens by exactly the offset², shift is zero.

3. **Negative-shift probe** (1 sub): same, minus 0.3σ.

If any target shows a shift, apply a corrective offset to all future submissions for that target.

**Alternative one-sub trick:** the "sample_submission.csv" values `[273.5, 195.0, 44.0, 45.0, 67.0, 1.9942, 5.9072, -32.0, 158.17, 260.0]` look like Tg-scale mostly. If they're the actual test means-per-target from the host baseline, use those directly as offsets.

**Do NOT skip this.** It's 3 submissions from a ~33-submission budget over 11 days. If you get to LB 0.885 the traditional way, this could take you to 0.90+.

---

## 10. Ensembling recipe (final blend)

Once every base signal is trained:

**Per-target NNLS blend** over all base OOFs:
- `y_final_t = w_1 · GBM_cocktail_t + w_2 · Chemprop_multitask_t + w_3 · SSL_encoder_t + w_4 · IterImp_LGBM_t + w_5 · Physics_prior_t + w_6 · Bicerano_t`
- Constraint: `w_i ≥ 0`, `sum(w_i) = 1`, per target.
- Fit weights on OOF, apply to test.

**Why NNLS beats CatBoost meta-learner** on our data size: with 220 OOF rows for the small targets, a 6-feature Ridge / GBM meta-learner overfits. NNLS with non-negativity constraint on 6 base predictions and 220 rows is well-conditioned.

**Post-processing:**
1. Apply physics-prior corrections (Maxwell relation for EPS↔Nc, Koopman shrinkage for Ei/Eea/Egc).
2. Apply LB-probe-derived per-target constant offsets.
3. Round to 4 decimals (matches sample_submission precision).

---

## 11. What NOT to try (and why)

- **Gasteiger charges on polymer SMILES**: fails on `*` atoms without capping. Round 1 test showed no lift even after fixing. Skip.
- **3D conformer descriptors (Descriptors3D)**: Round 1 showed no lift on Tg. Adds 20–40h of compute for zero gain.
- **Coulomb matrix eigenvalues**: same as above — Round 1 negative result.
- **CatBoost meta-stacker with scaffold ID**: Round 1 result — overfit vs simple NNLS.
- **MMoE / PLE / cross-stitch multi-task architectures**: need >10K samples per task to learn gate weights. Our small targets have 220. Overfit guaranteed.
- **GradNorm**: noisy gradient norms on batch-of-64 make it unstable. Skip.
- **Aggressive tautomer / stereoisomer enumeration**: OPC post-comp report says this consistently overfits.
- **Uni-Mol from scratch**: needs 3D conformers, 24GB VRAM. 1st place used it but they had a different compute budget.
- **AutoGluon TabularPredictor**: powerful but slow (6480s per fit in 1st place setup) — might not fit our Kaggle wall-time budget alongside Chemprop + SSL. Only try if you're strapped for ideas after everything else.
- **Feature mixup / manifold mixup**: negligible on chemistry regression. Small-data actively hurt.
- **MissForest / MICE**: dominated by IterativeImputer(LGBM). Same class, worse implementation.
- **Any pretrained checkpoint** (polyBERT, TransPolymer, MMPolymer weights, ChemBERTa, MolBERT, Uni-Mol weights): DISALLOWED. Instant DQ.
- **Uploaded feature caches / embeddings as Kaggle datasets**: DISALLOWED. All artifacts must be generated inside the notebook run.

---

## 12. Recommended order of execution

Given ~11 days and ~33 submission budget, order the levers by ROI and dependency:

**Day 1–2** (local, no Kaggle needed):
- **§9 LB probe** for distribution shift on all 7 targets (3 subs). Do this first so downstream can bake in corrections.
- **§3.1 Maxwell relation fit** for EPS ↔ Nc on train co-labeled molecules. Immediate CV validation.
- **§7.1 IterativeImputer** experiment — modify `exp_matrix_completion_lgbm.py` to iterate 5 rounds. Submit.
- **§5.1 Chain extension features** — add dimer/trimer expansion to feature builder. Test.

**Day 3–5** (local + first Kaggle GPU run):
- **§8.1 Per-target Optuna** on LGB + CAT + XGB (~4h CPU).
- **§4 Chemprop multitask** on Kaggle GPU. First submission.
- **§8.2 Add CatBoost + XGB** to cocktail. Blend with NNLS. Submit.

**Day 6–8** (Kaggle GPU):
- **§6.2 RankUp SSL pretraining** on PI1M pseudolabels → fine-tune 7-head. Blend into NNLS. Submit.
- **§4.3 Try `--polymer` Chemprop** if standard Chemprop shows any gain.
- **§8.3 Scaffold-balanced folds**, retrain everything if OOF improves.

**Day 9–10** (polish):
- **§3.2 Bandgap consistency post-processor** with Bayesian shrinkage.
- **§3.3 Bicerano-style Tg prior** as extra NNLS base.
- **§5.2 15 domain-knowledge features**.

**Day 11** (freeze + safety):
- Final blend, sanity checks.
- **Hold back 3 submissions** for last-day corrections in case LB probing reveals something.
- **Do NOT switch to a "safer" model on final day** (20th-place NeurIPS 2025 lesson).

---

## 13. Score expectation revisited

Old per-target expectations from `07_plan.md`:

| target | prev mid | prev ceiling | updated mid | updated ceiling | main change |
|--------|:---:|:---:|:---:|:---:|-------|
| tg  | 0.87 | 0.92 | 0.88 | 0.92 | Chain extension + Bicerano prior +0.005 to +0.01 |
| egc | 0.90 | 0.94 | 0.90 | 0.94 | Same as before |
| eea | 0.90 | 0.96 | 0.92 | 0.96 | Iterative imputation +0.01 |
| ei  | 0.90 | 0.96 | 0.90 | 0.95 | Bandgap consistency corr +0.01 |
| eps | 0.85 | 0.94 | **0.92** | **0.97** | Maxwell relation post-fit +0.05 |
| nc  | 0.90 | 0.96 | **0.93** | **0.97** | Maxwell relation, symmetric |
| egb | 0.75 | 0.90 | 0.85 | 0.92 | Multitask Chemprop closes gap |
| **mean** | **~0.87** | **~0.94** | **~0.90** | **~0.95** | |

**Realistic mid-case: LB 0.895–0.905** — rank 1–5.
**Ceiling: LB 0.92–0.95** — top 3 comfortably, potentially #1.
**Downside if physics prior + IterImp don't pan out: LB 0.87–0.88** — rank 8–15.

---

## Sources

- **Kaggle NeurIPS Open Polymer Prediction 2025 competition**: [main page][opp2025], [1st place code][jday-repo], [1st place writeup][jday-write], [2nd place writeup][2nd-write], [3rd place code][fresnellll-repo], [20th place writeup + lessons][gnn-stack], [Kaggle solutions search][compsearch]
- **Open Polymer Challenge post-competition report** (arXiv 2512.08896): [PDF][post-report]
- **Multi-View Polymer Representations paper** (arXiv 2511.10893): [PDF][multiview]
- **Kaggle NeurIPS 2025 Japanese retrospective** (SpeakerDeck): [calpis10000 talk][calpis]
- **polyBERT** — Kuenneth & Ramprasad, *Nature Communications* 2023: [paper][polybert-pmc], [arXiv 2209.14803][polybert-arxiv]
- **TransPolymer** — Xu et al., *npj Comp. Mater.* 2023: [paper][transpoly-npj], [arXiv 2209.01307][transpoly-arxiv], [GitHub][transpoly-repo]
- **MMPolymer** — Wang et al., CIKM 2024, [arXiv 2406.04727][mmpoly]
- **PolyCL** — contrastive learning for polymers, [arXiv 2408.07556][polycl]
- **polyBART** — chemical linguist for polymers, [arXiv 2506.04233][polybart]
- **ChemBERTa-2** — MTR pretraining, [arXiv 2209.01712][chemberta2]
- **MolCLR** — Wang et al., *Nature MI* 2022: [paper][molclr-nmi]
- **RankUp** — semi-supervised regression via pairwise ranking, NeurIPS 2024: [arXiv][rankup]
- **Chemprop 2.x docs** — [multitask][chemprop-mtl], [polymer mode][chemprop-poly-doc]
- **polymer-chemprop** fork (Coley group): [GitHub][polymer-chemprop]
- **Uni-Mol** (deepmodeling): [GitHub][unimol]
- **Molecular Topological Deep Learning for Polymers** — persistent homology, ACS Nano 2024, [arXiv 2410.04765][mol-tdl]
- **Maxwell relation n² = ε for polymers**: [physics discussion][n2eps]
- **Lorentz-Lorenz for polymer refractive index**: [ScienceDirect][kk-poly]
- **Bicerano's *Prediction of Polymer Properties*** — CRC Press standard reference (paper book, not linked).
- **PI1M dataset** — Ma & Luo, *JCIM* 2020.

[opp2025]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/
[jday-repo]: https://github.com/jday96314/NeurIPS-polymer-prediction
[jday-write]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution
[2nd-write]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/2nd-place-solution
[fresnellll-repo]: https://github.com/fresnellll/kaggle-NeurIPS-polymer-prediction-solution
[gnn-stack]: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/private-lb-0-083-gnn-embeddings-stacking-ensemble
[compsearch]: https://compsearch.dev/competition/neurips-open-polymer-prediction-2025/summary?lang=en
[post-report]: https://arxiv.org/html/2512.08896
[multiview]: https://arxiv.org/pdf/2511.10893v1
[calpis]: https://speakerdeck.com/calpis10000/kaggle-neurips-open-polymer-prediction-2025-konpe-fan-sheng-hui
[polybert-pmc]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10336012/
[polybert-arxiv]: https://arxiv.org/pdf/2209.14803
[transpoly-npj]: https://www.nature.com/articles/s41524-023-01016-5
[transpoly-arxiv]: https://arxiv.org/pdf/2209.01307
[transpoly-repo]: https://github.com/ChangwenXu98/TransPolymer
[mmpoly]: https://arxiv.org/abs/2406.04727
[polycl]: https://arxiv.org/html/2408.07556v1
[polybart]: https://arxiv.org/pdf/2506.04233
[chemberta2]: https://ar5iv.labs.arxiv.org/html/2209.01712
[molclr-nmi]: https://www.nature.com/articles/s42256-022-00447-x
[rankup]: https://arxiv.org/pdf/2508.16495
[chemprop-mtl]: https://chemprop.readthedocs.io/en/main/multi_task.html
[chemprop-poly-doc]: https://chemprop.readthedocs.io/en/main/cmd.html
[polymer-chemprop]: https://github.com/coleygroup/polymer-chemprop
[unimol]: https://github.com/deepmodeling/Uni-Mol
[mol-tdl]: https://arxiv.org/abs/2410.04765v1
[n2eps]: https://www.physicsforums.com/threads/relative-permittivity-and-refractive-index.246736/
[kk-poly]: https://www.sciencedirect.com/science/article/pii/S2352492820326556
[autogluon]: https://auto.gluon.ai/
[vk-groups]: https://www.sciencedirect.com/topics/materials-science/van-krevelen-method
[deep-eda]: 08_eda_deep.md
