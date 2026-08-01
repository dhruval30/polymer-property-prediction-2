# Proposed Plan — Round 2

Draft. Awaiting Dhruval's approval before any implementation.

## Score expectation (before we start)

Current public LB #1 is **0.899**. Rank 15 is **0.867**. Round 1 finish (Dhruval) was **0.911** on a *2-target* problem — Round 2 averages 7 R²s, several of which have <300 training rows, so absolute score ceilings are tighter.

Rough per-target expectation, based on Round 1 outcomes + the 5-pack cross-target leak:
| target | expected R² (mid) | expected R² (ceiling) | why |
|--------|:---:|:---:|-----|
| tg  | 0.87 | 0.92 | Round 1 hit 0.90+ on tg with the same recipe; large training set. |
| egc | 0.90 | 0.94 | Round 1 hit 0.91+ on egc; 2k rows, well-behaved. |
| eea | 0.90 | 0.96 | Matrix-completion — 141/147 test rows have cross-target info. |
| ei  | 0.90 | 0.96 | Same reason. |
| eps | 0.85 | 0.94 | Same reason but noisier / more skewed. |
| nc  | 0.90 | 0.96 | Same reason; tight scale (1.5–2.7). |
| egb | 0.75 | 0.90 | 80% cross-target coverage but 93 test rows have NO cross-target info; also 337-row training set is fairly small for SMILES-only prediction. |
| **mean** | **~0.87** | **~0.94** |  |

Realistic mid-case: **0.87–0.90** — competitive with the top 5 but not #1. Ceiling: **0.92–0.94** if the cross-target leverage really pays off and Chemprop + PI1M pretraining hit their upside. Above **0.90** likely places top 3.

## Strategy in one line
**Two-track:** (A) SMILES→property regression for `tg` and `egc`; (B) cross-target matrix completion + SMILES fallback for `eea, egb, ei, eps, nc`. Blend both tracks with per-target NNLS. Optional PI1M SSL pretraining boost if wall-time allows.

---

## Track A — tg, egc (SMILES → property)

Reuse the Round 1 playbook nearly verbatim:

### From Round 1 (reused)
- **Feature families for GBM cocktail** (~11K features): RDKit 2D descriptors, Morgan-r2/r3 count FPs (2048 each), MACCS keys, Avalon, Atom-Pair count FP, Topological-Torsion count FP.
- **GBM cocktail:** LightGBM + CatBoost + HistGradientBoosting, mean-blended. Same hyperparams that worked in Round 1 (see CLAUDE.md).
- **CV:** 5-fold stratified quantile split on the transformed target, 10 bins. Same folds for all base models to enable OOF stacking.
- **Chemprop D-MPNN** with the Round 1 config: `BondMessagePassing(d_h=300, depth=4, dropout=0.05)`, `MeanAggregation`, `RegressionFFN(hidden_dim=300, n_layers=2, dropout=0.05)`, `batch_norm=True`, `max_epochs=50`, `patience=10`, `batch_size=64`, `gradient_clip_val=1.0`. 5-fold × 3-seed bag.
- **Per-target NNLS blend** of GBM cocktail OOF + Chemprop OOF.
- **Target transforms:** identity for `tg` (has negatives), identity or log1p for `egc` (positive but roughly symmetric — will pick based on OOF).
- **Refit refit on full data** at `median(best_iters) × 1.10` for GBMs.
- **Wire EpochLogger** on Chemprop (Round 1 burned 9h on silent hangs).

### From Round 1 (skipped)
- ❌ 3D physics features (Coulomb-matrix, multi-conformer Descriptors3D aggregates) — did not lift Tg in Round 1.
- ❌ Gasteiger partial charges on polymer SMILES.
- ❌ Iterative pseudo-labeling (2 rounds) — added noise.
- ❌ CatBoost meta-stacker with scaffold ID — overfit vs simple NNLS.
- ❌ Dual-variant Chemprop (Mean+d2 vs Sum+d3) — only +0.003, not worth 2× compute unless we're at the ceiling.

### Changes for Round 2
- Chemprop trained as a **multitask model over all 7 targets simultaneously**, not per-target. Same molecular representation is shared; each head predicts its own target. This is important because tg is disjoint from the 5-pack, but egc has some overlap and we want to give the encoder every label it can see.
- Since we can't upload a feature cache (rules), the feature computation runs inside the Kaggle notebook. RDKit + fingerprints for ~12k SMILES is ~5–10 min single-threaded. Cache in memory only.

---

## Track B — eea, egb, ei, eps, nc (matrix completion + SMILES fallback)

**This is the key differentiator from Round 1.** These 5 targets have 221–337 training rows each, but 95%+ of their test rows correspond to SMILES that are labeled in train for at least one *other* electronic property. That's a matrix-completion problem, not a regression problem.

### Approach
1. **Build a per-SMILES 5-pack matrix** from train: rows = unique SMILES, cols = {eea, egb, ei, eps, nc}, values = measured target (NaN if unmeasured).
2. **Featurize each row as:** [SMILES-derived features] ⊕ [4 other 5-pack values with NaN-indicators].
3. **Train 5 GBMs** (LightGBM), one per target, on the rows that have the target measured. Each GBM sees SMILES features + values of the other 4 targets (when available, otherwise NaN → LightGBM handles natively).
4. **At inference time:** for each test row asking target T on SMILES S, look up S in train; grab whatever of the other 4 electronic properties are known; feed as inputs alongside SMILES features. If none known (~5% of eea/ei/eps/nc test rows, ~40% of egb test rows), fall back to the SMILES-only branch (which is the multitask Chemprop head + a SMILES-only GBM trained on that target alone).
5. **Chemprop multitask head predictions for these 5 targets** are still used, and blended in with NNLS. Chemprop implicitly does matrix completion via shared representation — including its OOF alongside the matrix-completion GBM gives us two independent views to blend.

### Key detail: OOF for matrix completion
For CV integrity, when computing OOF for a fold, we must **withhold that SMILES's target measurements** from the auxiliary-features view *for both fold-train and fold-val* — otherwise we leak the target we're predicting. Concretely: for row `(smiles=S, target_type=T)` in the val fold, the auxiliary features must exclude any train row where the same SMILES has target T. (Since 5-pack same-target duplicates within train are extremely rare, this is nearly a no-op, but must be right.)

### CV grouping
**GroupKFold on `smiles`** (not simple KFold). Otherwise the SMILES in train's `eea` row and the same SMILES in the val fold's `ei` row would leak via the matrix-completion features.

### Chemprop for the 5-pack
Multitask over all 7 targets. But because the 5-pack has so few labels, add **PI1M SSL pretraining** as an optional Phase 0. Simple recipe:
- Pretrain the Chemprop encoder on PI1M with an atom-masking self-supervised objective (randomly mask ~15% of atoms, predict atom type).
- Fine-tune on train.csv (all 7 targets, multitask heads).
- Time budget: PI1M has 995K SMILES; even 1 epoch at batch 512 on Kaggle GPU is ~30–45 min. 3 epochs = ~2h. If wall time is a concern, subsample to 200K.

---

## Feature engineering (both tracks)

Feature families — same set for all targets (unified featurizer):
1. RDKit 2D descriptors (~210) — via `Descriptors.CalcMolDescriptors`. Impute inf/NaN with median.
2. Morgan-r2 count FP, 2048 bits.
3. Morgan-r3 count FP, 2048 bits.
4. MACCS keys, 167 bits.
5. Avalon FP, 512 bits.
6. Atom-Pair count FP, 2048 bits.
7. Topological-Torsion count FP, 2048 bits.
**Total ≈ 11,235 features.** Same as Round 1.

For Track B, append 4 auxiliary numerical features (the 4 other 5-pack values) + 4 NaN-mask indicator features per row.

---

## Wall-time / compute budget

Everything must run in one Kaggle notebook. Kaggle GPU (T4) limit is typically ~9h continuous compute. Rough per-phase estimate:

| Phase | Est. wall-time (Kaggle GPU) | Notes |
|-------|:---:|-------|
| Load + featurize (RDKit + 6 FPs on ~12k unique SMILES) | 5–10 min | Single-threaded RDKit is the bottleneck; can multiprocess. |
| GBM cocktail — 5 folds × 3 models × 7 targets (with early stop) | 30–60 min | tg and egc dominate; small targets are seconds. |
| Track-B matrix-completion GBMs — 5 folds × 5 targets | 5–10 min | Small data. |
| Chemprop pretrain on PI1M (1 epoch @ 200K SMILES) | 30–45 min | Optional. Can skip if tight. |
| Chemprop fine-tune — 5-fold × 3-seed multitask | 90–150 min | Depends on epochs + patience. |
| NNLS blending + submission generation | <5 min | |
| **Total (without PI1M pretrain)** | **~3.5–4.5 h** | Comfortable under 9h Kaggle GPU limit. |
| **Total (with PI1M pretrain)** | **~5–6 h** | Still under. |

---

## Deliverables (in this order)

1. `experiments/_utils.py` — paths, `setup_logging`, per-target transforms + inverses, `stratified_quantile_folds`, `group_kfold_by_smiles`, feature-computation helpers with in-memory caching.
2. `experiments/exp_featurize_all.py` — build the ~11K feature matrix on train+test (in-memory), sanity checks. Sanity-runs standalone so we can validate the featurizer without training.
3. `experiments/exp_gbm_cocktail_multitarget.py` — Track A + Track B (Track B without the matrix-completion features first — just standard SMILES → target per-target GBMs). One script; writes per-target OOF and test preds under `results/exp_gbm_cocktail/`.
4. `experiments/exp_matrix_completion_5pack.py` — Track B upgraded with auxiliary-target features for the 5 small targets. Writes `results/exp_matrix_completion/`.
5. `experiments/exp_chemprop_multitask.py` — multitask Chemprop over 7 targets, 5-fold × 3-seed, EpochLogger wired. `results/exp_chemprop_multitask/`.
6. `experiments/exp_blend_nnls.py` — per-target NNLS blend across the previous experiments' OOF; produces final `submission.csv`.
7. (Stretch) `experiments/exp_pi1m_pretrain.py` — PI1M SSL pretraining; artifacts consumed by an updated Chemprop fine-tune script.

Each experiment writes: `run.log`, `oof.csv`, `submission.csv`, `cv_summary.json`, `checkpoint.pkl.gz`.

## Levers to hold in reserve
- Scaffold-stratified split (in addition to quantile) — may help egb whose test has 93 rows with no cross-target info.
- Per-target target transforms tuned by OOF (`identity`, `log1p`, `sqrt`, Yeo-Johnson).
- Second Chemprop variant (Sum aggregation, depth 3) — only if the single variant plateaus and we have headroom.
- LB probe: submit constant `train.mean()` per target (via long-form submission with a value per target_type) to detect distribution shift. Costs 1 submission.

## Open questions for Dhruval
1. **Aim:** are we optimizing for **top 3** (worth spending ~6h on the Kaggle notebook, including PI1M pretraining), or **top 15 safely** (skip the PI1M pretraining lever)?
2. **Development plan:** do we develop locally end-to-end and only port to Kaggle at the end, or develop in the Kaggle notebook from day 1? Local dev is faster to iterate but the pipeline is not final until it runs on Kaggle inside the wall-time budget.
3. **PI1M pretraining lever:** in for the first submission, or held back as a v2 improvement?
4. **Submission budget:** 3/day, timeline ~11 days if it's actually 11 days out → max ~33 submissions before final deadline. First submission should be Track A + Track B without matrix completion, just to confirm the plumbing.
