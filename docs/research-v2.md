# Research v2 — What's Left to Try From LB 0.897

Compiled 2026-08-06 in response to `docs/SESSION_HANDOFF.md`. Supersedes prioritization in `docs/research.md`; that v1 was written at LB 0.857 and most of its top levers (Chemprop multitask, Maxwell prior, per-target NNLS, chain-ext trimer, LB shift probe) are now done and in production.

**Current state:** LB **0.897** (`exp_blend_nnls_3seed`), rank 8. Best solo LB **0.894** (`exp_chain_ext_lgbm`, +0.028 OOF-LB gap). Top LB 0.916–0.918 (MUGABROS, Sandman, low sub counts). Top-3 requires LB ≥ **0.903** (**+0.006**). ~2–3 days remaining, 3 subs/day.

**Core problem, from the handoff**: past chain-ext v1, every proposed "improvement" has flipped a **+0.028 OOF-LB gap** into a negative gap and dropped LB by 0.026–0.293. Six documented failures fit the signature. This shapes every lever in this doc.

---

## 1. The meta-framework — OOF-LB gap preservation

Before proposing anything, restate what the failures teach:

| lever kind | historical effect on our gap | why |
|---|---|---|
| Add features to chain-ext v1 | destroys gap | features can capture fold-specific structural correlations that don't repeat on test |
| Per-target Optuna | destroys gap | search extracts every last drop of OOF, eliminating slack |
| Per-target transform search | destroys gap | 5 transforms × N trials = compound selection bias on 220 rows |
| IterativeImputer with any leak | catastrophic | val fold's target values contaminate features |
| Add correlated blend partner | destroys gap in blend | Chemprop + chain-ext LGB are too similar |
| Add uncorrelated but weaker blend partner | preserves gap, small LB gain | mono-LGB + Chemprop worked at 0.897 |
| Add data (more train rows) | **preserves gap** | data smooths error variance without changing fit target |
| Multiple seeds / bagging | preserves gap | pure variance reduction |
| Post-processing (physics) | preserves gap | acts on predictions, not fit |
| Different-math base | preserves gap | GP / TabPFN / RankUp have different failure modes than GBM/D-MPNN |

**Two rules that follow:**

1. **Prefer data-adding levers over feature-adding levers** for chain-ext v1. Never re-tune it, never add hand-engineered features to it. Only add rows (bagging, pseudo-labels), post-processing (physics), or blend partners with **structurally different math** (not just different feature encoding of the same math).

2. **Match OOF-LB gaps when blending.** A model with a +0.028 gap blended with one that has a -0.010 gap will get pulled down by the negative-gap partner. If you must blend across divergent gaps, apply the LB-bias mitigation pattern already in `exp_blend_nnls_3seed.py` (weight floor + additive bias).

---

## 2. Prioritized action list (v2)

Ranked by **expected LB lift × probability of preserving the gap ÷ compute cost**. Every lever below is untried, or explicitly identified in the handoff §8 as unexplored.

| # | lever | expected LB | gap-safe | compute | confidence | notes |
|---|---|:---:|:---:|:---:|:---:|-------|
| 1 | **PI1M pseudo-label augmentation done right** (see §5) | **+0.005 to +0.015** | Y (data-adding, not features) | 3–5 h CPU | Med-High | §8 handoff calls this the biggest untested lever. Sandman/MUGABROS almost certainly do this. |
| 2 | **Chemprop 5- or 6-seed bag** (from 3) | **+0.001 to +0.003** | Y (bagging = variance reduction) | +150 min per extra 2 seeds | High | Diminishing returns after 5. Blend NNLS weights should barely shift. |
| 3 | **RankUp pseudo-label pretraining of a small MLP** on PI1M, then use as 3rd NNLS base (see §6) | +0.003 to +0.010 | Y (different math, orthogonal error) | 4–6 h CPU / 2 h GPU | Med | The 1st place NeurIPS 2025 recipe. Untried by us. |
| 4 | **Gaussian Process with Tanimoto kernel per target** as 3rd NNLS base (see §7) | +0.002 to +0.008 | Y (different math) | 30–60 min per target × 7 | Med-High | Classical polymer regression winner. Errors orthogonal to GBM. |
| 5 | **Bicerano/van Krevelen physics prior as 3rd NNLS base** for Tg and eps (see §8) | +0.002 to +0.005 | Y (physics post-hoc) | 2–3 h CPU (implement) | Med | Structurally different errors. Only 2 targets get a boost. |
| 6 | **Rank-based blend** replacing NNLS (see §9) | 0 to +0.004 | Y (scale-invariant) | 5 min | Med | Robust to per-model bias without floor/bias hacks. Try as a safety net. |
| 7 | **TabPFN on top-500 features for small targets** (see §10) | 0 to +0.008 on ei/eea/eps/nc | Y (in-context, no training) | <10 min per fold | Med | Zero-training foundation model. Bounded to <500 features / <10K rows — fits our small targets perfectly. |
| 8 | **NGBoost per target** with LogNormal / Normal distribution, as 3rd base | +0.001 to +0.005 | Y (probabilistic GBM = different objective) | 30 min | Low-Med | Adds uncertainty; use as inverse-variance weight in blend. |
| 9 | **Chemprop `--polymer` mode** (Coley fork) with weighted repeat-unit bonds as 3rd base | +0.002 to +0.005 | Y (different molecular representation) | 3–5 h CPU | Med | Handoff §8 lists this. Fits Kaggle runtime. |
| 10 | **Mordred descriptors on dimer** as extra feature family in a NEW LGB (not on chain-ext v1) | +0.001 to +0.005 solo, blend more | Careful — features can overfit | 2 h featurize + 30 min train | Med | All top-5 NeurIPS 2025 solutions used Mordred. Compute on dimer per OPC post-comp report. Add as a SEPARATE base for blend, don't modify chain-ext v1. |
| 11 | **Scaffold-balanced GroupKFold** across all existing bases + refit | +0.001 to +0.003 | Y (better CV, not overfit source) | rerun all bases | Med | Fold 4 consistently trailing on small targets. |
| 12 | **PI1M nearest-neighbor smoothing** — post-fit predictions blended with nearest-PI1M-neighbor's teacher pseudo-label | +0.001 to +0.003 | Y (post-hoc) | 1 h | Low-Med | Cheap fallback if pseudo-label training is too risky. |
| 13 | **Cross-target OOF as Ridge meta features** on top of NNLS | +0.001 to +0.004 | Careful — small-data overfit risk | 1 h | Low-Med | Handoff §8 #9. Watch for 220-row overfit. |
| 14 | **Chemprop with SMILES enumeration TTA** at test time | +0.001 to +0.003 | Y (post-hoc, no train change) | +30% inference | Low-Med | Chemprop is theoretically perm-invariant but empirically has small SMILES-order variance. |

**Cumulative EV of executing 1 + 2 + 3 or 4 + 6:** roughly **+0.006 to +0.020 LB → target range 0.903–0.917**. That puts us at rank 1–5.

**Do NOT re-execute** (already tried, documented failures): per-target Optuna, per-target transform search, IterativeImputer for aux, LB distribution shift probe (no shift), 15 domain features, per-fold-agressive fitting, chain-ext v1 modifications.

---

## 3. Answers to handoff §9 questions

### Q1 — RankUp exact recipe for jday96314

**Loss formulation** ([arXiv 2410.22124][rankup]):
```
p̂_ij = softmax(r(x_i) − r(x_j))          # pair scored by regression-head diff
ℓ_arc = (1/N_lb²) Σ_ij CE(y_ij, p̂_ij)     # supervised pair CE on labeled
      + ω_ulb · (1/N_ulb²) Σ_ij 1[max(p̂_w_ij) > τ] · CE(argmax(p̂_w_ij), p̂_s_ij)
```
- Weak/strong augmentation (FixMatch pattern). For SMILES: weak = canonical; strong = randomized SMILES.
- Confidence threshold `τ = 0.95`, temperature `0.5`, `ω_arc = 0.2`, `ω_rda = 1.0`.
- **Regression Distribution Alignment (RDA)**: sort labeled labels, interpolate to unlabeled size, sort pseudo-labels the same way, replace each pseudo-label with the corresponding sorted labeled value. Refined every 1024 steps. Warm-up: `min(iter / α_warm, 1.0)`.
- Combined loss: `ℓ = ℓ_reg + ω_rda · ℓ_rda + ω_arc · ℓ_arc`.

**jday96314's application** ([tabular training script][jday-tab], [BERT pretrain][jday-repo]):
- Encoder: ModernBERT-base, CodeBERT-base, polyBERT (we can only use the first two — polyBERT weights are DISALLOWED).
- Pretraining data: **pseudolabeled 50k PI1M subset**, not the full 995k.
- LR 8e-5 (ModernBERT/CodeBERT), batch 32, 3–6 epochs, AdamW (weight_decay 0.01, fused), OneCycleLR with 10% warmup, mixed-precision bfloat16, gradient clip 1.0.
- **Margin threshold: 0.2 × per-target standard deviation** of the teacher predictions. Pairs whose |teacher_diff| < margin are masked from the loss.

**Kaggle-notebook feasibility for us**: yes, if we use a smaller encoder (RoBERTa-small, 6L/384h) instead of ModernBERT-base. Timing: 50k pseudolabeled × 3 epochs × ~22M params ≈ 90–120 min on Kaggle P100, ~3 h on CPU. Fits.

### Q2 — PI1M pseudo-labeling done right (§5 dedicated below)

The two failure modes that killed Round 1 pseudo-labeling were:
- **Fold-alignment leak**: pseudo-labels created from a teacher trained on all-train, then re-used inside CV → val fold's label information indirectly leaks through teacher trained on val.
- **Noise > signal on small teacher**: a teacher trained on 220-row targets produces noisy PI1M pseudo-labels, which then swamps the real signal at fine-tune time.

Both are avoidable — see §5.

### Q3 — Bicerano Tg implementation

Bicerano's method uses **66 group-contribution values** for very small subgroups on elements {C, N, O, H, S, P, Si, halogens}, correlated to bond indices from the polymer repeat unit ([review][bicerano-review]).

**No ready Python implementation** was surfaced by search. Two viable paths:
- **Ramprasad-Group's `psmiles`** ([GitHub][psmiles]) supports dimerize + Mordred fingerprints on polymer SMILES. Not Bicerano exactly, but the same class of macromolecular descriptors.
- **PolyMetriX** ([Nature 2025][polymetrix]) — new polymer descriptor ecosystem. Package is behind a Nature paywall for full details; worth `pip install polymetrix` and reading the README.
- **DIY**: encode Bicerano groups as ~30 SMARTS patterns matching each subgroup (functional/aromatic/etc.), fit a Ridge from group counts → Tg on our 4143 train rows. This is a knock-off Bicerano. Not the paper's exact coefficients but same-class-of-model errors will be orthogonal to a fingerprint GBM.

**Typical accuracy on independent polymer sets**: ~30–50 °C RMSE ([modified Bicerano review][bicerano-modified]) — much worse than GBM on our data (Tg R² 0.90+), so it's not a solo winner but a **diverse NNLS base**.

### Q4 — van Krevelen for dielectric

Van Krevelen's method for dielectric constant is documented in his book (Ch. 11, "Properties of Polymers") but no maintained Python package. Same DIY path as Bicerano: encode the ~30 group additivity terms as SMARTS, fit a Ridge / GBM, use as NNLS base for eps.

**Alternative**: the psmiles package's Mordred fingerprints computed on a **dimer** SMILES capture many of the same physics-additivity features Mordred was designed for. Solo LB from Mordred-dimer alone is likely 0.86–0.88 on eps (comparable to our current LGB base).

### Q5 — Mordred subsets that help polymers

Mordred: ~1600 descriptors. Not all useful; many degenerate on polymer-like molecules. Categories most cited in polymer regression:
- **Constitutional** (atom counts, ring counts by element/hybridization)
- **Autocorrelation** (Moreau-Broto, Moran, Geary — encode graph connectivity)
- **Topological** (Kier-Hall, BertzCT, BalabanJ, Wiener)
- **InformationContent** (structural entropy)
- **BCUT** (Burden eigenvalues weighted by atomic properties)
- **CPSA** (charged partial surface area — needs 2D)

The **OPC post-competition report** specifically notes: "Mordred descriptors computed on **dimers** often perform as well as or better than those computed on monomers, likely because dimers encode additional information including inter-monomer relationships."

**Practical recipe:** compute Mordred on dimer, drop columns with `std < 0.01` or `nunique < 3` (typical: ~800 survive on polymer sets). Filter further by mutual-information rank per target. Feed as an extra family — but into a **separate new LGB experiment**, not by extending chain-ext v1.

### Q6 — Was the NeurIPS 2025 Tg shift trick a metric/unit issue?

From the [Open Polymer Challenge post-competition report][post-report]: "Many Tg values evaluated with hyperbolic fits are higher than the corresponding values from the bi-linear fit. Public leaderboard showed mean Tg of 102.9 °C (σ=103.7), while the private leaderboard exhibited 179.8 °C (σ=134.9) — a substantial shift **unrelated to unit conversions**."

The shift was a **methodology difference in the ground-truth simulation** (which fitting method was used for the MD trace), NOT a unit conversion or metric issue. The 2nd-place team's +40 constant was a numerical fit to that shift, discovered by LB probing.

**Our LB shift probe returned R² = -0.007**, ruling out an equivalent per-target constant shift bigger than ~0.22σ on any target. So this specific trick is **not applicable to our competition**. Sandman/MUGABROS at 0.916 are winning through modeling, not through a probe trick. Move on.

### Q7 — Polymer-specific architectures fitting Kaggle 12h with ~6k train samples

Tractable in ≤ 8 h on a Kaggle notebook:
- **Chemprop 2.x multitask D-MPNN** — done.
- **Chemprop `--polymer` mode** ([Coley fork][polymer-chemprop-repo]) — weighted repeat-unit bonds. ~4h, different molecular representation.
- **GAUCHE + Tanimoto-kernel Gaussian Process** ([GAUCHE lib][gauche]) — per-target, ~10 min small targets, ~40 min tg. See §7.
- **TabPFN v2** with a feature-projection head ([TabPFN for chemistry][tabpfn-chem]) — ~1 min per prediction batch, no training. Bounded to ≤ 500 features / ≤ 10K rows.
- **NGBoost** with LogNormal/Normal distribution ([NGBoost paper][ngboost]) — 30 min for all 7 targets. Gives uncertainty as a bonus.
- **A small from-scratch RoBERTa (6L/384h)** with MLM or MTR or RankUp pretraining on PI1M — ~2–3 h GPU, ~6 h CPU.

Not tractable: Uni-Mol (24 GB VRAM, 3D conformers), MMPolymer (same), TransPolymer from scratch (85M params).

### Q8 — Handling small-data targets (200–400 rows) without overfit

Verified from NeurIPS 2025 top-solutions + our own failures:
- **Do NOT per-target Optuna** on ≤ 400-row targets. Selection bias across 30 trials at that scale kills the LB.
- **Do share representation via multitask** (Chemprop) — a shared encoder trained on all 7 targets uses the 4143-row Tg to smooth the small targets' encoder.
- **Do use robust base models** (LGB with fixed hyperparams from Round 1, CatBoost with defaults, GP with Tanimoto kernel).
- **Do augment via cross-target matrix completion** (we do; 96% of small-target test rows have same-molecule in train under another target).
- **Do augment via PI1M pseudo-labels done safely** (§5).
- **Consider TabPFN** — genuinely designed for < 10K samples, no per-target tuning needed.

### Q9 — How to preserve positive OOF-LB gap while adding features

Short answer: **you can't reliably**, but you can add things that aren't "features":
- **Add data** (bagged seeds, pseudo-labels, augmented rows) instead of adding features.
- **Add post-processing** (physics constraints, LB-shift offsets) — acts on predictions, not fit.
- **Add a whole separate model as a blend partner** with orthogonal errors, then blend with gap-matched weighting.
- **When you MUST add features**: monitor **gain share per feature block**. If a new k-feature block gets more than `3× (k / total_features)` of the total gain, it's over-earning and probably overfitting. The 15-feature domain-knowledge failure hit 30–39% gain share out of 14k features — that's 30× normalized gain-per-feature, which is exactly the warning sign.

### Q10 — Ensembling for divergent OOF-LB gaps

Three tools, in order of increasing sophistication:
1. **LB-bias mitigations** (already in `exp_blend_nnls_3seed.py`): weight floor for the model with the positive gap, additive bias transferring weight from negative-gap models. Requires knowing per-base LB, so costs 1 sub per new base.
2. **Rank-based blending** (see §9): scale-invariant, doesn't need bias calibration. Gets you close to the same result without bias hacks, at the cost of ignoring absolute values (may lose calibration on eps/nc where absolute scale matters for R²).
3. **Learn per-model calibration** with a small Ridge on OOF that outputs `y = α + β_1·y_hat_1 + β_2·y_hat_2 + ...` — but constrain β_i ≥ 0 to avoid inverting a base. On 220-row small targets, keep β count ≤ 3.

Rule: if two models have solo LBs within 0.005 of each other, blending them will either add or subtract ~0.003 depending on their correlation. If correlation > 0.98, blend hurts. Compute base-vs-base OOF residual correlation *before* submitting a new blend.

---

## 4. The "add data, not features" doctrine

Given the six documented failures all involve feature-adding or fit-tightening, the highest-EV levers are the ones that add data. Ranked:

- **PI1M pseudo-label augmentation** (§5) — biggest untested lever
- **Bagging more Chemprop seeds** (5 or 6, from 3) — cheapest
- **Bagging more LGB seeds** for chain-ext v1 (5 seeds, from 1) — cheapest, ~30 min
- **SMILES enumeration TTA** for Chemprop at test time — free, adds test-time robustness
- **Multi-scaffold multi-fold ensembling**: run chain-ext v1 with 3 different fold seeds, average — pure variance reduction

**Concrete action for the next 24 h:** run 5-seed chain-ext LGB. It's the cheapest gap-preserving lever and gives a clean read on how much of chain-ext v1's LB is variance vs signal. If solo LB moves +0.002, add it. If not, no harm done.

---

## 5. PI1M pseudo-labeling — the RIGHT way

Round 1 failed at this. The handoff and NeurIPS 2025 confirm the top teams do it. Two failure modes to avoid:

### 5.1 Failure mode #1 — teacher-CV leakage

**Wrong:** train teacher on all train, generate pseudo-labels on PI1M, add PI1M as extra train rows, run 5-fold CV. → OOF looks great because each fold's val rows implicitly see themselves through the pseudo-labels (the teacher was trained on them).

**Right:** **per-fold teacher pilots.** For each of the 5 folds, train a fold-teacher on that fold's train slice ONLY, generate pseudo-labels for a PI1M subset, add those as extra train rows for that fold's student. The val slice remains untainted.

Costs 5× teacher training time. Mitigation: use a fast teacher (LGB defaults, not chain-ext v1's full 4000-round training). ~5 × 10 min = 50 min.

### 5.2 Failure mode #2 — noise swamps signal

Teacher trained on 220-row `eea` produces very noisy PI1M pseudo-labels. Adding 100k noisy rows drowns the 220 real ones.

**Right:**
- **Small PI1M subset** (10k–20k, not 100k+). Trades label diversity for noise reduction.
- **Confidence filtering**: keep only PI1M rows where teacher's out-of-bag prediction variance is below some threshold (e.g., 5-seed teacher; drop rows where stdev of 5 preds > 0.3 × per-target std).
- **Downweight pseudo-labels**: sample_weight = 0.2–0.3 on pseudo rows, 1.0 on real rows. LGB natively supports this.
- **Tanimoto distance filtering**: keep only PI1M rows within Tanimoto 0.4–0.9 of some train molecule. Too far (< 0.4) = out of distribution; too close (> 0.9) = redundant with train and possibly leaked into test.

### 5.3 Recommended pipeline

```
For each of 7 targets:
    For each of 5 folds:
        1. Train teacher on fold-train (LGB defaults + full FP stack, ~2 min)
        2. Predict on 20k random PI1M rows (~10 sec)
        3. Filter by teacher's 5-seed prediction stdev < 0.3σ  (keeps ~30-50%)
        4. Filter by min Tanimoto to fold-train in [0.4, 0.9]   (keeps ~50-70%)
        5. Concatenate filtered PI1M rows (weight 0.25) to fold-train rows (weight 1.0)
        6. Train student = chain-ext v1 config on augmented fold-train
        7. Predict fold-val → OOF
    Refit student on full-train + full-filtered-PI1M for test predictions.
```

**Total compute:** 7 targets × 5 folds × (2 min teacher + ~12 min student full-feature train) ≈ 8–9 h CPU. Marginal for a Kaggle notebook alongside Chemprop — better to run locally overnight and submit result.

**Expected LB:** +0.005 to +0.015. The wide range reflects genuine uncertainty on how noisy the teachers are. Confidence-filter recall is the main knob.

**Safety valve:** if the pilot PI1M-augmented chain-ext OOF underperforms plain chain-ext v1 on any target, drop PI1M for that target and use plain chain-ext v1's predictions instead. Mix-and-match per target is legal and cheap.

---

## 6. RankUp pretraining — the 1st-place NeurIPS 2025 recipe adapted for us

RankUp trains a regression head with a joint (regression + pairwise ranking) loss on labeled data, plus (weak/strong-augmented pairwise ranking) loss on unlabeled data with confidence filtering, plus RDA (regression distribution alignment) to refine pseudo-labels. The pairwise ranking framing is key — it's robust to teacher noise because you only need the teacher to get the *ordering* right, not the exact values.

### 6.1 What we adapt

- **Encoder:** small from-scratch RoBERTa (6 layers, hidden 384, 8 heads, ~22M params, BPE vocab 2000). Not ModernBERT — that's pretrained. Trains from scratch inside the notebook.
- **Weak augmentation:** canonical SMILES. **Strong augmentation:** randomized SMILES via `Chem.MolToRandomSmilesVect`.
- **Loss:** `ℓ = ℓ_reg + ω_rda · ℓ_rda + ω_arc · ℓ_arc` with `ω_arc = 0.2`, `ω_rda = 1.0`, `τ = 0.95`.
- **Margin threshold:** 0.2 × per-target std of teacher predictions.
- **Pretraining data:** 20k–50k PI1M rows pseudolabeled by the LGB cocktail teacher (per §5.3's per-fold pilots).

### 6.2 Kaggle-runtime timing

- Featurize PI1M subset: ~2 min per 10k
- Encoder training on 20k pseudolabeled + 6k real, batch 128, 5 epochs: ~1 h on P100 GPU or ~2.5 h on CPU
- Per-fold RankUp fine-tune on real fold-train + pseudo: ~15 min × 5 folds = 75 min
- **Total ~4 h.** Fits alongside Chemprop 3-seed (225 min) in a single Kaggle notebook run.

### 6.3 How to integrate

Output the RankUp encoder's per-target predictions as `results/exp_rankup_ptrain/oof.csv` and `submission.csv`. Add as a 3rd NNLS base to `exp_blend_nnls_3seed`. Expected blend LB: **+0.003 to +0.010**.

If short on time, the **cheaper approximation**: skip the RoBERTa encoder, use RankUp only as a *loss function* on top of a fingerprint-input MLP. Trains in ~30 min. Weaker but still adds a diverse base signal.

---

## 7. Gaussian Process with Tanimoto kernel — orthogonal-math base signal

Historically the standard tool for polymer property regression pre-2023. `GAUCHE` ([lib][gauche]) is a maintained PyTorch+GPyTorch library with a `TanimotoKernel` implementation. `pip install gauche`.

### 7.1 Why this preserves the gap

GPs have **fundamentally different error modes** than GBM or D-MPNN:
- No fold-wise fit; posterior mean is a weighted average of training targets by kernel similarity.
- No feature-level overfitting because there are no learnable feature weights.
- Errors are large on molecules with low max-Tanimoto to train (correctly), small on high-similarity molecules.

**Blending a GP with a GBM adds structurally-diverse errors** the way Chemprop+LGB does — it isn't just "another way to weight the same features."

### 7.2 Practical recipe

```python
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# X = Morgan-r2 count fingerprint as binary/float tensor
# y = per-target train values
kernel = TanimotoKernel()
model = SingleTaskGP(X, y, covar_module=kernel)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)   # ~20-40 sec for 220 rows, ~5 min for 4143
```

Per target × 5 folds: ~15 min total for small targets, ~40 min for tg. Fits into Kaggle easily.

### 7.3 Where GP helps

- Small-data targets (`eea`, `ei`, `eps`, `nc`, `egb`): GP is strongest exactly where GBM struggles.
- `tg`: GP with Tanimoto kernel on 4143 rows is the historical gold standard for polymer Tg. Solo R² likely 0.85–0.90.
- Expected blend LB gain: **+0.002 to +0.008**. Highest confidence on the small-data 5-pack.

**Fallback:** if GAUCHE is too heavy for the Kaggle notebook, `sklearn.gaussian_process.GaussianProcessRegressor` with a custom `PairwiseKernel(metric='jaccard')` gets ~80% of the way there.

---

## 8. Bicerano / van Krevelen as physics-diverse NNLS bases

Handoff §8 lists these as untried. They're not going to solo-beat GBM, but their errors are **structurally different** (group-additivity vs fold-CV-fit), which is exactly what NNLS blends thrive on.

### 8.1 Quick-and-dirty Bicerano Tg

No ready Python. Build:
```python
BICERANO_SMARTS = {
    "sp3_C":     "[CX4;!R]",
    "sp3_C_ring":"[CX4;R]",
    "aromatic_C":"c",
    "sp2_C":     "[CX3;!R]",
    "amide_N":   "[NX3;$(NC=O)]",
    "amine_N":   "[NX3;!$(NC=O)]",
    "ether_O":   "[OX2;!$(O=C);!$(OC=O)]",
    "carbonyl_O":"[OX1]=C",
    "ester_C":   "[CX3](=O)[OX2H0]",
    "sulfone_S": "[SX4](=O)(=O)",
    "F":         "[F]",
    "Cl":        "[Cl]",
    "Br":        "[Br]",
    "Si":        "[Si]",
    "cyano":     "C#N",
    "phenyl":    "c1ccccc1",
    "biphenyl":  "c1ccc(-c2ccccc2)cc1",
    "thiophene": "c1ccsc1",
    "aliphatic_ring": "[CX4;R]1[CX4;R][CX4;R][CX4;R][CX4;R][CX4;R]1",
    ...
}
# Count matches per polymer → 20-30 features → fit Ridge / LGB to Tg
```

Expected solo Tg R²: **0.75–0.85** (vs chain-ext v1's 0.906). Weaker but diverse. Blend into `exp_blend_nnls_3seed` with a small weight; watch for whether the NNLS gives it any weight at all — that's the real test of diversity.

### 8.2 Van Krevelen dielectric

Same recipe, different SMARTS list biased toward dielectric-relevant motifs (`F`-substitution, `Si`, `sulfone`, `carbonyl` for polar/nonpolar contribution). Coefficients from van Krevelen's book (Ch. 11). Solo eps R²: **0.70–0.80**. Diverse from Maxwell prior because Maxwell is *only* through Nc; van Krevelen goes direct from structure.

### 8.3 Combined approach — polymer classical physics base

Rather than shipping Bicerano and van Krevelen as separate bases, build **one small physics-additivity model per target** using target-specific SMARTS lists and Ridge/LGB heads. Package as `exp_physics_prior_ridge.py` — 7 target-specific models, ~2 h to implement, ~30 min to train. Add to NNLS as 3rd base.

**Expected LB after adding to current 2-way blend: +0.001 to +0.004.** Small but genuinely diverse.

---

## 9. Rank-based blending — the safer alternative to NNLS

Rank averaging is scale-invariant. Instead of blending `y_pred` values (which requires bias calibration when models disagree on scale), you rank the predictions and blend the ranks.

### 9.1 Basic recipe

```python
from scipy.stats import rankdata

def rank_blend(preds_list, weights):
    ranks = np.stack([rankdata(p, method='average') for p in preds_list])
    blended_rank = np.average(ranks, weights=weights, axis=0)
    # Map back to the value scale using the mean prediction's distribution
    mean_pred = np.average(preds_list, weights=weights, axis=0)
    sort_idx = np.argsort(blended_rank)
    out = np.empty_like(mean_pred)
    sorted_vals = np.sort(mean_pred)
    out[sort_idx] = sorted_vals
    return out
```

Fit weights on OOF (still NNLS on ranked OOF). Apply to test.

### 9.2 Why it's gap-safe

Rank blending **discards absolute-value information**. Two models that disagree on scale but agree on ordering blend cleanly. This is exactly the case that broke our chain-ext + Chemprop blend: they agreed on which molecules had high/low targets (correlated ranks), but disagreed on absolute values (different scales due to different OOF-LB gaps). NNLS on raw values got confused; rank blending would just have agreed on the ordering.

### 9.3 Loss on R²

R² is scale-sensitive. Rank blending loses ~10-20% of the "calibrated absolute value" information. On our tightly-scaled targets (nc, eps), this can hurt. **Test on OOF first**; if per-target rank-blend R² is within 0.005 of NNLS-blend R², ship it — the gap-safety upside is worth it.

### 9.4 Geometric mean rank

`gmean` instead of arithmetic mean of ranks is even more robust to outlier models. Sometimes wins on Kaggle when one base is much worse than the others.

**Concrete action:** try rank-blend and gmean-rank-blend as parallel submissions once. If either beats 0.897, adopt.

---

## 10. TabPFN for the small-data 5-pack

TabPFN v2 ([Nature 2024][tabpfn-nature]) is a transformer foundation model that does **in-context tabular regression with no training** — you pass train `(X, y)` and test `X` to a pretrained network, and it returns predictions in seconds. Its performance holds on datasets ≤ 10k rows / ≤ 500 features.

### 10.1 Why this fits

- Our small-data targets are 221–337 rows: comfortably in TabPFN's sweet spot.
- Adding TabPFN as a base uses NO training compute (~1 min per fold).
- **Its errors are structurally different** from GBM (no gradient boosting; it's an amortized Bayesian inference).
- ChemRxiv 2025 report: TabPFN "performs on par with XGBoost in classification but demonstrates clear and stable advantages in regression, with its strongest gains on small and medium data sets."

### 10.2 Constraint: ≤ 500 features

Our feature stack is 14k. Solutions:
- **Per-target top-500 features by mutual information** with the target on train. Fast (~30 sec per target). Feature list per target.
- **PCA to 500 dims** on the full feature stack. Simple but loses interpretability.
- **Morgan-r2 top 500 bits + top 100 RDKit descriptors by MI**. Handpicked. Fastest.

### 10.3 Rules constraint — no pretrained weights

TabPFN is pretrained on synthetic tabular data. Rules disallow "pretrained model weights." **Ambiguous case.** The Kaggle rules say "no pretrained models" but TabPFN's weights are trained on SYNTHETIC tabular data, not on any chemistry dataset — arguably a "learned inductive prior" that's more like a novel algorithm.

**Recommendation**: DM the hosts to clarify before using. If disallowed, fall back to the GP-Tanimoto route (§7) which is unambiguously legal.

If allowed:
- Expected per-target OOF: 0.80–0.90 on small targets (competitive with GBM).
- Add to NNLS as 3rd base. Expected blend LB: **+0.002 to +0.008**.
- Compute: ~10 min end-to-end. Cheapest strong-lift lever if legal.

---

## 11. Chemprop `--polymer` mode + weighted repeat-unit bonds

The Coley group maintains a polymer fork of Chemprop ([`coleygroup/polymer-chemprop`][polymer-chemprop-repo]) that natively handles numbered wildcards `[*:1]`, `[*:2]` and specifies a weighted extra bond representing the periodic-boundary connection between repeat units.

Format: `SMILES <1-2:0.5:0.5 ~ 2` = the polymer's SMILES followed by `<atom-to-*1 to atom-to-*2 bond, forward-weight, reverse-weight> ~ degree_of_polymerization`.

### 11.1 Why this could preserve the gap

- **Different molecular representation than chain-ext trimer.** Trimer expansion changes the graph size; polymer mode keeps the monomer graph and adds a weighted `*→*` edge signaling the connectivity.
- **Chemprop with polymer mode should generalize differently** from standard Chemprop or from chain-ext-Chemprop (which we already tried at 27.5 h wall time and got LB -0.001).
- Handoff §8 lists this as untried and expects +0.002 to +0.005.

### 11.2 Compute estimate

Polymer-mode Chemprop's graph size is the same as monomer (not trimer), so runtime should be similar to our monomer Chemprop (~52 min single-seed, ~225 min 3-seed). **Fits Kaggle notebook budget.**

### 11.3 How to integrate

Ship as `exp_chemprop_polymer_mode_3seed.py`. Compare solo LB to `exp_chemprop_multitask_cpu_3seed` (0.892). If solo LB ≥ 0.890, add as 3rd NNLS base. Watch for correlation with standard Chemprop — if OOF residual correlation > 0.95, don't blend (they're the same signal).

---

## 12. Concrete 2–3 day execution plan

Given 3 subs/day and ~2–3 days, prioritize by (LB gain × gap-safety) / (compute + submission cost).

**Day 1 (highest-EV, low-risk):**
- **Morning:** implement and run **PI1M pseudo-label augmentation** with per-fold pilots (§5) locally. Expected wall time: 8–9 h on CPU. Kick off in background.
- **Morning parallel:** implement and run **5-seed chain-ext LGB** (~2 h). Submit if solo LB > 0.894.
- **Evening:** submit `exp_chain_ext_lgbm_pi1m` if OOF > 0.87. Track LB carefully — if OOF-LB gap drops below +0.020, be cautious.
- **Evening 2nd sub:** rank-blend of existing bases (`exp_blend_rank_v1`). Cheap, robust.

**Day 2:**
- **Morning:** implement and run **GAUCHE Tanimoto GP** per target (~1 h total). Get solo LB via sub. Add as 3rd base in a new 3-way NNLS blend if solo LB > 0.87 on average.
- **Afternoon:** implement Bicerano/van Krevelen physics prior (~2 h). Sub as additional NNLS base.
- **Evening:** re-blend with all new bases + existing chain-ext v1 + Chemprop 3-seed. Sub.

**Day 3 (safe finalization):**
- **Morning:** submit the best-OOF blend from Day 2, one variant.
- **Morning:** if time permits, run **Chemprop `--polymer` mode 3-seed** (~4 h). Add as 4th base.
- **Afternoon:** submit final blend.
- **HOLD 1 SUB SLOT** for last-day corrections in case of any distribution surprise.
- **DO NOT** switch to a "safer" model at the last minute (20th-place NeurIPS 2025 lesson).

### 12.1 Submission budget accounting

- Day 1: 3 subs (baseline sanity + PI1M v1 + rank blend)
- Day 2: 3 subs (GP solo + physics base solo + final blend)
- Day 3: 2 subs (final blend + optional polymer-mode blend), 1 held back

Total: 8 subs. Fits.

### 12.2 What to skip

- Chemprop on trimer (already 27.5 h, LB -0.001)
- Any per-target Optuna (proven overfit)
- Per-target transform search (proven overfit)
- LB distribution shift probe (already done, no shift)
- Feature-adding to chain-ext v1 (proven overfit)
- IterativeImputer (leak-prone; you'd need to re-derive the per-fold masking pattern from scratch)

---

## 13. Score expectation

If PI1M pseudo-labeling works as designed (+0.008), plus a diverse GP base (+0.002 in blend), we land at LB **~0.905–0.910** — **rank 3–5**.

If pseudo-labeling also fails to preserve the gap, expect ~0.898 (safe, still top-8).

**Ceiling:** if pseudo-labeling gives +0.015 (upper end) and GP + physics base each contribute +0.003, we hit **~0.918** — **rank 1–3**.

The 0.916 leaders (MUGABROS/Sandman) with <15 subs each are almost certainly doing exactly the PI1M pseudo-label augmentation trick. Doing it right is the only path to matching them.

---

## 14. Sources

- **Handoff document**: `docs/SESSION_HANDOFF.md` — the full context this v2 responds to.
- **Prior research doc**: `docs/research.md` — v1 written at LB 0.857. Levers 1–7 there are now done. This v2 supersedes.
- **Best experiment tracker**: `docs/best-experiment.md`
- **Best ensemble tracker**: `docs/best-ensemble.md`

- **RankUp** (NeurIPS 2024): [arXiv 2410.22124][rankup], [alphaXiv][rankup-alphaxiv], [NeurIPS poster][rankup-neurips]
- **1st place NeurIPS Open Polymer Prediction 2025**: [GitHub jday96314][jday-repo]
- **Open Polymer Challenge post-competition report**: [arXiv 2512.08896][post-report]
- **NeurIPS 2025 Kaggle retrospective (JP)**: [SpeakerDeck calpis10000][calpis]
- **psmiles package** (Ramprasad Group): [GitHub][psmiles], [ReadTheDocs][psmiles-docs]
- **PolyMetriX** (2025): [Nature Comp Mat][polymetrix]
- **GAUCHE** (Gaussian Processes for chemistry): [GitHub][gauche], [paper][gauche-paper]
- **TabPFN** foundation model: [Nature][tabpfn-nature], [chemistry benchmark][tabpfn-chem], [ChemRxiv application][tabpfn-chemrxiv]
- **NGBoost**: [arXiv 1910.03225][ngboost]
- **polymer-chemprop** (Coley Group fork): [GitHub][polymer-chemprop-repo]
- **Bicerano method review**: [Modified Bicerano (2020)][bicerano-modified]
- **Van Krevelen "Properties of Polymers"** — book, no free link.
- **4th place "Less is More" writeup** (Kaggle PSS6E2): [Kaggle][less-is-more] (JS-rendered; blocked from WebFetch)

[rankup]: https://arxiv.org/abs/2410.22124
[rankup-alphaxiv]: https://www.alphaxiv.org/abs/2410.22124
[rankup-neurips]: https://nips.cc/virtual/2024/poster/94365
[jday-repo]: https://github.com/jday96314/NeurIPS-polymer-prediction
[jday-tab]: https://github.com/jday96314/NeurIPS-polymer-prediction/blob/main/tabular/train.py
[post-report]: https://arxiv.org/html/2512.08896
[calpis]: https://speakerdeck.com/calpis10000/kaggle-neurips-open-polymer-prediction-2025-konpe-fan-sheng-hui
[psmiles]: https://github.com/Ramprasad-Group/psmiles
[psmiles-docs]: https://psmiles.readthedocs.io/
[polymetrix]: https://www.nature.com/articles/s41524-025-01823-y
[gauche]: https://github.com/leojklarner/gauche
[gauche-paper]: https://arxiv.org/pdf/2212.04450
[tabpfn-nature]: https://www.nature.com/articles/s41586-024-08328-6
[tabpfn-chem]: https://jonswain.github.io/ai/cheminformatics/data%20science/machine%20learning/2025/01/22/TabPFN-for-chemical-datasets.html
[tabpfn-chemrxiv]: https://chemrxiv.org/doi/10.26434/chemrxiv-2025-szk5s
[ngboost]: https://arxiv.org/pdf/1910.03225
[polymer-chemprop-repo]: https://github.com/coleygroup/polymer-chemprop
[bicerano-review]: https://www.researchgate.net/publication/231370667_A_New_Group_Contribution_Scheme_To_Estimate_the_Glass_Transition_Temperature_for_Polymers_and_Diluents
[bicerano-modified]: https://pubs.acs.org/doi/10.1021/acsomega.0c04499
[less-is-more]: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution
