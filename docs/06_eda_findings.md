# EDA Findings — train / test / PI1M

All numbers derived from `ppp-round-2/{train,test,PI1M}.csv` on 2026-08-01.

## Shapes
- `train.csv`: **7,409 rows, 3 cols** (`smiles`, `target`, `target_type`) — long form.
- `test.csv` : **4,940 rows, 3 cols** (`id`, `smiles`, `target_type`).
  - Note: the challenge overview said 4,497. That's the count of *unique test SMILES*. The actual number of test *rows* (i.e. test predictions to make) is **4,940**.
- `PI1M.csv` : **995,799 rows, 1 col** (`SMILES`).

## Per-target row counts (heavy imbalance)

| target_type | train rows | test rows | test/train ratio |
|-------------|-----------:|----------:|-----------------:|
| tg  | 4,143 | 2,763 | 0.67 |
| egc | 2,028 | 1,352 | 0.67 |
| egb | 337   | 224   | 0.66 |
| eps | 229   | 153   | 0.67 |
| nc  | 229   | 153   | 0.67 |
| ei  | 222   | 148   | 0.67 |
| eea | 221   | 147   | 0.67 |
| **total** | **7,409** | **4,940** | — |

- Test:train ratio is constant (~0.67) across all targets — the split is stratified by target.
- **Five of seven targets have <350 training rows.** Any target-specific model on eea/ei/eps/nc/egb is essentially a small-data problem — regularization will matter, augmentation via multitask training will matter, and PI1M pretraining may matter.
- `tg` and `egc` dominate row counts. They also drive most of the wall time in any model that trains per-row.
- **Mean-R² weighting reality check:** because the metric averages the 7 per-target R² values, the 221-row `eea` counts *just as much* as the 4,143-row `tg`. Doing well on the small-data targets is not optional.

## Per-target value distributions

| target | n | min | q10 | q50 | mean | q90 | max | std | skew | kurt | neg | zeros |
|--------|--:|----:|----:|----:|-----:|----:|----:|----:|-----:|-----:|----:|------:|
| eea | 221 |   0.394 |   0.770 |   2.272 |   2.278 |   3.770 |   5.144 |   1.107 | +0.22 | -0.79 |   0 | 0 |
| egb | 337 |   0.507 |   1.916 |   4.052 |   4.276 |   6.939 |  10.114 |   1.979 | +0.44 | -0.50 |   0 | 0 |
| egc | 2028 |  0.021 |   2.528 |   4.614 |   4.529 |   6.526 |   9.863 |   1.568 | -0.10 | -0.64 |   0 | 0 |
| ei  | 222 |   4.026 |   5.172 |   6.168 |   6.346 |   7.815 |   9.838 |   1.047 | +0.78 | +0.52 |   0 | 0 |
| eps | 229 |   2.610 |   3.494 |   4.320 |   4.577 |   6.196 |   9.090 |   1.094 | +1.21 | +1.67 |   0 | 0 |
| nc  | 229 |   1.560 |   1.657 |   1.900 |   1.934 |   2.253 |   2.758 |   0.235 | +0.88 | +0.78 |   0 | 0 |
| tg  | 4143 | -109.82 |   8.000 | 136.400 | 143.459 | 285.950 | 495.000 | 109.084 | +0.09 | -0.71 | 370 | 4 |

Suggested target transforms (informed by skew + support):
- `identity`: `tg`, `egc`, `eea` (roughly symmetric; large scale for `tg`).
- `log1p`: `eps`, `nc` (right-skewed, strictly positive).
- `sqrt` or `log1p`: `ei`, `egb` (mild-to-moderate right skew).
- For `tg`, **do not log1p** — it has ~370 negatives.

## Duplicates

- **Duplicate `(smiles, target_type)` in train:** 3 pairs (6 rows). All are `tg`. Targets disagree by 5–7 units on `tg` (`98.28 vs 105.00`, `61.10 vs 72.08`, `239 vs 244`) — measurement noise. **Action:** average them.
- **Duplicate `(smiles, target_type)` in test:** 2 rows — the same SMILES × target_type appears twice under different `id`s. **Action:** emit the same prediction for both; nothing to worry about.
- **Unparseable SMILES in train/test:** 0. Every SMILES parses with RDKit.
- **Unparseable SMILES in PI1M:** ~0.7% (14 / 2,000-sample). Filter before use.

## Chemistry composition

- All SMILES contain **exactly 2 wildcard atoms (`*`)** — the standard polymer-repeat-unit notation. Round 1 handling (cap-with-methyl or use `*`-aware fingerprints) transfers directly. **No SMILES have 0, 1, 3+ wildcards** — no edge cases to worry about.
- Elements observed across a 3,000-row (train+test) sample: `B, Br, C, Cl, F, Ge, H, I, N, Na, O, P, Pb, S, Se, Si, Sn, *`. Includes heavier / less common atoms (Ge, Sn, Pb, Se, Na) — some RDKit descriptors misbehave on these; must handle inf/NaN robustly.
- SMILES length (train): median 38 chars, 90th pct 97 chars, max 267.
- RDKit atom count (train, sampled): median 25 atoms, 90th pct 54, max 143. Small-to-medium molecules — Chemprop / GBM on molecular descriptors will train fast per-row.

## Train ↔ test SMILES overlap (**huge finding**)

- 457 unique SMILES appear in both `train.csv` and `test.csv`.
- Only **2** of those (both `tg`) are the same `(smiles, target_type)` — i.e. the same molecule with the same target measured in both train and test. Effectively zero same-target leak.
- **455 SMILES share across train and test with a different target being asked.** That is, we have the molecule's value for property A in train, and the test wants the same molecule's value for property B.

Per-test-target: percent of test SMILES that appear in train (for any target):

| test target | test rows | with any train row for same SMILES | % |
|-------------|----------:|-----------------------------------:|--:|
| **eea** | 147   | 141 | **95.9%** |
| **ei**  | 148   | 142 | **95.9%** |
| **eps** | 153   | 148 | **96.7%** |
| **nc**  | 153   | 148 | **96.7%** |
| **egb** | 224   | 180 | **80.4%** |
| egc     | 1,352 |  72 |  5.3% |
| tg      | 2,763 |   8 |  **0.3%** |

**Interpretation:**
- For eea/ei/eps/nc/egb, the test set is basically a **matrix-completion** problem — the same polymer was measured on some subset of these 5 electronic properties, and we're being asked to fill in the missing entries. Purely-SMILES-based prediction is only ~5% of the story for these; the other 95% is *"given the values of the other electronic properties for this molecule, predict this one"*.
- For `tg` and `egc`, it's an almost pure "predict from SMILES" problem — those 4,143 + 2,028 rows are the meat, and test SMILES are ~all unseen molecules.

For each of the 5 electronic test rows, distribution of how many of the *other 4* electronic properties are known in train for the same SMILES:

| test target | 0 others known | 1 | 2 | 3 | 4 |
|-------------|--:|--:|--:|--:|--:|
| eea | 6 | 20 | 56 | 45 | 20 |
| egb | 93 | 17 | 49 | 52 | 13 |
| ei  | 6 | 17 | 49 | 54 | 22 |
| eps | 5 | 23 | 60 | 51 | 14 |
| nc  | 5 | 21 | 58 | 52 | 17 |

For eea/ei/eps/nc: only ~5 test rows have zero cross-target info. For egb: 93 test rows have zero (egb has ~150 SMILES that are unique to egb and don't co-appear with the other 4).

## The 5-pack overlap in train

Restricting to the 5 electronic properties {eea, egb, ei, eps, nc}, per-SMILES coverage in train:

| # of these 5 targets labeled | SMILES count |
|-----------------------------:|-------------:|
| 1 | 147 |
| 2 | 102 |
| 3 | 134 |
| 4 |  90 |
| 5 |  25 |

498 unique SMILES have at least one 5-pack label. 115 SMILES have ≥4 of the 5. 25 have all 5.

## PI1M

- 995,799 unique SMILES, all with 2 wildcards (same polymer convention).
- Length: median 44, 90th pct 73. Very similar distribution to train.
- **PI1M ∩ train = 167 SMILES**, **PI1M ∩ test = 116 SMILES**. Small overlap — PI1M is *mostly* new molecules. No test-leak concern.
- Rules allow PI1M use only for pretraining/pseudo-labeling *inside the notebook*. See [02_rules_and_constraints.md](02_rules_and_constraints.md).

## Data quirks / gotchas

1. **Row count discrepancy** — challenge page says 4,497 test rows but actual `test.csv` has 4,940 rows (4,497 unique SMILES; test does repeat SMILES for the same target as well as across targets).
2. **Baseline notebook is stale** — only handles `tg` and `egc`, ignores the other 5 targets.
3. **Element set includes heavy metals (Pb, Sn, Ge, Se, Na)** — a small fraction of RDKit 3D-descriptor calculations will fail or return nonsense on these. Standard practice: replace `±inf`/NaN with the column median before feeding to GBMs.
4. **`tg` has 370 negative values.** Cannot `log1p`. Standard scaling or `identity` only.
5. **Test-target column values `tg` and `egc` are lowercase.** Round 1's playbook uses lowercase too — no rename needed, but must match exactly (`'tg'` not `'Tg'`).

## Immediate takeaways for the plan
- Two-track strategy:
  - **Track A (tg + egc):** classic SMILES → property regression. GBM cocktail + Chemprop. Round 1 recipe applies.
  - **Track B (eea, egb, ei, eps, nc):** matrix-completion / cross-target regressor. Use "other 5-pack values measured on the same molecule" as auxiliary features. SMILES-only is a fallback for the ~5% of rows with no cross-target info.
- Multitask Chemprop is doubly valuable: (a) shared representation for tiny-data targets, (b) can consume cross-target labels as auxiliary inputs at inference.
- PI1M pretraining: SSL objective (e.g. masked-atom, contrastive) is a real lever precisely because the 5 small-data targets have so few labels.
- CV must be **grouped by SMILES** so we don't leak cross-target info across folds.

Detailed plan proposal → [07_plan.md](07_plan.md).
