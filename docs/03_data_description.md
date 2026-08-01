# Data Description

Files live in `ppp-round-2/`.

## Files shipped
| File | Size | Purpose |
|------|------|---------|
| `train.csv` | ~444 KB | Labeled training set — 7,409 rows across 7 target types. |
| `test.csv` | ~284 KB | Test set — 4,497 rows. Each row is tagged with the `target_type` you must predict. |
| `PI1M.csv` | ~47.6 MB | ~995K unlabeled polymer SMILES. Explicitly permitted as **auxiliary data**. |
| `archive/base_line_model.ipynb` | — | Host-shipped Ridge-regression baseline. See [04_baseline.md](04_baseline.md). |
| `archive/sample_submission.csv` | — | Illustrates the required submission format. |
| `archive/{train,test}.csv` | — | Older copies of the training/test data. Not the ones to use. |

## `train.csv`
Long-form. **7,409 rows total**, not 7,409 polymers × 7 columns.

| Column | Description |
|--------|-------------|
| `smiles` | SMILES representation of the polymer structure (polymer SMILES with wildcard `*` atoms marking the repeat unit connection points, per Round 1 convention). |
| `target` | Experimental numeric value of one of the seven polymer properties. |
| `target_type` | Category label indicating which of the seven properties this row's `target` refers to. Values ∈ {Egc, Egb, Ei, Eea, EPS, Nc, Tg}. |

Because the format is long, one polymer SMILES may appear in multiple rows (once per property it was measured for). Duplicate `(smiles, target_type)` pairs are worth checking during EDA.

## `test.csv`
**4,497 rows.** Long-form.

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier (integer). Used as the submission key. |
| `smiles` | SMILES representation. |
| `target_type` | Which of the seven properties to predict for this row. |

## `PI1M.csv`
**~995K rows** of unlabeled polymer SMILES from the PI1M database.

| Column | Description |
|--------|-------------|
| `SMILES` | Polymer SMILES string (note the capitalized column name — will need normalization). |

Permitted uses (see [02_rules_and_constraints.md](02_rules_and_constraints.md)):
- Self-supervised pretraining of a graph or transformer encoder inside the Kaggle notebook.
- Pseudo-labeling via a teacher trained on `train.csv`.
- Distribution-aware augmentation / mixing.

Not permitted: using it as if it were the *test* set, or bringing in labels that don't exist in it.

## Sample submission
`archive/sample_submission.csv` (91 bytes) shows the required 2-column `id,target` layout. Full submission format spec in [01_submission_and_reproducibility.md](01_submission_and_reproducibility.md).

## Notes / open questions to resolve in EDA
- Distribution of rows per `target_type` in train and test — are all 7 targets equally represented, or is one target much rarer (which would bottleneck the mean-R² score)?
- Per-target value distributions — which need log/sqrt transforms, which are symmetric?
- SMILES token quirks — wildcard placement, unusual atoms, ring counts, aromatic-vs-Kekulé.
- Overlap between `train.smiles` and `test.smiles` (same polymer, different property being asked).
- Are there duplicate `(smiles, target_type)` pairs with conflicting `target` values?
- Basic PI1M sanity — is it disjoint from train/test SMILES?

All of these are addressed in `docs/06_eda_findings.md` (to be written after the EDA task runs).
