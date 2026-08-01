# Host Baseline

File: `ppp-round-2/archive/base_line_model.ipynb`.

## What it actually does (source-verified)
Not what the dataset description implies. The notebook is **a Round 1 leftover** and only handles two of the seven targets — `tg` and `egc`. The other five targets (`egb`, `ei`, `eea`, `eps`, `nc`) are **ignored entirely**: no features, no model, no predictions in `submission.csv`.

Pipeline for the two targets it does handle:
1. **Split by target_type** into `train_tg`, `train_egc`, `test_tg`, `test_egc`.
2. **Featurize each subset independently** using RDKit's `Descriptors.CalcMolDescriptors` (the ~210 default 2D descriptors), one row per SMILES.
3. **Clean:** replace ±inf with NaN, impute NaN with column mean.
4. **Fit Ridge per target:** grid over 20 alpha values on `np.logspace(-10, 1, 20)`, 5-fold `KFold(shuffle=True, random_state=42)`, `StandardScaler` refit inside each fold, score by RMSE. Pick alpha with lowest CV RMSE.
5. **Feature selection:** take the top 20 features by |coefficient|, retrain Ridge on that subset with the same CV-alpha procedure.
6. **Predict** on the test subset for that target, write `submission.csv` from `pd.concat([test_tg, test_egc])` with columns `id, target`.

## Score
- Not computed inside the notebook (the CV RMSE is per-target and never converted to the mean-R² metric).
- Because the notebook only fills predictions for `tg` and `egc` rows, the other 5 target_types would be **missing from `submission.csv`** if run as-is on the current test set. That either errors out at scoring or gives NaN R² on the missing targets — either way, effectively a non-starter as a real submission for Round 2.

## Useful ideas to lift from it
- The **RDKit descriptor featurizer function** (`calculate_descriptors`) is a clean starting point for the descriptor family in our feature stack.
- The **inf → NaN → column-mean impute** pattern is worth keeping; RDKit descriptors can return ±inf on certain molecules.
- The **top-20-feature reselection** trick is not worth copying — LightGBM/CatBoost handle feature importance internally, and Ridge on 20 features caps expressiveness.

## Ideas to explicitly NOT copy
- ❌ Per-target `StandardScaler` refit + Ridge — Ridge on raw descriptors is way underpowered vs GBM cocktails.
- ❌ Splitting the data 7 ways and training 7 independent models with no shared representation — the whole point of joint / multitask training (Chemprop) is that one target's labels improve another target's representation.
- ❌ Scoring in RMSE — the actual metric is per-target R², averaged.

## Sample submission — target scales
From `ppp-round-2/archive/sample_submission.csv`:

```
id, target
1, 273.5     ← Tg-scale
2, 195.0     ← Tg-scale
3,  44.0
4,  45.0
5,  67.0
6,   1.9942  ← refractive-index (Nc) scale
7,   5.9072  ← eV-scale (Ei / Egb / Egc)
8, -32.0
9, 158.17
10, 260.0
```

Confirms the huge dynamic-range spread across targets, hence the mean-R² metric normalizing per target.
