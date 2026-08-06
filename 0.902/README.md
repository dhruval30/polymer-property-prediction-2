# LB 0.902 — Reproduction

Standalone pipeline that reproduces the **LB 0.902** submission for the ANRF AISEHack 2.0 Round 2 polymer property prediction competition.

## What's in the box

| file | purpose |
|------|---------|
| `reproduce.py` | Single-file end-to-end pipeline (4 phases, ~1500 lines) |
| `README.md` | This file |

## The pipeline (4 phases, LB progression)

| # | phase | model | wall time | LB |
|---|-------|-------|:---------:|:--:|
| 1 | `phase1_lgb_maxwell` | LightGBM per-target (mono FP + aux + Maxwell physics blend) | ~15-20 min | 0.860 solo ref |
| 2 | `phase2_chemprop_3seed` | Chemprop multitask D-MPNN, 5-fold × 3-seed bag | ~3.5-4 h CPU / ~1.5 h GPU | 0.892 solo ref |
| 3 | `phase3_nnls_blend` | 2-way NNLS per-target blend of phases 1+2, with Chemprop weight floor 0.40 + bias +0.15 | <1 sec | 0.897 |
| 4 | `phase4_koopmans_postfit` | Physics rule `Egc ≈ Ei − Eea` as OOF-fit α blend on 3 bandgap targets | <5 sec | **0.902** |

**Total wall time: ~4 h on Mac CPU, ~2 h on Kaggle GPU.** Well within Kaggle notebook's 12h limit.

## Requirements

- **Python 3.11** (required for Chemprop 2.x — Python 3.10 will crash)
- Install:
  ```bash
  pip install numpy pandas scipy scikit-learn tqdm
  pip install rdkit
  pip install lightgbm
  pip install torch lightning chemprop
  ```

## Data setup

`reproduce.py` expects two CSVs in a directory referenced by the `DATA_DIR` variable at the top of the script:

```
<DATA_DIR>/
    train.csv    (columns: id, smiles, target_type, target)
    test.csv     (columns: id, smiles, target_type)
```

**Default `DATA_DIR`** (relative to the script): `../ppp-round-2/`

**On Kaggle**: edit line ~112 to something like:
```python
DATA_DIR = Path("/kaggle/input/anrf-aisehack2-round2/ppp-round-2")
```
Adjust to wherever the host mounts the dataset.

## How to run

### Local

```bash
cd 0.902/
python reproduce.py
```

Outputs land in `./work_0902/`. The final `submission.csv` will be at both:
- `work_0902/submission.csv` (final)
- `work_0902/koopmans/submission.csv` (source, before copy)

### Kaggle notebook

Copy the entire `reproduce.py` into a single notebook cell. Adjust `DATA_DIR` at the top. Add these installs at the top of a preceding cell (if not already in the Kaggle image):

```python
!pip install chemprop lightgbm rdkit -q
```

Then run the cell. Outputs land in `/kaggle/working/` (or wherever `WORK_DIR` points).

Enable GPU accelerator in Kaggle notebook settings and change:
```python
DEVICE = "gpu"    # was "cpu"
```
to cut Phase 2 from ~4h to ~1.5h.

## Directory layout after running

```
work_0902/
├── reproduce.log                       # full training log
├── submission.csv                      # FINAL — this is what you submit
│
├── lgb_maxwell/                        # Phase 1
│   ├── feature_cache.pkl               # ~500 MB — Morgan/MACCS/etc.
│   ├── oof.csv                         # OOF predictions
│   └── submission.csv                  # LGB-only test predictions
│
├── chemprop_3seed/                     # Phase 2
│   ├── checkpoint_fold_0.pkl.gz        # per-fold cached predictions
│   ├── checkpoint_fold_1.pkl.gz        # (resumes across kernel restarts)
│   ├── checkpoint_fold_2.pkl.gz
│   ├── checkpoint_fold_3.pkl.gz
│   ├── checkpoint_fold_4.pkl.gz
│   ├── refit_test_preds.pkl.gz         # 3-seed averaged test predictions
│   ├── oof.csv
│   └── submission.csv
│
├── blend_nnls/                         # Phase 3
│   ├── blend_summary.json              # per-target NNLS weights
│   └── submission.csv                  # LB 0.897 (before Koopmans)
│
└── koopmans/                           # Phase 4
    ├── koopmans_summary.json           # per-target α values + R² deltas
    └── submission.csv                  # LB 0.902 — copy at work_0902/submission.csv
```

## Checkpointing / resume behavior

Each phase checks whether its output files already exist. If so, it skips.

- **Phase 1** (`lgb_maxwell/submission.csv` present): skip entirely.
- **Phase 2** (per-fold `checkpoint_fold_k.pkl.gz` present): skip that fold's training.
- **Phase 2** (also `refit_test_preds.pkl.gz` present): skip the 3 refit models too.
- **Phases 3 & 4**: cheap, always re-run.

**To force a fresh run of a phase**: delete its directory in `work_0902/`.

## Pipeline design in one paragraph

Two proven-strong bases + physics prior for the bandgap targets. LGB captures monomer chemistry via fingerprints + descriptors + matrix-completion aux features + a Maxwell EPS↔Nc physics blend. Chemprop D-MPNN (multitask, 3-seed bag) captures polymer graph structure — same fold split as LGB so OOFs are per-target-alignable. NNLS blend combines them with a **Chemprop weight floor of 0.40 and additive bias of +0.15** — this corrects for the OOF-LB gap disparity between the two bases (Chemprop OOF understates its LB by ~+0.032, LGB with aux OOF overstates LB by ~-0.006). Koopmans post-processor applies `Egc ≈ Ei − Eea` (and its two rearrangements for Ei and Eea) as a per-target linear blend with weights `α ∈ [0.5, 1.0]` grid-searched on Chemprop OOF. Physics term at test uses Chemprop's refit predictions since Chemprop is multitask and produces all 7 target predictions per canon (even those without labels for that target).

## Key hyperparameters (at top of `reproduce.py`)

### LGB (Phase 1)
- `learning_rate=0.03, num_leaves=63, min_child_samples=10`
- `feature_fraction=0.5, bagging_fraction=0.85`
- `reg_lambda=1.0, n_estimators=4000, early_stopping=200`
- `refit_iters = median(best_iters) * 1.10`

### Chemprop (Phase 2)
- `BondMessagePassing(d_h=300, depth=4, dropout=0.05)`
- `MeanAggregation`, `RegressionFFN(hidden=300, layers=2, dropout=0.05)`
- `batch_norm=True`, `max_epochs=60`, `patience=10`, `batch_size=64`
- `lr: 1e-3 (max) → 1e-4 (final)`, `warmup=2`, `grad_clip=1.0`
- Seeds: (42, 43, 44); fold split seed: 42

### Blend (Phase 3)
- `CHEMPROP_WEIGHT_FLOOR = 0.40`
- `APPLY_CHEMPROP_BIAS = 0.15`

### Koopmans (Phase 4)
- Alpha grid: `np.arange(0.5, 1.001, 0.025)` (21 points)
- Physics recipes:
  - `Egc_new = α · Egc_pred + (1-α) · (Ei_pred - Eea_pred)`  (Koopmans)
  - `Ei_new  = α · Ei_pred  + (1-α) · (Egc_pred + Eea_pred)`
  - `Eea_new = α · Eea_pred + (1-α) · (Ei_pred - Egc_pred)`

## Sanity checks after running

The final log block will print:

```
FINAL SUBMISSION: work_0902/submission.csv
Total wall time: X.X min
Expected LB: 0.902
```

Expected per-target OOF R² (Chemprop 3-seed only, before Maxwell / Koopmans):

| target | Chemprop OOF |
|--------|:------------:|
| eea | 0.9082 |
| egb | 0.9305 |
| egc | 0.9070 |
| ei  | 0.7766 |
| eps | 0.7916 |
| nc  | 0.8681 |
| tg  | 0.9083 |
| **MEAN** | **0.8701** |

Expected Koopmans α values (grid-searched on OOF):

| target | best α | ΔR² |
|--------|:------:|:---:|
| egc | 0.900 | +0.0007 |
| ei  | 0.500 | +0.0193 |
| eea | 0.600 | +0.0078 |

If any of these diverges by more than 0.005 from these numbers, something is wrong — check that:
- `DATA_DIR` points to the correct train/test CSVs
- Split seed is 42 everywhere
- Chemprop version is 2.x (2.0.4 or similar; 1.x will fail)

## Known gotchas

- **Chemprop can silently hang on CPU** for hours if the per-epoch logger isn't wired. `EpochLogger` in the script handles this.
- **RDKit 2023+** required for `Chem.MolFromSmiles` behavior on polymer wildcards. Older versions may parse `*` differently.
- **Mac MPS** thermally throttles on long Chemprop runs. `DEVICE = "cpu"` is the reliable local option; use `"gpu"` (CUDA) on Kaggle.
- **Kaggle GPU CUDA mismatch**: some notebook sessions error with `no kernel image is available for execution on the device`. Fall back to `DEVICE = "cpu"` if you hit this.
- **First-run feature cache**: Phase 1 writes ~500 MB to `work_0902/lgb_maxwell/feature_cache.pkl`. Ensure Kaggle's `/kaggle/working/` has space (it does).

## Data reference

- Round 2 training data: 7,405 (canon, target_type, target) rows across 5,920 unique canonical SMILES. Per-target counts: `tg=4139, egc=2028, egb=337, eps=229, nc=229, ei=222, eea=221`.
- Test data: 4,940 (id, canon, target_type) rows across 4,133 unique canonical SMILES.
- Metric: mean R² across the 7 target types.

## Contact / issues

For questions on the pipeline design or gotchas, see the full experiment history in `docs/best-experiment.md` and `docs/best-ensemble.md` in the parent repo.
