# ANRF AISEHack 2.0 Round 2 — polymer property prediction

Final public LB: **0.904**. Two submissions locked as final; both under `final_submissions/`.

## Final submissions

- **`LB0904_chemprop_multimer_aug_bag_and_lgb_maxwell_and_koopmans_postfit.py`** — primary, LB 0.904
- **`LB0901_chemprop_7seed_bag_and_lgb_maxwell_and_koopmans_postfit.py`** — backup, LB 0.901 (7-seed Chemprop instead of 3, picked for private-LB robustness — variance reduction over the base)

Each script is single-file, self-contained, Kaggle-notebook-runnable. Auto-detects data path. ~6-7h wall on Kaggle T4 GPU for the 0.904 script.

## Pipeline (the 0.904 recipe)

Standard 2-way blend + physics-based post-processing. Nothing exotic — the win came from a polymer-specific augmentation, not architectural pivots.

1. **LGB per-target** — mono fingerprints (Morgan-r2, Morgan-r3, MACCS, Atom-Pair, Top-Torsion, Avalon = ~9K features) + RDKit 2D descriptors + 14 aux matrix-completion features (mean of other targets per canonical SMILES). 5-fold GroupKFold by canon, seed 42.
2. **Maxwell EPS↔Nc post-fit on LGB** — used the empirical Maxwell relation `EPS ≈ n²` as an OOF-tuned blend prior on LGB's EPS and Nc predictions.
3. **Chemprop 3-seed multitask D-MPNN** — shared graph encoder, 7 regression heads, `d_h=300, depth=4, dropout=0.05`. Multitask so molecules labeled for one target still improve the shared representation.
4. **Repeat-unit augmentation on Chemprop** — for each polymer, training set includes monomer, dimer, and trimer graphs as separate datapoints with the same target. Val/test stay canonical. **This was the lever that took us from 0.902 to 0.904.** Peer-reviewed for polymers ([arXiv 2505.10726](https://arxiv.org/abs/2505.10726), "Learning Repetition-Invariant Representations for Polymer Informatics").
5. **2-way NNLS blend** — per-target non-negative-least-squares of Chemprop + LGB-Maxwell. Chemprop weight floored at 0.40 with +0.15 bias, calibrated to the observed Chemprop OOF-LB gap.
6. **Koopmans bandgap post-fit** — physics rule `Egc ≈ Ei − Eea` (and rearrangements) applied as an OOF-tuned α blend on the 3 bandgap targets. Final α values: `α_egc=0.9, α_ei=0.5, α_eea=0.6`.

## What didn't work (honest list)

Tried a lot. Most of it regressed on LB despite positive OOF signal:

- **PI1M pseudo-labeling** — proper per-fold LGB teacher pilots + confidence/Tanimoto filters. Ran out of time on Kaggle (12h notebook timeout). Not in final subs.
- **3-way NNLS blends** — added CatBoost (chain-ext features), PolyMetriX (polymer topology descriptors), extra LGB variants as third bases. Even with 20-30% NNLS weight on the new base, LB never beat the 2-way blend. Consistent pattern: OOF gains up to +0.030 did not translate.
- **SMILES atom-order augmentation** on Chemprop — D-MPNN is permutation-invariant on the graph, so different SMILES orderings produce identical inputs. Confirmed empirically (LB 0.899).
- **Post-hoc Koopmans variants** — Egb 3-way, Moss rule for Nc, multi-signal NNLS, α refit on blend OOF — all flat or slight regression.
- **Isotonic calibration** — in-sample R² gain of +0.16 was pure overfitting. Honest nested CV showed −0.05.
- **Chain-length aug extended to tetramer [1,2,3,4]** — mechanism saturated at trimer; adding tetramer regressed vs [1,2,3].

## Gotchas we hit

- **Chemprop 2.x + Kaggle P100** throws `no kernel image is available for execution on the device` — P100 is sm_60, current PyTorch dropped support. Switch to T4 x2 accelerator (sm_75). Works fine.
- **Chemprop silently hangs on CPU** without per-epoch logging. Wired an `EpochLogger` callback.
- **LGB on augmented data** rarely early-stops (data too big) — augmented pipelines hit the 4000-iteration ceiling frequently, which was the root cause of our PI1M pipeline timing out.

## Rules compliance

- **No external data used** in the final submissions. PI1M was in the provided dataset but the pseudo-labeling attempt didn't complete.
- **No pretrained model weights.** Chemprop trained from scratch on the 5920 labeled polymers.
- **5-fold GroupKFold by canonical SMILES** (seed 42) throughout — no target leakage across folds.
- **Feature preprocessing statistics** (quantile clipping, median imputation) computed on train+test SMILES combined — labels never touched.

## Reproducing

```
Cell 1: !pip install -q rdkit lightgbm chemprop lightning
Cell 2: paste one of the scripts from final_submissions/, run.
```

Final submission writes to `/kaggle/working/submission.csv`. Wall time ~6-7h for 0.904, ~4h for 0.901 (both on T4 GPU).

---

Dhruval Padia
