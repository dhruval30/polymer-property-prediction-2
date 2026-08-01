# Rules & Constraints — What's Allowed and What Gets You DQ'd

Source: `challenge-description/rules.txt` (Sections 6 and 7) plus the "Requirements" screenshot.

## Bottom line
- **PI1M.csv (shipped in the competition data section) is ALLOWED as auxiliary data.**
- **All other external data is BANNED.**
- **All pretrained model weights are BANNED** (whether you download them at runtime or upload them as a dataset).
- **All uploaded artifacts (embeddings, feature caches, checkpoints, processed datasets) are BANNED.**
- The full pipeline — including any PI1M pretraining — must **execute inside the Kaggle notebook during a single run**.

Violating any of the above is **immediate disqualification**, regardless of leaderboard position.

## Section 6.2.1 — No External Data
> Participants must use only the official Competition Data.

Strictly prohibited:
- Use of any external, private, or previously prepared datasets (public or private).
- Attaching or accessing any external dataset within the submission notebook.
- Using data generated or collected outside notebook execution.

Requirements-tab screenshot clarifies the auxiliary-data carve-out:
> For this competition publicly available external data is not allowed, including pre-trained models. Instead participants can use the auxiliary data [that] is provided in the data section.

So **PI1M.csv** — provided in the data section — counts as *auxiliary data*, not *external data*. Anything else you'd normally reach for (ChEMBL, PubChem, QM9, PolyInfo, another polymer SMILES dump) is disqualifying.

## Section 6.2.2 — Notebook / Code-Only
This is a notebook/code-only competition. All submissions must be generated **entirely within a Kaggle Notebook** in a single run.

All of these must happen inside that one notebook execution:
- Data loading and preprocessing
- Train/val set preparation
- Model definition and initialization
- Training or fine-tuning
- Inference on test inputs
- Submission file generation

**Manual intervention at any stage is not permitted.**

## Section 6.2.3 — Public Code Usage
Publicly available code repos are OK (e.g. importing model architectures from a linked GitHub repo during execution, or the way scripts are used in the baseline notebook), **provided that**:
- No external data is used.
- No pretrained weights, checkpoints, embeddings, or processed artifacts are uploaded.
- The entire pipeline executes reproducibly within the Kaggle notebook environment.
- All training outputs are generated during notebook execution.

So we can `pip install chemprop`, `pip install rdkit`, import Chemprop's architecture, `git clone` a public repo of an untrained polymer model architecture — but we cannot download or ship pretrained weights.

## Section 6.2.4 — Uploaded Artifacts (PROHIBITED)
Strictly prohibited:
- Uploading pretrained model weights or checkpoints.
- Uploading embeddings, feature files, cached tensors, or processed datasets.
- Attaching any dataset or model artifact created outside notebook execution.
- Linking or using any private artifact or external file.

> All model weights and artifacts must be produced during the notebook run.

**What this rules out from the Round 1 playbook:**
- ❌ Local feature caches (`results/_feature_cache/`) uploaded as a Kaggle dataset.
- ❌ Locally-trained checkpoint files uploaded and loaded in the notebook.
- ❌ Any downloaded pretrained model (ChemBERTa, MolBERT, Uni-Mol, etc.).

**What is still fine:**
- ✅ Building the feature cache inside the notebook run.
- ✅ Training GBM / Chemprop / any model from scratch inside the notebook.
- ✅ Doing PI1M-based *self-supervised pretraining* inside the notebook, then fine-tuning on train.csv.

## Section 6.2.5 — Organizer Audit Rights
Hosts reserve the right to inspect:
- Attached datasets and files.
- Notebook code, outputs, and execution logs.
- Model loading paths and download sources.
- Training and inference procedures.

You may be required to explain any part of your pipeline. **Assume they will look.**

## Section 6.2.6 — Violations
Any external data or prohibited artifact usage = **immediate DQ**, regardless of leaderboard standing.

## Section 7 — Auditability and Reproducibility
Detailed in [01_submission_and_reproducibility.md](01_submission_and_reproducibility.md). Highlights:
- Submission description must link to the notebook.
- Pinned notebook version must match the submitted results exactly.
- Notebook shared with all named hosts.
- Post-competition rerun by hosts must reproduce the score.

## Practical do / don't list

**Do:**
- Train everything from scratch inside the notebook.
- Use PI1M for pretraining or pseudo-labeling — it's the *only* auxiliary source allowed.
- Set every random seed and document them.
- Keep total pipeline wall time within Kaggle's compute limit for a single run.

**Don't:**
- Upload feature caches or precomputed descriptors as a Kaggle dataset.
- Load any pretrained checkpoint (RDKit descriptor computation is *not* a pretrained model — that's fine).
- Bring in QM9, ChEMBL, PolyInfo, or any other polymer/chemistry corpus.
- Split the pipeline into "train locally, upload weights, inference in Kaggle notebook."
