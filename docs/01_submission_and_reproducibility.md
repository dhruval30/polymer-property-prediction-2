# Submission Format, Auditability & Reproducibility

## Submission file
- Name **must be** `submission.csv`.
- Columns: `id, target`.
- One row per `id` in the test set — each `id` is already tagged with a `target_type` in `test.csv`; you only output the predicted numeric `target` for that id.
- Header required.

Example (from the challenge description):

| id | target |
|----|--------|
| 1  | 220    |
| 2  | 2.3    |
| 3  | 110    |
| 4  | 70     |

## Submission limits
- **3 submissions per day.**
- **2 final submissions** may be selected for private-LB judging.

## Notebook-backed requirement (mandatory)
Every submission must be **backed by a Kaggle Notebook**:
- Submission description must include a link to the notebook that generated it.
- The **default/pinned version** of that notebook must exactly match what produced the submitted score.
- You may make new versions later, but the pinned version must remain the score-producing one.
- Notebook may be private, but must be **shared with view access** with all hosts:
  - Rohit Batra IITM
  - Rahulsundar
  - LaksmanN
  - VIJITH P
  - shreyasri0301
- Submissions without a linked notebook, without correct sharing, or with a mismatched pinned version are **invalidated**.

## End-to-end execution requirement
All ML pipeline stages must execute within a **single Kaggle notebook run**:
- Data loading + preprocessing
- Train/val split
- Model definition + initialization
- Training / fine-tuning
- Inference on the test set
- Submission file generation

**Manual intervention at any stage is not permitted.**

## Post-competition reproducibility validation
After the competition closes, hosts will re-execute the pinned notebook version of each finalist's selected best submission. To remain eligible:
- Notebook must run end-to-end **without manual intervention**.
- Execution must complete within Kaggle's compute + time limits.
- The reproduced results must **match the submitted results**.

If the pinned version does not reproduce the submitted score, the submission is **invalidated regardless of leaderboard position**.

Participants **must**:
- Explicitly set and document all random seeds.
- Pin the correct notebook version to the submission.

## Practical implications for our pipeline
- **Everything must fit in one Kaggle notebook run.** No feature-cache uploads, no pre-trained checkpoint uploads, no manually-computed embeddings brought in as datasets.
- Total wall-clock budget = Kaggle's compute limit (typically 9h GPU or 12h CPU per notebook, but confirm on the competition compute tab).
- Seed everything: numpy, torch, lightgbm, catboost, sklearn splits.
- If we use PI1M pretraining, that pretraining must also happen inside the notebook — see [02_rules_and_constraints.md](02_rules_and_constraints.md) for the full rules.
