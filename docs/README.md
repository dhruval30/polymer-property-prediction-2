# docs/ — Polymer Property Prediction, Round 2

Digitized challenge context, rules, and analysis notes. All content here is derived from `challenge-description/` (rules.txt + screenshots) and direct inspection of `ppp-round-2/`.

| # | File | Contents |
|---|------|----------|
| 00 | [challenge_overview.md](00_challenge_overview.md) | Competition summary, 7 targets, metric, timeline, top-line status. |
| 01 | [submission_and_reproducibility.md](01_submission_and_reproducibility.md) | Submission format, notebook-backed requirement, reproducibility validation. |
| 02 | [rules_and_constraints.md](02_rules_and_constraints.md) | Full external-data / pretrained-weights / uploaded-artifact rules. PI1M carve-out. |
| 03 | [data_description.md](03_data_description.md) | train.csv / test.csv / PI1M.csv schemas, row counts, quirks to check. |
| 04 | [baseline.md](04_baseline.md) | Host-shipped Ridge baseline. |
| 05 | [leaderboard.md](05_leaderboard.md) | 2026-08-01 public LB snapshot with commentary. |

| 06 | [eda_findings.md](06_eda_findings.md) | Per-target counts, distributions, SMILES stats, train/test overlap, duplicates, 5-pack matrix. |
| 07 | [plan.md](07_plan.md) | Proposed pipeline — Track A (tg/egc SMILES→property), Track B (5-pack matrix completion), Chemprop multitask, PI1M SSL, score expectation. |
| 08 | [eda_deep.md](08_eda_deep.md) | Deep EDA — chemistry per target, scaffolds, Tanimoto NN, 5-pack cross-target correlations, polymer class effects, PI1M distribution, CV viability, updated modeling implications + score expectations. |
| 09 | [data_exploration.md](09_data_exploration.md) | Massive edition — 17 sections. Row-level test accounting, per-target dives, descriptor→target correlations, Morgan bit analysis, UMAP, dedup rungs, rare atoms/charges, signal-to-noise ceilings, PI1M target slicing, Ridge floor baseline, feature engineering catalog, data quality risk register, splitting scheme decision. |
| — | [best-experiment.md](best-experiment.md) | **Living tracker** of the current best LB submission + full submission history. Updated every time a new experiment beats the previous best. |
