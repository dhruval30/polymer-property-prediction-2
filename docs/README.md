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
