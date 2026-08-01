# Challenge Overview — ANRF AISEHack 2.0, Theme 2, Round 2

**Competition:** Polymer Property Prediction (Round 2)
**Platform:** Kaggle — https://www.kaggle.com/competitions/aisehack-2-0/
**Host:** VIJITH P (ANRF)
**Prizes:** Kudos (no monetary prize)
**Points/Medals:** Does not award points or medals
**Team size limit:** 5

## Timeline
- **Start:** 29 June 2026
- **Final Submission Deadline:** XX July 2026 (~11 days out from 2026-08-01; screenshot shows "Close 11 days ago" so may be actively closing)
- **Status snapshot on 2026-08-01:** 154 entrants, 95 participants, 36 teams, 179 submissions

## Round context
- This is **Round 2**. Round 1 had two targets and forbade external data.
- Round 2 raises the bar: **7 targets** simultaneously, and PI1M auxiliary data is *permitted* (see [02_rules_and_constraints.md](02_rules_and_constraints.md)).
- Hosts describe the motivation as building "robust and generalizable models capable of accurately predicting multiple polymer properties from molecular structure."

## Problem statement
Given a polymer's **SMILES string** and a **target_type** label indicating *which* property to predict, output a single numeric value for that property.

The data is in **long form** — one row per (polymer, target_type) pair, not one row per polymer with 7 columns. See [03_data_description.md](03_data_description.md).

## The seven properties
| Symbol | Property | Description |
|--------|----------|-------------|
| Egc  | Chain Bandgap | Electronic bandgap of an isolated polymer chain. |
| Egb  | Bulk Bandgap | Electronic bandgap of the polymer in the bulk phase. |
| Ei   | Ionisation Energy | Energy required to remove an electron from the polymer. |
| Eea  | Electron Affinity | Energy released when the polymer accepts an electron. |
| EPS  | Dielectric Constant | Ability of the polymer to store electrical energy in an electric field. |
| Nc   | Refractive Index | Optical property describing the interaction of light with the polymer. |
| Tg   | Glass Transition Temperature | Temperature at which the polymer transitions from a glassy to a rubbery state. |

## Evaluation metric
Mean R² across the seven targets:

```
Score = ( R²_Tg + R²_Egc + R²_Egb + R²_Ei + R²_Eea + R²_Nc + R²_EPS ) / 7
```

with the standard R² definition:

```
R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
```

Higher is better; ceiling is 1.0. Each per-target R² is computed on that target's slice of the test set, then the seven are averaged. Because R² is scale-invariant per target, dominant-scale targets (Tg is much larger in magnitude than Egc etc.) cannot swamp the score. That means **weak-performing targets drag the mean disproportionately** — a 0.6 R² on one target pulls the mean down by ~0.06.

## Public vs Private leaderboard
- **Public LB:** scored on a subset of test.
- **Private LB:** hidden test set — determines final rank.
- Both use the same mean-R² formula.

## Current public LB snapshot (2026-08-01)
Top 5 shown from the leaderboard screenshot:
1. Kuch bhi Karna hai — 0.899
2. Opus 6.7 — 0.898
3. ShiokParikh08 — 0.893
4. TV0lEy — 0.893
5. The Invincibles — 0.891

Cluster is tight at the top (top 15 all ≥ 0.87). Non-trivial gains will require targeted per-target work, not another cocktail of the same base learners.

## Related docs
- [01_submission_and_reproducibility.md](01_submission_and_reproducibility.md)
- [02_rules_and_constraints.md](02_rules_and_constraints.md)
- [03_data_description.md](03_data_description.md)
- [04_baseline.md](04_baseline.md)
- [05_leaderboard.md](05_leaderboard.md)
