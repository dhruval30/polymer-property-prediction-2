# Leaderboard Snapshot — 2026-08-01

Captured from `challenge-description/leaderboard-status.png`. Public LB only (private LB is hidden until close).

## Top 27
| Rank | Team | Score | Members | Submissions | Last submitted |
|------|------|-------|---------|-------------|----------------|
| 1  | Kuch bhi Karna hai       | 0.899 | 1  | 11 | 4h |
| 2  | Opus 6.7                 | 0.898 | 1  | 5  | 13h |
| 3  | ShiokParikh08            | 0.893 | 1  | 3  | 13h |
| 4  | TV0lEy                   | 0.893 | 1  | 5  | 14h |
| 5  | The Invincibles          | 0.891 | 1  | 3  | 1h |
| 6  | Rish202410159            | 0.889 | 1  | 6  | 1h |
| 7  | Runtime Rebel's          | 0.881 | 1  | 8  | 1d |
| 8  | VibeCoders               | 0.880 | 2  | 5  | 2d |
| 9  | Prime Polymers           | 0.878 | 1  | 16 | 2h |
| 10 | Coding Brigades          | 0.876 | 1  | 6  | 2h |
| 11 | The Debuggers            | 0.875 | 1  | 5  | 1d |
| 12 | Aniruddha Shinde         | 0.875 | 1  | 2  | 2d |
| 13 | Cross Linkers            | 0.874 | 3  | 5  | 14h |
| 14 | Bond                     | 0.872 | 1  | 3  | 1d |
| 15 | The Team                 | 0.867 | 1  | 10 | 15h |
| 16 | 1nf1n1ty                 | 0.865 | 1  | 6  | 6h |
| 17 | Susegad Coderz           | 0.863 | 1  | 11 | 2h |
| 18 | The Polymaths            | 0.859 | 1  | 7  | 13h |
| 19 | Lakshya                  | 0.849 | 1  | 1  | 1d |
| 20 | ESPRIT                   | 0.847 | 1  | 7  | 12h |
| 21 | Epitome                  | 0.843 | 1  | 2  | 1d |
| 22 | Hackaholics              | 0.841 | 1  | 4  | 3h |
| 23 | Synapse Syndicate        | 0.840 | 1  | 5  | 1d |
| 24 | PS Square                | 0.835 | 1  | 4  | 1d |
| 25 | InfinityLoop             | 0.834 | 1  | 5  | 1d |
| 26 | Team XD                  | 0.834 | 1  | 5  | 1d |
| 27 | Vikas_23f1001674         | 0.822 | 1  | 6  | 5h |

## Reading the leaderboard
- **Top of pack is tight:** rank 1 (0.899) → rank 15 (0.867) spans only 0.032. Any of them could flip on private LB.
- **Number-1 has 11 submissions** and the last submission was 4h ago — they're still actively iterating. Some top teams have shipped 5 or fewer submissions and haven't touched it in a day, suggesting they hit a plateau.
- **Round 1 comparison:** Dhruval finished Round 1 at 0.911 (rank 15). Round 2 top scores are lower absolute numbers, but that's expected — R² averaged over 7 targets (some of which are harder than others) is a tougher metric than 2-target Round 1.
- **Prime Polymers (rank 9, 16 subs) and The Team (rank 15, 10 subs)** are the teams grinding submissions hardest without breaking into the top 5 — signals a ceiling that most conventional stacks hit around 0.87–0.88.

## Where the gains probably are
- Break above ~0.89 likely requires either (a) a better message-passing / transformer model pretrained on PI1M, (b) a per-target modeling strategy that beats the "one model on the full long table" default, or (c) careful attention to whichever of the 7 targets is dragging the mean (identify in EDA).
