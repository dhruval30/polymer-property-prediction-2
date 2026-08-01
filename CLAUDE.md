# CLAUDE.md — Polymer Property Prediction (Round 2)

Context for Claude when working with Dhruval on this Kaggle competition.

---

## Who I am

- Dhruval Padia. ML engineer building competition pipelines.
- Made it to Round 2 of ANRF AISEHack 2.0 (finished Round 1 at rank 15 with LB 0.911).
- Comfortable with sklearn, LightGBM, CatBoost, PyTorch, Chemprop, RDKit.
- I move fast and iterate — I care about *shipping working pipelines*, not perfect abstractions.

## How to talk to me

- **Terse. Direct. No fluff.** Skip preambles ("Let me...", "I'll now..."), skip trailing summaries. If I asked a yes/no, answer yes or no.
- Casual tone is fine — I say "bruh", "cool", "lets just do X". Match that register.
- When I ask an exploratory question ("what could we try?"), give 2–3 sentences with the main tradeoff, not a plan. Don't implement until I say go.
- If I'm about to do something dumb, tell me. Don't hedge.
- Give me score expectations when I ask. Be honest about ceilings. If a lever is unlikely to move the needle, say so before we spend hours on it.

## How I want code

- **No comments unless the WHY is non-obvious.** Well-named identifiers explain the WHAT.
- No docstrings unless it's a public API. Never multi-paragraph.
- No error handling for scenarios that can't happen. No defensive validation of internal code.
- No premature abstraction. Three similar lines beats a helper I use once.
- No half-finished implementations. Either do it or don't.
- Prefer editing existing files over creating new ones.
- Never create `*.md` files unless I ask.

## Project structure I like

```
experiments/
  exp_<descriptive_name>.py    # ONE self-contained standalone script per experiment
results/
  exp_<descriptive_name>/
    run.log                     # logs (file + stdout)
    oof.csv                     # OOF predictions
    submission.csv              # test predictions
    cv_summary.json             # metrics
    checkpoint.pkl.gz           # for resumable long runs
data/                            # symlinked or contains train.csv / test.csv
```

- **No shared `_utils.py`.** Every experiment script is self-contained — all imports, constants, helpers, featurization, CV, training, and output live in the one file. Yes this means duplication across scripts. Yes that's the point: I can look at one file six months later and reproduce it without archaeology through utils commits. If a common recipe stabilizes, I'll copy-paste it into the next experiment, not import it.
- Experiment names are descriptive (`exp_polymer_physics_stack.py`, not `exp1.py`).
- Feature caching lives under `results/_feature_cache/` keyed by a content hash — the caching code is inlined per script.
- Every script uses `tqdm` for visible progress on any loop that takes more than a couple seconds (featurization, per-fold training, etc.).
- Every experiment writes `run.log` (via python `logging` to both file + stdout) and a `cv_summary.json` I can grep later.
- Long runs must checkpoint per phase so I can Ctrl+C and resume.

## Commit style

- Format: `<type>: <lowercase description>`
- Types I use: `feat`, `fix`, `doc`, `refactor`, or a bare short label like `physics-stack + from-scratch pipeline experiments`.
- Keep the subject under ~70 chars. Body only if the "why" isn't obvious.
- Never attribute claude and NEVER end with:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- Use HEREDOC to pass the message (never `git commit -m "one line"` for multi-line).
- Only commit when I ask. Don't self-initiate.

## Environment

- Mac M-series (MPS available but throttles on long runs — treat CPU as the reliable local option).
- Python 3.11 required for Chemprop 2.x (3.10 breaks it).
- Local `.venv` in the repo. If deps missing, install into that venv, not global.
- Kaggle notebook runtime for final submission: has GPU, but **Kaggle GPUs sometimes throw `CUDA error: no kernel image is available for execution on the device`** — fall back to CPU-only if that happens.

## Round 1 playbook — what worked (LB 0.911, rank 5→15)

Round 1 was two targets (`tg`, `egc`), no external data / no pretrained models allowed. Round 2 is different (see data), but the *methodology* transferred, so bring it in as a starting point unless the new setup contradicts it.

### Winning recipe (in order of impact)

1. **Two-stage stack, per-target NNLS blend**
   - Phase 1: GBM cocktail (LightGBM + CatBoost + HistGradientBoosting), mean-blended.
   - Phase 2: Chemprop D-MPNN multitask (shared graph rep, one head per target). 5-fold × 3-seed bag = 15 models.
   - Blend: per-target NNLS over Phase 1 OOF + Chemprop OOF, weights normalized to sum-to-1, applied to test.

2. **Feature families for GBM** (all concatenated, ~11K features)
   - RDKit 2D descriptors (~210)
   - Morgan-r2 count FP (2048)
   - Morgan-r3 count FP (2048)
   - MACCS keys (167)
   - Avalon (512)
   - Atom-Pair count FP (2048)
   - Topological-Torsion count FP (2048)

3. **Target transforms per target**
   - `log1p` on skewed / non-negative targets (Egc)
   - `identity` on symmetric ones (Tg)
   - Always invert at prediction time.

4. **CV: 5-fold stratified quantile split** (10 quantile bins on the transformed target, `StratifiedKFold`). Same folds across all base models — critical for honest OOF stacking.

5. **Chemprop config that worked**
   - `BondMessagePassing(d_h=300, depth=4, dropout=0.05)`
   - `MeanAggregation`
   - `RegressionFFN(hidden_dim=300, n_layers=2, dropout=0.05)`
   - `batch_norm=True`, `max_epochs=50`, `patience=10`, `batch_size=64`, `gradient_clip_val=1.0`
   - Standardize targets per-fold (train-mean/std), un-standardize at predict.
   - Multitask joint head — molecules labeled for one target still improve the shared representation.

6. **GBM hyperparams that worked**
   - LGB: `n_est=4000, lr=0.03, num_leaves=63, min_child_samples=10, feature_fraction=0.5, bagging_fraction=0.85, reg_lambda=1.0` + early stopping 200
   - CAT: `iterations=4000, depth=8, lr=0.03, l2_leaf_reg=3.0` + early stopping 200
   - HGB: `max_iter=1000, lr=0.05, max_leaf_nodes=63, min_samples_leaf=20, l2_reg=1.0` + built-in early stop
   - Refit on full data at `median(best_iters) * 1.10` for test predictions.

7. **LB probe: submit `train.mean()` and `train.mean() + 30` to detect distribution shift.** Round 1 confirmed no shift (probe score matched the no-shift math exactly). Do this early — it's one submission slot to eliminate a whole hypothesis class.

### What did NOT work (skip these unless you have a specific reason)

- **3D physics features** — multi-conformer `Descriptors3D` aggregates + Coulomb-matrix eigenvalues *regressed* Phase 1 OOF (Tg 0.9017 → 0.8994). GBMs were feature-saturated.
- **Gasteiger partial charges on polymer SMILES** — wildcard atoms (`*`) return NaN unless capped with carbon first. Even after fixing, no measurable lift.
- **Iterative pseudo-labeling (2 rounds)** — added noise, no gain on the leaderboard.
- **CatBoost meta-stacker with scaffold ID** — overfit vs simple per-target NNLS.
- **Dual-variant Chemprop (Mean+depth2 vs Sum+depth3)** — only marginal (+0.003), not worth 2× compute unless we're already at the ceiling.

### Gotchas that burned hours in Round 1

- **Chemprop hung silently on CPU for 9+ hours** because it has no per-epoch logging by default. Always wire an `EpochLogger` callback or you can't tell if it's alive.
- **Mac MPS degrades on long Chemprop runs** — fold 1 took 1.7h, fold 5 took 8h+. Either use CPU (slow but stable) or a real CUDA GPU.
- **Kaggle CUDA kernel mismatch** — on some Kaggle GPU sessions, PyTorch throws `no kernel image is available for execution on the device`. Have a CPU fallback path in the notebook.
- **Chemprop 2.x + Python 3.10 = crash.** Force 3.11.
- **Feature cache invalidation** — if the featurizer code changes, the cache key must change. Hash the *code that generates the features*, not just the SMILES list.

## How I want you to work

- **Plan before coding for non-trivial work.** Show me the approach (which base learners, which features, which CV) before running anything long.
- **Ask me before starting anything that takes >30 min of wall time.** I want to know the expected score lift.
- **When exploring, use the `Explore` / `Plan` agents. When executing, do it inline.**
- **Multiple independent tool calls → parallel in one message.** Don't serialize what can go in parallel.
- **Track progress with TaskCreate for multi-step work.** Mark complete as you go, not in batches.
- **If a run is going to take hours, tell me first so I can decide whether to Ctrl+C it early.**
- Don't reuse Round 1 checkpoints in Round 2 — the data is different, everything trains from scratch here.

## Data note (Round 2)

- `ppp-round-2/train.csv`, `ppp-round-2/test.csv`, plus `ppp-round-2/PI1M.csv` (~995K unlabeled polymer SMILES from PI1M — check the rules for whether it's allowed as auxiliary data; Round 1 forbade external data but Round 2 shipping this file suggests it's fair game).
- Multiple target types this round (not just tg/egc). Do the EDA yourself, then plan.
- Baseline notebook exists at `ppp-round-2/archive/base_line_model.ipynb` — read it before designing anything, it tells you what the hosts consider a "reasonable" starting point.
- Rules and challenge screenshots are in `challenge-description/`. Read those first — rule violations disqualify.
