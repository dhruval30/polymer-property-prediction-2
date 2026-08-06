"""
exp_kernel_ridge_tanimoto.py — Kernel Ridge Regression with Tanimoto kernel
                              on chain-ext Morgan-r2 bit fingerprints.

============================================================================
WHY THIS EXISTS
============================================================================

Research-v2 §7 (the Fable follow-up to our session handoff) identifies
kernel-based methods with Tanimoto similarity as the highest-EV UNTRIED
lever that:
  1. Uses structurally different math from ALL our existing bases (LGB
     tree splits, Chemprop message passing, MLP nonlinear projection,
     CatBoost oblivious trees)
  2. Preserves the +0.028 OOF-LB gap by construction (no fold-CV fit,
     no hyperparameter search, no feature-selection risk)
  3. Fits in Kaggle-notebook runtime (~20-30 min end-to-end)
  4. Is the historical gold standard for polymer property regression
     pre-2023 (still competitive per research-v2 §7)

We use Kernel Ridge Regression (KRR) instead of a full Gaussian Process.
For point predictions on our data, KRR ≈ GP MAP estimate; KRR is:
  - Simpler (no gpytorch/botorch/GAUCHE dependency)
  - Faster (no marginal-log-likelihood optimization)
  - Same math (both weight training samples by kernel similarity to query)
  - Doesn't need "cool models" that risk unknown failure modes

**KRR predicts:**  y_new = k(X_new, X_train) · (K + αI)^-1 · y_train
where k(·,·) is Tanimoto similarity on binary fingerprints. Every test
prediction is a similarity-weighted average of the closest train targets.

============================================================================
KEY DESIGN CHOICES (all "gap-safe" per research-v2 §1)
============================================================================

- **Fixed alpha = 1e-3** (regularization strength). NO per-target tuning.
  If we OOF-search alpha, we introduce selection bias (same disease that
  killed v2 Optuna).
- **Standardize y per fold** (train mean/std). Un-standardize at predict.
- **Binary Morgan-r2 fingerprints** on BOTH monomer and trimer, concatenated
  → 4096 bits. Chain-ext is our proven signal source. Tanimoto handles
  binary bit vectors natively.
- **No aux features.** Tanimoto kernel operates on binary set-similarity;
  continuous aux target values don't fit the kernel interpretation. Blend
  partners (chain-ext LGB v1) still USE aux, so we don't lose that signal —
  the blend brings it back.
- **Same 5-fold GroupKFold with SEED=42** as v1 → OOFs are blend-alignable.
- **Maxwell EPS↔Nc physics blend** (same as v1) — the only post-processor.

============================================================================
DEPENDENCIES
============================================================================

  Data: ppp-round-2/{train,test}.csv
  Venv: poly2-venv with rdkit, sklearn, numpy, pandas, tqdm

============================================================================
OUTPUTS  (under results/exp_kernel_ridge_tanimoto/)
============================================================================

  run.log             — training log, per-target R², timing
  oof.csv             — OOF predictions after Maxwell blend
  submission.csv      — Kaggle format id, target
  cv_summary.json     — per-target R², Maxwell params, config

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_kernel_ridge_tanimoto.py

Then either:
  (a) submit results/exp_kernel_ridge_tanimoto/submission.csv directly
      (expected solo LB 0.85-0.88 — likely below LGB v1's 0.894)
  (b) write a NNLS blend script combining KR + chain-ext LGB v1 (or +
      3-seed Chemprop) — expected LB 0.898-0.902

============================================================================
WALL TIME (~20-30 min on Mac CPU)
============================================================================

  - Load + canonicalize: ~5 sec
  - Compute Morgan-r2 bits (mono + trimer): ~5-10 min
  - Per-target CV: ~10-20 min total (tg dominates at ~5-8 min, others fast)
  - Maxwell + outputs: <1 min

============================================================================
EXPECTED
============================================================================

Solo per-target OOF R²: 0.75-0.88 (kernel methods historically underperform
tuned GBMs on high-dim tabular but are competitive on chemistry with
Tanimoto). Solo LB: 0.85-0.88.

Blend with chain-ext LGB v1 (LB 0.894, OOF 0.866) — research-v2 predicts
+0.002 to +0.008 LB → target LB 0.896-0.902. Structurally-different-math
partner is exactly what NNLS blends thrive on.

Blend with prior best (blend_nnls_3seed, LB 0.897) as 3rd base — potential
LB 0.898-0.905 if errors are orthogonal to both LGB and Chemprop.

============================================================================
"""
from __future__ import annotations

# --- stdlib ---
import json
import logging
import random
import sys
import time
from pathlib import Path

# --- third-party ---
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_kernel_ridge_tanimoto"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42
CHAIN_N_UNITS = 3

MORGAN_NBITS = 2048    # per (mono OR tri), so total feat dim = 2 * 2048

# Kernel Ridge alpha — FIXED, no per-target search (that would be OOF-selection)
KR_ALPHA = 1e-3

BLEND_W_GRID = np.linspace(0.0, 1.0, 201)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(exp_dir: Path) -> logging.Logger:
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "run.log"
    logger = logging.getLogger(EXP_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


# ============================================================================
# DATA + CANONICALIZATION  (verbatim from v1)
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def load_and_canonicalize(log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading train.csv / test.csv")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log.info(f"train raw: {tr.shape}   test raw: {te.shape}")

    all_smi = pd.concat([tr["smiles"], te["smiles"]]).unique()
    log.info(f"canonicalizing {len(all_smi)} unique raw SMILES")
    canon_map = {s: canonical(s) for s in tqdm(all_smi, desc="canonical", ncols=100)}
    tr["canon"] = tr["smiles"].map(canon_map)
    te["canon"] = te["smiles"].map(canon_map)

    dupes = tr.groupby(["canon", "target_type"]).size()
    n_dup_rows = int((dupes[dupes > 1] - 1).sum())
    if n_dup_rows:
        log.info(f"collapsing {n_dup_rows} duplicate (canon, target_type) rows in train by mean")
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean"),
                 smiles=("smiles", "first")))
    log.info(f"train after dedup: {tr.shape}")
    log.info(f"per-target train counts: {tr['target_type'].value_counts().to_dict()}")
    log.info(f"per-target test  counts: {te['target_type'].value_counts().to_dict()}")
    return tr, te


# ============================================================================
# POLYMER CHAIN EXTENSION  (verbatim from v1)
# ============================================================================

def polymer_to_multimer(smi: str, n_units: int = CHAIN_N_UNITS) -> str:
    if n_units <= 1:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        return smi
    star_a, star_b = stars
    a_bonds = m.GetAtomWithIdx(star_a).GetBonds()
    b_bonds = m.GetAtomWithIdx(star_b).GetBonds()
    if len(a_bonds) != 1 or len(b_bonds) != 1:
        return smi
    connect_a = a_bonds[0].GetOtherAtomIdx(star_a)
    connect_b = b_bonds[0].GetOtherAtomIdx(star_b)
    bond_type_a = a_bonds[0].GetBondType()
    bond_type_b = b_bonds[0].GetBondType()
    editable = Chem.RWMol(m)
    for idx in sorted(stars, reverse=True):
        editable.RemoveAtom(idx)
    def adjust(orig_idx: int, removed_sorted: list[int]) -> int:
        return orig_idx - sum(1 for r in removed_sorted if r < orig_idx)
    removed_sorted = sorted(stars)
    ca = adjust(connect_a, removed_sorted)
    cb = adjust(connect_b, removed_sorted)
    core = editable.GetMol()
    n_atoms_core = core.GetNumAtoms()
    if n_atoms_core == 0:
        return smi
    result = Chem.RWMol(core)
    prev_cb = cb
    first_ca = ca
    for i in range(1, n_units):
        result = Chem.RWMol(Chem.CombineMols(result, core))
        offset = result.GetNumAtoms() - n_atoms_core
        new_ca = offset + ca
        new_cb = offset + cb
        result.AddBond(prev_cb, new_ca, bond_type_a)
        prev_cb = new_cb
    left_star = result.AddAtom(Chem.Atom(0))
    right_star = result.AddAtom(Chem.Atom(0))
    result.AddBond(first_ca, left_star, bond_type_a)
    result.AddBond(prev_cb, right_star, bond_type_b)
    try:
        final = result.GetMol()
        Chem.SanitizeMol(final)
        return Chem.MolToSmiles(final, canonical=True)
    except Exception:
        return smi


# ============================================================================
# BINARY MORGAN FINGERPRINT  (Tanimoto operates on binary set-similarity)
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def morgan_bits(smi: str, radius: int = 2, nbits: int = MORGAN_NBITS) -> np.ndarray:
    """Return Morgan-radius-r BINARY bit vector (uint8) for the SMILES.
    Returns zeros on parse failure."""
    m = Chem.MolFromSmiles(_cap(smi))
    if m is None:
        return np.zeros(nbits, dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    for bit in bv.GetOnBits():
        arr[bit] = 1
    return arr


# ============================================================================
# TANIMOTO KERNEL  (aka Jaccard for binary vectors)
# ============================================================================

def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto similarity matrix. A: (n_A, D) binary. B: (n_B, D) binary.
    Returns (n_A, n_B) matrix of Tanimoto similarities in [0, 1].

    Formula: T(a, b) = |a ∩ b| / |a ∪ b| = (a · b) / (||a||² + ||b||² - a·b)
    For binary vectors, dot product = intersection count, norm² = |a|."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A_dot_B = A @ B.T                          # (n_A, n_B)
    A_norm = (A * A).sum(axis=1)[:, None]      # (n_A, 1)
    B_norm = (B * B).sum(axis=1)[None, :]      # (1, n_B)
    denom = A_norm + B_norm - A_dot_B + 1e-9
    return (A_dot_B / denom).astype(np.float32)


# ============================================================================
# CV  (same fold structure as chain-ext v1)
# ============================================================================

def group_kfold_splits(
    canon_arr: np.ndarray | list,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# PER-TARGET KERNEL RIDGE
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    X_all: np.ndarray,
    canon_to_idx: dict[str, int],
    log: logging.Logger,
) -> dict:
    """Per-target Kernel Ridge with Tanimoto kernel + Maxwell-ready OOF/test outputs."""
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    # Slice features by canon index
    tr_idx = np.array([canon_to_idx[c] for c in g_tr["canon"]])
    te_idx = np.array([canon_to_idx[c] for c in g_te["canon"]])
    X_tr = X_all[tr_idx]   # (n_train_for_target, 4096)
    X_te = X_all[te_idx]   # (n_test_for_target, 4096)

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    fold_r2s = []

    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()
        # Standardize y on train fold (numerical stability for KR)
        y_mean = y[tri].mean()
        y_std = y[tri].std()
        y_std = max(y_std, 1e-6)
        y_tr_norm = (y[tri] - y_mean) / y_std

        # Precomputed Tanimoto kernel matrices
        K_train_train = tanimoto_kernel(X_tr[tri], X_tr[tri])   # (n_tr, n_tr)
        K_val_train   = tanimoto_kernel(X_tr[vai], X_tr[tri])   # (n_val, n_tr)

        model = KernelRidge(alpha=KR_ALPHA, kernel="precomputed")
        model.fit(K_train_train, y_tr_norm)
        pred_va_norm = model.predict(K_val_train)
        pred_va = pred_va_norm * y_std + y_mean
        oof[vai] = pred_va

        r2 = float(r2_score(y[vai], pred_va))
        fold_r2s.append(r2)
        log.info(f"[{target}] fold {k}: R²={r2:.4f}  n_val={len(vai)}  time={time.time()-t0:.1f}s")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² (KR only, pre-Maxwell) = {oof_r2:.4f}   "
             f"(fold mean {np.mean(fold_r2s):.4f})")

    # Refit on full train, predict test
    t0 = time.time()
    y_mean = y.mean()
    y_std = max(y.std(), 1e-6)
    y_norm = (y - y_mean) / y_std

    K_full_full = tanimoto_kernel(X_tr, X_tr)
    K_test_full = tanimoto_kernel(X_te, X_tr)

    model = KernelRidge(alpha=KR_ALPHA, kernel="precomputed")
    model.fit(K_full_full, y_norm)
    test_pred_norm = model.predict(K_test_full)
    test_pred = test_pred_norm * y_std + y_mean
    log.info(f"[{target}] refit + predict test done  time={time.time()-t0:.1f}s")

    return {
        "target":    target,
        "n_train":   int(len(g_tr)),
        "n_test":    int(len(g_te)),
        "oof": pd.DataFrame({
            "canon":       g_tr["canon"].values,
            "target_type": target,
            "y_true":      y,
            "y_pred":      oof,
        }),
        "test_pred": pd.DataFrame({
            "id":          g_te["id"].values,
            "canon":       g_te["canon"].values,
            "target_type": target,
            "target":      test_pred,
        }),
        "oof_r2":     oof_r2,
        "fold_r2s":   fold_r2s,
    }


# ============================================================================
# MAXWELL POST-FIT  (verbatim from v1)
# ============================================================================

def fit_maxwell_forward(nc_values, eps_values):
    x = nc_values ** 2
    y = eps_values
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b), float(r2_score(y, a * x + b))


def fit_maxwell_reverse(eps_values, nc_values):
    x = eps_values
    y = nc_values ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred_nc = np.sqrt(np.clip(a * x + b, 1e-9, None))
    return float(a), float(b), float(r2_score(nc_values, pred_nc))


def apply_maxwell_forward(nc_values, a, b):
    return a * (nc_values ** 2) + b


def apply_maxwell_reverse(eps_values, a, b):
    return np.sqrt(np.clip(a * eps_values + b, 1e-9, None))


def search_blend_weight(y_true, y_ml, y_prior, grid=BLEND_W_GRID):
    r2s = np.array([r2_score(y_true, w * y_ml + (1 - w) * y_prior) for w in grid])
    best_i = int(np.argmax(r2s))
    baseline_r2 = float(r2_score(y_true, y_ml))
    return float(grid[best_i]), float(r2s[best_i]), baseline_r2


def build_effective_value_lookup(train_df, oof_results, target):
    lookup: dict[str, float] = {}
    tr_t = train_df[train_df["target_type"] == target]
    for _, row in tr_t.iterrows():
        lookup[row["canon"]] = float(row["target"])
    if target in oof_results:
        oof_df = oof_results[target]["oof"]
        for _, row in oof_df.iterrows():
            if row["canon"] not in lookup:
                lookup[row["canon"]] = float(row["y_pred"])
    return lookup


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: n_splits={N_SPLITS}  seed={SEED}  chain_n_units={CHAIN_N_UNITS}  "
             f"morgan_nbits={MORGAN_NBITS}  KR_alpha={KR_ALPHA}")
    log.info(f"Feature: BINARY Morgan-r2 fingerprints on MONO + TRIMER, concatenated → "
             f"{2*MORGAN_NBITS} bits per canon")
    log.info(f"Kernel: Tanimoto (Jaccard on binary bits)")
    log.info(f"Post-fit: Maxwell EPS↔Nc physics prior blend")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = list(pd.concat([tr["canon"], te["canon"]]).drop_duplicates())
    log.info(f"unique canonical SMILES (train + test): {len(all_canon)}")
    canon_to_idx = {c: i for i, c in enumerate(all_canon)}

    # -------- Compute trimer SMILES --------
    log.info(f"generating {CHAIN_N_UNITS}-mer polymer SMILES...")
    t0 = time.time()
    all_tri = [polymer_to_multimer(s, CHAIN_N_UNITS) for s in tqdm(all_canon, desc="mono→trimer", ncols=100)]
    n_extended = sum(1 for m, t in zip(all_canon, all_tri) if m != t)
    log.info(f"chain extension: {n_extended}/{len(all_canon)} extended  time={time.time()-t0:.1f}s")

    # -------- Binary Morgan-r2 on mono + trimer --------
    log.info(f"computing Morgan-r2 BIT vectors on monomer ({MORGAN_NBITS} bits)...")
    t0 = time.time()
    X_mono = np.stack([morgan_bits(s, 2, MORGAN_NBITS) for s in tqdm(all_canon, desc="mono morgan-r2", ncols=100)])
    log.info(f"  X_mono: {X_mono.shape}  time={time.time()-t0:.1f}s")

    log.info(f"computing Morgan-r2 BIT vectors on trimer ({MORGAN_NBITS} bits)...")
    t0 = time.time()
    X_tri = np.stack([morgan_bits(s, 2, MORGAN_NBITS) for s in tqdm(all_tri, desc="tri morgan-r2", ncols=100)])
    log.info(f"  X_tri: {X_tri.shape}  time={time.time()-t0:.1f}s")

    # Concat: 4096 bits per canon
    X_all = np.concatenate([X_mono, X_tri], axis=1)
    log.info(f"X_all (mono + trimer concat): {X_all.shape}  "
             f"nnz frac={float((X_all > 0).mean()):.4f}")

    # -------- Train 7 targets --------
    log.info("=" * 60)
    log.info("PER-TARGET KERNEL RIDGE (Tanimoto kernel)")
    log.info("=" * 60)
    results: dict[str, dict] = {}
    for tgt in tqdm(TARGETS, desc="targets", ncols=100):
        log.info("-" * 60)
        results[tgt] = train_one_target(tgt, tr, te, X_all, canon_to_idx, log)

    baseline_mean_r2 = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("KR-only per-target OOF R²  (before Maxwell)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}")
    log.info(f"  MEAN R² (KR only) = {baseline_mean_r2:.4f}")

    # -------- Maxwell post-fit --------
    log.info("=" * 60)
    log.info("MAXWELL RELATION POST-FIT (EPS ↔ Nc)")
    log.info("=" * 60)
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    co = wide.dropna(subset=["eps", "nc"])
    log.info(f"co-labeled train molecules: n={len(co)}")

    a_fwd, b_fwd, r2_fwd = fit_maxwell_forward(co["nc"].values, co["eps"].values)
    a_rev, b_rev, r2_rev = fit_maxwell_reverse(co["eps"].values, co["nc"].values)
    log.info(f"forward EPS = {a_fwd:.4f}·Nc² + {b_fwd:.4f}   R²={r2_fwd:.4f}")
    log.info(f"reverse Nc² = {a_rev:.4f}·EPS + {b_rev:.4f}   R²(on Nc)={r2_rev:.4f}")

    canon_to_nc = build_effective_value_lookup(tr, results, "nc")
    canon_to_eps = build_effective_value_lookup(tr, results, "eps")

    # EPS blend
    eps_oof = results["eps"]["oof"].copy()
    nc_eff = eps_oof["canon"].map(canon_to_nc).values.astype(float)
    eps_maxwell_oof = apply_maxwell_forward(nc_eff, a_fwd, b_fwd)
    mask = np.isnan(eps_maxwell_oof)
    eps_maxwell_oof[mask] = eps_oof["y_pred"].values[mask]
    best_w_eps, best_r2_eps, baseline_r2_eps = search_blend_weight(
        eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_maxwell_oof
    )
    log.info(f"eps blend: KR R²={baseline_r2_eps:.4f}  best w={best_w_eps:.3f}  "
             f"blend R²={best_r2_eps:.4f}   Δ={best_r2_eps - baseline_r2_eps:+.4f}")
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred"].values + (1 - best_w_eps) * eps_maxwell_oof
    results["eps"]["oof"] = eps_oof
    results["eps"]["oof_r2"] = best_r2_eps

    # Nc blend
    nc_oof = results["nc"]["oof"].copy()
    eps_eff = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    nc_maxwell_oof = apply_maxwell_reverse(eps_eff, a_rev, b_rev)
    mask = np.isnan(nc_maxwell_oof)
    nc_maxwell_oof[mask] = nc_oof["y_pred"].values[mask]
    best_w_nc, best_r2_nc, baseline_r2_nc = search_blend_weight(
        nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_maxwell_oof
    )
    log.info(f"nc blend: KR R²={baseline_r2_nc:.4f}  best w={best_w_nc:.3f}  "
             f"blend R²={best_r2_nc:.4f}   Δ={best_r2_nc - baseline_r2_nc:+.4f}")
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_maxwell_oof
    results["nc"]["oof"] = nc_oof
    results["nc"]["oof_r2"] = best_r2_nc

    # Apply Maxwell to test predictions
    canon_to_nc_test = dict(zip(results["nc"]["test_pred"]["canon"], results["nc"]["test_pred"]["target"]))
    canon_to_eps_test = dict(zip(results["eps"]["test_pred"]["canon"], results["eps"]["test_pred"]["target"]))

    def get_nc_test(c):
        if c in canon_to_nc: return canon_to_nc[c]
        return canon_to_nc_test.get(c, float("nan"))

    def get_eps_test(c):
        if c in canon_to_eps: return canon_to_eps[c]
        return canon_to_eps_test.get(c, float("nan"))

    eps_te = results["eps"]["test_pred"].copy()
    nc_eff_te = np.array([get_nc_test(c) for c in eps_te["canon"]], dtype=float)
    eps_max_te = apply_maxwell_forward(nc_eff_te, a_fwd, b_fwd)
    m = np.isnan(eps_max_te); eps_max_te[m] = eps_te["target"].values[m]
    eps_te["target"] = best_w_eps * eps_te["target"].values + (1 - best_w_eps) * eps_max_te
    results["eps"]["test_pred"] = eps_te

    nc_te = results["nc"]["test_pred"].copy()
    eps_eff_te = np.array([get_eps_test(c) for c in nc_te["canon"]], dtype=float)
    nc_max_te = apply_maxwell_reverse(eps_eff_te, a_rev, b_rev)
    m = np.isnan(nc_max_te); nc_max_te[m] = nc_te["target"].values[m]
    nc_te["target"] = best_w_nc * nc_te["target"].values + (1 - best_w_nc) * nc_max_te
    results["nc"]["test_pred"] = nc_te

    final_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R² (post-Maxwell)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   R²={results[t]['oof_r2']:.4f}")
    log.info(f"  MEAN R² (final) = {final_mean:.4f}   (pre-Maxwell was {baseline_mean_r2:.4f})")

    # -------- Write outputs --------
    oof_all = pd.concat([results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                         for t in TARGETS], ignore_index=True)
    sub_all = pd.concat([results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                        ignore_index=True)

    oof_path = EXP_DIR / "oof.csv"
    oof_all.to_csv(oof_path, index=False)
    log.info(f"wrote {oof_path}  rows={len(oof_all)}")

    sub_out = sub_all.sort_values("id").reset_index(drop=True)
    sub_path = EXP_DIR / "submission.csv"
    sub_out.to_csv(sub_path, index=False)
    log.info(f"wrote {sub_path}  rows={len(sub_out)}")

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2_final":  final_mean,
        "mean_r2_pre_maxwell": baseline_mean_r2,
        "per_target":     {t: {"n_train": results[t]["n_train"],
                              "n_test":  results[t]["n_test"],
                              "oof_r2":  results[t]["oof_r2"],
                              "fold_r2s": results[t]["fold_r2s"]}
                          for t in TARGETS},
        "maxwell": {
            "n_co_labeled": int(len(co)),
            "forward_fit": {"a": a_fwd, "b": b_fwd, "r2": r2_fwd},
            "reverse_fit": {"a": a_rev, "b": b_rev, "r2_on_nc": r2_rev},
            "eps_blend": {"baseline_r2": baseline_r2_eps, "best_w": best_w_eps, "best_r2": best_r2_eps},
            "nc_blend":  {"baseline_r2": baseline_r2_nc,  "best_w": best_w_nc,  "best_r2": best_r2_nc},
        },
        "chain_extension": {
            "n_units":         CHAIN_N_UNITS,
            "n_extended":      int(n_extended),
            "n_total_smiles":  int(len(all_canon)),
        },
        "config": {
            "n_splits":       N_SPLITS,
            "seed":           SEED,
            "morgan_nbits":   MORGAN_NBITS,
            "kr_alpha":       KR_ALPHA,
            "feature_dim":    2 * MORGAN_NBITS,
            "kernel":         "Tanimoto (Jaccard on binary bits)",
            "cv_mode":        "no-aux + kernel ridge + Maxwell post-fit",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Comparison vs chain-ext v1 (LB 0.894)
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (KR + Maxwell)  vs chain-ext LGB v1 reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'KR':>10s}  {'LGB v1':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = results[t]["oof_r2"]
        ref = v1_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    v1_mean = float(np.mean(list(v1_ref.values())))
    log.info(f"  {'MEAN':>6s}  {final_mean:>10.4f}  {v1_mean:>10.4f}  {final_mean - v1_mean:>+8.4f}")
    log.info(f"  (chain-ext v1 LB reference: 0.894)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
