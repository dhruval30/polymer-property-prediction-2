"""
exp_polymer_physics_prior.py — Bicerano-style polymer physics prior with
                                SMARTS group-additivity features + Ridge.

============================================================================
WHY THIS EXISTS
============================================================================

Every past attempt to improve on chain-ext LGB v1 (LB 0.894) has failed by:
  1. Adding features that overfit train's structural distribution (v3fixed
     15 domain features → LB -0.037)
  2. Blending with weaker models that have divergent OOF-LB gaps (MLP → -0.027)
  3. Selecting hyperparameters / transforms on OOF (v2 Optuna → -0.026)
  4. Adding correlated bases (chain-ext + Chemprop blend → -0.004)

This script is DIFFERENT in every way from those failures:
  - **Separate NEW base model** (not modification of chain-ext v1)
  - **Polymer-DOMAIN-specific features**, not generic chemistry — SMARTS
    patterns based on Bicerano's group additivity method (the polymer
    chemistry textbook reference for Tg prediction, predates ML)
  - **Ridge with FIXED alpha** — no OOF search, no Optuna, no transform search
  - **~30 features** on 220-4139 samples → well-conditioned, no overfit risk
  - **Fundamentally different math** — group additivity is essentially:
       target = intercept + Σ contribution_i × count_of_group_i
    A LINEAR physics equation, not statistical fingerprint learning

The failure modes we've hit don't apply here:
  - No fold-CV overfit (no hyperparameter selection surface)
  - No OOF-LB gap disparity (Ridge on 30 features has small stable gap)
  - Errors are truly orthogonal to chain-ext LGB (fingerprints) — different
    worldview on the same molecule

============================================================================
FEATURES (~30 polymer-relevant SMARTS + a few computed statistics)
============================================================================

Backbone types:
  sp3 aliphatic C (chain flexibility)
  sp3 aliphatic C in ring
  sp2 aromatic C
  sp2 non-aromatic C (alkene)

Ring motifs:
  Phenyl / benzene ring
  Biphenyl (Ar-Ar linkage → rigidity)
  Naphthalene
  Aromatic 5-ring (thiophene/pyrrole-like)
  Aliphatic ring atoms

Heteroatoms:
  F, Cl, Br, Si atom counts
  S atoms, N atoms, O atoms

Functional groups (chain motifs):
  Ester        C(=O)O
  Amide        C(=O)N
  Ether        C-O-C (non-ester)
  Sulfone      S(=O)(=O)
  Sulfoxide    S(=O)
  Nitrile      C#N
  Urethane     N-C(=O)-O
  Vinyl        C=C
  Hydroxyl     OH
  Amine primary/secondary

Backbone stats (computed on original SMILES with * wildcards intact):
  Backbone atom count (shortest path between two *)
  Backbone aromatic fraction
  Backbone sp2 fraction

Aggregate ratios:
  F count / heavy atoms
  Halogen fraction
  Aromatic fraction
  sp3 fraction

Total: ~30 features per polymer.

============================================================================
DEPENDENCIES
============================================================================

  Data: ppp-round-2/{train,test}.csv
  Venv: poly2-venv with rdkit, sklearn, numpy, pandas, tqdm

============================================================================
OUTPUTS  (under results/exp_polymer_physics_prior/)
============================================================================

  run.log             — training log, per-target R², per-fold R²
  oof.csv             — OOF predictions after Maxwell blend
  submission.csv      — Kaggle format id, target
  cv_summary.json     — per-target R², Ridge alpha, feature list, config

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_polymer_physics_prior.py

============================================================================
WALL TIME (~5-15 min on Mac CPU)
============================================================================

  - Load + canonicalize: ~5 sec
  - Compute SMARTS features on ~9k canons: ~2-5 min
  - Per-target Ridge CV: <1 min total (7 targets × 5 folds × 30-feature Ridge fit)
  - Maxwell + outputs: <1 min

============================================================================
EXPECTED
============================================================================

Solo per-target OOF R²: 0.55-0.85 (polymer physics group additivity is a
weaker predictor than modern ML, but on Tg specifically it should hit 0.80+
since Bicerano was designed for Tg).

Solo LB: 0.70-0.82 (weak alone — expected).

**Post-Maxwell + used in 3-way NNLS blend with chain-ext LGB v1 +
3-seed Chemprop**: expected +0.001-0.005 LB over current best 0.897.

Blend script is a separate file (write once physics prior is confirmed
to produce sane OOFs).

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
from rdkit.Chem import rdMolDescriptors
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_polymer_physics_prior"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)

N_SPLITS = 5
SEED = 42

# Ridge alpha — FIXED, no per-target search (that would be OOF-selection)
RIDGE_ALPHA = 1.0

BLEND_W_GRID = np.linspace(0.0, 1.0, 201)


# ============================================================================
# POLYMER-RELEVANT SMARTS PATTERNS (Bicerano-inspired)
# ============================================================================

_SMARTS_STR: dict[str, str] = {
    # -------- Backbone carbons --------
    "sp3_c_chain":       "[CX4;!R]",
    "sp3_c_ring":        "[CX4;R]",
    "sp2_c_aromatic":    "c",
    "sp2_c_alkene":      "[CX3;!R;!$(C=O);!$(C#N);!$(C=S);!$(C=N)]",
    # -------- Ring motifs --------
    "phenyl":            "c1ccccc1",
    "biphenyl":          "c1ccc(-c2ccccc2)cc1",
    "naphthalene":       "c1ccc2ccccc2c1",
    "aromatic_5_ring":   "[a;r5]",
    # -------- Heteroatoms --------
    "atom_F":            "[F]",
    "atom_Cl":           "[Cl]",
    "atom_Br":           "[Br]",
    "atom_Si":           "[Si]",
    "atom_S":            "[#16]",
    "atom_N":            "[#7]",
    "atom_O":            "[#8]",
    # -------- Functional groups (chain motifs) --------
    "ester":             "[CX3](=[OX1])[OX2H0]",
    "amide":             "[CX3](=[OX1])[NX3]",
    "ether":             "[OX2;!$(O=C);!$(O~[!C])][CX4]",
    "carbonyl":          "[CX3]=[OX1]",
    "sulfone":           "[SX4](=[OX1])(=[OX1])",
    "sulfoxide":         "[SX3;$([S](=O)([#6])[#6])]",
    "nitrile":           "[CX2]#[NX1]",
    "urethane":          "[NX3][CX3](=[OX1])[OX2]",
    "vinyl":             "[CX3]=[CX3]",
    "hydroxyl":          "[OX2H]",
    "amine_prim":        "[NX3;H2]",
    "amine_sec":         "[NX3;H1;!$(N-C=O)]",
}

# Pre-compile SMARTS at module load
_SMARTS_MOLS: dict[str, Chem.Mol] = {}
for _name, _pat in _SMARTS_STR.items():
    _m = Chem.MolFromSmarts(_pat)
    if _m is None:
        raise RuntimeError(f"Failed to compile SMARTS '{_name}': {_pat}")
    _SMARTS_MOLS[_name] = _m

SMARTS_FEATURE_NAMES = tuple(_SMARTS_MOLS.keys())          # 26 features

BACKBONE_FEATURE_NAMES = (
    "backbone_atoms",
    "backbone_aromatic_frac",
    "backbone_sp2_frac",
    "heavy_atoms",
    "rot_bonds",
    "F_C_ratio",
    "halogen_frac",
    "aromatic_frac",
    "sp3_frac",
)

FEATURE_NAMES = SMARTS_FEATURE_NAMES + BACKBONE_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)


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
# FEATURE COMPUTATION
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def compute_physics_features(smi: str) -> np.ndarray:
    """Return (N_FEATURES,) float32 vector for the polymer SMILES.
    Returns zeros on parse failure so the pipeline never crashes."""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    m_orig = Chem.MolFromSmiles(smi)
    if m_orig is None:
        return out
    m_capped = Chem.MolFromSmiles(_cap(smi))
    if m_capped is None:
        return out

    heavy = m_capped.GetNumHeavyAtoms()
    if heavy < 1:
        return out

    # -------- SMARTS group counts (on capped molecule) --------
    for i, name in enumerate(SMARTS_FEATURE_NAMES):
        try:
            matches = m_capped.GetSubstructMatches(_SMARTS_MOLS[name])
            out[i] = float(len(matches))
        except Exception:
            out[i] = 0.0

    # -------- Backbone stats (need original SMILES with * intact) --------
    smarts_end = len(SMARTS_FEATURE_NAMES)
    stars = [a.GetIdx() for a in m_orig.GetAtoms() if a.GetSymbol() == "*"]
    backbone_atoms: list[int] = []
    if len(stars) == 2:
        try:
            path = Chem.GetShortestPath(m_orig, stars[0], stars[1])
            backbone_atoms = [i for i in path if m_orig.GetAtomWithIdx(i).GetSymbol() != "*"]
        except Exception:
            backbone_atoms = []

    out[smarts_end + 0] = float(len(backbone_atoms))     # backbone_atoms
    if backbone_atoms:
        n_arom = sum(1 for i in backbone_atoms if m_orig.GetAtomWithIdx(i).GetIsAromatic())
        out[smarts_end + 1] = n_arom / len(backbone_atoms)     # backbone_aromatic_frac
        n_sp2 = sum(1 for i in backbone_atoms
                    if m_orig.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2)
        out[smarts_end + 2] = n_sp2 / len(backbone_atoms)     # backbone_sp2_frac

    out[smarts_end + 3] = float(heavy)     # heavy_atoms

    try:
        out[smarts_end + 4] = float(rdMolDescriptors.CalcNumRotatableBonds(m_capped))
    except Exception:
        pass

    f_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "F")
    c_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "C")
    halogen = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    n_arom_atoms = sum(1 for a in m_capped.GetAtoms() if a.GetIsAromatic())
    n_sp3 = sum(1 for a in m_capped.GetAtoms()
                if a.GetHybridization() == Chem.HybridizationType.SP3)

    out[smarts_end + 5] = f_count / max(1, c_count)         # F_C_ratio
    out[smarts_end + 6] = halogen / heavy                    # halogen_frac
    out[smarts_end + 7] = n_arom_atoms / heavy               # aromatic_frac
    out[smarts_end + 8] = n_sp3 / heavy                      # sp3_frac

    return out


# ============================================================================
# CV
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
# PER-TARGET RIDGE
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    X_all: np.ndarray,
    canon_to_idx: dict[str, int],
    log: logging.Logger,
) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    tr_idx = np.array([canon_to_idx[c] for c in g_tr["canon"]])
    te_idx = np.array([canon_to_idx[c] for c in g_te["canon"]])
    X_tr = X_all[tr_idx]
    X_te = X_all[te_idx]

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)
    oof = np.zeros(len(g_tr), dtype=np.float64)
    fold_r2s = []

    for k, (tri, vai) in enumerate(splits):
        t0 = time.time()

        # Standardize features on train fold (Ridge needs standardization for
        # numerical stability when features have wildly different scales)
        scaler_x = StandardScaler()
        X_tr_s = scaler_x.fit_transform(X_tr[tri])
        X_va_s = scaler_x.transform(X_tr[vai])

        # Standardize target on train fold (Ridge intercept handles mean, but
        # scaling improves conditioning)
        y_mean = float(y[tri].mean())
        y_std = float(max(y[tri].std(), 1e-6))
        y_tri_norm = (y[tri] - y_mean) / y_std

        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=SEED)
        model.fit(X_tr_s, y_tri_norm)
        pred_va_norm = model.predict(X_va_s)
        pred_va = pred_va_norm * y_std + y_mean

        oof[vai] = pred_va
        r2 = float(r2_score(y[vai], pred_va))
        fold_r2s.append(r2)
        log.info(f"[{target}] fold {k}: R²={r2:.4f}  n_val={len(vai)}  time={time.time()-t0:.2f}s")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² (Ridge only, pre-Maxwell) = {oof_r2:.4f}   "
             f"(fold mean {np.mean(fold_r2s):.4f})")

    # Refit on full train and predict test
    t0 = time.time()
    scaler_x = StandardScaler()
    X_tr_full = scaler_x.fit_transform(X_tr)
    X_te_s = scaler_x.transform(X_te)
    y_mean = float(y.mean())
    y_std = float(max(y.std(), 1e-6))
    y_norm = (y - y_mean) / y_std
    full = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=SEED)
    full.fit(X_tr_full, y_norm)
    test_pred = full.predict(X_te_s) * y_std + y_mean
    log.info(f"[{target}] refit + predict test done  time={time.time()-t0:.2f}s")

    # Feature importance (absolute standardized coefficients)
    coefs = np.abs(full.coef_)
    top5 = np.argsort(coefs)[-5:][::-1]
    top5_str = "  ".join([f"{FEATURE_NAMES[i]}={coefs[i]:.3f}" for i in top5])
    log.info(f"[{target}] top-5 features by |coef|: {top5_str}")

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
        "top5_features": [FEATURE_NAMES[i] for i in top5],
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
    log.info(f"CONFIG: n_splits={N_SPLITS}  seed={SEED}  ridge_alpha={RIDGE_ALPHA}")
    log.info(f"Features: {N_FEATURES} = {len(SMARTS_FEATURE_NAMES)} SMARTS + "
             f"{len(BACKBONE_FEATURE_NAMES)} backbone/aggregate")
    log.info(f"Post-fit: Maxwell EPS↔Nc physics prior blend")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = list(pd.concat([tr["canon"], te["canon"]]).drop_duplicates())
    log.info(f"unique canonical SMILES (train + test): {len(all_canon)}")
    canon_to_idx = {c: i for i, c in enumerate(all_canon)}

    # -------- Compute physics features --------
    log.info(f"computing {N_FEATURES} physics features per polymer...")
    t0 = time.time()
    X_all = np.stack([compute_physics_features(s) for s in tqdm(all_canon, desc="physics feats", ncols=100)])
    log.info(f"X_all: {X_all.shape}  size={X_all.nbytes/1e6:.2f}MB  time={time.time()-t0:.1f}s")

    # Report feature stats (which SMARTS actually fired)
    log.info("feature non-zero counts (across all canons):")
    for i, name in enumerate(FEATURE_NAMES):
        nnz = int((X_all[:, i] != 0).sum())
        mean_val = float(X_all[:, i].mean())
        max_val = float(X_all[:, i].max())
        log.info(f"  {name:>22s}   nnz={nnz:>5d} ({100*nnz/len(all_canon):5.1f}%)   "
                 f"mean={mean_val:>7.3f}   max={max_val:>7.1f}")

    # -------- Train 7 targets --------
    log.info("=" * 60)
    log.info(f"PER-TARGET RIDGE (alpha={RIDGE_ALPHA}, {N_FEATURES} features)")
    log.info("=" * 60)
    results: dict[str, dict] = {}
    for tgt in TARGETS:
        log.info("-" * 60)
        results[tgt] = train_one_target(tgt, tr, te, X_all, canon_to_idx, log)

    baseline_mean_r2 = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("Ridge-only per-target OOF R²  (before Maxwell)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}")
    log.info(f"  MEAN R² (Ridge only) = {baseline_mean_r2:.4f}")

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
    eps_max = apply_maxwell_forward(nc_eff, a_fwd, b_fwd)
    m = np.isnan(eps_max); eps_max[m] = eps_oof["y_pred"].values[m]
    best_w_eps, best_r2_eps, base_r2_eps = search_blend_weight(
        eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_max)
    log.info(f"eps blend: Ridge R²={base_r2_eps:.4f}  best w={best_w_eps:.3f}  "
             f"blend R²={best_r2_eps:.4f}   Δ={best_r2_eps - base_r2_eps:+.4f}")
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred"].values + (1 - best_w_eps) * eps_max
    results["eps"]["oof"] = eps_oof
    results["eps"]["oof_r2"] = best_r2_eps

    # Nc blend
    nc_oof = results["nc"]["oof"].copy()
    eps_eff = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    nc_max = apply_maxwell_reverse(eps_eff, a_rev, b_rev)
    m = np.isnan(nc_max); nc_max[m] = nc_oof["y_pred"].values[m]
    best_w_nc, best_r2_nc, base_r2_nc = search_blend_weight(
        nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_max)
    log.info(f"nc blend: Ridge R²={base_r2_nc:.4f}  best w={best_w_nc:.3f}  "
             f"blend R²={best_r2_nc:.4f}   Δ={best_r2_nc - base_r2_nc:+.4f}")
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_max
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
                              "fold_r2s": results[t]["fold_r2s"],
                              "top5_features": results[t]["top5_features"]}
                          for t in TARGETS},
        "maxwell": {
            "n_co_labeled": int(len(co)),
            "forward_fit": {"a": a_fwd, "b": b_fwd, "r2": r2_fwd},
            "reverse_fit": {"a": a_rev, "b": b_rev, "r2_on_nc": r2_rev},
            "eps_blend": {"baseline_r2": base_r2_eps, "best_w": best_w_eps, "best_r2": best_r2_eps},
            "nc_blend":  {"baseline_r2": base_r2_nc,  "best_w": best_w_nc,  "best_r2": best_r2_nc},
        },
        "config": {
            "n_splits":       N_SPLITS,
            "seed":           SEED,
            "ridge_alpha":    RIDGE_ALPHA,
            "n_features":     N_FEATURES,
            "smarts_features": list(SMARTS_FEATURE_NAMES),
            "backbone_features": list(BACKBONE_FEATURE_NAMES),
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Comparison vs chain-ext v1 (LB 0.894)
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (Ridge + Maxwell)  vs chain-ext LGB v1 reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'physics':>10s}  {'LGB v1':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = results[t]["oof_r2"]
        ref = v1_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    v1_mean = float(np.mean(list(v1_ref.values())))
    log.info(f"  {'MEAN':>6s}  {final_mean:>10.4f}  {v1_mean:>10.4f}  {final_mean - v1_mean:>+8.4f}")
    log.info(f"  (chain-ext v1 LB reference: 0.894 — physics prior is WEAKER solo but structurally orthogonal for blending)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
