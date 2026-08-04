"""
exp_chain_ext_lgbm_v2.py — Chain-ext LGB v2: per-target Optuna + target-transform search
                          + nc-regression fix + bandgap consistency post-processor.

============================================================================
WHY THIS EXISTS
============================================================================

`exp_chain_ext_lgbm.py` (v1, LB 0.894) got us to best-solo but left three
weaknesses on the OOF:
  - nc REGRESSED -0.013 from mono-only (trimer noise on tight 1.5-2.7 range).
  - ei chronically low (OOF 0.804) — physical target, small data.
  - eps stuck at 0.822 despite Maxwell prior — bandgap physics not yet used.

v1 used ONE hyperparameter config for all 7 targets. Round 1 winner
(jday NeurIPS 2025) tuned per-target with Optuna. This script does that,
plus 3 more targeted upgrades:

  (1) Per-target Optuna LGB hyperparameter search
      - 30 trials for eea/ei/eps/nc/egb  (small-data targets)
      - 50 trials for egc/tg              (larger data)
      - Search space: n_est, lr, num_leaves, min_child_samples,
        feature_fraction, bagging_fraction, reg_lambda, reg_alpha, max_depth
      - Objective: 5-fold GroupKFold mean R² (same folds as v1 for blend align)

  (2) Per-target target transform search
      - Candidates: identity, log1p, sqrt (only if y >= 0), yeo-johnson, rank-Gauss
      - Pick winner by 5-fold OOF R² in ORIGINAL target space (unwind transform)
      - Round 1 winner recipe

  (3) Nc regression fix
      - Drop trimer features for the nc target ONLY. Keep mono + aux + Maxwell.
      - Other 6 targets keep the full mono + trimer stack.
      - Direct recovery of the -0.013 OOF regression on nc.

  (4) Bandgap consistency post-processor
      - Physics: Egc ≈ Ei - Eea (chain bandgap ≈ ionization − electron affinity).
      - Physics: Egb ≈ Egc (r = 0.93 in train EDA).
      - After all 7 LGB predictions + Maxwell EPS/Nc blend:
          pred_Egc_v2 = w1 * pred_Egc + (1-w1) * (pred_Ei - pred_Eea)
          pred_Egb_v2 = w2 * pred_Egb + (1-w2) * pred_Egc_v2
          pred_Ei_v2  = w3 * pred_Ei  + (1-w3) * (pred_Egc + pred_Eea)
          pred_Eea_v2 = w4 * pred_Eea + (1-w4) * (pred_Egc - pred_Ei)
      - Weights tuned per-target on OOF via search.

============================================================================
WHAT'S THE SAME AS v1
============================================================================

  - 5-fold GroupKFold on canonical SMILES, SPLIT_SEED=42 (blend-alignable)
  - Chain-extension via polymer_to_multimer (trimer, verbatim from v1)
  - Feature stack per SMILES:
      * MONOMER: RDKit desc + Morgan-r2/r3 count + MACCS + AtomPair + TopTorsion + Avalon
      * TRIMER:  RDKit desc + Morgan-r2 count + MACCS + AtomPair + Avalon
  - Aux matrix-completion features (14 = 7 values + 7 masks per row)
  - Maxwell EPS ↔ Nc physics prior post-fit
  - Output format: oof.csv (canon, target_type, y_true, y_pred) + submission.csv

============================================================================
WHAT'S ONE-SHOT
============================================================================

- Features computed from SCRATCH (no reuse of v1's feature_cache.pkl).
- Everything self-contained in this file (no shared _utils per CLAUDE.md).

============================================================================
WALL TIME
============================================================================

Rough plan on Mac M-series CPU:
  - Featurize (mono + trimer): ~10 min
  - Per-target target-transform search (7 targets × 5 transforms × 5 folds): ~15-20 min
  - Per-target Optuna (30-50 trials × 5 folds per target × 7 targets): ~2.5-3.5 h
  - Final per-target refit + Maxwell + bandgap post-fit: ~10-15 min
  - Total: **~3-4 hours**

============================================================================
OUTPUTS  (under results/exp_chain_ext_lgbm_v2/)
============================================================================

  run.log             — full training log
  oof.csv             — OOF predictions after Maxwell + bandgap post-processing
  submission.csv      — Kaggle format id, target
  cv_summary.json     — per-target R², Optuna best hparams, transform picks,
                        Maxwell params, bandgap blend weights, config

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chain_ext_lgbm_v2.py

============================================================================
EXPECTED
============================================================================

vs chain-ext v1 (LB 0.894):
  - Optuna tune:            +0.005 to +0.012 mean OOF
  - Target transform:       +0.002 to +0.005 (mostly on skewed targets)
  - Nc regression fix:      +0.013 on nc directly (+0.002 on mean)
  - Bandgap post-processor: +0.005 to +0.015 on Ei/Eea/Egc/Egb collectively
  - Compound (with diminishing returns): +0.005 to +0.015 mean OOF

Expected LB solo: 0.898-0.906.  If we break 0.897 we're new best overall.

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
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # silence Optuna's info spam


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_chain_ext_lgbm_v2"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42
CHAIN_N_UNITS = 3

# Fingerprint sizes (match v1 exactly)
MORGAN2_NBITS = 2048
MORGAN3_NBITS = 2048       # monomer only
ATOMPAIR_NBITS = 2048
TOPTORSION_NBITS = 2048    # monomer only
AVALON_NBITS = 512

# Optuna trial budget per target — small-data targets get fewer trials
OPTUNA_TRIALS = {
    "eea": 30, "egb": 30, "ei": 30, "eps": 30, "nc": 30,
    "egc": 50, "tg": 50,
}
OPTUNA_TIMEOUT_SECONDS = 60 * 45   # hard 45-min cap per target as a safety net
OPTUNA_STARTUP_TRIALS = 8          # random-search warmup before TPE kicks in

# LGB training bounds — used both by Optuna objective and by final refit
N_ESTIMATORS_MAX = 8000
EARLY_STOP_ROUNDS = 200
REFIT_ITER_MULTIPLIER = 1.10

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
# DATA + CANONICALIZATION
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
# POLYMER CHAIN EXTENSION  (verbatim from exp_chain_ext_lgbm.py)
# ============================================================================

def polymer_to_multimer(smi: str, n_units: int = CHAIN_N_UNITS) -> str:
    """Extend polymer *A* SMILES to n-mer chain (head-to-tail). Return canonical SMILES.
    Falls back to the original SMILES on any structural issue."""
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
# FEATURE COMPUTATION (per SMILES, works on both monomer AND trimer)
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def _mol(smi: str):
    return Chem.MolFromSmiles(_cap(smi))


def compute_rdkit_desc(smi: str) -> dict | None:
    m = _mol(smi)
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))


def _count_fp_to_arr(fp, nbits: int) -> np.ndarray:
    out = np.zeros(nbits, dtype=np.int32)
    for k, v in fp.GetNonzeroElements().items():
        out[k] = v
    return out


def compute_morgan_count(smi: str, radius: int, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(AllChem.GetHashedMorganFingerprint(m, radius, nBits=nbits), nbits)


def compute_maccs(smi: str) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(167, dtype=np.int8)
    return np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8)


def compute_atompair_count(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=nbits), nbits)


def compute_toptorsion_count(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=nbits), nbits)


def compute_avalon(smi: str, nbits: int) -> np.ndarray:
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int8)
    return np.array(pyAvalonTools.GetAvalonFP(m, nBits=nbits), dtype=np.int8)


def _sanitize_desc_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        med = df[c].median()
        if pd.isna(med): med = 0.0
        df[c] = df[c].fillna(med)
    for c in df.columns:
        lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
        if lo == hi: continue
        df[c] = df[c].clip(lo, hi)
    dropped = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=dropped)
    return df, dropped


def build_feature_bundle(canon_smiles: list[str], log: logging.Logger) -> dict:
    smis_mono = list(dict.fromkeys(canon_smiles))
    log.info(f"unique canonical SMILES: {len(smis_mono)}")

    log.info(f"generating {CHAIN_N_UNITS}-mer polymer SMILES...")
    t0 = time.time()
    smis_tri = [polymer_to_multimer(s, CHAIN_N_UNITS) for s in tqdm(smis_mono, desc=f"polymer→{CHAIN_N_UNITS}-mer", ncols=100)]
    n_extended = sum(1 for m, t in zip(smis_mono, smis_tri) if m != t)
    log.info(f"chain extension: {n_extended}/{len(smis_mono)} SMILES extended "
             f"({100*n_extended/len(smis_mono):.1f}%; rest kept as monomer)  time={time.time()-t0:.1f}s")

    parts, families_slice, cursor = [], {}, 0

    def _add(name: str, arr: np.ndarray):
        nonlocal cursor
        parts.append(arr)
        families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    log.info(f"MONOMER features")
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis_mono, desc="mono rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X = df_desc.values.astype(np.float32)
    _add("desc_mono", X)
    log.info(f"  desc_mono: {X.shape}  dropped={len(dropped)}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_mono, desc="mono morgan-r2", ncols=100)]).astype(np.float32)
    _add("morgan2c_mono", X)
    log.info(f"  morgan2c_mono: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis_mono, desc="mono morgan-r3", ncols=100)]).astype(np.float32)
    _add("morgan3c_mono", X)
    log.info(f"  morgan3c_mono: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_maccs(s) for s in tqdm(smis_mono, desc="mono maccs", ncols=100)]).astype(np.float32)
    _add("maccs_mono", X)
    log.info(f"  maccs_mono: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_mono, desc="mono atom-pair", ncols=100)]).astype(np.float32)
    _add("atompair_c_mono", X)
    log.info(f"  atompair_c_mono: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis_mono, desc="mono top-torsion", ncols=100)]).astype(np.float32)
    _add("toptorsion_c_mono", X)
    log.info(f"  toptorsion_c_mono: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_mono, desc="mono avalon", ncols=100)]).astype(np.float32)
    _add("avalon_mono", X)
    log.info(f"  avalon_mono: {X.shape}  time={time.time()-t0:.1f}s")

    log.info(f"TRIMER features (from {CHAIN_N_UNITS}-mer SMILES)")
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis_tri, desc="tri rdkit desc", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc_matrix(df_desc)
    X = df_desc.values.astype(np.float32)
    _add("desc_tri", X)
    log.info(f"  desc_tri: {X.shape}  dropped={len(dropped)}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_tri, desc="tri morgan-r2", ncols=100)]).astype(np.float32)
    _add("morgan2c_tri", X)
    log.info(f"  morgan2c_tri: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_maccs(s) for s in tqdm(smis_tri, desc="tri maccs", ncols=100)]).astype(np.float32)
    _add("maccs_tri", X)
    log.info(f"  maccs_tri: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_tri, desc="tri atom-pair", ncols=100)]).astype(np.float32)
    _add("atompair_c_tri", X)
    log.info(f"  atompair_c_tri: {X.shape}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    X = np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_tri, desc="tri avalon", ncols=100)]).astype(np.float32)
    _add("avalon_tri", X)
    log.info(f"  avalon_tri: {X.shape}  time={time.time()-t0:.1f}s")

    X_full = np.concatenate(parts, axis=1)
    log.info(f"FEATURE MATRIX TOTAL: {X_full.shape}  size≈{X_full.nbytes/1e6:.1f}MB")

    return {
        "X": X_full,
        "smiles_index": {s: i for i, s in enumerate(smis_mono)},
        "families_slice": families_slice,
        "n_extended": n_extended,
        "n_total_smiles": len(smis_mono),
    }


def slice_smiles_features(bundle: dict, canon_series: pd.Series, drop_trimer: bool = False) -> np.ndarray:
    """Return sliced feature matrix. If drop_trimer=True, only monomer columns are kept."""
    idx = canon_series.map(bundle["smiles_index"]).values
    if not drop_trimer:
        return bundle["X"][idx]
    # Only monomer families (name suffix "_mono")
    mono_cols = []
    for fam, sl in bundle["families_slice"].items():
        if fam.endswith("_mono"):
            mono_cols.extend(range(sl.start, sl.stop))
    return bundle["X"][idx][:, mono_cols]


# ============================================================================
# AUX MATRIX-COMPLETION FEATURES
# ============================================================================

def build_aux_lookup(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    lookup: dict[str, np.ndarray] = {}
    grouped = train_df.groupby("canon")
    for canon, g in tqdm(grouped, desc="build aux lookup", ncols=100, total=grouped.ngroups):
        row = empty.copy()
        for tt, gg in g.groupby("target_type"):
            if tt in TARGET_IDX:
                idx = TARGET_IDX[tt]
                row[idx] = float(gg["target"].mean())
                row[idx + N_TARGETS] = 1.0
        lookup[canon] = row
    return lookup


def aux_features_for_target(
    canon_series: pd.Series,
    target: str,
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


# ============================================================================
# CV
# ============================================================================

def group_kfold_splits(
    canon_arr: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


# ============================================================================
# TARGET TRANSFORMS  (each returns forward + inverse; fit on train fold only)
# ============================================================================

def _fit_identity(y: np.ndarray):
    return (lambda x: x, lambda x: x)


def _fit_log1p(y: np.ndarray):
    if y.min() < -1 + 1e-9:
        return None
    return (lambda x: np.log1p(x), lambda x: np.expm1(x))


def _fit_sqrt(y: np.ndarray):
    if y.min() < -1e-9:
        return None
    return (lambda x: np.sqrt(np.clip(x, 0, None)),
            lambda x: np.square(np.clip(x, 0, None)))


def _fit_yeo_johnson(y: np.ndarray):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(y.reshape(-1, 1))
    return (
        lambda x: pt.transform(np.asarray(x).reshape(-1, 1)).ravel(),
        lambda x: pt.inverse_transform(np.asarray(x).reshape(-1, 1)).ravel(),
    )


def _fit_rank_gauss(y: np.ndarray):
    """Rank-Gauss: rank → uniform in (0,1) → inverse normal CDF. Store sorted training targets for inverse."""
    n = len(y)
    order = np.argsort(y)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    # Store sorted y for inverse mapping
    y_sorted = np.sort(y)

    def forward(x):
        # rank via searchsorted against training distribution
        x = np.asarray(x)
        r = np.searchsorted(y_sorted, x, side="right").astype(np.float64)
        u = (r - 0.5) / n
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return norm.ppf(u)

    def inverse(z):
        z = np.asarray(z)
        u = norm.cdf(z)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        idx = np.clip((u * n).astype(int), 0, n - 1)
        return y_sorted[idx]

    return (forward, inverse)


TRANSFORM_FITTERS = {
    "identity": _fit_identity,
    "log1p":    _fit_log1p,
    "sqrt":     _fit_sqrt,
    "yeojohnson": _fit_yeo_johnson,
    "rankgauss": _fit_rank_gauss,
}


# ============================================================================
# LGB HELPERS
# ============================================================================

def _default_lgb_params() -> dict:
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=10,
        feature_fraction=0.5,
        bagging_fraction=0.85,
        bagging_freq=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_depth=-1,
        verbosity=-1,
        n_jobs=-1,
        seed=SEED,
    )


def train_lgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    canons: np.ndarray,
    params: dict,
    n_estimators: int,
    log: logging.Logger | None = None,
    ctx: str = "",
) -> tuple[np.ndarray, list[int], list[float]]:
    """5-fold GroupKFold CV, return OOF preds (in TRANSFORMED space), best_iters, per-fold R² (transformed)."""
    splits = group_kfold_splits(canons, N_SPLITS, SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    best_iters = []
    fold_r2s = []
    for k, (tri, vai) in enumerate(splits):
        d_tr = lgb.Dataset(X[tri], y[tri])
        d_va = lgb.Dataset(X[vai], y[vai], reference=d_tr)
        booster = lgb.train(
            params, d_tr,
            num_boost_round=n_estimators,
            valid_sets=[d_va], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred_va = booster.predict(X[vai], num_iteration=booster.best_iteration)
        oof[vai] = pred_va
        best_iters.append(int(booster.best_iteration))
        fold_r2s.append(float(r2_score(y[vai], pred_va)))
    return oof, best_iters, fold_r2s


# ============================================================================
# STEP 1: PER-TARGET TARGET-TRANSFORM SEARCH
# ============================================================================

def select_best_transform(
    target: str,
    y_orig: np.ndarray,
    X: np.ndarray,
    canons: np.ndarray,
    log: logging.Logger,
) -> str:
    """Test each candidate transform via a quick 5-fold LGB CV (default hparams)
       and pick the one with highest R² in the ORIGINAL target space."""
    log.info(f"[{target}] transform search")
    results = {}
    params = _default_lgb_params()
    for name, fitter in TRANSFORM_FITTERS.items():
        pair = fitter(y_orig)
        if pair is None:
            log.info(f"  [{target}] {name:>11s}: skipped (target range not compatible)")
            continue
        fwd, inv = pair
        y_tr = fwd(y_orig)
        try:
            oof_tr, _, _ = train_lgb_cv(X, y_tr, canons, params, n_estimators=1500, ctx=f"{target}/{name}")
        except Exception as e:
            log.info(f"  [{target}] {name:>11s}: crashed ({e})")
            continue
        oof_orig = inv(oof_tr)
        r2 = float(r2_score(y_orig, oof_orig))
        results[name] = r2
        log.info(f"  [{target}] {name:>11s}: OOF R² (orig space) = {r2:.4f}")

    best = max(results, key=results.get)
    log.info(f"  [{target}] BEST transform: {best}  (R²={results[best]:.4f})")
    return best


# ============================================================================
# STEP 2: PER-TARGET OPTUNA HYPERPARAMETER TUNE
# ============================================================================

def tune_target_optuna(
    target: str,
    y_tr_transformed: np.ndarray,
    y_orig: np.ndarray,
    X: np.ndarray,
    canons: np.ndarray,
    inverse_transform,
    n_trials: int,
    log: logging.Logger,
) -> tuple[dict, float, list[int]]:
    """Optuna-tune LGB hparams for this target. Returns (best_params, best_oof_r2_orig, best_iters)."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 4, 60),
            feature_fraction=trial.suggest_float("feature_fraction", 0.25, 0.9),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 14),
            verbosity=-1,
            n_jobs=-1,
            seed=SEED,
        )
        try:
            oof_tr, _, _ = train_lgb_cv(X, y_tr_transformed, canons, params,
                                         n_estimators=N_ESTIMATORS_MAX, ctx=f"{target}/tune")
            oof_orig = inverse_transform(oof_tr)
            r2 = float(r2_score(y_orig, oof_orig))
        except Exception as e:
            log.info(f"  [{target}/tune] trial crashed: {e}")
            return -10.0
        return r2

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=OPTUNA_STARTUP_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    log.info(f"[{target}] Optuna: {n_trials} trials, timeout {OPTUNA_TIMEOUT_SECONDS}s")
    t0 = time.time()
    # tqdm callback for visible progress
    tbar = tqdm(total=n_trials, desc=f"[{target}] optuna", ncols=100)
    def tqdm_cb(study_, trial_):
        tbar.update(1)
        tbar.set_postfix(best=f"{study_.best_value:.4f}")
        if len(study_.trials) % 5 == 0:
            log.info(f"  [{target}/tune] trial {len(study_.trials):>3d}: best R² so far = {study_.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT_SECONDS,
                   callbacks=[tqdm_cb], gc_after_trial=True)
    tbar.close()

    best_params_search = study.best_params
    best_r2 = float(study.best_value)
    log.info(f"[{target}] Optuna DONE: {len(study.trials)} trials  best R² = {best_r2:.4f}  time={(time.time()-t0)/60:.1f}min")
    log.info(f"  best_params: {best_params_search}")

    # Re-run best hparams to grab best_iters for refit
    full_params = dict(
        objective="regression",
        metric="rmse",
        bagging_freq=1,
        verbosity=-1,
        n_jobs=-1,
        seed=SEED,
        **best_params_search,
    )
    _, best_iters, _ = train_lgb_cv(X, y_tr_transformed, canons, full_params,
                                     n_estimators=N_ESTIMATORS_MAX, ctx=f"{target}/final-cv")
    return full_params, best_r2, best_iters


# ============================================================================
# STEP 3: FINAL PER-TARGET PIPELINE
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    bundle: dict,
    aux_lookup: dict,
    log: logging.Logger,
) -> dict:
    """Full per-target pipeline: transform selection → Optuna tune → refit → test preds.
       For nc, drop trimer features (nc regression fix)."""
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y_orig = g_tr["target"].astype(float).values

    drop_trimer = (target == "nc")
    if drop_trimer:
        log.info(f"[{target}] NC-FIX: dropping trimer features for this target only")

    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"], drop_trimer=drop_trimer)
    X_te_smi = slice_smiles_features(bundle, g_te["canon"], drop_trimer=drop_trimer)
    X_tr_aux = aux_features_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_features_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)
    canons_tr = g_tr["canon"].values

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y_orig.min():.4f}, {y_orig.max():.4f}]   std={y_orig.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}")

    # (a) Target transform search
    best_transform_name = select_best_transform(target, y_orig, X_tr, canons_tr, log)
    fitter = TRANSFORM_FITTERS[best_transform_name]
    fwd, inv = fitter(y_orig)
    y_tr_transformed = fwd(y_orig)

    # (b) Optuna hparam tune (in transformed target space, evaluated in original space)
    n_trials = OPTUNA_TRIALS.get(target, 30)
    best_params, best_tune_r2, best_iters = tune_target_optuna(
        target, y_tr_transformed, y_orig, X_tr, canons_tr, inv, n_trials, log,
    )

    # (c) Re-run 5-fold to get OOF at the best hparams (in transformed space)
    oof_tr, best_iters_final, fold_r2s_transformed = train_lgb_cv(
        X_tr, y_tr_transformed, canons_tr, best_params,
        n_estimators=N_ESTIMATORS_MAX, ctx=f"{target}/final-cv",
    )
    oof_orig = inv(oof_tr)
    oof_r2 = float(r2_score(y_orig, oof_orig))
    log.info(f"[{target}] FINAL OOF R² (orig space) = {oof_r2:.4f}   "
             f"transform={best_transform_name}   best_iters median={int(np.median(best_iters_final))}")

    # (d) Refit on full train and predict test
    refit_iters = max(50, int(np.median(best_iters_final) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y_tr_transformed)
    full_booster = lgb.train(
        best_params, d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred_transformed = full_booster.predict(X_te)
    test_pred_orig = inv(test_pred_transformed)

    # Feature importance breakdown
    imp = full_booster.feature_importance(importance_type="gain")
    n_smi = X_tr_smi.shape[1]
    aux_gain = int(imp[n_smi:].sum())
    total_gain = int(imp.sum())
    # For mono/tri gain, only meaningful when trimer wasn't dropped
    family_gains: dict[str, int] = {}
    if drop_trimer:
        # All mono, no tri
        # Build synthetic family map for mono-only slice
        cursor = 0
        for fam, sl in bundle["families_slice"].items():
            if fam.endswith("_mono"):
                width = sl.stop - sl.start
                family_gains[fam] = int(imp[cursor:cursor + width].sum())
                cursor += width
        mono_gain = sum(v for v in family_gains.values())
        tri_gain = 0
    else:
        for fam, sl in bundle["families_slice"].items():
            family_gains[fam] = int(imp[sl].sum())
        mono_gain = sum(v for k, v in family_gains.items() if k.endswith("_mono"))
        tri_gain = sum(v for k, v in family_gains.items() if k.endswith("_tri"))

    log.info(f"[{target}] gain totals: mono={mono_gain} ({100*mono_gain/max(1,total_gain):.1f}%)  "
             f"tri={tri_gain} ({100*tri_gain/max(1,total_gain):.1f}%)  "
             f"aux={aux_gain} ({100*aux_gain/max(1,total_gain):.1f}%)")

    return {
        "target": target,
        "n_train": int(len(g_tr)),
        "n_test":  int(len(g_te)),
        "transform": best_transform_name,
        "best_params": best_params,
        "oof": pd.DataFrame({
            "canon":       g_tr["canon"].values,
            "target_type": target,
            "y_true":      y_orig,
            "y_pred":      oof_orig,
        }),
        "test_pred": pd.DataFrame({
            "id":          g_te["id"].values,
            "canon":       g_te["canon"].values,
            "target_type": target,
            "target":      test_pred_orig,
        }),
        "oof_r2":       oof_r2,
        "fold_r2s_transformed": fold_r2s_transformed,
        "best_iters":   best_iters_final,
        "refit_iters":  refit_iters,
        "family_gains": family_gains,
        "mono_gain_share": float(100 * mono_gain / max(1, total_gain)),
        "tri_gain_share":  float(100 * tri_gain / max(1, total_gain)),
        "aux_gain_share":  float(100 * aux_gain / max(1, total_gain)),
        "drop_trimer": drop_trimer,
    }


# ============================================================================
# STEP 4: MAXWELL POST-FIT  (identical to v1)
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
# STEP 5: BANDGAP CONSISTENCY POST-PROCESSOR
# ============================================================================

def bandgap_post_process(
    results: dict[str, dict],
    tr: pd.DataFrame,
    log: logging.Logger,
) -> dict:
    """Apply physics-based cross-target adjustments.

    Priors (Round-2 EDA):
      - Egc ≈ Ei − Eea      (chain bandgap = ionization energy − electron affinity)
      - Egb ≈ Egc           (r = 0.93 in train)
      - Ei  ≈ Egc + Eea     (rearrangement of Koopman)
      - Eea ≈ Egc − Ei      (rearrangement)

    For each of {Egc, Egb, Ei, Eea}:
      1. Compute the physics-implied prediction from the other targets' OOF/test preds.
         Only defined where the source targets have valid predictions for the same canon.
      2. Grid-search a blend weight w on OOF: y_final = w * y_lgb + (1 - w) * y_physics.
      3. If OOF R² improves by > 0.001, keep the blend; else keep original.
      4. Apply the same weight to test predictions.

    Returns a summary dict for logging.
    """
    log.info("=" * 60)
    log.info("BANDGAP CONSISTENCY POST-PROCESSOR")
    log.info("=" * 60)

    summary = {}

    # Build per-target canon → prediction lookups from OOF + train truth (train truth > OOF)
    # For OOF adjustment, we use OTHER targets' OOF predictions (honest — same folds' held-out estimates).
    # For test adjustment, we use OTHER targets' TEST predictions (already refit on full train).
    oof_by_target = {t: dict(zip(results[t]["oof"]["canon"], results[t]["oof"]["y_pred"]))
                     for t in TARGETS}
    tr_truth_by_target = {t: dict(zip(tr[tr["target_type"] == t]["canon"],
                                       tr[tr["target_type"] == t]["target"]))
                          for t in TARGETS}
    test_by_target = {t: dict(zip(results[t]["test_pred"]["canon"], results[t]["test_pred"]["target"]))
                      for t in TARGETS}

    def get_pred_for_canon(t: str, canon: str, from_oof: bool):
        """Return prediction of target `t` for `canon`, from OOF or from test.
           If canon is a train molecule with a true label for `t`, prefer that (except when computing OOF for `t` itself)."""
        if from_oof:
            return oof_by_target[t].get(canon, np.nan)
        # For test: prefer train truth if available, else refit test pred
        if canon in tr_truth_by_target[t]:
            return tr_truth_by_target[t][canon]
        return test_by_target[t].get(canon, np.nan)

    # Physics recipes: for each target, define (source_targets, combine_fn)
    physics_recipes = {
        "egc": (["ei", "eea"], lambda ei, eea: ei - eea),
        "egb": (["egc"],       lambda egc: egc),                   # Egb ≈ Egc
        "ei":  (["egc", "eea"], lambda egc, eea: egc + eea),
        "eea": (["egc", "ei"], lambda egc, ei: egc - ei),
    }

    for target, (src_targets, combine) in physics_recipes.items():
        log.info(f"[{target}] physics recipe: from {src_targets}")
        oof_df = results[target]["oof"].copy()
        y_true = oof_df["y_true"].values
        y_lgb  = oof_df["y_pred"].values

        # Compute physics prediction on OOF
        srcs = [
            np.array([get_pred_for_canon(s, c, from_oof=True) for c in oof_df["canon"]], dtype=float)
            for s in src_targets
        ]
        y_phys = combine(*srcs)

        # Where physics is NaN, fall back to LGB
        mask_nan = np.isnan(y_phys)
        n_phys_valid = int((~mask_nan).sum())
        y_phys_filled = np.where(mask_nan, y_lgb, y_phys)

        r2_lgb  = float(r2_score(y_true, y_lgb))
        r2_phys = float(r2_score(y_true, y_phys_filled))
        best_w, best_r2, _ = search_blend_weight(y_true, y_lgb, y_phys_filled)
        delta = best_r2 - r2_lgb

        log.info(f"  [{target}] n_phys_valid={n_phys_valid}/{len(oof_df)}   "
                 f"LGB R²={r2_lgb:.4f}   pure-physics R²={r2_phys:.4f}   "
                 f"best w_lgb={best_w:.3f}   blend R²={best_r2:.4f}   Δ={delta:+.4f}")

        if delta > 0.001:
            log.info(f"  [{target}] APPLY bandgap blend (Δ={delta:+.4f} > 0.001)")
            # Apply to OOF
            oof_df["y_pred"] = best_w * y_lgb + (1 - best_w) * y_phys_filled
            results[target]["oof"] = oof_df
            results[target]["oof_r2"] = best_r2

            # Apply to test
            test_df = results[target]["test_pred"].copy()
            srcs_test = [
                np.array([get_pred_for_canon(s, c, from_oof=False) for c in test_df["canon"]], dtype=float)
                for s in src_targets
            ]
            y_phys_test = combine(*srcs_test)
            mask_nan_test = np.isnan(y_phys_test)
            y_phys_test_filled = np.where(mask_nan_test, test_df["target"].values, y_phys_test)
            test_df["target"] = best_w * test_df["target"].values + (1 - best_w) * y_phys_test_filled
            results[target]["test_pred"] = test_df

            summary[target] = {
                "applied": True, "w_lgb": best_w, "r2_before": r2_lgb, "r2_after": best_r2,
                "delta": delta, "source_targets": src_targets, "n_phys_valid": n_phys_valid,
            }
        else:
            log.info(f"  [{target}] SKIP bandgap blend (Δ={delta:+.4f} ≤ 0.001)")
            summary[target] = {
                "applied": False, "w_lgb": None, "r2_before": r2_lgb, "r2_after": r2_lgb,
                "delta": delta, "source_targets": src_targets, "n_phys_valid": n_phys_valid,
            }

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} chain_n_units={CHAIN_N_UNITS} "
             f"morgan2={MORGAN2_NBITS} morgan3={MORGAN3_NBITS} atompair={ATOMPAIR_NBITS} "
             f"toptorsion={TOPTORSION_NBITS} avalon={AVALON_NBITS}")
    log.info(f"CV mode: aux-augmented + per-target-optuna + per-target-transform + nc-fix + bandgap-physics")
    log.info(f"Optuna trials per target: {OPTUNA_TRIALS}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = build_feature_bundle(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"feature families: {fam_str}")
    log.info(f"total SMILES features (with trimer): {bundle['X'].shape[1]}")

    log.info(f"building aux lookup over {tr['canon'].nunique()} unique canonical SMILES in train")
    aux_lookup = build_aux_lookup(tr)
    log.info(f"aux lookup built.  aux features per row: {2*N_TARGETS}")

    # Train 7 targets
    results: dict[str, dict] = {}
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        log.info("=" * 60)
        log.info(f"START TARGET: {tgt}")
        log.info("=" * 60)
        results[tgt] = train_one_target(tgt, tr, te, bundle, aux_lookup, log)

    pre_maxwell_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("PER-TARGET OOF R²  (pre-Maxwell, post-Optuna+transform+nc-fix)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}   "
                 f"transform={results[t]['transform']:>11s}   "
                 f"gain: mono={results[t]['mono_gain_share']:.1f}% tri={results[t]['tri_gain_share']:.1f}% aux={results[t]['aux_gain_share']:.1f}%")
    log.info(f"  MEAN R² (pre-Maxwell) = {pre_maxwell_mean:.4f}")

    # ==== Maxwell EPS↔Nc ====
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
    log.info(f"eps blend: LGB R²={baseline_r2_eps:.4f}  pure-Maxwell R²={r2_score(eps_oof['y_true'].values, eps_maxwell_oof):.4f}  "
             f"best w={best_w_eps:.3f}  blend R²={best_r2_eps:.4f}   Δ={best_r2_eps - baseline_r2_eps:+.4f}")

    # Nc blend
    nc_oof = results["nc"]["oof"].copy()
    eps_eff = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    nc_maxwell_oof = apply_maxwell_reverse(eps_eff, a_rev, b_rev)
    mask = np.isnan(nc_maxwell_oof)
    nc_maxwell_oof[mask] = nc_oof["y_pred"].values[mask]
    best_w_nc, best_r2_nc, baseline_r2_nc = search_blend_weight(
        nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_maxwell_oof
    )
    log.info(f"nc blend: LGB R²={baseline_r2_nc:.4f}  pure-Maxwell R²={r2_score(nc_oof['y_true'].values, nc_maxwell_oof):.4f}  "
             f"best w={best_w_nc:.3f}  blend R²={best_r2_nc:.4f}   Δ={best_r2_nc - baseline_r2_nc:+.4f}")

    # Apply Maxwell blend to OOF
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred"].values + (1 - best_w_eps) * eps_maxwell_oof
    results["eps"]["oof"] = eps_oof
    results["eps"]["oof_r2"] = best_r2_eps
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_maxwell_oof
    results["nc"]["oof"] = nc_oof
    results["nc"]["oof_r2"] = best_r2_nc

    # Apply Maxwell blend to test
    canon_to_nc_test = dict(zip(results["nc"]["test_pred"]["canon"], results["nc"]["test_pred"]["target"]))
    canon_to_eps_test = dict(zip(results["eps"]["test_pred"]["canon"], results["eps"]["test_pred"]["target"]))

    def get_nc_for_test(canon):
        if canon in canon_to_nc:
            tr_val = tr[(tr["canon"] == canon) & (tr["target_type"] == "nc")]["target"]
            if len(tr_val) > 0: return float(tr_val.mean())
        if canon in canon_to_nc_test:
            return float(canon_to_nc_test[canon])
        return float("nan")

    def get_eps_for_test(canon):
        if canon in canon_to_eps:
            tr_val = tr[(tr["canon"] == canon) & (tr["target_type"] == "eps")]["target"]
            if len(tr_val) > 0: return float(tr_val.mean())
        if canon in canon_to_eps_test:
            return float(canon_to_eps_test[canon])
        return float("nan")

    eps_test = results["eps"]["test_pred"].copy()
    nc_eff_test = np.array([get_nc_for_test(c) for c in eps_test["canon"]], dtype=float)
    eps_maxwell_test = apply_maxwell_forward(nc_eff_test, a_fwd, b_fwd)
    mask = np.isnan(eps_maxwell_test)
    eps_maxwell_test[mask] = eps_test["target"].values[mask]
    eps_test["target"] = best_w_eps * eps_test["target"].values + (1 - best_w_eps) * eps_maxwell_test
    results["eps"]["test_pred"] = eps_test

    nc_test = results["nc"]["test_pred"].copy()
    eps_eff_test = np.array([get_eps_for_test(c) for c in nc_test["canon"]], dtype=float)
    nc_maxwell_test = apply_maxwell_reverse(eps_eff_test, a_rev, b_rev)
    mask = np.isnan(nc_maxwell_test)
    nc_maxwell_test[mask] = nc_test["target"].values[mask]
    nc_test["target"] = best_w_nc * nc_test["target"].values + (1 - best_w_nc) * nc_maxwell_test
    results["nc"]["test_pred"] = nc_test

    post_maxwell_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info(f"MEAN R² (post-Maxwell, pre-bandgap) = {post_maxwell_mean:.4f}")

    # ==== Bandgap consistency ====
    bandgap_summary = bandgap_post_process(results, tr, log)

    post_bandgap_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R² (post-Maxwell + post-bandgap)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   R²={results[t]['oof_r2']:.4f}   transform={results[t]['transform']}")
    log.info(f"  MEAN R² (final) = {post_bandgap_mean:.4f}")
    log.info(f"  Pipeline lift: pre-Maxwell {pre_maxwell_mean:.4f} → post-Maxwell {post_maxwell_mean:.4f} → post-bandgap {post_bandgap_mean:.4f}")

    # ==== Write outputs ====
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

    per_target = {t: {
        "n_train":         results[t]["n_train"],
        "n_test":          results[t]["n_test"],
        "oof_r2":          results[t]["oof_r2"],
        "transform":       results[t]["transform"],
        "best_params":     results[t]["best_params"],
        "best_iters":      results[t]["best_iters"],
        "refit_iters":     results[t]["refit_iters"],
        "mono_gain_share": results[t]["mono_gain_share"],
        "tri_gain_share":  results[t]["tri_gain_share"],
        "aux_gain_share":  results[t]["aux_gain_share"],
        "drop_trimer":     results[t]["drop_trimer"],
    } for t in TARGETS}

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2_final":  post_bandgap_mean,
        "mean_r2_pre_maxwell":  pre_maxwell_mean,
        "mean_r2_post_maxwell": post_maxwell_mean,
        "per_target":     per_target,
        "chain_extension": {
            "n_units":          CHAIN_N_UNITS,
            "n_extended":       int(bundle["n_extended"]),
            "n_total_smiles":   int(bundle["n_total_smiles"]),
            "pct_extended":     float(100 * bundle["n_extended"] / max(1, bundle["n_total_smiles"])),
        },
        "maxwell": {
            "n_co_labeled":         int(len(co)),
            "forward_fit":          {"a": a_fwd, "b": b_fwd, "r2": r2_fwd},
            "reverse_fit":          {"a": a_rev, "b": b_rev, "r2_on_nc": r2_rev},
            "eps_blend":            {"baseline_r2": baseline_r2_eps, "best_w": best_w_eps, "best_r2": best_r2_eps},
            "nc_blend":             {"baseline_r2": baseline_r2_nc,  "best_w": best_w_nc,  "best_r2": best_r2_nc},
        },
        "bandgap_post_process": bandgap_summary,
        "config": {
            "n_splits":         N_SPLITS,
            "seed":             SEED,
            "chain_n_units":    CHAIN_N_UNITS,
            "morgan2_nbits":    MORGAN2_NBITS,
            "morgan3_nbits":    MORGAN3_NBITS,
            "atompair_nbits":   ATOMPAIR_NBITS,
            "toptorsion_nbits": TOPTORSION_NBITS,
            "avalon_nbits":     AVALON_NBITS,
            "n_estimators_max": N_ESTIMATORS_MAX,
            "early_stop":       EARLY_STOP_ROUNDS,
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "optuna_trials":    OPTUNA_TRIALS,
            "optuna_timeout_s": OPTUNA_TIMEOUT_SECONDS,
            "optuna_startup":   OPTUNA_STARTUP_TRIALS,
            "smiles_families":  {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_smiles_features": bundle["X"].shape[1],
            "n_aux_features":   2 * N_TARGETS,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Comparison vs chain-ext v1 reference
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (v2)  vs chain-ext v1 reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'v2':>10s}  {'v1 ref':>10s}  {'delta':>8s}  {'transform':>11s}")
    for t in TARGETS:
        r2 = results[t]["oof_r2"]
        ref = v1_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}  {results[t]['transform']:>11s}")
    v1_mean = float(np.mean(list(v1_ref.values())))
    log.info(f"  {'MEAN':>6s}  {post_bandgap_mean:>10.4f}  {v1_mean:>10.4f}  {post_bandgap_mean - v1_mean:>+8.4f}")
    log.info(f"  (chain-ext v1 LB reference: 0.894)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
