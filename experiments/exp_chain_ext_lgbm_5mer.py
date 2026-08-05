"""
exp_chain_ext_lgbm_5mer.py — v1 pipeline with 5-mer (pentamer) chain instead of 3-mer.

============================================================================
WHAT'S DIFFERENT FROM exp_chain_ext_lgbm.py (v1, LB 0.894)
============================================================================

**One-line change**: CHAIN_N_UNITS = 5 instead of 3. Everything else — feature
stack, LGB hparams, folds, Maxwell prior, aux — is byte-for-byte identical to v1.

Rationale: v1 arbitrarily picked trimer (3 units). Research doc §5.1 lists
"dimer/trimer/tetramer" as chain-extension options — length wasn't optimized.
Pentamer (5 units) gives 2× more repeat-unit context, which may capture
longer-range backbone patterns (regularity, stacking, backbone rigidity)
that trimer misses. This is a pure ADDITIVE change — no OOF selection,
no ensembling risk.

Same 5-fold split_seed=42 as v1 so this OOF can be per-target NNLS-blended
with existing bases (Chemprop 3-seed, mono LGB, etc).

Trade-offs vs trimer:
  + More chain context → potentially better polymer-level features
  + Same LGB compute (same feature dim, same rows)
  − Featurization ~1.5-2× slower (bigger molecules → RDKit desc + FP slower)
  − Larger molecules → more RDKit desc overflow warnings (harmless, sanitizer clips)
  − Diminishing returns possible: 5-mer might not add much over 3-mer if v1 already
    captured the chain-level signal that mattered.

============================================================================
ORIGINAL v1 DESCRIPTION FOLLOWS  (feature stack, method, etc)
============================================================================

Extends each polymer `*A*` SMILES to a n-mer `*AAAAA*` (5 units chained
head-to-tail), computes fingerprints + descriptors on BOTH the monomer AND
the n-mer, and concatenates as separate feature families in a LightGBM
pipeline. Everything else (GroupKFold, aux matrix-completion features,
Maxwell EPS↔Nc post-fit) is identical to `exp_maxwell_prior_lgbm.py`
(current best LGB base, LB 0.860).

Why: polymer physical properties (bulk MolWt, chain flexibility, TPSA,
LogP, BertzCT, Kier-Hall indices) are properties of the CHAIN, not the
monomer. A trimer's descriptors are structurally closer to the actual
polymer's bulk chemistry. Morgan fingerprints on a trimer pick up bit
patterns that only form across two adjacent monomer units (backbone-
backbone junctions, side-chain-side-chain interactions from repeat units).

3rd place NeurIPS Open Polymer Prediction 2025 used chain extension. The
Open Polymer Challenge post-competition report (arXiv 2512.08896)
specifically calls out that "Mordred descriptors computed on dimers often
perform as well as or better than those computed on monomers, likely because
dimers encode additional information including inter-monomer relationships."

============================================================================
FEATURE STACK
============================================================================

Monomer (matches exp_maxwell_prior_lgbm.py exactly for direct comparability):
  - RDKit 2D descriptors        (~207)
  - Morgan-r2 count FP          (2048)
  - Morgan-r3 count FP          (2048)
  - MACCS keys                  (167)
  - Atom-pair count FP          (2048)
  - Topological-torsion count FP (2048)
  - Avalon FP                   (512)
  = ~9,078 monomer features

Trimer (streamlined — drops the weak morgan3c + toptorsion families
  per earlier diagnostics; those contributed <5% gain on monomer):
  - RDKit 2D descriptors        (~207)
  - Morgan-r2 count FP          (2048)
  - MACCS keys                  (167)
  - Atom-pair count FP          (2048)
  - Avalon FP                   (512)
  = ~4,982 trimer features

Plus 14 aux cross-target features (matrix completion, same as maxwell_prior).

Grand total: ~14,074 features per row.

============================================================================
DEPENDENCIES
============================================================================

  Data: ppp-round-2/{train,test}.csv
  Venv: poly2-venv with rdkit, lightgbm, sklearn, tqdm

============================================================================
OUTPUTS  (under results/exp_chain_ext_lgbm/)
============================================================================

  run.log            — training logs + Maxwell fit stats + per-family gain
  oof.csv            — OOF predictions after Maxwell blend
  submission.csv     — Kaggle format id, target
  cv_summary.json    — per-target R², Maxwell params, blend weights, config
  feature_cache.pkl  — reusable feature bundle (excluded from git via *.pkl)

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chain_ext_lgbm.py

============================================================================
WALL TIME
============================================================================

~45-60 min on Mac M-series CPU:
  - Featurize: ~10 min (mono ~5 min + trimer ~5 min)
  - LGB training × 7 targets × 5 folds: ~25-40 min (more features = slower)
  - Maxwell post-fit: <30 sec

============================================================================
EXPECTED
============================================================================

Vs exp_maxwell_prior_lgbm (LB 0.860):
  - OOF gain from trimer features: +0.005 to +0.015 on tg/egc (targets with
    SMILES-heavy skill; small-data 5-pack should see modest lift)
  - Solo LB: 0.862-0.870

Vs exp_blend_nnls_3seed (current best LB 0.897), when re-blended:
  - New 2-way blend with 3-seed Chemprop + this LGB: expected LB
    0.898-0.902 → rank 2-4
"""
from __future__ import annotations

# --- stdlib ---
import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

# --- third-party ---
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_chain_ext_lgbm_5mer"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42

# Number of monomer units in the chain extension  (5-mer / pentamer vs v1's 3-mer)
CHAIN_N_UNITS = 5

# Fingerprint sizes
MORGAN2_NBITS = 2048
MORGAN3_NBITS = 2048       # monomer only
ATOMPAIR_NBITS = 2048
TOPTORSION_NBITS = 2048    # monomer only
AVALON_NBITS = 512

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    feature_fraction=0.5,
    bagging_fraction=0.85,
    bagging_freq=1,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=-1,
    seed=SEED,
)
N_ESTIMATORS = 4000
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
# POLYMER CHAIN EXTENSION
# ============================================================================

def polymer_to_multimer(smi: str, n_units: int = CHAIN_N_UNITS) -> str:
    """Extend polymer *A* SMILES to n-mer chain (head-to-tail). Return canonical SMILES.

    Falls back to the original SMILES on any structural issue so featurization
    always succeeds. Handles the standard 2-wildcard polymer convention.

    For SMILES with != 2 wildcard atoms (rare edge cases), returns the input
    unchanged.
    """
    if n_units <= 1:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi

    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        return smi  # non-standard polymer, cannot extend

    star_a, star_b = stars
    a_bonds = m.GetAtomWithIdx(star_a).GetBonds()
    b_bonds = m.GetAtomWithIdx(star_b).GetBonds()
    if len(a_bonds) != 1 or len(b_bonds) != 1:
        return smi

    connect_a = a_bonds[0].GetOtherAtomIdx(star_a)
    connect_b = b_bonds[0].GetOtherAtomIdx(star_b)
    bond_type_a = a_bonds[0].GetBondType()
    bond_type_b = b_bonds[0].GetBondType()

    # Remove star atoms → "core" monomer
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
        return smi  # weird case: monomer had only 2 atoms both being *

    # Build n-mer by chaining copies
    result = Chem.RWMol(core)
    prev_cb = cb          # end of previous unit
    first_ca = ca         # start of first unit (kept for terminal *)
    for i in range(1, n_units):
        result = Chem.RWMol(Chem.CombineMols(result, core))
        offset = result.GetNumAtoms() - n_atoms_core
        new_ca = offset + ca
        new_cb = offset + cb
        # Use bond_type_a for the inter-unit bond (matches the original * bond)
        result.AddBond(prev_cb, new_ca, bond_type_a)
        prev_cb = new_cb

    # Add terminal * atoms at both ends
    left_star = result.AddAtom(Chem.Atom(0))
    right_star = result.AddAtom(Chem.Atom(0))
    result.AddBond(first_ca, left_star, bond_type_a)
    result.AddBond(prev_cb, right_star, bond_type_b)

    try:
        final = result.GetMol()
        Chem.SanitizeMol(final)
        return Chem.MolToSmiles(final, canonical=True)
    except Exception:
        return smi  # graceful fallback


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

    # -------- Compute trimer SMILES --------
    log.info(f"generating {CHAIN_N_UNITS}-mer polymer SMILES...")
    t0 = time.time()
    smis_tri = [polymer_to_multimer(s, CHAIN_N_UNITS) for s in tqdm(smis_mono, desc=f"polymer→{CHAIN_N_UNITS}-mer", ncols=100)]
    # Sanity: count how many actually extended vs fell back
    n_extended = sum(1 for m, t in zip(smis_mono, smis_tri) if m != t)
    log.info(f"chain extension: {n_extended}/{len(smis_mono)} SMILES extended "
             f"({100*n_extended/len(smis_mono):.1f}%; rest kept as monomer)  time={time.time()-t0:.1f}s")

    parts, families_slice, cursor = [], {}, 0

    def _add(name: str, arr: np.ndarray):
        nonlocal cursor
        parts.append(arr)
        families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    # -------- MONOMER features (matching exp_maxwell_prior_lgbm exactly) --------
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

    # -------- TRIMER features (streamlined: drop weak families morgan3c + toptorsion) --------
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


def get_or_build_features(all_canon: list[str], log: logging.Logger) -> dict:
    key = hashlib.md5(
        (str(sorted(set(all_canon))) +
         f"n={CHAIN_N_UNITS};m2={MORGAN2_NBITS};m3={MORGAN3_NBITS};"
         f"ap={ATOMPAIR_NBITS};tt={TOPTORSION_NBITS};av={AVALON_NBITS}"
         ).encode()
    ).hexdigest()[:12]

    if FEATURE_CACHE_PATH.exists():
        try:
            with open(FEATURE_CACHE_PATH, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded feature cache: {FEATURE_CACHE_PATH.name}  key={key}")
                return bundle
            log.info(f"feature cache key mismatch; rebuilding")
        except Exception as e:
            log.info(f"failed to load cache ({e}); rebuilding")

    bundle = build_feature_bundle(all_canon, log)
    bundle["_key"] = key
    with open(FEATURE_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached features to {FEATURE_CACHE_PATH.name}")
    return bundle


def slice_smiles_features(bundle: dict, canon_series: pd.Series) -> np.ndarray:
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# ============================================================================
# AUX FEATURES (matrix completion — same as exp_maxwell_prior_lgbm)
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
# PER-TARGET TRAIN (identical to exp_maxwell_prior_lgbm)
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    bundle: dict,
    aux_lookup: dict,
    log: logging.Logger,
) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"])
    X_te_smi = slice_smiles_features(bundle, g_te["canon"])
    X_tr_aux = aux_features_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_features_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X shape train={X_tr.shape}, test={X_te.shape}")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    fold_bar = tqdm(splits, desc=f"[{target}] folds", ncols=100, leave=False)
    for k, (tri, vai) in enumerate(fold_bar):
        d_tr = lgb.Dataset(X_tr[tri], y[tri])
        d_va = lgb.Dataset(X_tr[vai], y[vai], reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS, d_tr,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[d_va], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred_va = booster.predict(X_tr[vai], num_iteration=booster.best_iteration)
        oof[vai] = pred_va
        best_iters.append(int(booster.best_iteration))
        r2 = r2_score(y[vai], pred_va)
        fold_r2s.append(float(r2))
        fold_bar.set_postfix(fold=k, best_iter=booster.best_iteration, r2=f"{r2:.4f}")
        log.info(f"[{target}] fold {k}: best_iter={booster.best_iteration:>4d}   "
                 f"R²={r2:.4f}   n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² (LGB only, pre-Maxwell) = {oof_r2:.4f}   (fold mean {np.mean(fold_r2s):.4f})")

    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    log.info(f"[{target}] refitting on full train for {refit_iters} rounds")
    d_full = lgb.Dataset(X_tr, y)
    full_booster = lgb.train(
        LGB_PARAMS, d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred = full_booster.predict(X_te)

    # Per-family gain breakdown to see if trimer features are earning their spot
    imp = full_booster.feature_importance(importance_type="gain")
    n_smi = X_tr_smi.shape[1]
    family_gains: dict[str, int] = {}
    for fam, sl in bundle["families_slice"].items():
        family_gains[fam] = int(imp[sl].sum())
    aux_gain = int(imp[n_smi:].sum())
    total_gain = int(imp.sum())

    # Split into mono / tri / aux totals
    mono_gain = sum(v for k, v in family_gains.items() if k.endswith("_mono"))
    tri_gain = sum(v for k, v in family_gains.items() if k.endswith("_tri"))
    log.info(f"[{target}] gain totals: mono={mono_gain} ({100*mono_gain/max(1,total_gain):.1f}%)  "
             f"tri={tri_gain} ({100*tri_gain/max(1,total_gain):.1f}%)  "
             f"aux={aux_gain} ({100*aux_gain/max(1,total_gain):.1f}%)")

    # Log top-3 families overall for compactness
    ranked = sorted(family_gains.items(), key=lambda x: -x[1])
    top_str = "  ".join([f"{k}={100*v/max(1,total_gain):.1f}%" for k, v in ranked[:5]])
    log.info(f"[{target}] top-5 families by gain: {top_str}")

    return {
        "target": target,
        "n_train": int(len(g_tr)),
        "n_test":  int(len(g_te)),
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
        "oof_r2":       oof_r2,
        "fold_r2s":     fold_r2s,
        "best_iters":   best_iters,
        "refit_iters":  refit_iters,
        "family_gains": family_gains,
        "mono_gain_share": float(100 * mono_gain / max(1, total_gain)),
        "tri_gain_share":  float(100 * tri_gain / max(1, total_gain)),
        "aux_gain_share":  float(100 * aux_gain / max(1, total_gain)),
    }


# ============================================================================
# MAXWELL POST-FIT (identical to exp_maxwell_prior_lgbm)
# ============================================================================

def fit_maxwell_forward(nc_values: np.ndarray, eps_values: np.ndarray) -> tuple[float, float, float]:
    x = nc_values ** 2
    y = eps_values
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    r2 = r2_score(y, a * x + b)
    return float(a), float(b), float(r2)


def fit_maxwell_reverse(eps_values: np.ndarray, nc_values: np.ndarray) -> tuple[float, float, float]:
    x = eps_values
    y = nc_values ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred_nc = np.sqrt(np.clip(a * x + b, 1e-9, None))
    r2 = r2_score(nc_values, pred_nc)
    return float(a), float(b), float(r2)


def apply_maxwell_forward(nc_values: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (nc_values ** 2) + b


def apply_maxwell_reverse(eps_values: np.ndarray, a: float, b: float) -> np.ndarray:
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
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} chain_n_units={CHAIN_N_UNITS} "
             f"morgan2={MORGAN2_NBITS} morgan3={MORGAN3_NBITS} atompair={ATOMPAIR_NBITS} "
             f"toptorsion={TOPTORSION_NBITS} avalon={AVALON_NBITS} early_stop={EARLY_STOP_ROUNDS}")
    log.info(f"CV mode: aux-augmented (aux lookup from full train, target slot masked)")
    log.info(f"Post-fit: Maxwell EPS↔Nc physics prior blend")
    log.info(f"LGB_PARAMS = {LGB_PARAMS}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"feature families: {fam_str}")
    log.info(f"total SMILES features: {bundle['X'].shape[1]}   "
             f"({bundle['n_extended']}/{bundle['n_total_smiles']} SMILES got chain-extended)")

    log.info(f"building aux lookup over {tr['canon'].nunique()} unique canonical SMILES in train")
    aux_lookup = build_aux_lookup(tr)
    log.info(f"aux lookup built.  aux features per row: {2*N_TARGETS}")

    # Train 7 targets
    results: dict[str, dict] = {}
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        results[tgt] = train_one_target(tgt, tr, te, bundle, aux_lookup, log)

    baseline_mean_r2 = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("LGB-only per-target OOF R²  (before Maxwell)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}   "
                 f"gain: mono={results[t]['mono_gain_share']:.1f}% tri={results[t]['tri_gain_share']:.1f}% aux={results[t]['aux_gain_share']:.1f}%")
    log.info(f"  MEAN R² (LGB only) = {baseline_mean_r2:.4f}")

    # Maxwell post-fit (same as maxwell_prior_lgbm)
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

    # Update OOF with blended predictions
    eps_oof["y_pred"] = best_w_eps * eps_oof["y_pred"].values + (1 - best_w_eps) * eps_maxwell_oof
    results["eps"]["oof"] = eps_oof
    results["eps"]["oof_r2"] = best_r2_eps
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_maxwell_oof
    results["nc"]["oof"] = nc_oof
    results["nc"]["oof_r2"] = best_r2_nc

    # Apply Maxwell to test predictions
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

    # Write outputs
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
        "fold_r2s":        results[t]["fold_r2s"],
        "best_iters":      results[t]["best_iters"],
        "refit_iters":     results[t]["refit_iters"],
        "family_gains":    results[t]["family_gains"],
        "mono_gain_share": results[t]["mono_gain_share"],
        "tri_gain_share":  results[t]["tri_gain_share"],
        "aux_gain_share":  results[t]["aux_gain_share"],
    } for t in TARGETS}
    mean_r2 = float(np.mean([per_target[t]["oof_r2"] for t in TARGETS]))

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        mean_r2,
        "mean_r2_lgb_only_pre_maxwell": baseline_mean_r2,
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
        "config": {
            "n_splits":         N_SPLITS,
            "seed":             SEED,
            "chain_n_units":    CHAIN_N_UNITS,
            "morgan2_nbits":    MORGAN2_NBITS,
            "morgan3_nbits":    MORGAN3_NBITS,
            "atompair_nbits":   ATOMPAIR_NBITS,
            "toptorsion_nbits": TOPTORSION_NBITS,
            "avalon_nbits":     AVALON_NBITS,
            "n_estimators":     N_ESTIMATORS,
            "early_stop":       EARLY_STOP_ROUNDS,
            "refit_multiplier": REFIT_ITER_MULTIPLIER,
            "lgb_params":       LGB_PARAMS,
            "smiles_families":  {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_smiles_features": bundle["X"].shape[1],
            "n_aux_features":   2 * N_TARGETS,
            "n_features_total": bundle["X"].shape[1] + 2 * N_TARGETS,
            "cv_mode":          "aux-augmented",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Final comparison vs chain-ext v1 (3-mer) reference — LB 0.894
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (5-mer + Maxwell)  vs chain-ext v1 (3-mer) reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'5-mer':>10s}  {'3-mer v1':>10s}  {'delta':>8s}  {'chain gain':>11s}")
    for t in TARGETS:
        r2 = per_target[t]["oof_r2"]
        ref = v1_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}  {per_target[t]['tri_gain_share']:>10.1f}%")
    v1_mean = float(np.mean(list(v1_ref.values())))
    log.info(f"  {'MEAN':>6s}  {mean_r2:>10.4f}  {v1_mean:>10.4f}  {mean_r2 - v1_mean:>+8.4f}")
    log.info(f"  (chain-ext v1 3-mer LB reference: 0.894)")
    log.info(f"  Note: 'chain gain %' column is the pentamer feature gain share (was named 'tri' in v1)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
