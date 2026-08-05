"""
exp_chain_ext_lgbm_v3.py — chain-ext LGB with ADDITIVE-only upgrades.

============================================================================
WHY THIS EXISTS  (learned from v2's failure)
============================================================================

v2 tried to lift chain-ext v1 (LB 0.894) via per-target Optuna + transform
search + nc-fix + bandgap physics. Result: OOF 0.878 (+0.011) but LB 0.868
(-0.026). The OOF-LB gap flipped from +0.028 → -0.010, a 0.038 swing.

Root cause: chain-ext v1's LB win came from being *under-fit* on OOF — the
positive gap was real generalization slack. Optuna's OOF-maximization
consumed that slack. Compounded by transform-search selection bias and
yeojohnson/rankgauss per-fold-fit leaks.

**Rule for v3: ADDITIVE-only upgrades. NO OOF selection.**
  ✅ ADDITIVE (adds signal, doesn't optimize OOF metric directly):
     bagging, more data, more features, cross-target propagation,
     physics-based post-processors with grid-searched weights.
  ❌ SELECTIVE (picks between alternatives on OOF metric):
     Optuna hparam search, target transform search, feature selection.

============================================================================
WHAT v3 ADDS TO v1
============================================================================

Same feature stack, same LGB hparams, same 5-fold GroupKFold, same Maxwell
prior. Additions:

  (1) Nc-fix — drop trimer features for nc target only
      Chain-ext v1 log showed nc regressed -0.013 vs mono-only. Direct fix.
      Other 6 targets keep the full mono+trimer stack.

  (2) IterativeImputer for aux features  (research doc §7.1)
      v1 aux = 14 sparse features (7 values + 7 masks) from single-pass
      lookup. Most cells are NaN.
      v3 aux = same 14 features but with IterativeImputer(LGBMRegressor)
      filling the NaNs via 5-round cross-target propagation. Per fold, the
      val fold's target values are held out from the imputation input, so
      no leakage.
      Expected: +0.005 to +0.015 on 5-pack targets (eps/nc/egb).

  (3) 15 domain-knowledge features  (research doc §5.2)
      Hand-engineered features that go BEYOND RDKit descriptors:
        - F/Cl/Br atom counts (fluorinated polymers = low ε)
        - F/C ratio, F/heavy ratio, halogen fraction
        - Backbone atom count (shortest path between wildcards)
        - Backbone aromatic fraction
        - Backbone sp² fraction
        - Sidechain heavy atom count
        - Rotatable bond / heavy atom ratio (chain flexibility → Tg)
        - HBD/HBA ratio
        - Vinyl-polymer SMARTS flag
        - Condensation-polymer SMARTS flag
      All computed on the MONOMER SMILES (with * atoms).
      Expected: +0.003 to +0.008.

  (4) Bandgap consistency post-processor  (research doc §3.2 + v2-validated)
      Applied to egb + ei only (the two v2 confirmed):
        pred_egb ← w_egb * pred_egb + (1 - w_egb) * pred_egc      (r=0.93)
        pred_ei  ← w_ei  * pred_ei  + (1 - w_ei)  * (pred_egc + pred_eea)
      Weight is grid-searched on OOF; applied only if OOF Δ > 0.001.
      Skipped for egc/eea in v3 (v2 log showed they gained +0.0006 and
      +0.0000 respectively — not worth the risk).

Skipped from v2:
  ❌ Per-target Optuna (destroyed OOF-LB gap)
  ❌ Per-target target transform search (compound selection bias + leaks)

Also NOT included this pass (potential v4 additions if v3 wins):
  - 3-seed LGB bagging  (user explicitly deferred)
  - 5-mer chain extension (larger compute)
  - PI1M pseudo-labeling (Round 1 failed at this, high risk)
  - Bicerano-style Tg prior as separate base signal

============================================================================
DEPENDENCIES
============================================================================

  Data: ppp-round-2/{train,test}.csv
  Venv: poly2-venv with rdkit, lightgbm, sklearn, tqdm

============================================================================
OUTPUTS  (under results/exp_chain_ext_lgbm_v3/)
============================================================================

  run.log             — full training log
  oof.csv             — OOF predictions after Maxwell + bandgap post-proc
  submission.csv      — Kaggle format id, target
  cv_summary.json     — per-target R², Maxwell params, bandgap weights,
                        IterImputer stats, domain feature stats

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chain_ext_lgbm_v3.py

============================================================================
WALL TIME
============================================================================

Rough plan on Mac M-series CPU:
  - Featurize (mono + trimer + domain + IterImputer aux): ~30-45 min
      * mono/trimer: ~15 min (mostly RDKit desc on trimer — same as v1)
      * domain: ~5 min (backbone SMARTS + halogen counts)
      * IterativeImputer: 5 folds × 5 rounds × 7 columns of LGB = ~15-25 min
  - LGB per-target training × 7 targets × 5 folds: ~15-20 min (same as v1)
  - Maxwell + bandgap post-fit: <1 min
  - Total: ~50-70 min

============================================================================
EXPECTED
============================================================================

vs chain-ext v1 (LB 0.894):
  - Nc-fix:             +0.001 to +0.002
  - Bandgap egb+ei:     +0.002 (v2 log validated +0.004+0.004 OOF gain
                              → LB roughly half of OOF gain)
  - IterativeImputer:   +0.003 to +0.008 on 5-pack (biggest lift)
  - Domain features:    +0.002 to +0.005

Compound (diminishing returns since some overlap): +0.004 to +0.010 LB
Expected solo LB: 0.897-0.903. If blended with 3-seed Chemprop: 0.900-0.906.

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
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG  (matches v1 exactly for the parts we're keeping)
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_chain_ext_lgbm_v3"
EXP_DIR = REPO / "results" / EXP_NAME

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS = len(TARGETS)
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS = 5
SEED = 42
CHAIN_N_UNITS = 3

MORGAN2_NBITS = 2048
MORGAN3_NBITS = 2048
ATOMPAIR_NBITS = 2048
TOPTORSION_NBITS = 2048
AVALON_NBITS = 512

# LGB hparams — verbatim from v1 (proven to hit LB 0.894).  DO NOT tune.
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

# IterativeImputer config (kept lightweight — imputes only the 7-target matrix)
ITERIMP_MAX_ITER = 5
ITERIMP_LGB_ESTIMATORS = 100      # small LGB per column (7 columns × 5 iterations)
ITERIMP_LGB_LR = 0.1

# Bandgap post-processor — apply to these targets only (v2 validated)
BANDGAP_TARGETS = ("egb", "ei")
BANDGAP_MIN_DELTA = 0.001

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
# FEATURE COMPUTATION  (verbatim from v1)
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


# ============================================================================
# NEW: 15 DOMAIN-KNOWLEDGE FEATURES  (research doc §5.2)
# ============================================================================

VINYL_SMARTS = Chem.MolFromSmarts("[*]-[CH2]-[CH]-[*]")
ESTER_SMARTS = Chem.MolFromSmarts("C(=O)O")
AMIDE_SMARTS = Chem.MolFromSmarts("C(=O)N")


def compute_domain_features(smi: str) -> np.ndarray:
    """Compute 15 domain-knowledge features on the MONOMER SMILES (with *).
    Returns np.array of shape (15,) with float values (NaN not allowed for LGB
    is OK, but we use 0.0 for missing since these are additive counts/flags)."""
    out = np.zeros(15, dtype=np.float32)
    m_orig = Chem.MolFromSmiles(smi)
    if m_orig is None:
        return out
    m_capped = _mol(smi)   # * → C for descriptors
    if m_capped is None:
        return out

    heavy = m_capped.GetNumHeavyAtoms()
    if heavy < 1:
        return out

    # (1) F count
    f_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "F")
    out[0] = f_count
    # (2) Cl count
    cl_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "Cl")
    out[1] = cl_count
    # (3) Br count
    br_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "Br")
    out[2] = br_count
    # (4) F/C ratio
    c_count = sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "C")
    out[3] = f_count / max(1, c_count)
    # (5) F/heavy ratio
    out[4] = f_count / heavy
    # (6) halogen fraction
    halogen = f_count + cl_count + br_count + sum(1 for a in m_capped.GetAtoms() if a.GetSymbol() == "I")
    out[5] = halogen / heavy

    # Backbone: shortest path between the two * atoms in ORIGINAL SMILES
    stars = [a.GetIdx() for a in m_orig.GetAtoms() if a.GetSymbol() == "*"]
    backbone_atoms: list[int] = []
    if len(stars) == 2:
        try:
            path = Chem.GetShortestPath(m_orig, stars[0], stars[1])
            backbone_atoms = [i for i in path if m_orig.GetAtomWithIdx(i).GetSymbol() != "*"]
        except Exception:
            backbone_atoms = []
    # (7) backbone atom count
    out[6] = len(backbone_atoms)
    # (8) backbone aromatic fraction
    if backbone_atoms:
        n_arom = sum(1 for i in backbone_atoms if m_orig.GetAtomWithIdx(i).GetIsAromatic())
        out[7] = n_arom / len(backbone_atoms)
        # (9) backbone sp² fraction
        n_sp2 = sum(1 for i in backbone_atoms
                    if m_orig.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2)
        out[8] = n_sp2 / len(backbone_atoms)
    # (10) sidechain heavy atom count = heavy - backbone (approximate)
    n_heavy_orig = sum(1 for a in m_orig.GetAtoms() if a.GetSymbol() != "*")
    out[9] = max(0, n_heavy_orig - len(backbone_atoms))

    # (11) rotatable / heavy ratio
    try:
        rot = rdMolDescriptors.CalcNumRotatableBonds(m_capped)
    except Exception:
        rot = 0
    out[10] = rot / heavy

    # (12) HBD/HBA ratio
    try:
        hbd = rdMolDescriptors.CalcNumHBD(m_capped)
        hba = rdMolDescriptors.CalcNumHBA(m_capped)
    except Exception:
        hbd, hba = 0, 0
    out[11] = hbd / max(1, hba)

    # (13) vinyl polymer flag
    try:
        out[12] = 1.0 if VINYL_SMARTS is not None and m_orig.HasSubstructMatch(VINYL_SMARTS) else 0.0
    except Exception:
        out[12] = 0.0

    # (14) ester in backbone flag
    try:
        out[13] = 1.0 if ESTER_SMARTS is not None and m_capped.HasSubstructMatch(ESTER_SMARTS) else 0.0
    except Exception:
        out[13] = 0.0

    # (15) amide in backbone flag
    try:
        out[14] = 1.0 if AMIDE_SMARTS is not None and m_capped.HasSubstructMatch(AMIDE_SMARTS) else 0.0
    except Exception:
        out[14] = 0.0

    return out


DOMAIN_FEATURE_NAMES = (
    "F_count", "Cl_count", "Br_count",
    "F_C_ratio", "F_heavy_ratio", "halogen_frac",
    "backbone_atoms", "backbone_aromatic_frac", "backbone_sp2_frac",
    "sidechain_heavy",
    "rot_per_heavy",
    "HBD_HBA_ratio",
    "is_vinyl", "has_ester", "has_amide",
)


# ============================================================================
# FEATURE BUNDLE BUILDER
# ============================================================================

def build_feature_bundle(canon_smiles: list[str], log: logging.Logger) -> dict:
    smis_mono = list(dict.fromkeys(canon_smiles))
    log.info(f"unique canonical SMILES: {len(smis_mono)}")

    log.info(f"generating {CHAIN_N_UNITS}-mer polymer SMILES...")
    t0 = time.time()
    smis_tri = [polymer_to_multimer(s, CHAIN_N_UNITS) for s in tqdm(smis_mono, desc=f"polymer→{CHAIN_N_UNITS}-mer", ncols=100)]
    n_extended = sum(1 for m, t in zip(smis_mono, smis_tri) if m != t)
    log.info(f"chain extension: {n_extended}/{len(smis_mono)} SMILES extended  time={time.time()-t0:.1f}s")

    parts, families_slice, cursor = [], {}, 0

    def _add(name: str, arr: np.ndarray):
        nonlocal cursor
        parts.append(arr)
        families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    # -------- MONOMER features (matching v1 exactly) --------
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

    # -------- TRIMER features (matching v1) --------
    log.info(f"TRIMER features")
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

    # -------- NEW: 15 domain features on MONOMER --------
    log.info(f"DOMAIN features (15 hand-engineered)")
    t0 = time.time()
    X = np.stack([compute_domain_features(s) for s in tqdm(smis_mono, desc="domain features", ncols=100)]).astype(np.float32)
    _add("domain_mono", X)
    log.info(f"  domain_mono: {X.shape}  time={time.time()-t0:.1f}s")

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
    """Return sliced feature matrix. If drop_trimer=True, only monomer + domain columns kept."""
    idx = canon_series.map(bundle["smiles_index"]).values
    if not drop_trimer:
        return bundle["X"][idx]
    keep_cols = []
    for fam, sl in bundle["families_slice"].items():
        if fam.endswith("_mono"):
            keep_cols.extend(range(sl.start, sl.stop))
    return bundle["X"][idx][:, keep_cols]


# ============================================================================
# AUX  (raw single-pass — kept as base signal, then upgraded via IterImputer)
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
# NEW: ITERATIVE IMPUTATION FOR AUX  (research doc §7.1)
# ============================================================================
#
# Design:
#   - Input: per-target per-canon aux value matrix (n_canons × 7 targets),
#     NaN where target not labeled for that canon.
#   - IterativeImputer with LGBMRegressor iterates 5 rounds, using the OTHER
#     6 columns to predict each column, filling NaN with model predictions.
#   - Result: for every canon, a dense 7-target aux vector where missing
#     values are imputed cross-target-model predictions rather than NaN.
#   - Fold-safe: per fold, mask val rows' targets before imputation.
#
# NOTE ON SIGNAL: this replaces NaN in aux features with predicted values.
# LGB handles NaN natively so v1 was fine, but IterImputer's predictions
# often carry real cross-target signal (especially for the 5-pack: eps↔nc,
# egb↔egc, etc). Research doc §7.1 expects +0.005 to +0.015 on 5-pack.

def build_target_matrix(train_df: pd.DataFrame, all_canons: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (n_canons × 7) matrix of aux target values, NaN where unlabeled.
    canon_list is the ordered index of canonical SMILES."""
    canon_to_idx = {c: i for i, c in enumerate(all_canons)}
    matrix = np.full((len(all_canons), N_TARGETS), np.nan, dtype=np.float32)
    for _, row in train_df.iterrows():
        i = canon_to_idx.get(row["canon"])
        if i is None: continue
        t_idx = TARGET_IDX.get(row["target_type"])
        if t_idx is None: continue
        matrix[i, t_idx] = float(row["target"])
    return matrix, list(all_canons)


def iterimpute_target_matrix(
    base_matrix: np.ndarray,
    log: logging.Logger,
    ctx: str = "",
) -> np.ndarray:
    """Run IterativeImputer with LightGBM base to fill NaNs in a target matrix."""
    estimator = lgb.LGBMRegressor(
        n_estimators=ITERIMP_LGB_ESTIMATORS,
        learning_rate=ITERIMP_LGB_LR,
        num_leaves=31,
        min_child_samples=5,
        verbosity=-1,
        n_jobs=-1,
        random_state=SEED,
    )
    imp = IterativeImputer(
        estimator=estimator,
        max_iter=ITERIMP_MAX_ITER,
        initial_strategy="mean",
        imputation_order="ascending",
        random_state=SEED,
        verbose=0,
    )
    t0 = time.time()
    imputed = imp.fit_transform(base_matrix)
    log.info(f"  IterImputer {ctx}: {base_matrix.shape} → imputed in {time.time()-t0:.1f}s  "
             f"n_iter={imp.n_iter_}  n_iter_capped_at={ITERIMP_MAX_ITER}")
    return imputed.astype(np.float32)


def build_iterimputed_aux_per_fold(
    tr: pd.DataFrame, te: pd.DataFrame,
    all_canons: list[str],
    fold_of_canon: dict[str, int],
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    """Compute per-fold (and full-train) iterated aux features.

    Returns dict keyed by:
      - 'fold_k' for k in [0, N_SPLITS): (n_canons × 7) imputed matrix
        where the val fold's target values are held out from the fit.
      - 'full': (n_canons × 7) imputed matrix using ALL train labels
        (used for refit + test prediction).
    """
    base_matrix, _ = build_target_matrix(tr, all_canons)
    n_missing_by_target = np.isnan(base_matrix).sum(axis=0)
    log.info(f"aux target matrix: {base_matrix.shape}   "
             f"NaN per target: {dict(zip(TARGETS, n_missing_by_target.tolist()))}")

    out = {}
    # Per-fold imputations: mask val rows' target values
    for k in range(N_SPLITS):
        matrix_masked = base_matrix.copy()
        # For each canon, if it's in val fold k, mask its 7-target row
        for canon_idx, canon in enumerate(all_canons):
            if fold_of_canon.get(canon) == k:
                matrix_masked[canon_idx, :] = np.nan
        out[f"fold_{k}"] = iterimpute_target_matrix(matrix_masked, log, ctx=f"fold {k}")

    # Full-train imputation for refit + test predictions
    out["full"] = iterimpute_target_matrix(base_matrix, log, ctx="full-train")

    return out


def iter_aux_features_for_target(
    canon_series: pd.Series,
    target: str,
    iter_aux_matrix: np.ndarray,
    all_canons: list[str],
) -> np.ndarray:
    """Return (n_rows × 14) aux features using iterimputed values.

    Columns 0-6:  imputed target values (7)
    Columns 7-13: original mask indicators (7)  — preserved from single-pass
    The target's own slot is set to NaN (unknown at predict time).
    """
    canon_to_idx = {c: i for i, c in enumerate(all_canons)}
    t_idx = TARGET_IDX[target]
    values_part = np.zeros((len(canon_series), N_TARGETS), dtype=np.float32)
    for row_i, c in enumerate(canon_series):
        i = canon_to_idx.get(c)
        if i is None:
            values_part[row_i, :] = np.nan
        else:
            values_part[row_i, :] = iter_aux_matrix[i]
    # Mask target's own slot
    values_part[:, t_idx] = np.nan
    # Mask indicator: 1 if this cell was originally observed (imputer preserves observed values,
    # so ≠ nan in the imputed matrix means either observed or imputed — for simplicity we mark all as 1)
    mask_part = np.ones((len(canon_series), N_TARGETS), dtype=np.float32)
    mask_part[:, t_idx] = 0.0
    return np.concatenate([values_part, mask_part], axis=1)


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


def build_fold_of_canon(all_canons: list[str], n_splits: int = N_SPLITS, seed: int = SEED) -> dict[str, int]:
    """For each canon, which fold does it belong to? (train canons get their fold; test-only canons get fold -1)"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_canons))
    shuffled = [all_canons[i] for i in order]
    return {c: i % n_splits for i, c in enumerate(shuffled)}


# ============================================================================
# PER-TARGET TRAIN
# ============================================================================

def train_one_target(
    target: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    bundle: dict,
    iter_aux_by_fold: dict[str, np.ndarray],   # keys: 'fold_k' and 'full'
    all_canons: list[str],
    fold_of_canon: dict[str, int],
    log: logging.Logger,
) -> dict:
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    drop_trimer = (target == "nc")
    if drop_trimer:
        log.info(f"[{target}] NC-FIX: dropping trimer features for this target")

    X_tr_smi = slice_smiles_features(bundle, g_tr["canon"], drop_trimer=drop_trimer)
    X_te_smi = slice_smiles_features(bundle, g_te["canon"], drop_trimer=drop_trimer)

    log.info(f"[{target}] train rows={len(g_tr)}   test rows={len(g_te)}   "
             f"y range=[{y.min():.4f}, {y.max():.4f}]   std={y.std():.4f}")
    log.info(f"[{target}] X_smi shape train={X_tr_smi.shape}, test={X_te_smi.shape}")

    splits = group_kfold_splits(g_tr["canon"].values, N_SPLITS, SEED)

    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []
    fold_bar = tqdm(splits, desc=f"[{target}] folds", ncols=100, leave=False)
    for k, (tri, vai) in enumerate(fold_bar):
        # Per-fold IterImputer aux (val rows' targets were masked during imputation)
        aux_matrix_k = iter_aux_by_fold[f"fold_{k}"]
        X_tr_aux_k = iter_aux_features_for_target(g_tr["canon"], target, aux_matrix_k, all_canons)
        X_tr_k = np.concatenate([X_tr_smi, X_tr_aux_k], axis=1)

        d_tr = lgb.Dataset(X_tr_k[tri], y[tri])
        d_va = lgb.Dataset(X_tr_k[vai], y[vai], reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS, d_tr,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[d_va], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred_va = booster.predict(X_tr_k[vai], num_iteration=booster.best_iteration)
        oof[vai] = pred_va
        best_iters.append(int(booster.best_iteration))
        r2 = r2_score(y[vai], pred_va)
        fold_r2s.append(float(r2))
        fold_bar.set_postfix(fold=k, best_iter=booster.best_iteration, r2=f"{r2:.4f}")
        log.info(f"[{target}] fold {k}: best_iter={booster.best_iteration:>4d}   "
                 f"R²={r2:.4f}   n_val={len(vai)}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[{target}] OOF R² (LGB only, pre-Maxwell/bandgap) = {oof_r2:.4f}   "
             f"(fold mean {np.mean(fold_r2s):.4f})")

    # Refit on full train (use 'full' iterimputed aux)
    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    aux_matrix_full = iter_aux_by_fold["full"]
    X_tr_aux_full = iter_aux_features_for_target(g_tr["canon"], target, aux_matrix_full, all_canons)
    X_te_aux_full = iter_aux_features_for_target(g_te["canon"], target, aux_matrix_full, all_canons)
    X_tr_full = np.concatenate([X_tr_smi, X_tr_aux_full], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux_full], axis=1)

    log.info(f"[{target}] refitting on full train for {refit_iters} rounds  X_tr shape={X_tr_full.shape}")
    d_full = lgb.Dataset(X_tr_full, y)
    full_booster = lgb.train(
        LGB_PARAMS, d_full,
        num_boost_round=refit_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    test_pred = full_booster.predict(X_te)

    # Feature importance breakdown
    imp = full_booster.feature_importance(importance_type="gain")
    n_smi = X_tr_smi.shape[1]
    aux_gain = int(imp[n_smi:].sum())
    total_gain = int(imp.sum())
    family_gains: dict[str, int] = {}
    if drop_trimer:
        cursor = 0
        for fam, sl in bundle["families_slice"].items():
            if fam.endswith("_mono"):
                width = sl.stop - sl.start
                family_gains[fam] = int(imp[cursor:cursor + width].sum())
                cursor += width
    else:
        for fam, sl in bundle["families_slice"].items():
            family_gains[fam] = int(imp[sl].sum())
    mono_gain = sum(v for k, v in family_gains.items() if k.endswith("_mono") and not k.startswith("domain"))
    tri_gain = sum(v for k, v in family_gains.items() if k.endswith("_tri"))
    domain_gain = family_gains.get("domain_mono", 0)
    log.info(f"[{target}] gain totals: mono={mono_gain} ({100*mono_gain/max(1,total_gain):.1f}%)  "
             f"tri={tri_gain} ({100*tri_gain/max(1,total_gain):.1f}%)  "
             f"domain={domain_gain} ({100*domain_gain/max(1,total_gain):.1f}%)  "
             f"aux={aux_gain} ({100*aux_gain/max(1,total_gain):.1f}%)")

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
        "mono_gain_share":   float(100 * mono_gain / max(1, total_gain)),
        "tri_gain_share":    float(100 * tri_gain / max(1, total_gain)),
        "domain_gain_share": float(100 * domain_gain / max(1, total_gain)),
        "aux_gain_share":    float(100 * aux_gain / max(1, total_gain)),
        "drop_trimer":  drop_trimer,
    }


# ============================================================================
# MAXWELL POST-FIT (verbatim from v1)
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
# BANDGAP CONSISTENCY POST-PROCESSOR  (egb + ei only, v2-validated)
# ============================================================================

def bandgap_post_process(
    results: dict[str, dict],
    tr: pd.DataFrame,
    log: logging.Logger,
) -> dict:
    """Apply physics-based cross-target adjustments to egb and ei only."""
    log.info("=" * 60)
    log.info("BANDGAP CONSISTENCY POST-PROCESSOR  (egb + ei only)")
    log.info("=" * 60)

    summary = {}
    oof_by_target = {t: dict(zip(results[t]["oof"]["canon"], results[t]["oof"]["y_pred"]))
                     for t in TARGETS}
    tr_truth_by_target = {t: dict(zip(tr[tr["target_type"] == t]["canon"],
                                       tr[tr["target_type"] == t]["target"]))
                          for t in TARGETS}
    test_by_target = {t: dict(zip(results[t]["test_pred"]["canon"], results[t]["test_pred"]["target"]))
                      for t in TARGETS}

    def get_pred(t: str, canon: str, from_oof: bool):
        if from_oof:
            return oof_by_target[t].get(canon, np.nan)
        if canon in tr_truth_by_target[t]:
            return tr_truth_by_target[t][canon]
        return test_by_target[t].get(canon, np.nan)

    physics_recipes = {
        "egb": (["egc"],       lambda egc: egc),
        "ei":  (["egc", "eea"], lambda egc, eea: egc + eea),
    }

    for target in BANDGAP_TARGETS:
        src_targets, combine = physics_recipes[target]
        log.info(f"[{target}] physics recipe: from {src_targets}")
        oof_df = results[target]["oof"].copy()
        y_true = oof_df["y_true"].values
        y_lgb  = oof_df["y_pred"].values

        srcs = [
            np.array([get_pred(s, c, from_oof=True) for c in oof_df["canon"]], dtype=float)
            for s in src_targets
        ]
        y_phys = combine(*srcs)
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

        if delta > BANDGAP_MIN_DELTA:
            log.info(f"  [{target}] APPLY bandgap blend (Δ={delta:+.4f} > {BANDGAP_MIN_DELTA})")
            oof_df["y_pred"] = best_w * y_lgb + (1 - best_w) * y_phys_filled
            results[target]["oof"] = oof_df
            results[target]["oof_r2"] = best_r2

            test_df = results[target]["test_pred"].copy()
            srcs_test = [
                np.array([get_pred(s, c, from_oof=False) for c in test_df["canon"]], dtype=float)
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
            log.info(f"  [{target}] SKIP bandgap blend (Δ={delta:+.4f} ≤ {BANDGAP_MIN_DELTA})")
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
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} chain_n_units={CHAIN_N_UNITS}")
    log.info(f"CV mode: aux-augmented via IterativeImputer + nc-fix + bandgap-physics (egb+ei)")
    log.info(f"LGB_PARAMS = {LGB_PARAMS}")
    log.info(f"IterImputer: max_iter={ITERIMP_MAX_ITER} lgb_estimators={ITERIMP_LGB_ESTIMATORS} lr={ITERIMP_LGB_LR}")

    random.seed(SEED); np.random.seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).drop_duplicates().tolist()
    bundle = build_feature_bundle(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"feature families: {fam_str}")

    log.info("=" * 60)
    log.info("BUILDING ITERATIVELY IMPUTED AUX FEATURES (per-fold + full)")
    log.info("=" * 60)
    fold_of_canon = build_fold_of_canon(all_canon)
    iter_aux_by_fold = build_iterimputed_aux_per_fold(tr, te, all_canon, fold_of_canon, log)

    # Train 7 targets
    results: dict[str, dict] = {}
    tgt_bar = tqdm(TARGETS, desc="targets", ncols=100)
    for tgt in tgt_bar:
        tgt_bar.set_postfix(target=tgt)
        log.info("=" * 60)
        log.info(f"START TARGET: {tgt}")
        log.info("=" * 60)
        results[tgt] = train_one_target(tgt, tr, te, bundle, iter_aux_by_fold, all_canon, fold_of_canon, log)

    pre_maxwell_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("PER-TARGET OOF R²  (pre-Maxwell, post-nc-fix + IterImputer aux + domain feats)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   n={results[t]['n_train']:>5d}   R²={results[t]['oof_r2']:.4f}   "
                 f"gain: mono={results[t]['mono_gain_share']:.1f}% tri={results[t]['tri_gain_share']:.1f}% "
                 f"domain={results[t]['domain_gain_share']:.1f}% aux={results[t]['aux_gain_share']:.1f}%")
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

    post_maxwell_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info(f"MEAN R² (post-Maxwell, pre-bandgap) = {post_maxwell_mean:.4f}")

    # ==== Bandgap consistency (egb + ei only) ====
    bandgap_summary = bandgap_post_process(results, tr, log)

    post_bandgap_mean = float(np.mean([results[t]["oof_r2"] for t in TARGETS]))
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R² (post-Maxwell + post-bandgap)")
    log.info("=" * 60)
    for t in TARGETS:
        log.info(f"  {t:>4s}   R²={results[t]['oof_r2']:.4f}")
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
        "n_train":            results[t]["n_train"],
        "n_test":             results[t]["n_test"],
        "oof_r2":             results[t]["oof_r2"],
        "fold_r2s":           results[t]["fold_r2s"],
        "best_iters":         results[t]["best_iters"],
        "refit_iters":        results[t]["refit_iters"],
        "mono_gain_share":    results[t]["mono_gain_share"],
        "tri_gain_share":     results[t]["tri_gain_share"],
        "domain_gain_share":  results[t]["domain_gain_share"],
        "aux_gain_share":     results[t]["aux_gain_share"],
        "drop_trimer":        results[t]["drop_trimer"],
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
            "n_splits":             N_SPLITS,
            "seed":                 SEED,
            "chain_n_units":        CHAIN_N_UNITS,
            "morgan2_nbits":        MORGAN2_NBITS,
            "morgan3_nbits":        MORGAN3_NBITS,
            "atompair_nbits":       ATOMPAIR_NBITS,
            "toptorsion_nbits":     TOPTORSION_NBITS,
            "avalon_nbits":         AVALON_NBITS,
            "n_estimators":         N_ESTIMATORS,
            "early_stop":           EARLY_STOP_ROUNDS,
            "refit_multiplier":     REFIT_ITER_MULTIPLIER,
            "lgb_params":           LGB_PARAMS,
            "iterimp_max_iter":     ITERIMP_MAX_ITER,
            "iterimp_lgb_estimators": ITERIMP_LGB_ESTIMATORS,
            "iterimp_lgb_lr":       ITERIMP_LGB_LR,
            "bandgap_targets":      list(BANDGAP_TARGETS),
            "bandgap_min_delta":    BANDGAP_MIN_DELTA,
            "smiles_families":      {k: v.stop - v.start for k, v in bundle["families_slice"].items()},
            "n_smiles_features":    bundle["X"].shape[1],
            "n_aux_features":       2 * N_TARGETS,
            "n_domain_features":    15,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Comparison vs chain-ext v1 reference
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (v3)  vs chain-ext v1 reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'v3':>10s}  {'v1 ref':>10s}  {'delta':>8s}")
    for t in TARGETS:
        r2 = results[t]["oof_r2"]
        ref = v1_ref[t]
        d = r2 - ref
        log.info(f"  {t:>6s}  {r2:>10.4f}  {ref:>10.4f}  {d:>+8.4f}")
    v1_mean = float(np.mean(list(v1_ref.values())))
    log.info(f"  {'MEAN':>6s}  {post_bandgap_mean:>10.4f}  {v1_mean:>10.4f}  {post_bandgap_mean - v1_mean:>+8.4f}")
    log.info(f"  (chain-ext v1 LB reference: 0.894)")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
