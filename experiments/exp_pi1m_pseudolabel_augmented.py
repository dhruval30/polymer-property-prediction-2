"""
exp_pi1m_pseudolabel_augmented.py
=====================================================================
  PI1M pseudo-label augmentation of the LB-0.902 pipeline.
  Single-file, Kaggle-ready (T4/T4x2/L4 tested), ~4-6 hours end-to-end.
=====================================================================

THE IDEA (in one line)
----------------------
Use per-fold LGB teachers to pseudo-label ~20K filtered PI1M rows, add them
to Chemprop's training data with weight=0.25, retrain everything. Adds
graph-diversity that the current pipeline can't get from 5.9K labeled rows.

WHY PER-FOLD TEACHERS
---------------------
Round 1's pseudo-labeling failed because a global teacher leaked val-fold
labels through the pseudo-label predictions. Here, for each fold k, the
teacher is trained ONLY on fold k's train slice, then predicts PI1M. That
fold's student then trains on (fold_k_train + fold_k_pseudo). Val slice
stays pristine.

FOUR FILTERS ON PI1M PSEUDO-LABELS (all must fire)
--------------------------------------------------
1. Sample cap: 20K rows max (random subsample).
2. Confidence: cross-fold-teacher stdev < 0.3 * per-target train std.
3. Similarity: min-Tanimoto to train UNION ∈ [0.4, 0.9]
     (< 0.4 = teacher extrapolating; > 0.9 = redundant with training).
4. Sample weight: 0.25 on pseudo rows in student training.

CHECKPOINTING
-------------
Each phase writes to WORK_DIR/<phase>/ and skips if outputs exist.
Delete a phase directory to force re-run.

WALL-TIME BUDGET (Kaggle T4 GPU, single kernel)
-----------------------------------------------
  Phase 0  Load + canonicalize train/test/PI1M         5 min
  Phase 1  Feature matrices (mono LGB features)       20 min
  Phase 2  Morgan fingerprints for Tanimoto            5 min
  Phase 3  Per-fold LGB teachers × 7 targets          20 min
  Phase 4  Filter PI1M (stdev + Tanimoto)              3 min
  Phase 5  LGB student on augmented data              15 min
  Phase 6  Maxwell physics post-fit on LGB             2 min
  Phase 7  Chemprop 5-fold × 3-seed on aug'd data   ~150 min
  Phase 8  Chemprop 3-seed refit on aug'd full        30 min
  Phase 9  NNLS blend + Koopmans post-fit              1 min
  TOTAL                                             ~4-5 h
=====================================================================
"""
from __future__ import annotations

# --- stdlib ---
import gzip
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

# --- third-party ---
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import CombineMols
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from chemprop import data, featurizers, nn
from chemprop.models import MPNN

RDLogger.DisableLog("rdApp.*")


# =====================================================================
#                              CONFIG
# =====================================================================

ON_KAGGLE = Path("/kaggle/input").exists()

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                    # notebook kernel
    _HERE = Path.cwd()


def _find_data_dir() -> Path:
    candidates = []
    if ON_KAGGLE:
        root = Path("/kaggle/input")
        candidates += sorted(root.glob("*"))
        candidates += sorted(root.glob("*/*"))
    candidates += [_HERE.parent / "ppp-round-2", _HERE / "ppp-round-2", _HERE]
    for c in candidates:
        if c.is_dir() and (c / "train.csv").exists() and (c / "test.csv").exists():
            return c
    raise FileNotFoundError(
        "Could not find train.csv + test.csv. On Kaggle run "
        "`!ls -R /kaggle/input | head -50` and hardcode DATA_DIR."
    )


DATA_DIR       = _find_data_dir()
PI1M_PATH      = DATA_DIR / "PI1M.csv"
WORK_DIR       = (Path("/kaggle/working") if ON_KAGGLE else _HERE) / "work_pi1m_aug"
FINAL_SUB_PATH = Path("/kaggle/working/submission.csv") if ON_KAGGLE else _HERE / "submission_pi1m_aug.csv"

TARGETS      = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
N_TARGETS    = len(TARGETS)
TARGET_IDX   = {t: i for i, t in enumerate(TARGETS)}

N_SPLITS     = 5
SPLIT_SEED   = 42

# ---- LGB features (mono + trimer chain-ext, matches exp_chain_ext_catboost) ----
# Chain-extension: repeat each polymer *A* → *AAA* (trimer). Encodes inter-monomer
# junctions that pure monomer fingerprints can't see. Chain-ext LGB solo went
# 0.860 → 0.894 LB in Round 2, so this is a real +0.03 signal that also
# strengthens per-fold teachers → better PI1M pseudo-labels.
CHAIN_N_UNITS    = 3           # trimer
MORGAN2_NBITS    = 2048
MORGAN3_NBITS    = 2048
ATOMPAIR_NBITS   = 2048
TOPTORSION_NBITS = 2048
AVALON_NBITS     = 512
TANIMOTO_FP_NBITS = 2048       # Morgan-r2 bit fp for Tanimoto filter

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
    seed=SPLIT_SEED,
)
N_ESTIMATORS      = 4000
EARLY_STOP_ROUNDS = 200
REFIT_ITER_MULTIPLIER = 1.10
BLEND_W_GRID      = np.linspace(0.0, 1.0, 201)

# ---- PI1M pseudo-label config ----
PI1M_SAMPLE_CAP           = 20_000       # random subsample cap before filters
PI1M_SUBSAMPLE_SEED       = 42
CONFIDENCE_STDEV_FACTOR   = 0.30         # keep pseudo if stdev < 0.30 * train_std
TANIMOTO_LOW              = 0.40         # min sim to train union
TANIMOTO_HIGH             = 0.90         # max sim to train union
PSEUDO_WEIGHT             = 0.25         # sample weight for pseudo rows

# ---- Chemprop student ----
MODEL_SEEDS  = (42, 43, 44)              # 3-seed bag
D_H          = 300
DEPTH        = 4
MP_DROPOUT   = 0.05
FFN_HIDDEN   = 300
FFN_LAYERS   = 2
FFN_DROPOUT  = 0.05
BATCH_NORM   = True
MAX_EPOCHS   = 60
PATIENCE     = 10
BATCH_SIZE   = 64
GRAD_CLIP    = 1.0
LR_INIT      = 1e-3
LR_MAX       = 1e-3
LR_FINAL     = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS  = 0
DEVICE       = "gpu" if torch.cuda.is_available() else "cpu"

# ---- Blend + Koopmans (identical to reproduce.py) ----
CHEMPROP_WEIGHT_FLOOR = 0.40
APPLY_CHEMPROP_BIAS   = 0.15

PHYSICS_TARGETS = ("egc", "ei", "eea")
ALPHA_GRID      = np.arange(0.5, 1.001, 0.025)
PHYSICS_RECIPES = {
    "egc": ("ei",  "eea", lambda ei,  eea: ei  - eea),
    "ei":  ("egc", "eea", lambda egc, eea: egc + eea),
    "eea": ("ei",  "egc", lambda ei,  egc: ei  - egc),
}


# =====================================================================
#                            LOGGING
# =====================================================================

def setup_logging(work_dir: Path) -> logging.Logger:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "reproduce.log"
    logger = logging.getLogger("pi1m_aug")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout);       sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"logging to {log_path}")
    return logger


class EpochLogger(L.Callback):
    def __init__(self, logger, ctx: str):
        self.logger = logger; self.ctx = ctx; self.t_start = None
    def on_train_epoch_start(self, trainer, pl_module):
        if self.t_start is None: self.t_start = time.time()
        self._epoch_t = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch
        m = trainer.callback_metrics
        tl = float(m.get("train_loss", float("nan")))
        vl = float(m.get("val_loss", float("nan")))
        et = time.time() - self._epoch_t
        el = time.time() - self.t_start
        self.logger.info(f"[{self.ctx}] epoch {e:>3d}  "
                         f"train_loss={tl:.4f}  val_loss={vl:.4f}  "
                         f"epoch_time={et:.1f}s  elapsed={el/60:.1f}min")


# =====================================================================
#              DATA LOADING + CANONICALIZATION
# =====================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def _mol(smi: str):
    return Chem.MolFromSmiles(_cap(smi))


def polymer_to_multimer(smi: str, n_units: int = CHAIN_N_UNITS) -> str:
    """Extend *A* SMILES to n-mer chain head-to-tail. Returns canonical SMILES.
    Falls back to input SMILES on any structural issue (non-standard wildcards,
    parse failure, etc.) so featurization always succeeds."""
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

    def _adjust(orig_idx, removed_sorted):
        return orig_idx - sum(1 for r in removed_sorted if r < orig_idx)

    removed_sorted = sorted(stars)
    ca = _adjust(connect_a, removed_sorted)
    cb = _adjust(connect_b, removed_sorted)
    core = editable.GetMol()
    n_atoms_core = core.GetNumAtoms()
    if n_atoms_core == 0:
        return smi

    result = Chem.RWMol(core)
    prev_cb = cb
    first_ca = ca
    for _ in range(1, n_units):
        result = Chem.RWMol(CombineMols(result, core))
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


def load_train_test(log) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"loading train/test from {DATA_DIR}")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log.info(f"  train raw {tr.shape}  test raw {te.shape}")
    all_smi = pd.concat([tr["smiles"], te["smiles"]]).unique()
    log.info(f"  canonicalizing {len(all_smi)} unique raw SMILES")
    cmap = {s: canonical(s) for s in tqdm(all_smi, desc="canon(tr+te)", ncols=100)}
    tr["canon"] = tr["smiles"].map(cmap)
    te["canon"] = te["smiles"].map(cmap)
    tr = (tr.groupby(["canon", "target_type"], as_index=False)
            .agg(target=("target", "mean"), smiles=("smiles", "first")))
    log.info(f"  train after dedup {tr.shape}")
    return tr, te


def load_pi1m(log) -> pd.DataFrame:
    log.info(f"loading PI1M from {PI1M_PATH}")
    pi = pd.read_csv(PI1M_PATH)
    smi_col = "SMILES" if "SMILES" in pi.columns else pi.columns[0]
    pi = pi.rename(columns={smi_col: "smiles"})[["smiles"]]
    log.info(f"  PI1M raw {pi.shape}  (subsampling to {PI1M_SAMPLE_CAP})")

    rng = np.random.default_rng(PI1M_SUBSAMPLE_SEED)
    if len(pi) > PI1M_SAMPLE_CAP:
        idx = rng.choice(len(pi), size=PI1M_SAMPLE_CAP, replace=False)
        pi = pi.iloc[idx].reset_index(drop=True)
    log.info(f"  canonicalizing {len(pi)} PI1M SMILES")
    pi["canon"] = [canonical(s) for s in tqdm(pi["smiles"], desc="canon(PI1M)", ncols=100)]
    pi = pi.dropna(subset=["canon"]).drop_duplicates(subset=["canon"]).reset_index(drop=True)
    log.info(f"  PI1M after canon+dedup {pi.shape}")
    return pi


# =====================================================================
#            LGB FEATURE COMPUTATION (mono-only)
# =====================================================================

def compute_rdkit_desc(smi):
    m = _mol(smi)
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))


def _count_fp_to_arr(fp, nbits: int):
    out = np.zeros(nbits, dtype=np.int32)
    for k, v in fp.GetNonzeroElements().items():
        out[k] = v
    return out


def compute_morgan_count(smi, radius, nbits):
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(AllChem.GetHashedMorganFingerprint(m, radius, nBits=nbits), nbits)


def compute_maccs(smi):
    m = _mol(smi)
    if m is None: return np.zeros(167, dtype=np.int8)
    return np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8)


def compute_atompair_count(smi, nbits):
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=nbits), nbits)


def compute_toptorsion_count(smi, nbits):
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int32)
    return _count_fp_to_arr(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=nbits), nbits)


def compute_avalon(smi, nbits):
    m = _mol(smi)
    if m is None: return np.zeros(nbits, dtype=np.int8)
    return np.array(pyAvalonTools.GetAvalonFP(m, nBits=nbits), dtype=np.int8)


def _sanitize_desc(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        med = df[c].median()
        if pd.isna(med): med = 0.0
        df[c] = df[c].fillna(med)
    for c in df.columns:
        lo, hi = df[c].quantile(0.005), df[c].quantile(0.995)
        if lo == hi or not np.isfinite(lo) or not np.isfinite(hi): continue
        df[c] = df[c].clip(lo, hi)
    df = df.clip(-1e10, 1e10)
    dropped = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=dropped)
    return df, dropped


def build_lgb_features(canon_smiles: list[str], log) -> dict:
    """Compute mono + trimer feature stack for LGB.
      Mono   (7 families ~9078): desc + morgan-r2 + morgan-r3 + maccs + atompair + toptorsion + avalon
      Trimer (5 families ~4982): desc + morgan-r2 + maccs + atompair + avalon
                                 (drops weakest trimer families per prior diagnostics)
      Total ~14060 features per polymer."""
    smis_mono = list(dict.fromkeys(canon_smiles))
    log.info(f"unique canonical SMILES: {len(smis_mono)}")

    log.info(f"building {CHAIN_N_UNITS}-mer SMILES for chain extension")
    t0 = time.time()
    smis_tri = [polymer_to_multimer(s, CHAIN_N_UNITS)
                for s in tqdm(smis_mono, desc=f"→ {CHAIN_N_UNITS}-mer", ncols=100)]
    n_extended = sum(1 for m, t in zip(smis_mono, smis_tri) if m != t)
    log.info(f"  chain-extended {n_extended}/{len(smis_mono)} polymers "
             f"(others were unextendable, kept as monomer)  time={time.time()-t0:.1f}s")

    parts, families_slice, cursor = [], {}, 0
    def _add(name, arr):
        nonlocal cursor
        parts.append(arr); families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    # -------- MONO (all 7 families) --------
    log.info("computing MONO RDKit descriptors")
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis_mono, desc="mono rdkit", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc(df_desc)
    X = df_desc.values.astype(np.float32)
    _add("desc_mono", X)
    log.info(f"  desc_mono {X.shape} dropped={len(dropped)} time={time.time()-t0:.1f}s")

    for name, fn in [
        ("morgan2c_mono",     lambda: np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_mono, desc="mono morgan-r2", ncols=100)])),
        ("morgan3c_mono",     lambda: np.stack([compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis_mono, desc="mono morgan-r3", ncols=100)])),
        ("maccs_mono",        lambda: np.stack([compute_maccs(s) for s in tqdm(smis_mono, desc="mono maccs", ncols=100)])),
        ("atompair_c_mono",   lambda: np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_mono, desc="mono atompair", ncols=100)])),
        ("toptorsion_c_mono", lambda: np.stack([compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis_mono, desc="mono toptorsion", ncols=100)])),
        ("avalon_mono",       lambda: np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_mono, desc="mono avalon", ncols=100)])),
    ]:
        t0 = time.time()
        X = fn().astype(np.float32)
        _add(name, X)
        log.info(f"  {name} {X.shape} time={time.time()-t0:.1f}s")

    # -------- TRIMER (5 strongest families, drops weak morgan-r3 + toptorsion) --------
    log.info(f"computing TRIMER RDKit descriptors")
    t0 = time.time()
    rows = [compute_rdkit_desc(s) or {} for s in tqdm(smis_tri, desc="tri rdkit", ncols=100)]
    df_desc = pd.DataFrame(rows).astype(float)
    df_desc, dropped = _sanitize_desc(df_desc)
    X = df_desc.values.astype(np.float32)
    _add("desc_tri", X)
    log.info(f"  desc_tri {X.shape} dropped={len(dropped)} time={time.time()-t0:.1f}s")

    for name, fn in [
        ("morgan2c_tri",   lambda: np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_tri, desc="tri morgan-r2", ncols=100)])),
        ("maccs_tri",      lambda: np.stack([compute_maccs(s) for s in tqdm(smis_tri, desc="tri maccs", ncols=100)])),
        ("atompair_c_tri", lambda: np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_tri, desc="tri atompair", ncols=100)])),
        ("avalon_tri",     lambda: np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_tri, desc="tri avalon", ncols=100)])),
    ]:
        t0 = time.time()
        X = fn().astype(np.float32)
        _add(name, X)
        log.info(f"  {name} {X.shape} time={time.time()-t0:.1f}s")

    X_full = np.concatenate(parts, axis=1)
    log.info(f"FEATURE MATRIX (mono+tri) {X_full.shape}  size≈{X_full.nbytes/1e6:.1f}MB")
    return {
        "X": X_full,
        "smiles_index": {s: i for i, s in enumerate(smis_mono)},
        "families_slice": families_slice,
    }


def get_or_build_lgb_features(all_canon, cache_path, log):
    key = hashlib.md5(
        (str(sorted(set(all_canon))) +
         f"m2={MORGAN2_NBITS};m3={MORGAN3_NBITS};ap={ATOMPAIR_NBITS};tt={TOPTORSION_NBITS};"
         f"av={AVALON_NBITS};chain={CHAIN_N_UNITS}"
         ).encode()
    ).hexdigest()[:12]
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("_key") == key:
                log.info(f"loaded LGB feature cache {cache_path.name} key={key}")
                return bundle
            log.info("LGB feature cache key mismatch; rebuilding")
        except Exception as e:
            log.info(f"cache load failed ({e}); rebuilding")
    bundle = build_lgb_features(all_canon, log)
    bundle["_key"] = key
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"cached LGB features to {cache_path.name}")
    return bundle


def slice_features(bundle, canon_series: pd.Series) -> np.ndarray:
    idx = canon_series.map(bundle["smiles_index"]).values
    return bundle["X"][idx]


# =====================================================================
#          AUX (matrix completion) FEATURES for LGB
# =====================================================================

def build_aux_lookup(train_df):
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    lookup = {}
    grouped = train_df.groupby("canon")
    for canon, g in tqdm(grouped, desc="aux lookup", ncols=100, total=grouped.ngroups):
        row = empty.copy()
        for tt, gg in g.groupby("target_type"):
            if tt in TARGET_IDX:
                idx = TARGET_IDX[tt]
                row[idx] = float(gg["target"].mean())
                row[idx + N_TARGETS] = 1.0
        lookup[canon] = row
    return lookup


def aux_for_target(canon_series, target, lookup):
    t_idx = TARGET_IDX[target]
    empty = np.full(2 * N_TARGETS, np.nan, dtype=np.float32)
    empty[N_TARGETS:] = 0.0
    out = np.stack([lookup.get(c, empty).copy() for c in canon_series])
    out[:, t_idx] = np.nan
    out[:, t_idx + N_TARGETS] = 0.0
    return out


# =====================================================================
#                       CV SPLITS
# =====================================================================

def group_kfold_splits(canon_arr, n_splits=N_SPLITS, seed=SPLIT_SEED):
    canon_arr = np.asarray(canon_arr)
    uniq = pd.Series(pd.unique(canon_arr))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_arr = np.array([fold_of_group[g] for g in canon_arr])
    return [(np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]) for k in range(n_splits)]


def canon_to_fold(canons, n_splits=N_SPLITS, seed=SPLIT_SEED):
    """Return dict canon -> fold_id (0..n_splits-1), identical assignment across pipeline."""
    uniq = pd.Series(pd.unique(canons))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    shuffled = uniq.iloc[order].values
    return {g: i % n_splits for i, g in enumerate(shuffled)}


# =====================================================================
#              PHASE 3: PER-FOLD LGB TEACHERS
#     For each fold k and target t: train teacher on fold_k_train
#     (canonical), predict PI1M rows. Save (n_pi1m, 7) per fold.
# =====================================================================

def train_lgb_teacher_one(target, tr, canon_to_fold_map, fold_k, bundle, aux_lookup, log):
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    fold_arr = g_tr["canon"].map(canon_to_fold_map).values
    mask = fold_arr != fold_k
    g_tr = g_tr[mask].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_smi = slice_features(bundle, g_tr["canon"])
    X_aux = aux_for_target(g_tr["canon"], target, aux_lookup)
    X = np.concatenate([X_smi, X_aux], axis=1)

    d_tr = lgb.Dataset(X, y)
    booster = lgb.train(
        LGB_PARAMS, d_tr,
        num_boost_round=int(N_ESTIMATORS * 0.4),   # teacher: shorter, no ES
        callbacks=[lgb.log_evaluation(0)],
    )
    log.info(f"[TEACHER fold {fold_k} {target}] trained on {len(g_tr)} rows, "
             f"{booster.best_iteration or booster.num_trees()} iters")
    return booster


def phase3_teachers_predict_pi1m(tr, pi1m, bundle, aux_lookup, canon_to_fold_map, log):
    phase_dir = WORK_DIR / "teachers"
    phase_dir.mkdir(parents=True, exist_ok=True)
    pred_path = phase_dir / "pi1m_teacher_preds.pkl.gz"

    if pred_path.exists():
        log.info(f"[PHASE 3] loading cached teacher predictions {pred_path.name}")
        with gzip.open(pred_path, "rb") as f:
            return pickle.load(f)

    log.info("=" * 60)
    log.info("PHASE 3: PER-FOLD LGB TEACHERS -> PI1M PREDICTIONS")
    log.info("=" * 60)

    n_pi1m = len(pi1m)
    preds_by_fold = np.full((N_SPLITS, n_pi1m, N_TARGETS), np.nan, dtype=np.float32)

    X_pi_smi = slice_features(bundle, pi1m["canon"])

    for k in range(N_SPLITS):
        for target in TARGETS:
            booster = train_lgb_teacher_one(
                target, tr, canon_to_fold_map, k, bundle, aux_lookup, log,
            )
            # PI1M rows have no labels — aux_for_target uses empty=(nan, 0.0) safely
            X_pi_aux = aux_for_target(pi1m["canon"], target, {})
            X_pi = np.concatenate([X_pi_smi, X_pi_aux], axis=1)
            preds_by_fold[k, :, TARGET_IDX[target]] = booster.predict(X_pi)

    with gzip.open(pred_path, "wb") as f:
        pickle.dump(preds_by_fold, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"[PHASE 3] wrote {pred_path.name}  shape={preds_by_fold.shape}")
    return preds_by_fold


# =====================================================================
#      PHASE 4: FILTER PI1M
#   Compute per-target train std, cross-fold-teacher stdev per PI1M row,
#   Tanimoto to train union. Emit mask (n_pi1m, 7) of usable rows.
# =====================================================================

def _morgan_bit(smi, nbits=TANIMOTO_FP_NBITS):
    m = _mol(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)


def compute_tanimoto_max_to_train(pi1m_smiles, train_smiles, log) -> np.ndarray:
    log.info(f"computing Morgan-r2 bit fingerprints for {len(train_smiles)} train + {len(pi1m_smiles)} PI1M")
    train_fps = [_morgan_bit(s) for s in tqdm(train_smiles, desc="fp train", ncols=100)]
    pi1m_fps  = [_morgan_bit(s) for s in tqdm(pi1m_smiles,  desc="fp PI1M",  ncols=100)]
    train_fps_valid = [fp for fp in train_fps if fp is not None]
    log.info(f"  valid train fps: {len(train_fps_valid)}")
    max_sim = np.zeros(len(pi1m_smiles), dtype=np.float32)
    for i, fp in enumerate(tqdm(pi1m_fps, desc="tanimoto max", ncols=100)):
        if fp is None:
            max_sim[i] = np.nan; continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_valid)
        max_sim[i] = max(sims) if sims else 0.0
    return max_sim


def phase4_filter_pi1m(tr, pi1m, preds_by_fold, log) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      pseudo_labels (n_pi1m, 7)   -- fold-0 teacher preds (used for refit; per-fold uses preds_by_fold[k])
      mask          (n_pi1m, 7)   -- True if this PI1M row is usable as pseudo for this target
      stats         dict          -- per-target retained counts, thresholds, etc.
    """
    phase_dir = WORK_DIR / "filter"
    phase_dir.mkdir(parents=True, exist_ok=True)
    mask_path  = phase_dir / "pseudo_mask.pkl.gz"
    stats_path = phase_dir / "filter_stats.json"

    if mask_path.exists():
        log.info(f"[PHASE 4] loading cached filter {mask_path.name}")
        with gzip.open(mask_path, "rb") as f:
            payload = pickle.load(f)
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return payload["labels"], payload["mask"], stats

    log.info("=" * 60)
    log.info("PHASE 4: PI1M PSEUDO-LABEL FILTERING")
    log.info("=" * 60)

    n_pi1m = len(pi1m)
    mask = np.zeros((n_pi1m, N_TARGETS), dtype=bool)

    # Confidence: cross-fold stdev per target
    stdev = np.nanstd(preds_by_fold, axis=0)         # (n_pi1m, 7)
    mean_pred = np.nanmean(preds_by_fold, axis=0)    # (n_pi1m, 7)

    # Tanimoto max to train union
    train_canons = pd.unique(tr["canon"])
    max_sim = compute_tanimoto_max_to_train(pi1m["canon"].tolist(), train_canons.tolist(), log)
    tanimoto_mask = (max_sim >= TANIMOTO_LOW) & (max_sim <= TANIMOTO_HIGH)
    log.info(f"[PHASE 4] Tanimoto ∈ [{TANIMOTO_LOW}, {TANIMOTO_HIGH}] retained: "
             f"{tanimoto_mask.sum()}/{n_pi1m} ({tanimoto_mask.mean()*100:.1f}%)")

    stats = {"pi1m_n": int(n_pi1m), "tanimoto_retained": int(tanimoto_mask.sum()),
             "tanimoto_low": TANIMOTO_LOW, "tanimoto_high": TANIMOTO_HIGH,
             "confidence_stdev_factor": CONFIDENCE_STDEV_FACTOR,
             "per_target": {}}

    for t_idx, tgt in enumerate(TARGETS):
        train_std = float(tr.loc[tr["target_type"] == tgt, "target"].std())
        thresh = CONFIDENCE_STDEV_FACTOR * train_std
        conf_mask = stdev[:, t_idx] < thresh
        combined = conf_mask & tanimoto_mask
        mask[:, t_idx] = combined
        log.info(f"[PHASE 4] {tgt:>4s}: train_std={train_std:.4f}  thresh={thresh:.4f}  "
                 f"conf_ok={conf_mask.sum()}   final={combined.sum()} "
                 f"({combined.mean()*100:.1f}% of PI1M)")
        stats["per_target"][tgt] = {
            "train_std": train_std, "confidence_thresh": thresh,
            "conf_retained": int(conf_mask.sum()),
            "final_retained": int(combined.sum()),
        }

    with gzip.open(mask_path, "wb") as f:
        pickle.dump({"labels": mean_pred, "mask": mask}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info(f"[PHASE 4] wrote {mask_path.name} and {stats_path.name}")
    return mean_pred, mask, stats


# =====================================================================
#         PHASE 5: LGB STUDENT ON AUGMENTED DATA
# =====================================================================

def train_lgb_student_one_target(target, tr, te, bundle, aux_lookup,
                                  pi1m, pi1m_preds_by_fold, mask,
                                  canon_to_fold_map, log):
    g_tr = tr[tr["target_type"] == target].reset_index(drop=True)
    g_te = te[te["target_type"] == target].reset_index(drop=True)
    y = g_tr["target"].astype(float).values

    X_tr_smi = slice_features(bundle, g_tr["canon"])
    X_te_smi = slice_features(bundle, g_te["canon"])
    X_tr_aux = aux_for_target(g_tr["canon"], target, aux_lookup)
    X_te_aux = aux_for_target(g_te["canon"], target, aux_lookup)
    X_tr = np.concatenate([X_tr_smi, X_tr_aux], axis=1)
    X_te = np.concatenate([X_te_smi, X_te_aux], axis=1)

    # PI1M features (aux empty since unlabeled)
    X_pi_smi = slice_features(bundle, pi1m["canon"])
    X_pi_aux = aux_for_target(pi1m["canon"], target, {})
    X_pi = np.concatenate([X_pi_smi, X_pi_aux], axis=1)

    t_idx = TARGET_IDX[target]
    pi_mask_target = mask[:, t_idx]

    splits = group_kfold_splits(g_tr["canon"].values)
    oof = np.zeros(len(g_tr), dtype=np.float64)
    best_iters, fold_r2s = [], []

    for k, (tri, vai) in enumerate(splits):
        # Pseudo rows for fold k: use fold-k-teacher preds, filtered by mask
        pseudo_pred_k = pi1m_preds_by_fold[k, :, t_idx]
        keep_pseudo = pi_mask_target & ~np.isnan(pseudo_pred_k)

        X_pseudo = X_pi[keep_pseudo]
        y_pseudo = pseudo_pred_k[keep_pseudo]
        w_real   = np.ones(len(tri), dtype=np.float32)
        w_pseudo = np.full(len(y_pseudo), PSEUDO_WEIGHT, dtype=np.float32)

        X_train = np.concatenate([X_tr[tri], X_pseudo], axis=0)
        y_train = np.concatenate([y[tri], y_pseudo], axis=0)
        w_train = np.concatenate([w_real, w_pseudo], axis=0)

        d_tr = lgb.Dataset(X_train, y_train, weight=w_train)
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
        r2 = float(r2_score(y[vai], pred_va))
        fold_r2s.append(r2)
        log.info(f"[LGB-STU {target}] fold {k}: n_real={len(tri)} n_pseudo={len(y_pseudo)} "
                 f"best_iter={booster.best_iteration} R²={r2:.4f}")

    oof_r2 = float(r2_score(y, oof))
    log.info(f"[LGB-STU {target}] OOF R² = {oof_r2:.4f}  (fold mean {np.mean(fold_r2s):.4f})")

    # Refit on full labeled + all pseudo (bagged mean preds)
    refit_iters = max(50, int(np.median(best_iters) * REFIT_ITER_MULTIPLIER))
    pseudo_pred_bag = np.nanmean(pi1m_preds_by_fold[:, :, t_idx], axis=0)
    keep = pi_mask_target & ~np.isnan(pseudo_pred_bag)
    X_pseudo_full = X_pi[keep]
    y_pseudo_full = pseudo_pred_bag[keep]
    w_full = np.concatenate([
        np.ones(len(y), dtype=np.float32),
        np.full(len(y_pseudo_full), PSEUDO_WEIGHT, dtype=np.float32)
    ])
    X_full = np.concatenate([X_tr, X_pseudo_full], axis=0)
    y_full = np.concatenate([y, y_pseudo_full], axis=0)
    log.info(f"[LGB-STU {target}] refit on {len(y)} labeled + {len(y_pseudo_full)} pseudo, "
             f"{refit_iters} iters")
    d_full = lgb.Dataset(X_full, y_full, weight=w_full)
    full_booster = lgb.train(LGB_PARAMS, d_full, num_boost_round=refit_iters,
                              callbacks=[lgb.log_evaluation(0)])
    test_pred = full_booster.predict(X_te)

    return {
        "target": target,
        "oof": pd.DataFrame({
            "canon": g_tr["canon"].values, "target_type": target,
            "y_true": y, "y_pred": oof,
        }),
        "test_pred": pd.DataFrame({
            "id": g_te["id"].values, "canon": g_te["canon"].values,
            "target_type": target, "target": test_pred,
        }),
        "oof_r2": oof_r2,
        "best_iters": best_iters,
        "refit_iters": refit_iters,
    }


# ---- Maxwell physics prior (reused from reproduce.py) ----

def fit_maxwell_forward(nc, eps):
    x = nc ** 2; y = eps
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b), float(r2_score(y, a * x + b))


def fit_maxwell_reverse(eps, nc):
    x = eps; y = nc ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.sqrt(np.clip(a * x + b, 1e-9, None))
    return float(a), float(b), float(r2_score(nc, pred))


def apply_maxwell_forward(nc, a, b):  return a * (nc ** 2) + b
def apply_maxwell_reverse(eps, a, b): return np.sqrt(np.clip(a * eps + b, 1e-9, None))


def search_blend_weight(y_true, y_ml, y_prior, grid=BLEND_W_GRID):
    r2s = np.array([r2_score(y_true, w * y_ml + (1 - w) * y_prior) for w in grid])
    best_i = int(np.argmax(r2s))
    return float(grid[best_i]), float(r2s[best_i]), float(r2_score(y_true, y_ml))


def effective_value_lookup(train_df, oof_results, target):
    lookup = {}
    for _, row in train_df[train_df["target_type"] == target].iterrows():
        lookup[row["canon"]] = float(row["target"])
    if target in oof_results:
        for _, row in oof_results[target]["oof"].iterrows():
            if row["canon"] not in lookup:
                lookup[row["canon"]] = float(row["y_pred"])
    return lookup


def phase5_6_lgb_student_with_maxwell(tr, te, bundle, aux_lookup,
                                        pi1m, pi1m_preds_by_fold, mask,
                                        canon_to_fold_map, log) -> Path:
    phase_dir = WORK_DIR / "lgb_student"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"
    oof_path = phase_dir / "oof.csv"
    if sub_path.exists() and oof_path.exists():
        log.info(f"[PHASE 5-6] found existing outputs at {phase_dir} — skipping")
        return phase_dir

    log.info("=" * 60)
    log.info("PHASE 5: LGB STUDENT ON AUGMENTED DATA (per-fold pseudo)")
    log.info("=" * 60)

    results = {}
    for tgt in tqdm(TARGETS, desc="[LGB-STU] targets", ncols=100):
        results[tgt] = train_lgb_student_one_target(
            tgt, tr, te, bundle, aux_lookup,
            pi1m, pi1m_preds_by_fold, mask,
            canon_to_fold_map, log,
        )

    log.info("=" * 60)
    log.info("PHASE 6: MAXWELL PHYSICS POST-FIT (EPS <-> Nc)")
    log.info("=" * 60)
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    co = wide.dropna(subset=["eps", "nc"])
    log.info(f"  co-labeled n={len(co)}")
    a_fwd, b_fwd, r2_fwd = fit_maxwell_forward(co["nc"].values, co["eps"].values)
    a_rev, b_rev, r2_rev = fit_maxwell_reverse(co["eps"].values, co["nc"].values)
    log.info(f"  forward EPS = {a_fwd:.4f}·Nc² + {b_fwd:.4f}   R²={r2_fwd:.4f}")
    log.info(f"  reverse Nc² = {a_rev:.4f}·EPS + {b_rev:.4f}   R²(on Nc)={r2_rev:.4f}")

    canon_to_nc  = effective_value_lookup(tr, results, "nc")
    canon_to_eps = effective_value_lookup(tr, results, "eps")

    # EPS OOF blend
    eps_oof = results["eps"]["oof"].copy()
    nc_eff = eps_oof["canon"].map(canon_to_nc).values.astype(float)
    eps_max = apply_maxwell_forward(nc_eff, a_fwd, b_fwd)
    m = np.isnan(eps_max); eps_max[m] = eps_oof["y_pred"].values[m]
    w_eps, r2_e, base_e = search_blend_weight(eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_max)
    log.info(f"  eps blend: base R²={base_e:.4f}  w={w_eps:.3f}  R²={r2_e:.4f}  Δ={r2_e-base_e:+.4f}")
    eps_oof["y_pred"] = w_eps * eps_oof["y_pred"].values + (1 - w_eps) * eps_max
    results["eps"]["oof"] = eps_oof

    # Nc OOF blend
    nc_oof = results["nc"]["oof"].copy()
    eps_eff = nc_oof["canon"].map(canon_to_eps).values.astype(float)
    nc_max = apply_maxwell_reverse(eps_eff, a_rev, b_rev)
    m = np.isnan(nc_max); nc_max[m] = nc_oof["y_pred"].values[m]
    w_nc, r2_n, base_n = search_blend_weight(nc_oof["y_true"].values, nc_oof["y_pred"].values, nc_max)
    log.info(f"  nc  blend: base R²={base_n:.4f}  w={w_nc:.3f}  R²={r2_n:.4f}  Δ={r2_n-base_n:+.4f}")
    nc_oof["y_pred"] = w_nc * nc_oof["y_pred"].values + (1 - w_nc) * nc_max
    results["nc"]["oof"] = nc_oof

    # Apply to test
    c2nc_te = dict(zip(results["nc"]["test_pred"]["canon"], results["nc"]["test_pred"]["target"]))
    c2eps_te = dict(zip(results["eps"]["test_pred"]["canon"], results["eps"]["test_pred"]["target"]))
    def _nc(c):  return canon_to_nc.get(c,  c2nc_te.get(c,  float("nan")))
    def _eps(c): return canon_to_eps.get(c, c2eps_te.get(c, float("nan")))

    eps_te = results["eps"]["test_pred"].copy()
    nc_eff_te = np.array([_nc(c) for c in eps_te["canon"]], dtype=float)
    eps_max_te = apply_maxwell_forward(nc_eff_te, a_fwd, b_fwd)
    m = np.isnan(eps_max_te); eps_max_te[m] = eps_te["target"].values[m]
    eps_te["target"] = w_eps * eps_te["target"].values + (1 - w_eps) * eps_max_te
    results["eps"]["test_pred"] = eps_te

    nc_te = results["nc"]["test_pred"].copy()
    eps_eff_te = np.array([_eps(c) for c in nc_te["canon"]], dtype=float)
    nc_max_te = apply_maxwell_reverse(eps_eff_te, a_rev, b_rev)
    m = np.isnan(nc_max_te); nc_max_te[m] = nc_te["target"].values[m]
    nc_te["target"] = w_nc * nc_te["target"].values + (1 - w_nc) * nc_max_te
    results["nc"]["test_pred"] = nc_te

    oof_all = pd.concat([results[t]["oof"][["canon", "target_type", "y_true", "y_pred"]]
                         for t in TARGETS], ignore_index=True)
    sub_all = pd.concat([results[t]["test_pred"][["id", "target"]] for t in TARGETS],
                        ignore_index=True).sort_values("id").reset_index(drop=True)
    oof_all.to_csv(oof_path, index=False)
    sub_all.to_csv(sub_path, index=False)
    log.info(f"[PHASE 5-6] wrote {oof_path}  ({len(oof_all)} rows)")
    log.info(f"[PHASE 5-6] wrote {sub_path}  ({len(sub_all)} rows)")
    return phase_dir


# =====================================================================
#             PHASE 7-8: CHEMPROP STUDENT ON AUGMENTED DATA
# =====================================================================

def build_wide_train_chemprop(tr):
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns: wide[t] = np.nan
    wide = wide[list(TARGETS)]
    return wide.index.tolist(), wide.values.astype(np.float32)


def build_chemprop_model(output_transform=None):
    mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH, dropout=MP_DROPOUT)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        n_tasks=N_TARGETS, input_dim=D_H, hidden_dim=FFN_HIDDEN,
        n_layers=FFN_LAYERS, dropout=FFN_DROPOUT,
        output_transform=output_transform,
    )
    return MPNN(
        mp, agg, ffn, batch_norm=BATCH_NORM,
        init_lr=LR_INIT, max_lr=LR_MAX, final_lr=LR_FINAL,
        warmup_epochs=WARMUP_EPOCHS,
    )


def _make_datapoints(canons, y_matrix, weight=1.0, idxs=None):
    pts = []
    it = idxs if idxs is not None else range(len(canons))
    for i in it:
        m = Chem.MolFromSmiles(canons[i])
        if m is None: continue
        pts.append(data.MoleculeDatapoint(mol=m, y=y_matrix[i], weight=weight))
    return pts


def make_train_val_datasets_aug(labeled_canons, y_labeled, train_idxs, val_idxs,
                                  pseudo_canons, y_pseudo, featurizer):
    """train set = labeled train + pseudo (weight=PSEUDO_WEIGHT); val set = labeled val (weight=1)."""
    train_pts_lab = _make_datapoints(labeled_canons, y_labeled, weight=1.0, idxs=train_idxs)
    train_pts_pse = _make_datapoints(pseudo_canons, y_pseudo, weight=PSEUDO_WEIGHT)
    train_pts = train_pts_lab + train_pts_pse
    val_pts = _make_datapoints(labeled_canons, y_labeled, weight=1.0, idxs=val_idxs)
    train_dset = data.MoleculeDataset(train_pts, featurizer=featurizer)
    val_dset   = data.MoleculeDataset(val_pts,   featurizer=featurizer)
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)
    return train_dset, val_dset, scaler


def make_full_dataset_aug(labeled_canons, y_labeled, pseudo_canons, y_pseudo, featurizer):
    lab_pts = _make_datapoints(labeled_canons, y_labeled, weight=1.0)
    pse_pts = _make_datapoints(pseudo_canons, y_pseudo, weight=PSEUDO_WEIGHT)
    full_dset = data.MoleculeDataset(lab_pts + pse_pts, featurizer=featurizer)
    scaler = full_dset.normalize_targets()
    return full_dset, scaler


def train_chemprop_cv_model(train_dset, val_dset, scaler, seed, ctx, log):
    L.seed_everything(seed, workers=True)
    train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_chemprop_model(output_transform=output_transform)
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=False)
    epoch_logger = EpochLogger(log, ctx)
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, accelerator=DEVICE, devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        callbacks=[early_stop, epoch_logger], deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    wall = (time.time() - t0) / 60
    model.eval()
    preds_list = trainer.predict(model, val_loader)
    val_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    best_epoch = trainer.current_epoch - PATIENCE if early_stop.stopped_epoch > 0 else trainer.current_epoch
    log.info(f"[{ctx}] done. time={wall:.1f}min  best_epoch≈{best_epoch}")
    return val_preds, int(best_epoch), float(wall)


def train_chemprop_refit(full_dset, scaler, seed, n_epochs, test_canons, featurizer, ctx, log):
    L.seed_everything(seed, workers=True)
    full_loader = data.build_dataloader(full_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    model = build_chemprop_model(output_transform=output_transform)
    epoch_logger = EpochLogger(log, ctx)
    trainer = L.Trainer(
        max_epochs=n_epochs, accelerator=DEVICE, devices=1,
        gradient_clip_val=GRAD_CLIP,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        callbacks=[epoch_logger], deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, full_loader)
    wall = (time.time() - t0) / 60
    log.info(f"[{ctx}] refit done. time={wall:.1f}min")

    test_pts = []
    valid_mask = []
    for smi in test_canons:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            test_pts.append(data.MoleculeDatapoint(mol=m, y=np.zeros(N_TARGETS, dtype=np.float32)))
    test_dset = data.MoleculeDataset(test_pts, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds_list = trainer.predict(model, test_loader)
    test_preds_valid = torch.cat(preds_list, dim=0).cpu().numpy()

    aligned = np.zeros((len(test_canons), N_TARGETS), dtype=np.float32)
    j = 0
    for i, v in enumerate(valid_mask):
        if v: aligned[i] = test_preds_valid[j]; j += 1
        else: aligned[i] = np.nan
    return aligned, wall


def phase7_8_chemprop_augmented(tr, te, pi1m, pi1m_preds_by_fold, mask, canon_to_fold_map, log) -> Path:
    phase_dir = WORK_DIR / "chemprop_aug"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path   = phase_dir / "submission.csv"
    oof_path   = phase_dir / "oof.csv"
    refit_path = phase_dir / "refit_test_preds.pkl.gz"
    if sub_path.exists() and oof_path.exists() and refit_path.exists():
        log.info(f"[PHASE 7-8] outputs exist at {phase_dir} — skipping")
        return phase_dir

    log.info("=" * 60)
    log.info("PHASE 7: CHEMPROP 3-SEED 5-FOLD ON AUGMENTED DATA")
    log.info("=" * 60)

    random.seed(SPLIT_SEED); np.random.seed(SPLIT_SEED); torch.manual_seed(SPLIT_SEED)
    L.seed_everything(SPLIT_SEED, workers=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log.info(f"torch threads: {torch.get_num_threads()}, device: {DEVICE}")

    labeled_canons, y_labeled = build_wide_train_chemprop(tr)
    test_canons = te["canon"].drop_duplicates().tolist()
    log.info(f"labeled canons: {len(labeled_canons)}  test canons: {len(test_canons)}")

    # Determine each labeled canon's fold
    lab_fold_arr = np.array([canon_to_fold_map[c] for c in labeled_canons])

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pi1m_canons = pi1m["canon"].tolist()

    # Pseudo y-matrix: for CV, use per-fold-teacher preds; for refit, bagged mean.
    # NaN-out targets whose mask=False (Chemprop supports per-target NaN masking natively)
    pi1m_labels_bagged = np.nanmean(pi1m_preds_by_fold, axis=0).copy()   # (n_pi1m, 7)

    # ---- 5-fold CV × 3-seed with per-fold checkpointing ----
    splits = group_kfold_splits(labeled_canons)
    fold_results = []
    for k, (tri, vai) in enumerate(splits):
        cp_path = phase_dir / f"checkpoint_fold_{k}.pkl.gz"
        if cp_path.exists():
            log.info(f"[CP fold {k}] loading checkpoint (skip training)")
            with gzip.open(cp_path, "rb") as f:
                fold_results.append(pickle.load(f))
            continue

        # Build pseudo y-matrix for fold k, masking rejected rows to NaN
        y_pseudo_k = pi1m_preds_by_fold[k].copy()               # (n_pi1m, 7)
        y_pseudo_k[~mask] = np.nan
        # Drop rows that are fully NaN (rejected for every target)
        keep_row = ~np.isnan(y_pseudo_k).all(axis=1)
        pseudo_canons_k = [c for c, kk in zip(pi1m_canons, keep_row) if kk]
        y_pseudo_k = y_pseudo_k[keep_row]

        log.info("=" * 60)
        log.info(f"CHEMPROP FOLD {k}  (3-seed bag)")
        log.info(f"  n_train_labeled={len(tri)}   n_val={len(vai)}   n_pseudo={len(pseudo_canons_k)}")

        val_preds_per_seed, best_epochs, wall_times = [], [], []
        for si, seed in enumerate(MODEL_SEEDS):
            ctx = f"CP fold {k} seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
            log.info(f"[{ctx}] starting...")
            train_dset, val_dset, scaler = make_train_val_datasets_aug(
                labeled_canons, y_labeled, tri, vai,
                pseudo_canons_k, y_pseudo_k, featurizer,
            )
            val_preds, best_epoch, wall = train_chemprop_cv_model(
                train_dset, val_dset, scaler, seed, ctx, log)
            val_preds_per_seed.append(val_preds)
            best_epochs.append(best_epoch)
            wall_times.append(wall)

        val_preds_avg = np.mean(np.stack(val_preds_per_seed, axis=0), axis=0)
        result = {
            "fold_k": k, "val_idxs": vai,
            "val_preds_per_seed": val_preds_per_seed,
            "val_preds_avg": val_preds_avg,
            "val_true": y_labeled[vai],
            "best_epochs_per_seed": best_epochs,
            "wall_times_per_seed_min": wall_times,
            "seeds_used": list(MODEL_SEEDS),
            "n_pseudo_used": len(pseudo_canons_k),
        }
        with gzip.open(cp_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"[CP fold {k}] wrote checkpoint")
        fold_results.append(result)

    # Assemble OOF
    oof_preds = np.full((len(labeled_canons), N_TARGETS), np.nan, dtype=np.float32)
    for r in fold_results:
        oof_preds[r["val_idxs"]] = r["val_preds_avg"]
    log.info("PER-TARGET OOF R² (Chemprop augmented)")
    for t_idx, tgt in enumerate(TARGETS):
        m = ~np.isnan(y_labeled[:, t_idx])
        r2 = float(r2_score(y_labeled[m, t_idx], oof_preds[m, t_idx]))
        log.info(f"  {tgt:>4s}  n={int(m.sum()):>5d}  OOF R²={r2:.4f}")

    # ---- Refit epochs ----
    all_best = [e for r in fold_results for e in r["best_epochs_per_seed"]]
    refit_epochs = max(15, int(np.median(all_best) * REFIT_ITER_MULTIPLIER))
    log.info(f"refit epochs (median * 1.10): {refit_epochs}")

    log.info("=" * 60)
    log.info("PHASE 8: CHEMPROP REFIT ON FULL LABELED + BAGGED PSEUDO")
    log.info("=" * 60)

    if refit_path.exists():
        log.info(f"loading cached refit test preds {refit_path.name}")
        with gzip.open(refit_path, "rb") as f:
            cache = pickle.load(f)
        test_preds_avg = cache["test_preds_avg"]
    else:
        # bagged pseudo — mask + drop fully-NaN rows
        y_pseudo_full = pi1m_labels_bagged.copy()
        y_pseudo_full[~mask] = np.nan
        keep_row = ~np.isnan(y_pseudo_full).all(axis=1)
        pseudo_canons_full = [c for c, kk in zip(pi1m_canons, keep_row) if kk]
        y_pseudo_full = y_pseudo_full[keep_row]
        log.info(f"  refit augment: n_labeled={len(labeled_canons)}  n_pseudo={len(pseudo_canons_full)}")

        test_preds_per_seed = []
        for si, seed in enumerate(MODEL_SEEDS):
            ctx = f"CP REFIT seed {seed} ({si+1}/{len(MODEL_SEEDS)})"
            log.info(f"[{ctx}] starting refit for {refit_epochs} epochs")
            full_dset, scaler = make_full_dataset_aug(
                labeled_canons, y_labeled, pseudo_canons_full, y_pseudo_full, featurizer,
            )
            test_preds, wall = train_chemprop_refit(
                full_dset, scaler, seed, refit_epochs, test_canons, featurizer, ctx, log,
            )
            test_preds_per_seed.append(test_preds)
        test_preds_avg = np.mean(np.stack(test_preds_per_seed, axis=0), axis=0)
        with gzip.open(refit_path, "wb") as f:
            pickle.dump({"test_preds_avg": test_preds_avg,
                         "test_preds_per_seed": test_preds_per_seed}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"cached refit test preds -> {refit_path.name}")

    # Build OOF and submission csvs
    test_preds_by_canon = {c: test_preds_avg[i] for i, c in enumerate(test_canons)}
    labeled_c2i = {c: i for i, c in enumerate(labeled_canons)}
    oof_rows = []
    for _, row in tr.iterrows():
        c = row["canon"]; t = row["target_type"]
        i = labeled_c2i.get(c)
        if i is None: continue
        t_idx = TARGET_IDX[t]
        oof_rows.append({
            "canon": c, "target_type": t,
            "y_true": float(row["target"]),
            "y_pred": float(oof_preds[i, t_idx]) if not np.isnan(oof_preds[i, t_idx]) else np.nan,
        })
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False)
    log.info(f"[PHASE 7-8] wrote {oof_path}  ({len(oof_rows)} rows)")

    sub_rows = []
    for _, row in te.iterrows():
        c = row["canon"]; t = row["target_type"]
        t_idx = TARGET_IDX[t]
        preds = test_preds_by_canon.get(c)
        sub_rows.append({"id": int(row["id"]),
                          "target": float(preds[t_idx]) if preds is not None else float("nan")})
    pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True).to_csv(sub_path, index=False)
    log.info(f"[PHASE 7-8] wrote {sub_path}")
    return phase_dir


# =====================================================================
#            PHASE 9: NNLS BLEND + KOOPMANS POST-FIT
# =====================================================================

def fit_target_weights_nnls(y_true, y_c, y_l, log, target):
    A = np.vstack([y_c, y_l]).T
    x, _ = nnls(A, y_true)
    w_c_raw, w_l_raw = float(x[0]), float(x[1])
    s = w_c_raw + w_l_raw
    if s < 1e-9:
        log.warning(f"[BLEND {target}] NNLS collapsed to 0; using 50/50")
        w_c_norm, w_l_norm = 0.5, 0.5
    else:
        w_c_norm, w_l_norm = w_c_raw / s, w_l_raw / s
    if APPLY_CHEMPROP_BIAS != 0.0:
        w_c_b = min(1.0, w_c_norm + APPLY_CHEMPROP_BIAS)
        w_l_b = max(0.0, 1.0 - w_c_b)
    else:
        w_c_b, w_l_b = w_c_norm, w_l_norm
    if w_c_b < CHEMPROP_WEIGHT_FLOOR:
        w_c_f, w_l_f = CHEMPROP_WEIGHT_FLOOR, 1.0 - CHEMPROP_WEIGHT_FLOOR
    else:
        w_c_f, w_l_f = w_c_b, w_l_b
    log.info(f"[BLEND {target}] NNLS raw w_c={w_c_raw:.3f} w_l={w_l_raw:.3f}   "
             f"final w_c={w_c_f:.3f} w_l={w_l_f:.3f}")
    return w_c_f, w_l_f


def phase9_blend(chemprop_dir, lgb_dir, log) -> Path:
    phase_dir = WORK_DIR / "blend_nnls"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"
    log.info("=" * 60)
    log.info("PHASE 9a: 2-WAY NNLS BLEND (Chemprop-aug + LGB-aug)")
    log.info("=" * 60)
    log.info(f"  CHEMPROP_WEIGHT_FLOOR={CHEMPROP_WEIGHT_FLOOR}  APPLY_CHEMPROP_BIAS={APPLY_CHEMPROP_BIAS}")

    oof_c = pd.read_csv(chemprop_dir / "oof.csv")
    oof_l = pd.read_csv(lgb_dir / "oof.csv")
    oof = (oof_c.rename(columns={"y_pred": "y_pred_chemprop"})
           .merge(oof_l.rename(columns={"y_pred": "y_pred_lgb"})[["canon", "target_type", "y_pred_lgb"]],
                  on=["canon", "target_type"], how="inner"))
    sub_c = pd.read_csv(chemprop_dir / "submission.csv").rename(columns={"target": "target_chemprop"})
    sub_l = pd.read_csv(lgb_dir / "submission.csv").rename(columns={"target": "target_lgb"})
    te = pd.read_csv(DATA_DIR / "test.csv")[["id", "target_type"]]
    sub = te.merge(sub_c, on="id", how="left").merge(sub_l, on="id", how="left")

    per_target = {}
    sub = sub.copy(); sub["target_blend"] = np.nan
    for target in TARGETS:
        g = oof[oof["target_type"] == target].dropna(subset=["y_true", "y_pred_chemprop", "y_pred_lgb"])
        y_true, y_c, y_l = g["y_true"].values, g["y_pred_chemprop"].values, g["y_pred_lgb"].values
        r2_c = float(r2_score(y_true, y_c)); r2_l = float(r2_score(y_true, y_l))
        log.info(f"[BLEND {target}] CP OOF R²={r2_c:.4f}  LGB OOF R²={r2_l:.4f}")
        w_c, w_l = fit_target_weights_nnls(y_true, y_c, y_l, log, target)
        y_blend = w_c * y_c + w_l * y_l
        r2_b = float(r2_score(y_true, y_blend))
        log.info(f"[BLEND {target}] blend OOF R²={r2_b:.4f}   Δ vs best={r2_b - max(r2_c, r2_l):+.4f}")
        m = sub["target_type"] == target
        sub.loc[m, "target_blend"] = w_c * sub.loc[m, "target_chemprop"] + w_l * sub.loc[m, "target_lgb"]
        per_target[target] = {"w_chemprop": w_c, "w_lgb": w_l, "r2_blend": r2_b}
    out = sub[["id", "target_blend"]].rename(columns={"target_blend": "target"}).sort_values("id").reset_index(drop=True)
    out.to_csv(sub_path, index=False)
    log.info(f"[PHASE 9a] wrote {sub_path}")
    with open(phase_dir / "blend_summary.json", "w") as f:
        json.dump(per_target, f, indent=2, default=str)
    return phase_dir


def load_chemprop_oof_matrix(canons, chemprop_dir, log):
    n = len(canons)
    oof = np.full((n, 7), np.nan, dtype=np.float32)
    for k in range(N_SPLITS):
        with gzip.open(chemprop_dir / f"checkpoint_fold_{k}.pkl.gz", "rb") as f:
            r = pickle.load(f)
        oof[r["val_idxs"]] = r["val_preds_avg"]
    log.info(f"[KOOPMANS] Chemprop OOF matrix {oof.shape}   "
             f"missing rows: {int(np.isnan(oof).all(axis=1).sum())}")
    return oof


def tune_alpha_koopmans(target, oof, y_matrix, log):
    a_name, b_name, combine = PHYSICS_RECIPES[target]
    t_own = TARGET_IDX[target]; t_a = TARGET_IDX[a_name]; t_b = TARGET_IDX[b_name]
    mask = ~np.isnan(y_matrix[:, t_own])
    own = oof[mask, t_own]; sa = oof[mask, t_a]; sb = oof[mask, t_b]
    y = y_matrix[mask, t_own]
    valid = ~(np.isnan(own) | np.isnan(sa) | np.isnan(sb))
    own = own[valid]; sa = sa[valid]; sb = sb[valid]; y = y[valid]
    physics = combine(sa, sb)
    r2_base = float(r2_score(y, own))
    r2_phy = float(r2_score(y, physics))
    best_r2, best_alpha = -np.inf, 1.0
    for a in ALPHA_GRID:
        r2 = float(r2_score(y, a * own + (1 - a) * physics))
        if r2 > best_r2: best_r2, best_alpha = r2, float(a)
    log.info(f"[KOOPMANS {target}] base R²={r2_base:.4f}  pure-phys R²={r2_phy:.4f}  "
             f"best α={best_alpha:.3f}  blend R²={best_r2:.4f}  Δ={best_r2-r2_base:+.4f}")
    return {"best_alpha": best_alpha, "r2_baseline": r2_base,
            "r2_blend": best_r2, "delta_r2": best_r2 - r2_base}


def phase9b_koopmans(blend_dir, chemprop_dir, tr, te, log) -> Path:
    phase_dir = WORK_DIR / "koopmans"
    phase_dir.mkdir(parents=True, exist_ok=True)
    sub_path = phase_dir / "submission.csv"

    log.info("=" * 60)
    log.info("PHASE 9b: KOOPMANS BANDGAP POST-FIT (Egc ≈ Ei - Eea)")
    log.info("=" * 60)

    canons, y_matrix = build_wide_train_chemprop(tr)
    oof = load_chemprop_oof_matrix(canons, chemprop_dir, log)

    alpha_results = {}
    for tgt in PHYSICS_TARGETS:
        alpha_results[tgt] = tune_alpha_koopmans(tgt, oof, y_matrix, log)
    alphas = {t: alpha_results[t]["best_alpha"] for t in PHYSICS_TARGETS}
    total_delta = sum(alpha_results[t]["delta_r2"] for t in PHYSICS_TARGETS)
    log.info(f"[KOOPMANS] sum ΔR² = {total_delta:+.4f}   "
             f"expected 7-target mean uplift = {total_delta/7:+.4f}")

    sub = pd.read_csv(blend_dir / "submission.csv")
    te_look = te[["id", "canon", "target_type"]].copy()
    sub_full = sub.merge(te_look, on="id", how="left")

    with gzip.open(chemprop_dir / "refit_test_preds.pkl.gz", "rb") as f:
        cache = pickle.load(f)
    chem_test = cache["test_preds_avg"]
    test_canons = te["canon"].drop_duplicates().tolist()
    c2i = {c: i for i, c in enumerate(test_canons)}

    for tgt in PHYSICS_TARGETS:
        a = alphas[tgt]
        sa_name, sb_name, combine = PHYSICS_RECIPES[tgt]
        sa_idx, sb_idx = TARGET_IDX[sa_name], TARGET_IDX[sb_name]
        m = sub_full["target_type"] == tgt
        rows = sub_full[m].copy()
        cidx = np.array([c2i[c] for c in rows["canon"]])
        chem_a = chem_test[cidx, sa_idx]; chem_b = chem_test[cidx, sb_idx]
        physics_te = combine(chem_a, chem_b)
        own_te = rows["target"].values
        new_pred = a * own_te + (1 - a) * physics_te
        d = np.abs(new_pred - own_te)
        log.info(f"[KOOPMANS {tgt}] α={a:.3f}  n_rows={len(rows)}  mean|Δ|={d.mean():.4f}  max|Δ|={d.max():.4f}")
        sub_full.loc[m, "target"] = new_pred

    out = sub_full[["id", "target"]].sort_values("id").reset_index(drop=True)
    out.to_csv(sub_path, index=False)
    log.info(f"[PHASE 9b] wrote {sub_path}")
    with open(phase_dir / "koopmans_summary.json", "w") as f:
        json.dump({"alphas": alphas, "alpha_results": alpha_results,
                   "sum_delta_r2": total_delta,
                   "expected_mean_uplift": total_delta / 7}, f, indent=2, default=str)
    return phase_dir


# =====================================================================
#                              MAIN
# =====================================================================

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(WORK_DIR)
    log.info("=" * 60)
    log.info(" PI1M PSEUDO-LABEL AUGMENTED PIPELINE — target LB > 0.902")
    log.info("=" * 60)
    log.info(f"DATA_DIR: {DATA_DIR}")
    log.info(f"WORK_DIR: {WORK_DIR}")
    log.info(f"DEVICE:   {DEVICE}")
    log.info(f"PI1M cap: {PI1M_SAMPLE_CAP}  pseudo weight: {PSEUDO_WEIGHT}")
    log.info(f"Tanimoto range: [{TANIMOTO_LOW}, {TANIMOTO_HIGH}]  stdev factor: {CONFIDENCE_STDEV_FACTOR}")

    t_total = time.time()

    # ---- Phase 0: load train/test/PI1M, canonicalize ----
    tr, te = load_train_test(log)
    pi1m = load_pi1m(log)

    # ---- Phase 1: LGB features for train+test+PI1M ----
    all_canon = pd.concat([tr["canon"], te["canon"], pi1m["canon"]]).tolist()
    (WORK_DIR / "features").mkdir(parents=True, exist_ok=True)
    bundle = get_or_build_lgb_features(
        all_canon, WORK_DIR / "features" / "feature_cache.pkl", log,
    )
    aux_lookup = build_aux_lookup(tr)

    # ---- Fold assignment for the whole pipeline ----
    canon_to_fold_map = canon_to_fold(pd.unique(tr["canon"]))

    # ---- Phase 3: per-fold LGB teachers -> PI1M preds ----
    pi1m_preds_by_fold = phase3_teachers_predict_pi1m(
        tr, pi1m, bundle, aux_lookup, canon_to_fold_map, log,
    )
    # ---- Phase 4: filter PI1M ----
    pseudo_labels_bagged, mask, filter_stats = phase4_filter_pi1m(tr, pi1m, pi1m_preds_by_fold, log)

    # ---- Phase 5-6: LGB student on augmented + Maxwell ----
    lgb_dir = phase5_6_lgb_student_with_maxwell(
        tr, te, bundle, aux_lookup,
        pi1m, pi1m_preds_by_fold, mask, canon_to_fold_map, log,
    )

    # ---- Phase 7-8: Chemprop student on augmented ----
    chemprop_dir = phase7_8_chemprop_augmented(
        tr, te, pi1m, pi1m_preds_by_fold, mask, canon_to_fold_map, log,
    )

    # ---- Phase 9: NNLS blend + Koopmans ----
    blend_dir    = phase9_blend(chemprop_dir, lgb_dir, log)
    koopmans_dir = phase9b_koopmans(blend_dir, chemprop_dir, tr, te, log)

    final_src = koopmans_dir / "submission.csv"
    shutil.copy2(final_src, WORK_DIR / "submission.csv")
    shutil.copy2(final_src, FINAL_SUB_PATH)

    check = pd.read_csv(FINAL_SUB_PATH)
    n_nan = int(check["target"].isna().sum())
    log.info(f"submission rows={len(check)} cols={list(check.columns)} NaNs={n_nan}")
    if n_nan:
        log.warning(f"{n_nan} NaNs — Kaggle will reject. Investigate.")

    log.info("=" * 60)
    log.info(f"FINAL SUBMISSION: {FINAL_SUB_PATH}")
    log.info(f"                  (also at {final_src})")
    log.info(f"Total wall time: {(time.time() - t_total) / 60:.1f} min")
    log.info(f"Target LB: 0.905-0.912  (baseline was 0.902)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
