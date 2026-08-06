"""
exp_chain_ext_mlp.py — Multitask MLP on chain-ext v1 feature stack.

============================================================================
WHY THIS EXISTS
============================================================================

Every "clever variant" of chain-ext LGB (v2 Optuna, v3 IterImputer, v3fixed
domain features, 5-mer, chain-ext Chemprop, chain-ext blends, PI1M pseudo)
has failed to beat chain-ext v1's LB 0.894. The pattern is clear: our LGB
overfits fold structure any time we add ambitious ingredients.

The one thing that HAS blended well was Chemprop (LB 0.892) with mono LGB
(LB 0.860) → blend LB 0.897. Why? Chemprop is a STRUCTURALLY DIFFERENT
model family (graph message passing vs axis-aligned splits). Different
error patterns → real ensemble lift.

But Chemprop is Kaggle-runtime-incompatible (12h notebook limit; our
mono 3-seed took 225 min just for one variant). We need a similarly-
different model family that FITS in Kaggle runtime.

**A multitask MLP is that model.**
  - Different family from trees (nonlinear feature interactions vs
    axis-aligned splits)
  - Fast on CPU (~20-30 min for full pipeline)
  - Shared representation across 7 targets replaces the cross-target
    signal that aux features gave to LGB
  - Same fold structure as v1 → OOFs directly blend-alignable

============================================================================
DESIGN
============================================================================

**Architecture (multitask):**
  - Shared trunk: Linear(14074, 512) → BN → ReLU → Dropout(0.3)
                → Linear(512, 256)   → BN → ReLU → Dropout(0.3)
                → Linear(256, 128)   → BN → ReLU → Dropout(0.3)
  - Per-target head: Linear(128, 7)  (7 targets predicted jointly)

  ~7.3M params. Aggressive dropout (0.3) + weight decay (1e-3) +
  batch norm + early stopping to prevent overfit on small-data targets.

**Training:**
  - Loss: masked MSE across 7 targets (only present targets contribute)
  - Optimizer: AdamW, lr=1e-3, weight_decay=1e-3
  - Scheduler: CosineAnnealingLR to 1e-4
  - Batch size 64
  - Max 60 epochs, patience 10 (early stop on val loss)

**Features (SAME as chain-ext v1, but NO aux):**
  - Monomer: RDKit desc + Morgan-r2/r3 + MACCS + AtomPair + TopTorsion + Avalon
  - Trimer:  RDKit desc + Morgan-r2 + MACCS + AtomPair + Avalon
  - Total: ~14,074 features
  - **No aux matrix-completion features** — multitask MLP's shared trunk
    replaces that role (and using aux with multitask MLP would leak
    target values through the aux vector)

**Standardization:**
  - StandardScaler fit on TRAIN FOLD only, transform val+test — no leak
  - Target standardization: per-target mean/std from train fold, un-scale
    at predict

**Cross-validation:**
  - Same 5-fold GroupKFold with seed=42 as v1
  - OOF is per-canon (all 7 targets predicted jointly)
  - Filter by target_type at output for per-target blending compatibility

**Post-fit:**
  - Maxwell EPS↔Nc physics blend (same as v1)

============================================================================
DEPENDENCIES
============================================================================

  Data: ppp-round-2/{train,test}.csv
  Venv: poly2-venv with rdkit, torch, sklearn, tqdm

============================================================================
OUTPUTS  (under results/exp_chain_ext_mlp/)
============================================================================

  run.log             — training log, per-epoch loss, per-fold R², final summary
  oof.csv             — OOF predictions after Maxwell blend
  submission.csv      — Kaggle format id, target
  cv_summary.json     — per-target R², Maxwell params, MLP config, timing

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_chain_ext_mlp.py

============================================================================
WALL TIME (~30-45 min on Mac CPU)
============================================================================

  - Featurize (mono + trimer, same as v1): ~15-20 min
  - MLP CV (5 folds × ~2-4 min): ~10-20 min
  - Refit + test predict: ~2-3 min
  - Maxwell + output: <1 min

============================================================================
EXPECTED
============================================================================

Solo LB: 0.85-0.88 (MLP alone typically underperforms tuned LGB on tabular)
Blend LB (MLP + chain-ext LGB v1): 0.895-0.902 (target)
  - Similar dynamic to mono-LGB (0.860) + Chemprop (0.892) → blend 0.897
  - Different model family = different errors = real blend lift possible

If OOF looks reasonable (mean R² > 0.83), write blend script and submit.

============================================================================
"""
from __future__ import annotations

# --- stdlib ---
import hashlib
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

# --- third-party ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_chain_ext_mlp"
EXP_DIR = REPO / "results" / EXP_NAME
FEATURE_CACHE_PATH = EXP_DIR / "feature_cache.pkl"

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

# MLP config
HIDDEN_DIMS = (512, 256, 128)
DROPOUT = 0.3
LR_INIT = 1e-3
LR_FINAL = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 10
GRAD_CLIP = 1.0
REFIT_EPOCH_MULTIPLIER = 1.10

DEVICE = "cpu"

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
# DATA + CANONICALIZATION  (verbatim from chain-ext v1)
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
# POLYMER CHAIN EXTENSION  (verbatim from chain-ext v1)
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
# FEATURE COMPUTATION  (verbatim from chain-ext v1)
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
        if lo == hi or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        df[c] = df[c].clip(lo, hi)
    # Hard safety cap for float32 (BertzCT etc. can overflow float32 max ~3.4e38)
    df = df.clip(-1e10, 1e10)
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
    log.info(f"chain extension: {n_extended}/{len(smis_mono)} SMILES extended  time={time.time()-t0:.1f}s")

    parts, families_slice, cursor = [], {}, 0

    def _add(name: str, arr: np.ndarray):
        nonlocal cursor
        parts.append(arr)
        families_slice[name] = slice(cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]

    log.info(f"MONOMER features")
    for name, fn in [
        ("desc_mono",           lambda: (pd.DataFrame([compute_rdkit_desc(s) or {} for s in tqdm(smis_mono, desc="mono desc", ncols=100)]).astype(float))),
        ("morgan2c_mono",       lambda: np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_mono, desc="mono m2", ncols=100)])),
        ("morgan3c_mono",       lambda: np.stack([compute_morgan_count(s, 3, MORGAN3_NBITS) for s in tqdm(smis_mono, desc="mono m3", ncols=100)])),
        ("maccs_mono",          lambda: np.stack([compute_maccs(s) for s in tqdm(smis_mono, desc="mono maccs", ncols=100)])),
        ("atompair_c_mono",     lambda: np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_mono, desc="mono ap", ncols=100)])),
        ("toptorsion_c_mono",   lambda: np.stack([compute_toptorsion_count(s, TOPTORSION_NBITS) for s in tqdm(smis_mono, desc="mono tt", ncols=100)])),
        ("avalon_mono",         lambda: np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_mono, desc="mono av", ncols=100)])),
    ]:
        t0 = time.time()
        X = fn()
        if name == "desc_mono":
            df_desc, dropped = _sanitize_desc_matrix(X)
            X = df_desc.values.astype(np.float32)
        else:
            X = X.astype(np.float32)
        _add(name, X)
        log.info(f"  {name}: {X.shape}  time={time.time()-t0:.1f}s")

    log.info(f"TRIMER features (from {CHAIN_N_UNITS}-mer SMILES)")
    for name, fn in [
        ("desc_tri",            lambda: (pd.DataFrame([compute_rdkit_desc(s) or {} for s in tqdm(smis_tri, desc="tri desc", ncols=100)]).astype(float))),
        ("morgan2c_tri",        lambda: np.stack([compute_morgan_count(s, 2, MORGAN2_NBITS) for s in tqdm(smis_tri, desc="tri m2", ncols=100)])),
        ("maccs_tri",           lambda: np.stack([compute_maccs(s) for s in tqdm(smis_tri, desc="tri maccs", ncols=100)])),
        ("atompair_c_tri",      lambda: np.stack([compute_atompair_count(s, ATOMPAIR_NBITS) for s in tqdm(smis_tri, desc="tri ap", ncols=100)])),
        ("avalon_tri",          lambda: np.stack([compute_avalon(s, AVALON_NBITS) for s in tqdm(smis_tri, desc="tri av", ncols=100)])),
    ]:
        t0 = time.time()
        X = fn()
        if name == "desc_tri":
            df_desc, dropped = _sanitize_desc_matrix(X)
            X = df_desc.values.astype(np.float32)
        else:
            X = X.astype(np.float32)
        _add(name, X)
        log.info(f"  {name}: {X.shape}  time={time.time()-t0:.1f}s")

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


# ============================================================================
# CV  (same as v1)
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
# BUILD WIDE TRAINING MATRIX  (canon × 7 targets; NaN where unlabeled)
# ============================================================================

def build_wide_train(tr: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    wide = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[list(TARGETS)]
    canons = wide.index.tolist()
    y_matrix = wide.values.astype(np.float32)
    return canons, y_matrix


# ============================================================================
# MLP MODEL
# ============================================================================

class MultitaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
                 n_out: int = N_TARGETS, dropout: float = DROPOUT):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_out)

    def forward(self, x):
        return self.head(self.trunk(x))


def masked_mse_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE across present targets only. mask: 1 = present, 0 = NaN."""
    sq = (pred - y) ** 2 * mask
    n_present = mask.sum().clamp(min=1.0)
    return sq.sum() / n_present


# ============================================================================
# TRAIN ONE FOLD
# ============================================================================

def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    max_epochs: int, seed: int,
    log: logging.Logger, ctx: str,
) -> tuple[np.ndarray, int, float]:
    """Standardize features + targets on train; train MLP; return val_preds (in ORIGINAL target space), best_epoch, wall."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Standardize features (fit on train fold)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler_x.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler_x.transform(X_va).astype(np.float32)

    # Standardize targets per-column (train mean/std, using nanmean/nanstd)
    y_mean = np.nanmean(y_tr, axis=0)
    y_std = np.nanstd(y_tr, axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)   # avoid div0 for a target with no train rows
    y_tr_norm = (y_tr - y_mean) / y_std
    y_va_norm = (y_va - y_mean) / y_std

    # Build masks (1 where target present, 0 where NaN); zero-fill NaN in y for tensor math
    mask_tr = (~np.isnan(y_tr_norm)).astype(np.float32)
    mask_va = (~np.isnan(y_va_norm)).astype(np.float32)
    y_tr_norm = np.nan_to_num(y_tr_norm, nan=0.0).astype(np.float32)
    y_va_norm = np.nan_to_num(y_va_norm, nan=0.0).astype(np.float32)

    # Torch tensors + dataloaders
    tr_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_norm), torch.from_numpy(mask_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va_norm), torch.from_numpy(mask_va))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = MultitaskMLP(in_dim=X_tr_s.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=LR_FINAL)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb, mb in tr_ld:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse_loss(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        n_va_batches = 0
        with torch.no_grad():
            for xb, yb, mb in va_ld:
                xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
                pred = model(xb)
                val_loss_sum += float(masked_mse_loss(pred, yb, mb).item())
                n_va_batches += 1
        val_loss = val_loss_sum / max(1, n_va_batches)

        sched.step()
        elapsed = time.time() - t0
        log.info(f"  [{ctx}] epoch {epoch:>3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  elapsed={elapsed/60:.1f}min")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log.info(f"  [{ctx}] early stop at epoch {epoch} (best epoch {best_epoch}, val_loss {best_val:.4f})")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict val (in ORIGINAL target space)
    model.eval()
    val_preds_norm = np.zeros((len(X_va_s), N_TARGETS), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for xb, _, _ in va_ld:
            xb = xb.to(DEVICE)
            p = model(xb).cpu().numpy()
            val_preds_norm[offset:offset + len(p)] = p
            offset += len(p)
    val_preds = val_preds_norm * y_std + y_mean

    wall = (time.time() - t0) / 60
    return val_preds, best_epoch, float(wall)


def train_refit(
    X_full: np.ndarray, y_full: np.ndarray,
    X_te: np.ndarray, n_epochs: int, seed: int,
    log: logging.Logger, ctx: str,
) -> np.ndarray:
    """Refit on full train, predict test.  Return test_preds (original target space)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    scaler_x = StandardScaler(with_mean=True, with_std=True)
    X_full_s = scaler_x.fit_transform(X_full).astype(np.float32)
    X_te_s = scaler_x.transform(X_te).astype(np.float32)

    y_mean = np.nanmean(y_full, axis=0)
    y_std = np.nanstd(y_full, axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_full_norm = (y_full - y_mean) / y_std
    mask_full = (~np.isnan(y_full_norm)).astype(np.float32)
    y_full_norm = np.nan_to_num(y_full_norm, nan=0.0).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_full_s), torch.from_numpy(y_full_norm), torch.from_numpy(mask_full))
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = MultitaskMLP(in_dim=X_full_s.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=LR_FINAL)

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0.0
        n_b = 0
        for xb, yb, mb in ld:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse_loss(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item())
            n_b += 1
        sched.step()
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            log.info(f"  [{ctx}] epoch {epoch:>3d}  train_loss={loss_sum/max(1,n_b):.4f}  elapsed={(time.time()-t0)/60:.1f}min")

    # Test predictions
    te_ds = TensorDataset(torch.from_numpy(X_te_s))
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    test_preds_norm = np.zeros((len(X_te_s), N_TARGETS), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for (xb,) in te_ld:
            xb = xb.to(DEVICE)
            p = model(xb).cpu().numpy()
            test_preds_norm[offset:offset + len(p)] = p
            offset += len(p)
    test_preds = test_preds_norm * y_std + y_mean
    return test_preds


# ============================================================================
# MAXWELL POST-FIT  (same as v1)
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: n_splits={N_SPLITS} seed={SEED} chain_n_units={CHAIN_N_UNITS}")
    log.info(f"MLP: hidden={HIDDEN_DIMS} dropout={DROPOUT} lr={LR_INIT} wd={WEIGHT_DECAY} "
             f"batch={BATCH_SIZE} max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    log.info(f"NOTE: multitask, NO aux features (shared representation replaces cross-target signal)")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    log.info(f"torch threads: {torch.get_num_threads()}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t_start = time.time()

    tr, te = load_and_canonicalize(log)

    all_canon = pd.concat([tr["canon"], te["canon"]]).tolist()
    bundle = get_or_build_features(all_canon, log)
    fam_str = ", ".join(f"{k}={v.stop-v.start}" for k, v in bundle["families_slice"].items())
    log.info(f"feature families: {fam_str}")
    log.info(f"total SMILES features: {bundle['X'].shape[1]}")

    # Sklearn StandardScaler chokes on inf/NaN even after cast to float32 — nuke them.
    # (LGB tolerates inf natively; MLP + StandardScaler does not.)
    n_bad = int((~np.isfinite(bundle["X"])).sum())
    if n_bad > 0:
        log.warning(f"cleaning {n_bad} non-finite values in feature matrix "
                    f"({100*n_bad/bundle['X'].size:.4f}% of cells)")
        bundle["X"] = np.nan_to_num(bundle["X"], nan=0.0, posinf=1e10, neginf=-1e10)
        # Also clip extreme finite values that would break StandardScaler variance
        bundle["X"] = np.clip(bundle["X"], -1e10, 1e10).astype(np.float32)

    # Build wide train matrix (canon × 7 targets)
    canons, y_matrix = build_wide_train(tr)
    log.info(f"wide train: n_canon={len(canons)}  y shape={y_matrix.shape}  "
             f"NaN fraction={100*np.isnan(y_matrix).mean():.1f}%")

    # Slice X for train canons + test canons
    smiles_idx = bundle["smiles_index"]
    X_train = bundle["X"][[smiles_idx[c] for c in canons]]

    test_canons_unique = te["canon"].drop_duplicates().tolist()
    X_test = bundle["X"][[smiles_idx[c] for c in test_canons_unique]]

    log.info(f"X_train shape: {X_train.shape}   X_test shape: {X_test.shape}")

    # 5-fold GroupKFold on canons
    splits = group_kfold_splits(canons, N_SPLITS, SEED)

    oof_preds = np.full((len(canons), N_TARGETS), np.nan, dtype=np.float32)
    best_epochs = []
    fold_wall_times = []

    for k, (tri, vai) in enumerate(splits):
        log.info("=" * 60)
        log.info(f"FOLD {k}: n_train_canon={len(tri)}  n_val_canon={len(vai)}")
        log.info("=" * 60)
        val_preds, best_epoch, wall = train_one_fold(
            X_train[tri], y_matrix[tri],
            X_train[vai], y_matrix[vai],
            MAX_EPOCHS, SEED, log, f"fold {k}",
        )
        oof_preds[vai] = val_preds
        best_epochs.append(best_epoch)
        fold_wall_times.append(wall)
        # Per-target R² on this fold
        for t_idx, tgt in enumerate(TARGETS):
            mask = ~np.isnan(y_matrix[vai, t_idx])
            if mask.sum() < 5:
                continue
            r2 = float(r2_score(y_matrix[vai, t_idx][mask], val_preds[mask, t_idx]))
            log.info(f"  [fold {k}] {tgt:>4s}: R²={r2:.4f}  n_val={int(mask.sum())}")

    # Assemble per-target OOF R²
    log.info("=" * 60)
    log.info("PER-TARGET OOF R² (MLP alone, pre-Maxwell)")
    log.info("=" * 60)
    per_target: dict[str, dict] = {}
    for t_idx, tgt in enumerate(TARGETS):
        mask = ~np.isnan(y_matrix[:, t_idx])
        y_true = y_matrix[mask, t_idx]
        y_pred = oof_preds[mask, t_idx]
        r2 = float(r2_score(y_true, y_pred))
        per_target[tgt] = {"n_train": int(mask.sum()), "oof_r2": r2}
        log.info(f"  {tgt:>4s}  n={int(mask.sum()):>5d}  OOF R²={r2:.4f}")
    baseline_mean = float(np.mean([per_target[t]["oof_r2"] for t in TARGETS]))
    log.info(f"  MEAN R² (MLP only) = {baseline_mean:.4f}")

    # Refit on full train
    refit_epochs = max(15, int(np.median(best_epochs) * REFIT_EPOCH_MULTIPLIER))
    log.info(f"best_epochs per fold: {best_epochs}")
    log.info(f"refitting on full train for {refit_epochs} epochs")
    test_preds_matrix = train_refit(X_train, y_matrix, X_test, refit_epochs, SEED, log, "refit")

    test_by_canon = {c: test_preds_matrix[i] for i, c in enumerate(test_canons_unique)}

    # Build per-target results dict (mimics chain-ext v1's format for blending)
    results: dict[str, dict] = {}
    canon_to_idx = {c: i for i, c in enumerate(canons)}
    for t_idx, tgt in enumerate(TARGETS):
        # OOF frame — only rows where target is present in train
        oof_rows = []
        for _, row in tr[tr["target_type"] == tgt].iterrows():
            i = canon_to_idx.get(row["canon"])
            if i is None:
                continue
            oof_rows.append({
                "canon":       row["canon"],
                "target_type": tgt,
                "y_true":      float(row["target"]),
                "y_pred":      float(oof_preds[i, t_idx]),
            })
        oof_df = pd.DataFrame(oof_rows)

        # Test frame — one row per (id, target_type)
        te_t = te[te["target_type"] == tgt].reset_index(drop=True)
        test_rows = []
        for _, row in te_t.iterrows():
            preds = test_by_canon.get(row["canon"])
            test_rows.append({
                "id":          int(row["id"]),
                "canon":       row["canon"],
                "target_type": tgt,
                "target":      float(preds[t_idx]) if preds is not None else float("nan"),
            })
        test_df = pd.DataFrame(test_rows)

        results[tgt] = {"oof": oof_df, "test_pred": test_df,
                        "oof_r2": per_target[tgt]["oof_r2"]}

    # ==== Maxwell EPS↔Nc ====
    log.info("=" * 60)
    log.info("MAXWELL RELATION POST-FIT (EPS ↔ Nc)")
    log.info("=" * 60)
    wide_full = tr.pivot_table(index="canon", columns="target_type", values="target", aggfunc="mean")
    co = wide_full.dropna(subset=["eps", "nc"])
    log.info(f"co-labeled train molecules: n={len(co)}")

    a_fwd, b_fwd, r2_fwd = fit_maxwell_forward(co["nc"].values, co["eps"].values)
    a_rev, b_rev, r2_rev = fit_maxwell_reverse(co["eps"].values, co["nc"].values)
    log.info(f"forward EPS = {a_fwd:.4f}·Nc² + {b_fwd:.4f}   R²={r2_fwd:.4f}")
    log.info(f"reverse Nc² = {a_rev:.4f}·EPS + {b_rev:.4f}   R²(on Nc)={r2_rev:.4f}")

    # Build "effective" lookups (train truth first, then OOF prediction)
    def build_eff_lookup(target: str) -> dict[str, float]:
        lookup: dict[str, float] = {}
        for _, r in tr[tr["target_type"] == target].iterrows():
            lookup[r["canon"]] = float(r["target"])
        for _, r in results[target]["oof"].iterrows():
            if r["canon"] not in lookup:
                lookup[r["canon"]] = float(r["y_pred"])
        return lookup

    canon_to_nc = build_eff_lookup("nc")
    canon_to_eps = build_eff_lookup("eps")

    # EPS blend
    eps_oof = results["eps"]["oof"].copy()
    nc_eff = eps_oof["canon"].map(canon_to_nc).values.astype(float)
    eps_max = apply_maxwell_forward(nc_eff, a_fwd, b_fwd)
    m = np.isnan(eps_max); eps_max[m] = eps_oof["y_pred"].values[m]
    best_w_eps, best_r2_eps, base_r2_eps = search_blend_weight(
        eps_oof["y_true"].values, eps_oof["y_pred"].values, eps_max)
    log.info(f"eps blend: MLP R²={base_r2_eps:.4f}  best w={best_w_eps:.3f}  "
             f"blend R²={best_r2_eps:.4f}  Δ={best_r2_eps - base_r2_eps:+.4f}")
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
    log.info(f"nc blend: MLP R²={base_r2_nc:.4f}  best w={best_w_nc:.3f}  "
             f"blend R²={best_r2_nc:.4f}  Δ={best_r2_nc - base_r2_nc:+.4f}")
    nc_oof["y_pred"] = best_w_nc * nc_oof["y_pred"].values + (1 - best_w_nc) * nc_max
    results["nc"]["oof"] = nc_oof
    results["nc"]["oof_r2"] = best_r2_nc

    # Apply Maxwell to test
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
    log.info(f"  MEAN R² = {final_mean:.4f}   (pre-Maxwell was {baseline_mean:.4f})")

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

    summary = {
        "exp_name":       EXP_NAME,
        "mean_r2":        final_mean,
        "mean_r2_pre_maxwell": baseline_mean,
        "per_target":     per_target,
        "best_epochs":    best_epochs,
        "refit_epochs":   refit_epochs,
        "fold_wall_times_min": fold_wall_times,
        "maxwell": {
            "n_co_labeled": int(len(co)),
            "forward_fit": {"a": a_fwd, "b": b_fwd, "r2": r2_fwd},
            "reverse_fit": {"a": a_rev, "b": b_rev, "r2_on_nc": r2_rev},
            "eps_blend":   {"baseline_r2": base_r2_eps, "best_w": best_w_eps, "best_r2": best_r2_eps},
            "nc_blend":    {"baseline_r2": base_r2_nc,  "best_w": best_w_nc,  "best_r2": best_r2_nc},
        },
        "config": {
            "hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT,
            "lr_init": LR_INIT, "lr_final": LR_FINAL, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "grad_clip": GRAD_CLIP, "refit_epoch_multiplier": REFIT_EPOCH_MULTIPLIER,
            "n_smiles_features": bundle["X"].shape[1],
            "cv_mode": "multitask + StandardScaler(per-fold) + Maxwell",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'cv_summary.json'}")

    # Comparison vs chain-ext v1 (LB 0.894)
    log.info("=" * 60)
    log.info("FINAL PER-TARGET OOF R²  (MLP + Maxwell)  vs chain-ext LGB v1 reference")
    log.info("=" * 60)
    v1_ref = {"eea": 0.8734, "egb": 0.9087, "egc": 0.9023,
              "ei":  0.8041, "eps": 0.8218, "nc":  0.8471, "tg":  0.9063}
    log.info(f"  {'target':>6s}  {'MLP':>10s}  {'LGB v1':>10s}  {'delta':>8s}")
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
