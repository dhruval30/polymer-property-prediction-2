"""
exp_pi1m_diagnostic.py — before spending 3h on pseudo-labeling, check whether
                          PI1M is chemically similar enough to train/test to
                          make pseudo-labels useful signal (not noise).

============================================================================
WHY
============================================================================

Round 1 tried pseudo-labeling and it failed. Research doc §6 warns that
PI1M is broader / more aliphatic than our train (PI1M FractionCSP3=0.46 vs
train 0.28; only 6.7% of train molecules have PI1M NN > 0.9).

Before committing 3-4h to the full pseudo-label experiment, run this
diagnostic to answer:

  1. How chemically similar are TEST polymers to PI1M?
       - If most test polymers have a PI1M neighbor with Tanimoto > 0.5,
         pseudo-labels probably help (test-like polymers in PI1M).
       - If most test polymers are Tanimoto < 0.3 from any PI1M polymer,
         pseudo-labels are noise (PI1M covers different chemistry).

  2. How chemically similar are TRAIN vs TEST vs PI1M distributions?
       - Adversarial validation: can we distinguish train from PI1M by
         Morgan fingerprints? AUC > 0.85 = very different.
       - Distribution stats: MolWt, LogP, TPSA, FractionCSP3, halogen frac.

  3. What fraction of PI1M is "polymer-like" (has 2 wildcards, no metals,
     parses cleanly, etc.)?

  4. Adversarial validation on train vs test: is our OWN train close to
     test? (If not, that itself explains our OOF-LB gap swings.)

============================================================================
DATA + SAMPLING
============================================================================

  - Train: 5920 unique canons (from ppp-round-2/train.csv)
  - Test: 4133 unique canons (from ppp-round-2/test.csv)
  - PI1M: 995,800 rows in ppp-round-2/PI1M.csv. We sample the first ~100K
          parseable, polymer-like ones for the diagnostic.

============================================================================
OUTPUTS  (under results/exp_pi1m_diagnostic/)
============================================================================

  run.log             — full log with distribution stats + AUC + recommendation
  diagnostic_stats.json — machine-readable summary
  nn_hist_test.csv     — nearest-neighbor Tanimoto distribution for test rows
  nn_hist_train.csv    — nearest-neighbor Tanimoto distribution for train rows

============================================================================
USAGE
============================================================================

  poly2-venv/bin/python experiments/exp_pi1m_diagnostic.py

============================================================================
WALL TIME
============================================================================

  ~35-40 min on Mac CPU:
    - PI1M parse + canonicalize + filter: ~5 min
    - Fingerprint computation (~65K molecules): ~1 min
    - Nearest-neighbor Tanimoto (test x PI1M + train x PI1M): ~20-25 min
    - Adversarial validation LGB: ~5 min
    - Distribution stats: <1 min

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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CONFIG
# ============================================================================

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "ppp-round-2"
EXP_NAME = "exp_pi1m_diagnostic"
EXP_DIR = REPO / "results" / EXP_NAME

PI1M_SAMPLE = 100_000        # sample first-N parseable polymer-like PI1M rows
NN_BATCH = 500               # rows per batch for nearest-neighbor Tanimoto
ADVERSARIAL_TRIALS = 3       # 3-fold AUC estimate for adversarial validation

TANIMOTO_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)   # buckets for reporting


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
# CANONICALIZE + POLYMER FILTER
# ============================================================================

def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def is_valid_polymer(smi: str) -> bool:
    """Polymer-like: parseable, has exactly 2 * wildcards, no metals."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    stars = [a for a in m.GetAtoms() if a.GetSymbol() == "*"]
    if len(stars) != 2:
        return False
    # Filter out organometallics — check for transition metals or unusual atoms
    allowed = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "*"}
    if any(a.GetSymbol() not in allowed for a in m.GetAtoms()):
        return False
    return True


# ============================================================================
# DISTRIBUTION STATS
# ============================================================================

def _cap(smi: str) -> str:
    return smi.replace("*", "C")


def compute_basic_stats(smi: str) -> dict[str, float] | None:
    """Cheap descriptor set for distribution comparison."""
    m = Chem.MolFromSmiles(_cap(smi))
    if m is None:
        return None
    heavy = m.GetNumHeavyAtoms()
    if heavy < 1:
        return None
    try:
        return {
            "MolWt":          float(Descriptors.MolWt(m)),
            "LogP":           float(Descriptors.MolLogP(m)),
            "TPSA":           float(Descriptors.TPSA(m)),
            "n_heavy":        int(heavy),
            "n_aromatic":     int(sum(1 for a in m.GetAtoms() if a.GetIsAromatic())),
            "aromatic_frac":  float(sum(1 for a in m.GetAtoms() if a.GetIsAromatic()) / heavy),
            "FractionCSP3":   float(rdMolDescriptors.CalcFractionCSP3(m)),
            "n_rot":          int(rdMolDescriptors.CalcNumRotatableBonds(m)),
            "F_count":        int(sum(1 for a in m.GetAtoms() if a.GetSymbol() == "F")),
            "halogen_frac":   float(sum(1 for a in m.GetAtoms()
                                        if a.GetSymbol() in ("F", "Cl", "Br", "I")) / heavy),
        }
    except Exception:
        return None


def summarize_dist(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Given list of feature dicts, return per-feature mean/median/std/q25/q75."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        v = df[col].dropna()
        if len(v) == 0:
            continue
        out[col] = {
            "n":     int(len(v)),
            "mean":  float(v.mean()),
            "median": float(v.median()),
            "std":   float(v.std()),
            "q25":   float(v.quantile(0.25)),
            "q75":   float(v.quantile(0.75)),
        }
    return out


# ============================================================================
# FINGERPRINT + TANIMOTO
# ============================================================================

def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    m = Chem.MolFromSmiles(_cap(smi))
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def compute_nn_tanimoto(query_fps: list, ref_fps: list, log: logging.Logger,
                        ctx: str) -> np.ndarray:
    """For each query fingerprint, return max Tanimoto similarity across all ref fingerprints."""
    out = np.zeros(len(query_fps), dtype=np.float32)
    for i in tqdm(range(0, len(query_fps), NN_BATCH), desc=f"NN Tanimoto {ctx}", ncols=100):
        end = min(i + NN_BATCH, len(query_fps))
        for j in range(i, end):
            if query_fps[j] is None:
                out[j] = 0.0
                continue
            sims = DataStructs.BulkTanimotoSimilarity(query_fps[j], ref_fps)
            out[j] = float(max(sims)) if sims else 0.0
    return out


def bucket_report(nn_scores: np.ndarray, log: logging.Logger, label: str,
                  thresholds=TANIMOTO_THRESHOLDS) -> dict[str, float]:
    """Report fraction of rows with nearest-neighbor Tanimoto above each threshold."""
    n = len(nn_scores)
    stats = {
        "n":     int(n),
        "mean":  float(nn_scores.mean()),
        "median": float(np.median(nn_scores)),
        "std":   float(nn_scores.std()),
    }
    log.info(f"  [{label}] mean={stats['mean']:.3f}  median={stats['median']:.3f}  std={stats['std']:.3f}")
    for t in thresholds:
        frac = float((nn_scores >= t).mean())
        stats[f"pct_geq_{t}"] = frac
        log.info(f"    frac ≥ Tanimoto {t}: {frac:.3f}")
    return stats


# ============================================================================
# ADVERSARIAL VALIDATION  (train vs PI1M classifier)
# ============================================================================

def adversarial_validation(fps_a: list, fps_b: list, log: logging.Logger,
                            label_a: str, label_b: str) -> dict[str, float]:
    """LGB classifier: label 0 for fps_a, label 1 for fps_b. Report 3-fold AUC.
    AUC ~0.5 → indistinguishable. AUC > 0.85 → very different distributions."""
    valid_a = [(fp, 0) for fp in fps_a if fp is not None]
    valid_b = [(fp, 1) for fp in fps_b if fp is not None]
    log.info(f"[adv val {label_a} vs {label_b}] n_{label_a}={len(valid_a)}  n_{label_b}={len(valid_b)}")

    all_fps = [fp for fp, _ in valid_a] + [fp for fp, _ in valid_b]
    y = np.array([lbl for _, lbl in valid_a] + [lbl for _, lbl in valid_b])

    # Convert fps to numpy dense matrix (2048-bit Morgan)
    X = np.zeros((len(all_fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(tqdm(all_fps, desc="fp→matrix", ncols=100)):
        DataStructs.ConvertToNumpyArray(fp, X[i])

    skf = StratifiedKFold(n_splits=ADVERSARIAL_TRIALS, shuffle=True, random_state=42)
    aucs = []
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, feature_fraction=0.5, bagging_fraction=0.8,
            verbosity=-1, n_jobs=-1, random_state=42,
        )
        model.fit(X[tri], y[tri], eval_set=[(X[vai], y[vai])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = model.predict_proba(X[vai])[:, 1]
        auc = float(roc_auc_score(y[vai], pred))
        aucs.append(auc)
        log.info(f"  fold {k}: AUC = {auc:.4f}")
    mean_auc = float(np.mean(aucs))
    log.info(f"[adv val {label_a} vs {label_b}] MEAN AUC = {mean_auc:.4f}  "
             f"(0.5=indistinguishable, >0.85=very different)")
    return {"mean_auc": mean_auc, "per_fold_aucs": aucs,
            f"n_{label_a}": len(valid_a), f"n_{label_b}": len(valid_b)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(EXP_DIR)
    log.info(f"=== {EXP_NAME} ===")
    log.info(f"CONFIG: PI1M_SAMPLE={PI1M_SAMPLE}  NN_BATCH={NN_BATCH}")

    random.seed(42); np.random.seed(42)
    t_start = time.time()

    # ---- Load train + test ----
    log.info("loading train.csv / test.csv")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    log.info(f"train raw: {tr.shape}   test raw: {te.shape}")

    # Canonicalize
    all_smi_tt = pd.concat([tr["smiles"], te["smiles"]]).unique()
    log.info(f"canonicalizing {len(all_smi_tt)} unique train+test SMILES")
    canon_map = {s: canonical(s) for s in tqdm(all_smi_tt, desc="canonical tt", ncols=100)}
    tr["canon"] = tr["smiles"].map(canon_map)
    te["canon"] = te["smiles"].map(canon_map)
    train_canons = sorted(set(tr["canon"].dropna()))
    test_canons = sorted(set(te["canon"].dropna()))
    log.info(f"  unique train canons: {len(train_canons)}  unique test canons: {len(test_canons)}")

    # ---- Load PI1M and sample ----
    log.info(f"loading PI1M.csv (995K rows expected)")
    pi1m_raw = pd.read_csv(DATA_DIR / "PI1M.csv")
    log.info(f"PI1M raw shape: {pi1m_raw.shape}")

    pi1m_smi = pi1m_raw["SMILES"].tolist()
    log.info(f"filtering PI1M to polymer-like (2 wildcards, no metals) — target sample {PI1M_SAMPLE}")

    pi1m_valid_canons: list[str] = []
    n_parsed = n_filtered = 0
    for smi in tqdm(pi1m_smi, desc="PI1M filter", ncols=100):
        if len(pi1m_valid_canons) >= PI1M_SAMPLE:
            break
        if not is_valid_polymer(smi):
            n_filtered += 1
            continue
        c = canonical(smi)
        if c is None:
            n_filtered += 1
            continue
        n_parsed += 1
        pi1m_valid_canons.append(c)
    log.info(f"PI1M: kept {len(pi1m_valid_canons)} valid polymer-like SMILES  "
             f"(parsed {n_parsed}, filtered {n_filtered})")
    # De-duplicate
    pi1m_valid_canons = list(dict.fromkeys(pi1m_valid_canons))
    log.info(f"PI1M after dedup: {len(pi1m_valid_canons)}")

    # Overlap with train/test?
    train_set = set(train_canons)
    test_set = set(test_canons)
    pi1m_set = set(pi1m_valid_canons)
    overlap_train = len(train_set & pi1m_set)
    overlap_test  = len(test_set & pi1m_set)
    log.info(f"canonical SMILES overlap: PI1M ∩ train = {overlap_train}   PI1M ∩ test = {overlap_test}")

    # ---- Distribution stats ----
    log.info("=" * 60)
    log.info("BASIC DISTRIBUTION COMPARISON (MolWt, LogP, TPSA, FractionCSP3, halogen_frac)")
    log.info("=" * 60)
    log.info("computing basic stats for train / test / PI1M")
    stats_train = [s for s in (compute_basic_stats(c) for c in tqdm(train_canons, desc="train stats", ncols=100)) if s]
    stats_test  = [s for s in (compute_basic_stats(c) for c in tqdm(test_canons,  desc="test stats",  ncols=100)) if s]
    stats_pi1m  = [s for s in (compute_basic_stats(c) for c in tqdm(pi1m_valid_canons, desc="PI1M stats", ncols=100)) if s]

    summ_train = summarize_dist(stats_train)
    summ_test  = summarize_dist(stats_test)
    summ_pi1m  = summarize_dist(stats_pi1m)

    log.info(f"  {'feature':>18s}  {'train':>10s}  {'test':>10s}  {'PI1M':>10s}  {'tr-pi1m Δ':>10s}  {'te-pi1m Δ':>10s}")
    for f in ["MolWt", "LogP", "TPSA", "n_heavy", "aromatic_frac", "FractionCSP3", "n_rot", "F_count", "halogen_frac"]:
        if f not in summ_train:
            continue
        tr_m = summ_train[f]["mean"]
        te_m = summ_test[f]["mean"]
        pi_m = summ_pi1m[f]["mean"]
        log.info(f"  {f:>18s}  {tr_m:>10.3f}  {te_m:>10.3f}  {pi_m:>10.3f}  "
                 f"{tr_m - pi_m:>+10.3f}  {te_m - pi_m:>+10.3f}")

    # ---- Morgan fingerprints ----
    log.info("=" * 60)
    log.info("COMPUTING MORGAN FINGERPRINTS  (r=2, 2048 bits)")
    log.info("=" * 60)
    t0 = time.time()
    fps_train = [morgan_fp(c) for c in tqdm(train_canons, desc="train fps", ncols=100)]
    fps_test  = [morgan_fp(c) for c in tqdm(test_canons,  desc="test fps",  ncols=100)]
    fps_pi1m  = [morgan_fp(c) for c in tqdm(pi1m_valid_canons, desc="PI1M fps", ncols=100)]
    log.info(f"  fingerprint time: {time.time()-t0:.1f}s")

    # Filter Nones and keep aligned
    fps_train_v = [f for f in fps_train if f is not None]
    fps_test_v  = [f for f in fps_test  if f is not None]
    fps_pi1m_v  = [f for f in fps_pi1m  if f is not None]
    log.info(f"  valid fps: train={len(fps_train_v)}  test={len(fps_test_v)}  PI1M={len(fps_pi1m_v)}")

    # ---- Nearest-neighbor Tanimoto: test vs PI1M ----
    log.info("=" * 60)
    log.info("NEAREST-NEIGHBOR TANIMOTO SIMILARITY")
    log.info("=" * 60)
    log.info("test polymers → nearest PI1M polymer:")
    nn_test = compute_nn_tanimoto(fps_test_v, fps_pi1m_v, log, "test→PI1M")
    nn_test_stats = bucket_report(nn_test, log, "test→PI1M")

    log.info("train polymers → nearest PI1M polymer:")
    nn_train = compute_nn_tanimoto(fps_train_v, fps_pi1m_v, log, "train→PI1M")
    nn_train_stats = bucket_report(nn_train, log, "train→PI1M")

    # Also test → train (control comparison)
    log.info("test polymers → nearest TRAIN polymer (control comparison):")
    nn_test_train = compute_nn_tanimoto(fps_test_v, fps_train_v, log, "test→train")
    nn_test_train_stats = bucket_report(nn_test_train, log, "test→train")

    # Save NN distributions to CSV for later plotting
    pd.DataFrame({"nn_tanimoto": nn_test}).to_csv(EXP_DIR / "nn_hist_test.csv", index=False)
    pd.DataFrame({"nn_tanimoto": nn_train}).to_csv(EXP_DIR / "nn_hist_train.csv", index=False)
    pd.DataFrame({"nn_tanimoto": nn_test_train}).to_csv(EXP_DIR / "nn_hist_test_to_train.csv", index=False)
    log.info(f"wrote nn_hist_*.csv")

    # ---- Adversarial validation ----
    log.info("=" * 60)
    log.info("ADVERSARIAL VALIDATION (Morgan-r2 classifier)")
    log.info("=" * 60)
    adv_train_pi1m = adversarial_validation(fps_train_v, fps_pi1m_v, log, "train", "pi1m")
    adv_test_pi1m  = adversarial_validation(fps_test_v,  fps_pi1m_v, log, "test",  "pi1m")
    adv_train_test = adversarial_validation(fps_train_v, fps_test_v, log, "train", "test")

    # ---- Summary + recommendation ----
    log.info("=" * 60)
    log.info("SUMMARY + RECOMMENDATION")
    log.info("=" * 60)
    log.info(f"Test → PI1M NN Tanimoto: mean={nn_test_stats['mean']:.3f}  "
             f"pct≥0.5={nn_test_stats['pct_geq_0.5']:.3f}  pct≥0.7={nn_test_stats['pct_geq_0.7']:.3f}")
    log.info(f"Train → PI1M NN Tanimoto: mean={nn_train_stats['mean']:.3f}  "
             f"pct≥0.5={nn_train_stats['pct_geq_0.5']:.3f}  pct≥0.7={nn_train_stats['pct_geq_0.7']:.3f}")
    log.info(f"Test → Train NN Tanimoto (control): mean={nn_test_train_stats['mean']:.3f}  "
             f"pct≥0.5={nn_test_train_stats['pct_geq_0.5']:.3f}")
    log.info(f"Adversarial AUC train vs PI1M: {adv_train_pi1m['mean_auc']:.4f}")
    log.info(f"Adversarial AUC test vs PI1M:  {adv_test_pi1m['mean_auc']:.4f}")
    log.info(f"Adversarial AUC train vs test: {adv_train_test['mean_auc']:.4f}")

    # Decision heuristics
    log.info("")
    log.info("DECISION:")
    frac_test_close = nn_test_stats["pct_geq_0.5"]
    train_pi1m_auc = adv_train_pi1m["mean_auc"]

    if frac_test_close >= 0.5 and train_pi1m_auc < 0.85:
        log.info(f"  ✅ RECOMMEND PROCEED with pseudo-labeling.")
        log.info(f"     {frac_test_close:.1%} of test polymers have a PI1M neighbor at Tanimoto ≥ 0.5,")
        log.info(f"     and train vs PI1M is not too distinct (AUC {train_pi1m_auc:.3f}).")
        log.info(f"     Chain-ext v1's pseudo-labels on PI1M should carry usable signal.")
        rec = "PROCEED"
    elif frac_test_close >= 0.3:
        log.info(f"  ⚠️  MARGINAL: consider pseudo-labeling but with LOW weight (0.05-0.10)")
        log.info(f"     Only {frac_test_close:.1%} of test polymers have Tanimoto ≥ 0.5 in PI1M.")
        log.info(f"     Adversarial AUC {train_pi1m_auc:.3f} suggests some distribution shift.")
        log.info(f"     Higher risk of adding noise than signal.")
        rec = "MARGINAL"
    else:
        log.info(f"  ❌ RECOMMEND SKIP pseudo-labeling.")
        log.info(f"     Only {frac_test_close:.1%} of test polymers have Tanimoto ≥ 0.5 in PI1M.")
        log.info(f"     Adversarial AUC {train_pi1m_auc:.3f}.")
        log.info(f"     PI1M covers different chemistry — pseudo-labels would be noise.")
        rec = "SKIP"

    # Write JSON summary
    summary = {
        "exp_name":  EXP_NAME,
        "config": {"pi1m_sample": PI1M_SAMPLE, "nn_batch": NN_BATCH,
                   "adversarial_trials": ADVERSARIAL_TRIALS,
                   "tanimoto_thresholds": list(TANIMOTO_THRESHOLDS)},
        "counts": {
            "train_canons":         len(train_canons),
            "test_canons":          len(test_canons),
            "pi1m_valid_canons":    len(pi1m_valid_canons),
            "overlap_pi1m_train":   overlap_train,
            "overlap_pi1m_test":    overlap_test,
        },
        "basic_stats": {
            "train": summ_train,
            "test":  summ_test,
            "pi1m":  summ_pi1m,
        },
        "nn_tanimoto": {
            "test_to_pi1m":  nn_test_stats,
            "train_to_pi1m": nn_train_stats,
            "test_to_train": nn_test_train_stats,
        },
        "adversarial_validation": {
            "train_vs_pi1m": adv_train_pi1m,
            "test_vs_pi1m":  adv_test_pi1m,
            "train_vs_test": adv_train_test,
        },
        "recommendation": rec,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "diagnostic_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"wrote {EXP_DIR / 'diagnostic_stats.json'}")
    log.info(f"wall time: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
