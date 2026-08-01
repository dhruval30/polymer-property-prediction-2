"""Tanimoto NN similarity: how OOD is the test set per target?"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

# Precompute FPs for all UNIQUE smiles once
all_smi = pd.concat([tr['smiles'], te['smiles']]).unique()
print(f"Computing Morgan-r2 FPs for {len(all_smi)} unique SMILES...")
fp_map = {s: fp(s) for s in tqdm(all_smi, ncols=80)}

# For each target: compute nearest-neighbor Tanimoto for each test row
# vs. TRAIN rows OF THE SAME TARGET (target-specific NN)
print("\n=== per-target: nearest-neighbor Tanimoto (test row → same-target train row) ===")
print(f"  {'target':>6s} {'n_test':>7s} {'nn_min':>8s} {'nn_q10':>8s} {'nn_med':>8s} {'nn_q90':>8s} {'nn_max':>8s} {'%<.3':>7s} {'%<.5':>7s} {'%>.9':>7s}")
results = {}
for tt in sorted(tr['target_type'].unique()):
    tr_tt = tr[tr['target_type']==tt]
    te_tt = te[te['target_type']==tt]
    tr_fps = [fp_map[s] for s in tr_tt['smiles'] if fp_map[s] is not None]
    nn_sims = []
    for s in te_tt['smiles']:
        f = fp_map[s]
        if f is None:
            nn_sims.append(np.nan); continue
        sims = DataStructs.BulkTanimotoSimilarity(f, tr_fps)
        nn_sims.append(max(sims))
    nn = np.array(nn_sims)
    results[tt] = nn
    print(f"  {tt:>6s} {len(nn):>7d} {nn.min():>8.3f} {np.quantile(nn,.1):>8.3f} {np.median(nn):>8.3f} {np.quantile(nn,.9):>8.3f} {nn.max():>8.3f} {100*(nn<.3).mean():>6.1f}% {100*(nn<.5).mean():>6.1f}% {100*(nn>.9).mean():>6.1f}%")

# Also: NN vs ALL train (not just same target). This tells us if PI1M / multitask feats help.
print("\n=== per-target: nearest-neighbor Tanimoto (test row → ANY train row, all targets pooled) ===")
tr_all_fps = [fp_map[s] for s in tr['smiles'].unique() if fp_map[s] is not None]
print(f"  {'target':>6s} {'n_test':>7s} {'nn_min':>8s} {'nn_q10':>8s} {'nn_med':>8s} {'nn_q90':>8s} {'nn_max':>8s} {'%<.3':>7s} {'%<.5':>7s} {'%>.9':>7s}")
for tt in sorted(te['target_type'].unique()):
    te_tt = te[te['target_type']==tt]
    nn_sims = []
    for s in te_tt['smiles']:
        f = fp_map[s]
        if f is None:
            nn_sims.append(np.nan); continue
        sims = DataStructs.BulkTanimotoSimilarity(f, tr_all_fps)
        nn_sims.append(max(sims))
    nn = np.array(nn_sims)
    print(f"  {tt:>6s} {len(nn):>7d} {nn.min():>8.3f} {np.quantile(nn,.1):>8.3f} {np.median(nn):>8.3f} {np.quantile(nn,.9):>8.3f} {nn.max():>8.3f} {100*(nn<.3).mean():>6.1f}% {100*(nn<.5).mean():>6.1f}% {100*(nn>.9).mean():>6.1f}%")

# Save the per-target NN vectors so we can plot histograms in the doc
np.savez_compressed(
    "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures/tanimoto_nn.npz",
    **{tt: results[tt] for tt in results}
)
print("\nsaved: docs/figures/tanimoto_nn.npz")
