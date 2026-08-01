"""PI1M target-similarity slicing: what fraction of PI1M is 'usable' per target?"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
pi = pd.read_csv(f"{DATA}/PI1M.csv")

def fp(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

# Sample PI1M to 50K for the analysis
pi_sample = pi['SMILES'].dropna().sample(50000, random_state=42).unique()
print(f"PI1M sample: {len(pi_sample)} unique")

print("Computing FPs...")
tr_uniq = tr['smiles'].unique()
tr_fp_map = {s: fp(s) for s in tr_uniq}
pi_fps = [fp(s) for s in tqdm(pi_sample, ncols=80)]
pi_valid = [(s, f) for s, f in zip(pi_sample, pi_fps) if f is not None]
print(f"PI1M valid: {len(pi_valid)}")

# For each target: nearest same-target train NN for each PI1M mol
# If any target-specific NN > threshold, that PI1M mol is "usable" for that target
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(f"\n=== per target: PI1M coverage by max-Tanimoto-to-same-target-train ===")
print(f"  target  n_pi1m  " + " ".join([f"%>{t:.1f}   " for t in THRESHOLDS]))
per_target_usable = {}
for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt]
    tr_fps = [tr_fp_map[s] for s in g['smiles'] if tr_fp_map[s] is not None]
    if not tr_fps:
        continue
    max_sims = []
    for s, f in pi_valid:
        sims = DataStructs.BulkTanimotoSimilarity(f, tr_fps)
        max_sims.append(max(sims))
    m = np.array(max_sims)
    per_target_usable[tt] = m
    print(f"  {tt:>4s}  {len(m):>5d}  " + " ".join([f"{100*(m>t).mean():>5.1f}%" for t in THRESHOLDS]))

# Save the per-target "distance from train" arrays for later use
np.savez_compressed(
    "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures/pi1m_max_sim_per_target.npz",
    smiles=[s for s,_ in pi_valid],
    **{f"sim_{tt}": per_target_usable[tt] for tt in per_target_usable}
)
print("\nsaved: docs/figures/pi1m_max_sim_per_target.npz")
