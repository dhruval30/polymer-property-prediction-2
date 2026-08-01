"""Signal-to-noise: estimate measurement noise from duplicate measurements + variance across close chemistry."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

def canon(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToSmiles(m) if m else None

tr['canon'] = tr['smiles'].map(canon)

# Method 1: same (canon, target_type) duplicates
print("=== METHOD 1: measurement noise from same-mol same-target duplicates ===")
for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt]
    dups = g.groupby('canon')['target'].agg(['nunique','count','min','max','std','mean']).reset_index()
    dups = dups[dups['count']>1]
    if len(dups)==0:
        print(f"  {tt}: 0 duplicate measurements")
        continue
    dups['range'] = dups['max'] - dups['min']
    print(f"  {tt}: n_dup_groups={len(dups)}  mean range={dups['range'].mean():.4f}  mean std={dups['std'].mean():.4f}  max range={dups['range'].max():.4f}")

# Method 2: noise floor via near-neighbor variance
print("\n=== METHOD 2: variance among near-neighbors (Tanimoto > 0.90) ===")
def fp(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

uniq_smi = tr['smiles'].unique()
print(f"  computing FPs for {len(uniq_smi)} SMILES...")
fps = {s: fp(s) for s in uniq_smi}
print(f"  done. now searching NNs per target (Tanimoto > 0.9)...")

for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt].reset_index(drop=True)
    if len(g) < 20:
        print(f"  {tt}: n<20"); continue
    smis = g['smiles'].tolist(); ys = g['target'].astype(float).values
    fps_g = [fps[s] for s in smis]
    # for each mol, find same-target NNs with sim > 0.9
    nn_diffs = []
    for i, f in enumerate(fps_g):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, fps_g))
        sims[i] = 0  # exclude self
        nb = np.where(sims > 0.9)[0]
        for j in nb:
            nn_diffs.append(abs(ys[i] - ys[j]))
    if not nn_diffs:
        print(f"  {tt}: no same-target neighbors with sim>0.9"); continue
    d = np.array(nn_diffs)
    # implied R² ceiling: 1 - (MSE_noise / Var(y))
    y_var = np.var(ys)
    mse_nn = np.mean(d**2)
    r2_ceil = max(0, 1 - mse_nn / (2*y_var))  # factor 2 because both mols contribute
    print(f"  {tt}: n_NN_pairs={len(d):>6d}  mean|Δ|={d.mean():.4f}  median|Δ|={np.median(d):.4f}  MSE_NN={mse_nn:.4f}  implied R² ceiling≈{r2_ceil:.3f}")

# Method 3: coefficient of variation
print("\n=== TARGET VARIANCE + BASELINE (constant-predictor R²=0) ===")
for tt in sorted(tr['target_type'].unique()):
    v = tr[tr['target_type']==tt]['target'].astype(float)
    print(f"  {tt}: mean={v.mean():.4f}  std={v.std():.4f}  cov={v.std()/abs(v.mean()):.3f}  var={v.var():.4f}")
