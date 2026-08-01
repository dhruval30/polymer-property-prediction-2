"""Morgan bit frequency + per-target discrimination."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pointbiserialr
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

NBITS = 2048
def fp(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=NBITS)
    return np.array(fp, dtype=np.int8)

uniq = tr['smiles'].unique()
print(f"Computing Morgan-r2 FPs for {len(uniq)} unique train SMILES...")
fps = {s: fp(s) for s in uniq}
print("done")

# Build per-target matrix
tr = tr.copy()
tr['fp'] = tr['smiles'].map(fps)
tr = tr.dropna(subset=['fp'])

print(f"\n=== per-target: total bit density (mean # of ON bits per molecule) ===")
for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt]
    mat = np.stack(g['fp'].values)
    print(f"  {tt}: n={len(g)}, mean on-bits={mat.mean(0).sum():.1f}, per-mol density = {mat.sum(1).mean():.1f}")

print(f"\n=== per-target: top 10 most-frequent Morgan bits ===")
for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt]
    mat = np.stack(g['fp'].values)
    freq = mat.mean(0)
    top10 = np.argsort(-freq)[:10]
    print(f"\n  {tt}: bits {list(top10)} freq {[float(f'{freq[i]:.2f}') for i in top10]}")

# Bits that DISCRIMINATE targets: for each target, find bits whose activation-rate is very different vs OTHER targets
print(f"\n=== per-target: top 10 bits with highest activation-rate contrast vs other targets ===")
targets = sorted(tr['target_type'].unique())
for tt in targets:
    g = tr[tr['target_type']==tt]
    other = tr[tr['target_type']!=tt]
    mat_t = np.stack(g['fp'].values); mat_o = np.stack(other['fp'].values)
    rt = mat_t.mean(0); ro = mat_o.mean(0)
    diff = rt - ro
    top10 = np.argsort(-diff)[:10]
    bot10 = np.argsort(diff)[:10]
    print(f"\n  {tt}: MORE-FREQUENT (bit: rate_this/rate_others):")
    for b in top10:
        print(f"    bit {int(b):>4d}: {rt[b]:.2f} / {ro[b]:.2f}   Δ={diff[b]:+.2f}")
    print(f"  {tt}: LESS-FREQUENT:")
    for b in bot10:
        print(f"    bit {int(b):>4d}: {rt[b]:.2f} / {ro[b]:.2f}   Δ={diff[b]:+.2f}")

# For each target, find top-15 bits by |point-biserial correlation| with target value
print(f"\n=== per-target: TOP 15 BITS by point-biserial correlation with target value ===")
for tt in targets:
    g = tr[tr['target_type']==tt]
    if len(g) < 20:
        print(f"  {tt}: n<20, skip"); continue
    mat = np.stack(g['fp'].values); y = g['target'].astype(float).values
    rs = []
    for b in range(NBITS):
        col = mat[:, b].astype(float)
        if col.std() < 1e-9: continue
        try:
            res = pointbiserialr(col, y)
            r = float(res.statistic if hasattr(res,'statistic') else res[0])
        except Exception:
            r = 0.0
        rs.append((b, r))
    rs.sort(key=lambda x: -abs(x[1]))
    print(f"\n  {tt}:")
    for b, r in rs[:15]:
        freq = mat[:, b].mean()
        print(f"    bit {int(b):>4d}: r={r:>+7.3f}  freq_in_target={freq:.2f}")
