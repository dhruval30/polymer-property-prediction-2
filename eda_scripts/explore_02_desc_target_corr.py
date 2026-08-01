"""RDKit descriptor → per-target correlations. Pearson + Spearman, top 15 each direction."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

def all_desc(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))

# One row per unique SMILES first (cache)
uniq = tr['smiles'].unique()
print(f"Computing descriptors for {len(uniq)} unique train SMILES...")
d_map = {}
for i, s in enumerate(uniq):
    if i % 1000 == 0: print(f"  {i}/{len(uniq)}")
    d = all_desc(s)
    if d: d_map[s] = d
print("done")

df = pd.DataFrame.from_dict(d_map, orient='index').reset_index().rename(columns={'index':'smiles'})
# clean inf, drop constants
df = df.replace([np.inf, -np.inf], np.nan)
for c in df.columns:
    if c=='smiles': continue
    df[c] = df[c].fillna(df[c].median())
const_cols = [c for c in df.columns if c!='smiles' and df[c].nunique()<=1]
df = df.drop(columns=const_cols)
print(f"dropped {len(const_cols)} constant columns; {df.shape[1]-1} descriptors remain")

merged = tr.merge(df, on='smiles', how='left')

descriptors = [c for c in df.columns if c!='smiles']
print(f"\n=== TOP DESCRIPTOR → TARGET CORRELATIONS (Pearson, |r| ranked) ===\n")
for tt in sorted(tr['target_type'].unique()):
    g = merged[merged['target_type']==tt].dropna(subset=descriptors)
    y = g['target'].values
    rows = []
    for d in descriptors:
        x = g[d].values
        if np.std(x) < 1e-9: continue
        res = pearsonr(x, y)
        r = float(res.statistic if hasattr(res,'statistic') else res[0])
        rows.append((d, r))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"\n--- {tt} (n={len(g)}) ---")
    print(f"  TOP 15 POSITIVE:")
    pos = [r for r in rows if r[1] > 0][:15]
    for d, r in pos:
        print(f"    {r:>+7.3f}   {d}")
    print(f"  TOP 15 NEGATIVE:")
    neg = [r for r in rows if r[1] < 0][:15]
    for d, r in neg:
        print(f"    {r:>+7.3f}   {d}")

# Spearman too, only top-5 to keep output short
print(f"\n\n=== TOP 5 (Spearman |r|) PER TARGET ===\n")
for tt in sorted(tr['target_type'].unique()):
    g = merged[merged['target_type']==tt].dropna(subset=descriptors)
    y = g['target'].values
    rows = []
    for d in descriptors:
        x = g[d].values
        if np.std(x) < 1e-9: continue
        sr = spearmanr(x, y)
        r = float(sr.statistic if hasattr(sr,'statistic') else sr.correlation)
        rows.append((d, r))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"  {tt}: " + ", ".join([f"{d}({r:+.2f})" for d,r in rows[:5]]))
