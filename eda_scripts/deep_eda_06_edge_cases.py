"""Outliers, edge cases, RDKit descriptor degeneracy, GroupKFold viability."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.model_selection import GroupKFold, StratifiedKFold
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

# ============================================================
# 1. tg outliers — what are the extreme values?
# ============================================================
print("=" * 60)
print("tg extreme values")
print("=" * 60)
tg = tr[tr['target_type']=='tg'].sort_values('target')
print("\n=== 5 smallest tg ===")
print(tg.head(5)[['target','smiles']].to_string(index=False))
print("\n=== 5 largest tg ===")
print(tg.tail(5)[['target','smiles']].to_string(index=False))
print(f"\ntg zeros: {(tg['target']==0).sum()}")
print(tg[tg['target']==0][['smiles','target']].to_string(index=False))
print(f"\ntg negatives (n={(tg['target']<0).sum()}):")
print(f"  min: {tg['target'].min()}")
print(f"  q10: {tg['target'].quantile(.10)}")
print(f"  fraction with target < 0: {(tg['target']<0).mean():.3f}")
print(f"  fraction with target < 25: {(tg['target']<25).mean():.3f}")
print(f"  fraction with target > 300: {(tg['target']>300).mean():.3f}")

# ============================================================
# 2. RDKit descriptor degeneracy / inf/NaN on our data
# ============================================================
print("\n" + "=" * 60)
print("RDKit descriptor inf/NaN rates")
print("=" * 60)

def cap(smi): return smi.replace('*','C')

def all_desc(smi):
    m = Chem.MolFromSmiles(cap(smi))
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))

# Sample 2000 unique smiles from train+test
sample = pd.concat([tr['smiles'], te['smiles']]).unique()
sample = pd.Series(sample).sample(2000, random_state=0).tolist()
descs = [all_desc(s) for s in sample]
descs = [d for d in descs if d]
df_d = pd.DataFrame(descs)
print(f"\nsample size: {len(df_d)} unique SMILES")
print(f"total descriptors computed: {df_d.shape[1]}")

n_inf = np.isinf(df_d.values).sum()
n_nan = df_d.isna().sum().sum()
print(f"\ntotal ±inf cells: {n_inf}")
print(f"total NaN cells:  {n_nan}")

# per-column
inf_per_col = np.isinf(df_d.values).sum(axis=0)
nan_per_col = df_d.isna().sum().values
bad_cols = [(c, int(inf_per_col[i]), int(nan_per_col[i])) for i, c in enumerate(df_d.columns) if inf_per_col[i] > 0 or nan_per_col[i] > 0]
print(f"\ncolumns with any inf or NaN: {len(bad_cols)}")
for c, ninf, nnan in sorted(bad_cols, key=lambda x: -(x[1]+x[2]))[:15]:
    print(f"  {c:>30s}  inf={ninf:>4d}  nan={nnan:>4d}")

# degenerate columns (constant)
constant_cols = [c for c in df_d.columns if df_d[c].nunique(dropna=False) <= 1]
print(f"\nconstant / degenerate descriptor columns: {len(constant_cols)}")
if constant_cols:
    print(f"  first 10: {constant_cols[:10]}")

# ============================================================
# 3. GroupKFold viability by smiles
# ============================================================
print("\n" + "=" * 60)
print("GroupKFold(5) by SMILES viability")
print("=" * 60)
for tt, g in tr.groupby('target_type'):
    n_uniq = g['smiles'].nunique()
    if n_uniq < 5:
        print(f"  {tt}: only {n_uniq} unique SMILES — cannot 5-fold group")
        continue
    gkf = GroupKFold(n_splits=5)
    fold_sizes = []
    fold_uniq = []
    for tr_idx, val_idx in gkf.split(g, groups=g['smiles']):
        fold_sizes.append(len(val_idx))
        fold_uniq.append(g.iloc[val_idx]['smiles'].nunique())
    print(f"  {tt:>4s}  n={len(g):>4d}  n_unique={n_uniq:>4d}  fold sizes: {fold_sizes}  fold unique groups: {fold_uniq}")

# ============================================================
# 4. Quantile-stratified fold viability (Round 1 approach)
# ============================================================
print("\n" + "=" * 60)
print("StratifiedKFold(5) on quantile bins by target — viable per target?")
print("=" * 60)
for tt, g in tr.groupby('target_type'):
    v = g['target'].values.astype(float)
    if len(v) < 10:
        continue
    try:
        bins = pd.qcut(v, q=10, labels=False, duplicates='drop')
    except Exception as e:
        bins = np.zeros_like(v, dtype=int)
    n_uniq_bins = pd.Series(bins).nunique()
    # is any bin size <5? then stratified 5-fold might fail
    min_bin = pd.Series(bins).value_counts().min()
    print(f"  {tt:>4s}  n={len(v):>4d}  unique quantile bins: {n_uniq_bins}  smallest bin size: {min_bin}")

# ============================================================
# 5. Cross-target overlap in test — does a single test SMILES need multiple targets?
# ============================================================
print("\n" + "=" * 60)
print("Test SMILES with multiple target_types asked")
print("=" * 60)
te_gr = te.groupby('smiles')['target_type'].apply(list)
n_multi = sum(1 for lst in te_gr if len(lst) > 1)
print(f"# test SMILES with ≥2 target_types asked: {n_multi}")
print("\nfor multi-target test SMILES, what's the combo distribution?")
print(te_gr.map(lambda lst: '+'.join(sorted(set(lst)))).value_counts().head(15))
