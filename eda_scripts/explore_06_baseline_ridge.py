"""Baseline: per-target GroupKFold Ridge on RDKit descriptors — establishes score floor."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

def canon(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToSmiles(m, canonical=True) if m else None
def desc(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    return dict(Descriptors.CalcMolDescriptors(m))

uniq = tr['smiles'].unique()
print(f"Featurizing {len(uniq)} unique SMILES with RDKit descriptors...")
d_map = {s: desc(s) for s in uniq}
canon_map = {s: canon(s) for s in uniq}
print("done")
df_d = pd.DataFrame.from_dict(d_map, orient='index').reset_index().rename(columns={'index':'smiles'})

# clean
df_d = df_d.replace([np.inf, -np.inf], np.nan)
for c in df_d.columns:
    if c == 'smiles': continue
    df_d[c] = df_d[c].fillna(df_d[c].median())
const = [c for c in df_d.columns if c!='smiles' and df_d[c].nunique()<=1]
df_d = df_d.drop(columns=const)
print(f"dropped {len(const)} constant cols; features = {df_d.shape[1]-1}")

merged = tr.merge(df_d, on='smiles', how='left')
merged['canon'] = merged['smiles'].map(canon_map)
feats = [c for c in df_d.columns if c != 'smiles']

# Defensive: clip extreme feature values (99.5 pct)
for c in feats:
    q_lo, q_hi = merged[c].quantile(0.005), merged[c].quantile(0.995)
    merged[c] = merged[c].clip(q_lo, q_hi)

# Baselines
print(f"\n=== BASELINE PREDICTORS PER TARGET ===")
print(f"  target       n  mean-pred R²      Ridge-desc R²")
print(f"  -----------------------------------------------")
for tt in sorted(merged['target_type'].unique()):
    g = merged[merged['target_type']==tt].reset_index(drop=True)
    y = g['target'].astype(float).values
    # mean predictor (should be ~0 by construction of R²)
    mean_r2 = r2_score(y, np.full_like(y, y.mean()))

    # 5-fold GroupKFold Ridge with y-standardization + robust alpha range
    X = g[feats].values
    groups = g['canon'].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros_like(y)
    for tri, vai in gkf.split(X, groups=groups):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X[tri]); Xv = scaler.transform(X[vai])
        ymu, ysd = y[tri].mean(), y[tri].std()
        ynorm = (y[tri] - ymu) / max(ysd, 1e-9)
        model = RidgeCV(alphas=np.logspace(0, 5, 30)).fit(Xs, ynorm)
        oof[vai] = model.predict(Xv) * ysd + ymu
    ridge_r2 = r2_score(y, oof)
    print(f"  {tt:>4s}  {len(g):>5d}  {mean_r2:>10.3f}       {ridge_r2:>10.3f}")
