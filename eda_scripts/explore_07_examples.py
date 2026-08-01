"""Sample molecules from bottom-10 / median / top-10 percentiles of each target."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

for tt in ['tg','egc','egb','eea','ei','eps','nc']:
    g = tr[tr['target_type']==tt].sort_values('target').reset_index(drop=True)
    n = len(g)
    print("=" * 60)
    print(f"{tt}   n={n}, range=[{g['target'].min():.4f}, {g['target'].max():.4f}], median={g['target'].median():.4f}")
    print("=" * 60)

    print(f"\n  --- LOWEST 5 ---")
    for _, r in g.head(5).iterrows():
        s = r['smiles']
        if len(s) > 90: s = s[:87] + '...'
        print(f"    {r['target']:>10.4f}   {s}")

    print(f"\n  --- AROUND MEDIAN (5 rows) ---")
    mid = n // 2
    for _, r in g.iloc[mid-2:mid+3].iterrows():
        s = r['smiles']
        if len(s) > 90: s = s[:87] + '...'
        print(f"    {r['target']:>10.4f}   {s}")

    print(f"\n  --- HIGHEST 5 ---")
    for _, r in g.tail(5).iterrows():
        s = r['smiles']
        if len(s) > 90: s = s[:87] + '...'
        print(f"    {r['target']:>10.4f}   {s}")
    print()
