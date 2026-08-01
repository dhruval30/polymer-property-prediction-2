"""Rare-atom and unusual-structure hunt."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def elems(smi):
    m = Chem.MolFromSmiles(smi)
    return set(a.GetSymbol() for a in m.GetAtoms()) if m else set()

all_smi = pd.concat([tr['smiles'], te['smiles']]).unique()
print(f"Scanning {len(all_smi)} unique SMILES for rare atoms...")
elem_map = {s: elems(s) for s in all_smi}
# universe
allelem = set().union(*elem_map.values())
print(f"element universe: {sorted(allelem)}\n")

RARE = ['Pb','Cd','K','Li','Se','Ge','Sn','Na','B','As','Fe','I','P','Si']
print(f"=== per rare-atom: count of molecules containing it ===")
for e in RARE:
    smiles_with = [s for s, elset in elem_map.items() if e in elset]
    n_train = tr[tr['smiles'].isin(smiles_with)].shape[0]
    n_test  = te[te['smiles'].isin(smiles_with)].shape[0]
    print(f"  {e:>4s}  train_rows={n_train:>5d}  test_rows={n_test:>4d}  unique_mols={len(smiles_with):>4d}")

# For the rarest, show examples
for e in ['Pb','Cd','K','Li','As','Fe']:
    smis = [s for s, els in elem_map.items() if e in els][:5]
    if smis:
        print(f"\n  Example SMILES containing {e}:")
        for s in smis:
            if len(s)>90: s = s[:87]+'...'
            print(f"    {s}")

# Unusual bond types
print(f"\n=== bond types across all SMILES ===")
from collections import Counter
btype_c = Counter()
for s in all_smi[:3000]:  # sample for speed
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    for b in m.GetBonds():
        btype_c[str(b.GetBondType())] += 1
for bt, n in btype_c.most_common():
    print(f"  {bt}: {n}")

# Molecules with 0 rings
noring = [s for s in all_smi if Chem.MolFromSmiles(s) and Chem.MolFromSmiles(s).GetRingInfo().NumRings() == 0]
print(f"\n=== molecules with ZERO rings (pure acyclic) ===")
print(f"  count: {len(noring)} / {len(all_smi)} = {100*len(noring)/len(all_smi):.1f}%")

# Very large molecules
def na(s):
    m = Chem.MolFromSmiles(s); return m.GetNumHeavyAtoms() if m else 0
sizes = {s: na(s) for s in all_smi}
big = sorted(sizes.items(), key=lambda x: -x[1])[:5]
print(f"\n=== top 5 largest molecules (by heavy atoms) ===")
for s, n in big:
    ss = s if len(s)<100 else s[:97]+'...'
    print(f"  n={n:>4d}  {ss}")

# smallest
smallest = sorted(sizes.items(), key=lambda x: x[1])[:5]
print(f"\n=== smallest 5 molecules ===")
for s, n in smallest:
    print(f"  n={n:>4d}  {s}")
