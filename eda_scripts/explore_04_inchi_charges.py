"""InChI dedup + formal charge + valence audit."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def canon(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToSmiles(m, canonical=True) if m else None
def inchi_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return MolToInchiKey(m)
    except: return None

all_s = pd.concat([tr['smiles'], te['smiles']]).unique()
print(f"Computing InChI keys for {len(all_s)} unique SMILES...")
inchi_map = {s: inchi_key(s) for s in all_s}
canon_map = {s: canon(s) for s in all_s}
print("done")

tr['canon'] = tr['smiles'].map(canon_map)
tr['ik'] = tr['smiles'].map(inchi_map)
te['canon'] = te['smiles'].map(canon_map)
te['ik'] = te['smiles'].map(inchi_map)

print("\n=== dedup rung: raw < canonical < InChI ===")
print(f"  train unique raw:    {tr['smiles'].nunique()}")
print(f"  train unique canon:  {tr['canon'].nunique()}")
print(f"  train unique InChI:  {tr['ik'].nunique()}")
print(f"  test  unique raw:    {te['smiles'].nunique()}")
print(f"  test  unique canon:  {te['canon'].nunique()}")
print(f"  test  unique InChI:  {te['ik'].nunique()}")

# canon vs inchi extra dupe?
c_over_i = tr['canon'].nunique() - tr['ik'].nunique()
print(f"\n  InChI collapses {c_over_i} additional train pairs beyond canonical (usually stereo isomers)")

# Train↔test overlap under each dedup
tr_c = set(tr['canon']); te_c = set(te['canon'])
tr_i = set(tr['ik'].dropna()); te_i = set(te['ik'].dropna())
print(f"\n=== train↔test overlap ===")
print(f"  by raw:    {len(set(tr['smiles']) & set(te['smiles']))}")
print(f"  by canon:  {len(tr_c & te_c)}")
print(f"  by InChI:  {len(tr_i & te_i)}")

# any (InChI, target_type) dupes in train with disagreeing values?
d = tr.groupby(['ik','target_type'])['target'].agg(['nunique','min','max','mean','count']).reset_index()
d = d[d['nunique']>1]
print(f"\n(InChI, target_type) dupes in train with disagreement: {len(d)}")
print(d.head(10).to_string(index=False))

# Charge / valence audit
def charge_valence_stats(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    total = sum(charges)
    n_charged = sum(1 for c in charges if c != 0)
    return {'total_formal_charge': total, 'n_atoms_with_charge': n_charged}

cv_map = {s: charge_valence_stats(s) for s in all_s}
tr['tot_q'] = tr['smiles'].map(lambda s: cv_map[s]['total_formal_charge'] if cv_map[s] else None)
tr['n_charged'] = tr['smiles'].map(lambda s: cv_map[s]['n_atoms_with_charge'] if cv_map[s] else None)
te['tot_q'] = te['smiles'].map(lambda s: cv_map[s]['total_formal_charge'] if cv_map[s] else None)
te['n_charged'] = te['smiles'].map(lambda s: cv_map[s]['n_atoms_with_charge'] if cv_map[s] else None)

print("\n=== formal charges ===")
print(f"train: total_charge value_counts: {tr['tot_q'].value_counts().to_dict()}")
print(f"train: n_charged_atoms value_counts: {tr['n_charged'].value_counts().to_dict()}")
print(f"test:  total_charge value_counts: {te['tot_q'].value_counts().to_dict()}")
print(f"test:  n_charged_atoms value_counts: {te['n_charged'].value_counts().to_dict()}")

# Show any train molecule with charged atoms
charged_tr = tr[tr['n_charged'] > 0]
print(f"\n=== train molecules with formally-charged atoms: {len(charged_tr)}")
if len(charged_tr):
    print(charged_tr[['smiles','target','target_type','tot_q','n_charged']].head(10).to_string(index=False))
charged_te = te[te['n_charged'] > 0]
print(f"\n=== test molecules with formally-charged atoms: {len(charged_te)}")
