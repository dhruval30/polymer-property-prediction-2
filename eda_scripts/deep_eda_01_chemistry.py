"""Chemistry-aware descriptor EDA per target."""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
OUT = "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def cap_wildcards_with_methyl(smi):
    """Replace * with C so descriptors that dislike wildcards behave."""
    return smi.replace('*', 'C')

def mol_props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m_cap = Chem.MolFromSmiles(cap_wildcards_with_methyl(smi))
    if m_cap is None:
        m_cap = m
    return {
        'HeavyAtoms': m.GetNumHeavyAtoms(),
        'NumAtoms': m.GetNumAtoms(),
        'NumRings': rdMolDescriptors.CalcNumRings(m),
        'NumAromRings': rdMolDescriptors.CalcNumAromaticRings(m),
        'NumAliphRings': rdMolDescriptors.CalcNumAliphaticRings(m),
        'NumRotBonds': rdMolDescriptors.CalcNumRotatableBonds(m),
        'NumHBD': rdMolDescriptors.CalcNumHBD(m),
        'NumHBA': rdMolDescriptors.CalcNumHBA(m),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(m),
        'MolWt': Descriptors.MolWt(m_cap),
        'HeavyMolWt': Descriptors.HeavyAtomMolWt(m_cap),
        'MolLogP': Descriptors.MolLogP(m_cap),
        'TPSA': Descriptors.TPSA(m_cap),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(m),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(m),
        'BertzCT': Descriptors.BertzCT(m_cap),
        'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(m),
    }

# Compute for all unique SMILES first (cache)
all_smiles = pd.concat([tr['smiles'], te['smiles']]).unique()
print(f"unique SMILES total: {len(all_smiles)}")
props = {}
for s in all_smiles:
    p = mol_props(s)
    if p:
        props[s] = p

df_all = pd.DataFrame.from_dict(props, orient='index').reset_index().rename(columns={'index':'smiles'})
df_all.to_csv(f"{DATA}/../docs/figures/all_mol_props.csv", index=False)

# Per-target stats (train only, one row per SMILES per target)
tr_x = tr.merge(df_all, on='smiles', how='left')
te_x = te.merge(df_all, on='smiles', how='left')

# Number of test-only vs train-only SMILES
tr_s = set(tr['smiles']); te_s = set(te['smiles'])
print(f"\ntrain-only SMILES: {len(tr_s - te_s)}")
print(f"test-only  SMILES: {len(te_s - tr_s)}")
print(f"shared     SMILES: {len(tr_s & te_s)}")

props_to_summarize = ['HeavyAtoms','NumRings','NumAromRings','NumAliphRings','NumRotBonds',
                       'NumHBD','NumHBA','FractionCSP3','MolWt','MolLogP','TPSA',
                       'NumHeteroatoms','BertzCT']

print("\n=== per-target chemistry summary (mean ± std) ===")
summary = tr_x.groupby('target_type')[props_to_summarize].agg(['mean','std','min','max','median'])
# Print compact table
for col in props_to_summarize:
    print(f"\n{col}")
    print(f"  {'target':>6s} {'mean':>10s} {'std':>10s} {'min':>10s} {'med':>10s} {'max':>10s}")
    for tt, g in tr_x.groupby('target_type'):
        v = g[col].dropna()
        print(f"  {tt:>6s} {v.mean():>10.2f} {v.std():>10.2f} {v.min():>10.2f} {v.median():>10.2f} {v.max():>10.2f}")

# test set counterpart to spot drift
print("\n=== TRAIN vs TEST per-target chemistry (mean) ===")
print(f"  {'target':>6s} {'MolWt tr/te':>18s} {'HeavyAtoms tr/te':>22s} {'NumRings tr/te':>20s} {'MolLogP tr/te':>18s}")
for tt in sorted(tr_x['target_type'].unique()):
    gr = tr_x[tr_x['target_type']==tt]
    ge = te_x[te_x['target_type']==tt]
    print(f"  {tt:>6s}   "
          f"{gr['MolWt'].mean():>7.1f}/{ge['MolWt'].mean():>7.1f}     "
          f"{gr['HeavyAtoms'].mean():>7.1f}/{ge['HeavyAtoms'].mean():>7.1f}       "
          f"{gr['NumRings'].mean():>7.2f}/{ge['NumRings'].mean():>7.2f}     "
          f"{gr['MolLogP'].mean():>7.2f}/{ge['MolLogP'].mean():>7.2f}")

# Backbone: distance between the 2 wildcard atoms in the graph
def backbone_atoms(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    star_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol()=='*']
    if len(star_idx) != 2: return None
    try:
        path = Chem.GetShortestPath(m, star_idx[0], star_idx[1])
        return len(path)  # number of atoms in shortest path incl. both ends
    except Exception:
        return None

sample = df_all['smiles'].values
back = [backbone_atoms(s) for s in sample]
df_all['BackboneAtoms'] = back
tr_x = tr.merge(df_all[['smiles','BackboneAtoms']], on='smiles', how='left')
te_x = te.merge(df_all[['smiles','BackboneAtoms']], on='smiles', how='left')

print("\n=== backbone atom count (shortest path between the two '*' atoms, per target) ===")
print(f"  {'target':>6s} {'mean':>8s} {'std':>8s} {'min':>4s} {'med':>4s} {'max':>4s} {'na':>4s}")
for tt in sorted(tr_x['target_type'].unique()):
    g = tr_x[tr_x['target_type']==tt]
    v = g['BackboneAtoms'].dropna()
    print(f"  {tt:>6s} {v.mean():>8.2f} {v.std():>8.2f} {int(v.min()):>4d} {int(v.median()):>4d} {int(v.max()):>4d} {g['BackboneAtoms'].isna().sum():>4d}")

df_all.to_csv(f"{OUT}/all_mol_props.csv", index=False)
print(f"\nsaved: {OUT}/all_mol_props.csv")
