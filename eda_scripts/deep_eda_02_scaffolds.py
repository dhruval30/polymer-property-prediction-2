"""Scaffold analysis + canonical-SMILES dedup."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m, canonical=True)

def scaffold(smi):
    m = Chem.MolFromSmiles(smi.replace('*', 'C'))
    if m is None: return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc, canonical=True) if sc else ''
    except Exception:
        return None

def scaffold_generic(smi):
    m = Chem.MolFromSmiles(smi.replace('*', 'C'))
    if m is None: return None
    try:
        sc = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m))
        return Chem.MolToSmiles(sc, canonical=True) if sc else ''
    except Exception:
        return None

# ============================================================
# 1. Canonical SMILES dedup (beyond raw string dedup)
# ============================================================
print("=" * 60)
print("CANONICAL SMILES DEDUP")
print("=" * 60)
all_smi = pd.concat([tr[['smiles']], te[['smiles']]], ignore_index=True)['smiles'].unique()
canon_map = {s: canon(s) for s in all_smi}
tr['canon'] = tr['smiles'].map(canon_map)
te['canon'] = te['smiles'].map(canon_map)

print(f"\ntrain unique raw smiles:  {tr['smiles'].nunique()}")
print(f"train unique canon smiles: {tr['canon'].nunique()}")
print(f"delta (raw→canon collapse in train): {tr['smiles'].nunique() - tr['canon'].nunique()}")

print(f"\ntest unique raw smiles:  {te['smiles'].nunique()}")
print(f"test unique canon smiles: {te['canon'].nunique()}")
print(f"delta (raw→canon collapse in test): {te['smiles'].nunique() - te['canon'].nunique()}")

# Check for (canon, target_type) duplicates in train
canon_dupes_tr = tr.groupby(['canon','target_type']).size().reset_index(name='n').query("n>1")
print(f"\n(canon, target_type) dupes in train: {len(canon_dupes_tr)}")
# any additional dupes vs raw?
raw_dupes_tr = tr.groupby(['smiles','target_type']).size().reset_index(name='n').query("n>1")
print(f"(raw,   target_type) dupes in train: {len(raw_dupes_tr)}")
print(f"EXTRA dupes exposed only by canonicalization: {len(canon_dupes_tr) - len(raw_dupes_tr)}")

# check disagreement among canon dupes
disagree = tr.groupby(['canon','target_type'])['target'].agg(['nunique','min','max','mean','count']).reset_index()
disagree = disagree[disagree['nunique']>1]
print(f"\ncanon dupes with disagreeing targets: {len(disagree)}")
if len(disagree):
    for _, r in disagree.iterrows():
        print(f"  {r['target_type']}: {r['canon'][:80]}... n={r['count']} range=[{r['min']:.3f}, {r['max']:.3f}]")

# canonical overlap train↔test
tr_c = set(tr['canon']); te_c = set(te['canon'])
print(f"\ncanonical overlap train↔test: {len(tr_c & te_c)}")

# ============================================================
# 2. Murcko scaffolds
# ============================================================
print("\n" + "=" * 60)
print("MURCKO SCAFFOLDS")
print("=" * 60)

scaf_map = {s: scaffold(s) for s in all_smi}
tr['scaf'] = tr['smiles'].map(scaf_map)
te['scaf'] = te['smiles'].map(scaf_map)

# Scaffolds where the SMILES has no rings at all (acyclic → scaffold is empty string '')
print(f"\ntrain SMILES with acyclic (empty) scaffold: {(tr['scaf']=='').sum()} / {len(tr)}")
print(f"test  SMILES with acyclic (empty) scaffold: {(te['scaf']=='').sum()} / {len(te)}")

print(f"\ntrain unique scaffolds: {tr['scaf'].nunique()}")
print(f"test  unique scaffolds: {te['scaf'].nunique()}")

tr_sc = set(tr['scaf'].dropna()); te_sc = set(te['scaf'].dropna())
common_sc = tr_sc & te_sc
print(f"\nscaffold overlap train ↔ test: {len(common_sc)}")
print(f"  ...as % of test scaffolds: {100*len(common_sc)/len(te_sc):.1f}%")
print(f"scaffolds in TEST but not in TRAIN (potential OOD): {len(te_sc - tr_sc)}")

# per-target scaffold coverage
print(f"\n=== per-target: unique scaffolds in train / test, test scaffolds not in train ===")
print(f"  {'target':>6s} {'#scaf-train':>12s} {'#scaf-test':>12s} {'#test-scaf-not-in-train':>25s} {'%test-OOD-scaf':>16s}")
for tt in sorted(tr['target_type'].unique()):
    grsc = set(tr[tr['target_type']==tt]['scaf'].dropna())
    gesc = set(te[te['target_type']==tt]['scaf'].dropna())
    ood = gesc - grsc  # scaffolds appearing in test but never in train FOR THIS TARGET
    # actually more useful: test scaffolds not in ANY train
    ood_all = gesc - tr_sc
    print(f"  {tt:>6s}  {len(grsc):>10d}   {len(gesc):>10d}     {len(ood_all):>10d}                {100*len(ood_all)/max(1,len(gesc)):>10.1f}")

# Fraction of test ROWS whose scaffold is unseen anywhere in train
print(f"\n=== per-target: TEST ROWS whose scaffold is unseen in ANY train row ===")
print(f"  {'target':>6s} {'test rows':>10s} {'unseen-scaf rows':>18s} {'%':>8s}")
for tt in sorted(te['target_type'].unique()):
    g = te[te['target_type']==tt]
    n_unseen = (~g['scaf'].isin(tr_sc)).sum()
    print(f"  {tt:>6s}  {len(g):>10d}   {n_unseen:>15d}   {100*n_unseen/len(g):>6.1f}%")

# Top-10 most common scaffolds per target
print(f"\n=== top-5 scaffolds per target (rank, count, scaffold) ===")
for tt in sorted(tr['target_type'].unique()):
    g = tr[tr['target_type']==tt]
    vc = g['scaf'].value_counts().head(5)
    print(f"\n  {tt}:")
    for i, (sc, n) in enumerate(vc.items(), 1):
        display = sc if sc else "(acyclic - no rings)"
        display = display if len(display)<70 else display[:67]+'...'
        print(f"    {i}. n={n:>4d}  {display}")

# ============================================================
# 3. Save canon + scaffold enrichment
# ============================================================
tr[['smiles','canon','scaf','target','target_type']].to_csv(
    "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures/train_with_scaf.csv", index=False)
te[['id','smiles','canon','scaf','target_type']].to_csv(
    "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures/test_with_scaf.csv", index=False)
print("\nsaved: docs/figures/{train,test}_with_scaf.csv")
