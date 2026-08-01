"""Row-level test accounting: classify every test row by train-availability."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def canon(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToSmiles(m, canonical=True) if m else None
def scaf(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if not m: return None
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except: return None

all_s = pd.concat([tr['smiles'], te['smiles']]).unique()
canon_map = {s: canon(s) for s in all_s}
scaf_map = {s: scaf(s) for s in all_s}

tr['canon'] = tr['smiles'].map(canon_map); tr['scaf'] = tr['smiles'].map(scaf_map)
te['canon'] = te['smiles'].map(canon_map); te['scaf'] = te['smiles'].map(scaf_map)

# training index: canon -> set of target_types
tr_canon_targets = tr.groupby('canon')['target_type'].apply(set).to_dict()
tr_canon_set = set(tr['canon'])
tr_scaf_targets = tr.groupby('scaf')['target_type'].apply(set).to_dict()
tr_scaf_set = set(tr['scaf'])

def classify(row):
    c, s, t = row['canon'], row['scaf'], row['target_type']
    if c in tr_canon_set and t in tr_canon_targets[c]:
        return 'A_same_canon_same_target'
    if c in tr_canon_set:
        return 'B_same_canon_diff_target'
    if s in tr_scaf_set and t in tr_scaf_targets.get(s, set()):
        return 'C_same_scaf_same_target'
    if s in tr_scaf_set:
        return 'D_same_scaf_diff_target'
    return 'E_novel_scaffold'

te['bucket'] = te.apply(classify, axis=1)

print("=== FULL TEST-ROW ACCOUNTING (all 4940 test rows) ===\n")
print("Legend:")
print("  A = same canonical SMILES + same target measured in train (near-leak, exceptional)")
print("  B = same canonical SMILES in train but under a DIFFERENT target (multitask leverage)")
print("  C = same scaffold in train + same target seen there (in-distribution)")
print("  D = same scaffold in train + only diff-target seen (scaffold-level transfer)")
print("  E = novel scaffold — pure OOD")
print()
print(f"  {'target':>6s} {'total':>6s} "
      + " ".join([f"{b[0]:>7s}" for b in ['A_','B_','C_','D_','E_']]))
for tt in sorted(te['target_type'].unique()):
    g = te[te['target_type']==tt]
    print(f"  {tt:>6s} {len(g):>6d} "
          + " ".join([f"{(g['bucket']==k).sum():>7d}" for k in
                      ['A_same_canon_same_target','B_same_canon_diff_target',
                       'C_same_scaf_same_target','D_same_scaf_diff_target',
                       'E_novel_scaffold']]))
tot = te.groupby('bucket').size()
print(f"\n  {'ALL':>6s} {len(te):>6d} "
      + " ".join([f"{tot.get(k,0):>7d}" for k in
                  ['A_same_canon_same_target','B_same_canon_diff_target',
                   'C_same_scaf_same_target','D_same_scaf_diff_target',
                   'E_novel_scaffold']]))

# Percent version
print("\n=== SAME AS %s OF TARGET'S TEST ROWS ===\n")
print(f"  {'target':>6s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s} {'E':>7s}")
for tt in sorted(te['target_type'].unique()):
    g = te[te['target_type']==tt]
    print(f"  {tt:>6s} " + " ".join(
        [f"{100*(g['bucket']==k).mean():>6.1f}%" for k in
         ['A_same_canon_same_target','B_same_canon_diff_target',
          'C_same_scaf_same_target','D_same_scaf_diff_target',
          'E_novel_scaffold']]))

# Detailed B rows: what OTHER targets are known for the same canon?
print("\n=== B-bucket (same-canon-diff-target): distribution of number of ALL OTHER targets known in train ===")
te_b = te[te['bucket']=='B_same_canon_diff_target']
print(f"  {'target':>6s} {'n_B':>5s} {'0':>4s} {'1':>4s} {'2':>4s} {'3':>4s} {'4':>4s} {'5':>4s} {'6':>4s}")
from collections import Counter
for tt in sorted(te_b['target_type'].unique()):
    g = te_b[te_b['target_type']==tt]
    known_counts = []
    for _, r in g.iterrows():
        others = tr_canon_targets.get(r['canon'], set()) - {r['target_type']}
        known_counts.append(len(others))
    c = Counter(known_counts)
    print(f"  {tt:>6s} {len(g):>5d} "
          + " ".join([f"{c.get(k,0):>4d}" for k in range(7)]))

te.to_csv(f"{DATA}/../docs/figures/test_bucketed.csv", index=False)
print("\nsaved: docs/figures/test_bucketed.csv")
