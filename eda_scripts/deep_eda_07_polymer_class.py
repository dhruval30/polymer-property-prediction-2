"""Classify polymers by common functional groups / backbone motifs via SMARTS."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

# SMARTS patterns for common polymer backbone functionalities
# These are just detection — a polymer often has multiple simultaneously.
CLASSES = {
    'ester':          '[CX3](=O)[OX2H0]',          # C(=O)-O-C
    'amide':          '[NX3][CX3](=[OX1])',         # N-C(=O)
    'urea':           '[NX3][CX3](=[OX1])[NX3]',
    'carbonate':      '[OX2][CX3](=[OX1])[OX2]',
    'urethane':       '[NX3][CX3](=[OX1])[OX2]',   # N-C(=O)-O
    'imide':          '[NX3]([CX3]=O)[CX3]=O',      # 2 C=O bonded to same N
    'ether':          '[OD2]([#6])[#6]',
    'aromatic_C':     'c',                          # any aromatic C
    'aromatic_ring':  'a1aaaaa1',                   # generic 6-membered arom
    'thiophene':      'c1ccsc1',
    'furan':          'c1ccoc1',
    'pyrrole':        'c1cc[nH]c1',
    'pyridine':       'n1ccccc1',
    'siloxane':       '[Si][O][Si]',
    'silicon':        '[Si]',
    'fluorine':       '[F]',
    'chlorine':       '[Cl]',
    'sulfone':        '[SX4](=O)(=O)',
    'sulfide':        '[SX2]([#6])[#6]',
    'nitrile':        '[NX1]#[CX2]',
    'phosphate':      '[PX4](=O)',
    'boron':          '[B]',
    'triple_bond':    '#',
    'double_bond':    '=[CX3]',
    'CH2_chain':      '[CH2][CH2][CH2][CH2]',       # 4x -CH2- in a row (aliphatic backbone)
    'vinyl_polymer':  '[CX4]([*])[CX4]',            # -C(*)(H)-CH2-
    'polystyrene_like': 'c1ccccc1[CH2][CH2]',
}

# Pre-compile
patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in CLASSES.items()}
# drop any that failed to parse
patterns = {k:v for k,v in patterns.items() if v is not None}
print(f"active SMARTS classes: {len(patterns)}")

def match_all(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return {}
    return {name: int(m.HasSubstructMatch(pat)) for name, pat in patterns.items()}

# Compute for all unique SMILES
all_smi = tr['smiles'].unique()
print(f"scanning {len(all_smi)} unique train SMILES...")
mats = {s: match_all(s) for s in all_smi}
df_m = pd.DataFrame.from_dict(mats, orient='index').reset_index().rename(columns={'index':'smiles'})
tr_m = tr.merge(df_m, on='smiles', how='left')

# Class prevalence per target
print("\n=== class prevalence per target (fraction of rows with match) ===")
cols = list(patterns.keys())
print(f"  {'class':>18s} " + ' '.join(f"{t:>5s}" for t in sorted(tr_m['target_type'].unique())))
for c in cols:
    row = [f"  {c:>18s}"]
    for tt in sorted(tr_m['target_type'].unique()):
        g = tr_m[tr_m['target_type']==tt]
        pct = 100 * g[c].mean()
        row.append(f" {pct:>4.0f}%")
    print(''.join(row))

# Target values per class (for tg — highest variance)
print("\n=== mean tg by class membership (does class predict tg?) ===")
g = tr_m[tr_m['target_type']=='tg']
print(f"  {'class':>18s} {'n_yes':>7s} {'n_no':>7s} {'tg_mean_yes':>14s} {'tg_mean_no':>14s} {'diff':>8s}")
for c in cols:
    yes = g[g[c]==1]['target']
    no = g[g[c]==0]['target']
    if len(yes)<5 or len(no)<5:
        print(f"  {c:>18s} {len(yes):>7d} {len(no):>7d} {'--':>14s} {'--':>14s} {'--':>8s}")
        continue
    print(f"  {c:>18s} {len(yes):>7d} {len(no):>7d} {yes.mean():>14.1f} {no.mean():>14.1f} {yes.mean()-no.mean():>+8.1f}")

# and for egc
print("\n=== mean egc by class membership ===")
g = tr_m[tr_m['target_type']=='egc']
print(f"  {'class':>18s} {'n_yes':>7s} {'n_no':>7s} {'egc_mean_yes':>14s} {'egc_mean_no':>14s} {'diff':>8s}")
for c in cols:
    yes = g[g[c]==1]['target']
    no = g[g[c]==0]['target']
    if len(yes)<5 or len(no)<5:
        print(f"  {c:>18s} {len(yes):>7d} {len(no):>7d} {'--':>14s} {'--':>14s} {'--':>8s}")
        continue
    print(f"  {c:>18s} {len(yes):>7d} {len(no):>7d} {yes.mean():>14.2f} {no.mean():>14.2f} {yes.mean()-no.mean():>+8.2f}")
