"""PI1M vs train: distribution + chemical similarity."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
pi = pd.read_csv(f"{DATA}/PI1M.csv")

# Sample PI1M to something tractable — 20k random rows
pi_sample = pi.sample(20000, random_state=42)['SMILES'].dropna().unique()
print(f"PI1M sample size (unique after dropna): {len(pi_sample)}")

def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    mc = Chem.MolFromSmiles(smi.replace('*','C'))
    if mc is None: mc = m
    return {
        'HeavyAtoms': m.GetNumHeavyAtoms(),
        'NumRings': rdMolDescriptors.CalcNumRings(m),
        'NumAromRings': rdMolDescriptors.CalcNumAromaticRings(m),
        'NumRotBonds': rdMolDescriptors.CalcNumRotatableBonds(m),
        'MolWt': Descriptors.MolWt(mc),
        'MolLogP': Descriptors.MolLogP(mc),
        'TPSA': Descriptors.TPSA(mc),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(m),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(m),
    }

def elem_set(smi):
    m = Chem.MolFromSmiles(smi)
    if not m: return set()
    return {a.GetSymbol() for a in m.GetAtoms()}

print("Computing props for train...")
tr_props = [props(s) for s in tqdm(tr['smiles'].unique(), ncols=80)]
tr_props = pd.DataFrame([p for p in tr_props if p])

print("Computing props for PI1M sample...")
pi_props = [props(s) for s in tqdm(pi_sample, ncols=80)]
pi_props = pd.DataFrame([p for p in pi_props if p])

print(f"\ntrain: {len(tr_props)} valid mols, PI1M sample: {len(pi_props)} valid mols")

print("\n=== TRAIN vs PI1M (property distributions) ===")
print(f"  {'prop':>16s} {'train mean':>12s} {'PI1M mean':>12s} {'train med':>12s} {'PI1M med':>12s} {'train q90':>12s} {'PI1M q90':>12s}")
for c in tr_props.columns:
    t = tr_props[c]; p = pi_props[c]
    print(f"  {c:>16s} {t.mean():>12.2f} {p.mean():>12.2f} {t.median():>12.2f} {p.median():>12.2f} {t.quantile(.9):>12.2f} {p.quantile(.9):>12.2f}")

# Element set: what elements exist in PI1M vs train?
print("\n=== elements observed ===")
tr_elem = set()
for s in tqdm(tr['smiles'].unique(), desc='train', ncols=80):
    tr_elem |= elem_set(s)
pi_elem = set()
for s in tqdm(pi_sample[:5000], desc='PI1M sample', ncols=80):
    pi_elem |= elem_set(s)
print(f"train elements: {sorted(tr_elem)}")
print(f"PI1M sample:    {sorted(pi_elem)}")
print(f"in train but not in PI1M sample: {sorted(tr_elem - pi_elem)}")
print(f"in PI1M sample but not in train: {sorted(pi_elem - tr_elem)}")

# NN Tanimoto: for each train SMILES, find nearest PI1M SMILES.
# If train molecules have high NN sim to PI1M, then PI1M covers our distribution well
print("\n=== NN Tanimoto: train SMILES → nearest PI1M SMILES ===")
def fp(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)

tr_smiles_sample = pd.Series(tr['smiles'].unique()).sample(min(1500, tr['smiles'].nunique()), random_state=0).tolist()
tr_fps = [fp(s) for s in tqdm(tr_smiles_sample, desc='train FPs', ncols=80)]
pi_fps = [fp(s) for s in tqdm(pi_sample[:15000], desc='PI1M FPs', ncols=80)]
tr_fps = [f for f in tr_fps if f is not None]
pi_fps = [f for f in pi_fps if f is not None]

nn_sims = []
for f in tqdm(tr_fps, desc='NN search', ncols=80):
    sims = DataStructs.BulkTanimotoSimilarity(f, pi_fps)
    nn_sims.append(max(sims))
nn = np.array(nn_sims)
print(f"\nn={len(nn)}")
print(f"NN sim (train→PI1M): min={nn.min():.3f}, q10={np.quantile(nn,.1):.3f}, med={np.median(nn):.3f}, q90={np.quantile(nn,.9):.3f}, max={nn.max():.3f}")
print(f"%<.3: {100*(nn<.3).mean():.1f}, %<.5: {100*(nn<.5).mean():.1f}, %>.9: {100*(nn>.9).mean():.1f}")

# Also: reverse direction for a smaller sample. NN (PI1M sample → train)
pi_sample_small = pd.Series(pi_sample).sample(1000, random_state=0).tolist()
pi_fps_small = [fp(s) for s in pi_sample_small]
pi_fps_small = [f for f in pi_fps_small if f is not None]
tr_all_fps = [fp(s) for s in tr['smiles'].unique()]
tr_all_fps = [f for f in tr_all_fps if f is not None]
print(f"\nreverse NN (PI1M sample → nearest train): n={len(pi_fps_small)}")
rev_nn = []
for f in tqdm(pi_fps_small, desc='rev NN', ncols=80):
    sims = DataStructs.BulkTanimotoSimilarity(f, tr_all_fps)
    rev_nn.append(max(sims))
rev = np.array(rev_nn)
print(f"NN sim (PI1M→train): min={rev.min():.3f}, q10={np.quantile(rev,.1):.3f}, med={np.median(rev):.3f}, q90={np.quantile(rev,.9):.3f}")
print(f"%<.3: {100*(rev<.3).mean():.1f}, %<.5: {100*(rev<.5).mean():.1f}, %>.9: {100*(rev>.9).mean():.1f}")

# Scaffold coverage
print("\n=== scaffold overlap (train vs PI1M sample) ===")
def scaf(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None

tr_scafs = set()
for s in tqdm(tr['smiles'].unique(), desc='train scaf', ncols=80):
    sc = scaf(s)
    if sc: tr_scafs.add(sc)
pi_scafs = set()
for s in tqdm(pi_sample[:15000], desc='PI1M scaf', ncols=80):
    sc = scaf(s)
    if sc: pi_scafs.add(sc)
print(f"\ntrain unique scaffolds: {len(tr_scafs)}")
print(f"PI1M sample unique scaffolds: {len(pi_scafs)}")
print(f"overlap: {len(tr_scafs & pi_scafs)}")
print(f"% of train scaffolds also in PI1M sample: {100*len(tr_scafs & pi_scafs)/len(tr_scafs):.1f}%")
