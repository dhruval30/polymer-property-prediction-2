"""UMAP embedding of Morgan fingerprints across all train+test SMILES."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import umap
RDLogger.DisableLog('rdApp.*')

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
OUT = "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures"
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")

def fp(smi):
    m = Chem.MolFromSmiles(smi.replace('*','C'))
    if m is None: return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), dtype=np.uint8)

all_smi = pd.concat([tr['smiles'], te['smiles']]).unique()
print(f"Computing FPs for {len(all_smi)} unique SMILES...")
fps = np.stack([fp(s) for s in all_smi])
print(f"FP matrix: {fps.shape}")

print("Fitting UMAP (this takes ~2-4 min)...")
reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.15,
                    metric='jaccard', random_state=42, verbose=False)
emb = reducer.fit_transform(fps)
print(f"UMAP done, embedding shape: {emb.shape}")

emb_map = {s: emb[i] for i, s in enumerate(all_smi)}
tr['x'] = tr['smiles'].map(lambda s: emb_map[s][0])
tr['y'] = tr['smiles'].map(lambda s: emb_map[s][1])
te['x'] = te['smiles'].map(lambda s: emb_map[s][0])
te['y'] = te['smiles'].map(lambda s: emb_map[s][1])

# ---- fig: UMAP colored by target_type ----
targets = ['tg','egc','egb','eea','ei','eps','nc']
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(9, 8))
# Test in grey
ax.scatter(te['x'], te['y'], s=4, c='lightgrey', alpha=0.5, label='test (all)', linewidths=0)
for i, t in enumerate(targets):
    g = tr[tr['target_type']==t]
    ax.scatter(g['x'], g['y'], s=6, c=[colors[i]], alpha=0.7, label=f'train {t} (n={len(g)})', linewidths=0)
ax.legend(loc='best', fontsize=9, markerscale=2)
ax.set_title('UMAP of polymer chemistry (Morgan-r2, Jaccard, 1024b)\ntrain colored by target · test in grey')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
plt.tight_layout()
plt.savefig(f"{OUT}/fig07_umap_all.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig07_umap_all.png")

# ---- fig: UMAP colored by tg VALUE (only tg rows) ----
fig, ax = plt.subplots(figsize=(9, 8))
g = tr[tr['target_type']=='tg']
sc = ax.scatter(g['x'], g['y'], c=g['target'], cmap='RdYlBu_r', s=8, alpha=0.85, linewidths=0)
plt.colorbar(sc, ax=ax, label='tg (°C)')
ax.set_title('UMAP colored by tg value (train tg rows only)')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
plt.tight_layout()
plt.savefig(f"{OUT}/fig08_umap_tg.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig08_umap_tg.png")

# ---- fig: UMAP colored by egc VALUE (only egc rows) ----
fig, ax = plt.subplots(figsize=(9, 8))
g = tr[tr['target_type']=='egc']
sc = ax.scatter(g['x'], g['y'], c=g['target'], cmap='viridis', s=8, alpha=0.85, linewidths=0)
plt.colorbar(sc, ax=ax, label='egc (eV)')
ax.set_title('UMAP colored by egc value (train egc rows only)')
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
plt.tight_layout()
plt.savefig(f"{OUT}/fig09_umap_egc.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig09_umap_egc.png")

# ---- fig: UMAP colored by test bucket ----
try:
    bucketed = pd.read_csv(f"{OUT}/test_bucketed.csv")
    te2 = te.merge(bucketed[['id','bucket']], on='id', how='left')
    fig, ax = plt.subplots(figsize=(9, 8))
    # background train
    ax.scatter(tr['x'], tr['y'], s=3, c='lightgrey', alpha=0.4, linewidths=0)
    bucket_colors = {'A_same_canon_same_target':'red',
                     'B_same_canon_diff_target':'orange',
                     'C_same_scaf_same_target':'green',
                     'D_same_scaf_diff_target':'blue',
                     'E_novel_scaffold':'magenta'}
    for k, c in bucket_colors.items():
        g = te2[te2['bucket']==k]
        ax.scatter(g['x'], g['y'], s=5, c=c, alpha=0.6, label=f'{k} (n={len(g)})', linewidths=0)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title('UMAP: test rows colored by "how supported by train" bucket\n(train shown in grey)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig10_umap_test_buckets.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved: {OUT}/fig10_umap_test_buckets.png")
except Exception as e:
    print(f"(skipping bucket plot: {e})")

# save embeddings
np.savez_compressed(f"{OUT}/umap_embedding.npz", all_smiles=all_smi, emb=emb)
print(f"saved: {OUT}/umap_embedding.npz")
