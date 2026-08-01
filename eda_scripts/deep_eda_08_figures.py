"""Generate the plots referenced from docs/08_eda_deep.md."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 9, 'figure.dpi': 110})

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
OUT = "/Users/dhruval/Documents/polymer-property-prediction-2/docs/figures"
tr = pd.read_csv(f"{DATA}/train.csv")

# ---------- 1. per-target value histograms ----------
targets = ['tg','egc','egb','eea','ei','eps','nc']
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()
for ax, t in zip(axes, targets):
    v = tr[tr['target_type']==t]['target'].values
    ax.hist(v, bins=40, edgecolor='k', linewidth=0.4, alpha=0.85)
    ax.set_title(f"{t}  (n={len(v)})")
    ax.axvline(np.mean(v), color='r', ls='--', lw=1, label=f'mean={np.mean(v):.1f}')
    ax.axvline(np.median(v), color='k', ls=':', lw=1, label=f'med={np.median(v):.1f}')
    ax.legend(fontsize=7)
    ax.set_xlabel(t); ax.set_ylabel('count')
axes[-1].axis('off')
plt.suptitle('Per-target value distributions (train)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_per_target_hist.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig01_per_target_hist.png")

# ---------- 2. per-target row counts ----------
counts_tr = tr.groupby('target_type').size().reindex(targets)
te = pd.read_csv(f"{DATA}/test.csv")
counts_te = te.groupby('target_type').size().reindex(targets)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(targets))
w = 0.4
ax.bar(x-w/2, counts_tr, w, label='train', color='#4a90d9', edgecolor='k')
ax.bar(x+w/2, counts_te, w, label='test',  color='#f27f39', edgecolor='k')
for i, (a, b) in enumerate(zip(counts_tr, counts_te)):
    ax.text(i-w/2, a+40, str(int(a)), ha='center', fontsize=8)
    ax.text(i+w/2, b+40, str(int(b)), ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(targets)
ax.set_ylabel('rows'); ax.legend()
ax.set_title('Row counts per target — train vs test')
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_row_counts.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig02_row_counts.png")

# ---------- 3. 5-pack correlation heatmap ----------
wide = tr.groupby(['smiles','target_type'])['target'].mean().unstack()
five = ['eea','egb','egc','ei','eps','nc']
from scipy.stats import pearsonr
mat = np.eye(len(five))
ns  = np.zeros_like(mat, dtype=int)
for i,a in enumerate(five):
    for j,b in enumerate(five):
        if i==j:
            ns[i,j] = wide[a].notna().sum(); continue
        m = wide[[a,b]].dropna()
        ns[i,j] = len(m)
        if len(m)>=5:
            res = pearsonr(m[a].values, m[b].values)
            mat[i,j] = float(res.statistic if hasattr(res,'statistic') else res[0])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(five))); ax.set_xticklabels(five)
ax.set_yticks(range(len(five))); ax.set_yticklabels(five)
for i in range(len(five)):
    for j in range(len(five)):
        color = 'white' if abs(mat[i,j])>0.5 else 'black'
        ax.text(j, i, f"{mat[i,j]:+.2f}\nn={ns[i,j]}", ha='center', va='center', fontsize=8, color=color)
plt.colorbar(im, ax=ax, fraction=0.045)
ax.set_title('Pairwise Pearson corr (co-measured molecules)\n5-pack + egc')
plt.tight_layout()
plt.savefig(f"{OUT}/fig03_corr_heatmap.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig03_corr_heatmap.png")

# ---------- 4. Tanimoto NN histograms per target ----------
nn = np.load(f"{OUT}/tanimoto_nn.npz")
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()
for ax, t in zip(axes, targets):
    v = nn[t]
    ax.hist(v, bins=30, range=(0,1), edgecolor='k', linewidth=0.4, alpha=0.85, color='#5cae7a')
    ax.axvline(np.median(v), color='k', ls='--', lw=1, label=f'med={np.median(v):.2f}')
    ax.set_title(f"{t}  (n={len(v)})")
    ax.set_xlim(0, 1)
    ax.set_xlabel('NN Tanimoto (test → same-target train)')
    ax.legend(fontsize=7)
axes[-1].axis('off')
plt.suptitle('Same-target NN Tanimoto: test → train (Morgan r=2, 2048b, wildcards → C)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig04_tanimoto_nn.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig04_tanimoto_nn.png")

# ---------- 5. 5-pack "matrix availability" per SMILES ----------
five_wide = wide[five]
# how many of the 5 are labeled per row
n_labeled = five_wide.notna().sum(axis=1)
# only keep SMILES with >=1 label in five-pack
n_labeled = n_labeled[n_labeled >= 1]
fig, ax = plt.subplots(figsize=(6, 4))
vc = n_labeled.value_counts().sort_index()
ax.bar(vc.index, vc.values, edgecolor='k', color='#9673d1')
for i, v in zip(vc.index, vc.values):
    ax.text(i, v+3, str(v), ha='center', fontsize=9)
ax.set_xlabel('# of {eea,egb,ei,eps,nc} labels per SMILES (train)')
ax.set_ylabel('# of SMILES')
ax.set_title(f'5-pack label coverage in train  (n={int(n_labeled.sum())} labels over {len(n_labeled)} SMILES)')
plt.tight_layout()
plt.savefig(f"{OUT}/fig05_5pack_coverage.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig05_5pack_coverage.png")

# ---------- 6. MolWt / backbone-length per target box ----------
props = pd.read_csv(f"{OUT}/all_mol_props.csv")
mtr = tr.merge(props, on='smiles', how='left')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for i, col in enumerate(['MolWt','HeavyAtoms']):
    ax = axes[i]
    data = [mtr[mtr['target_type']==t][col].dropna().values for t in targets]
    bp = ax.boxplot(data, tick_labels=targets, showfliers=False, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.tab10.colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(f'{col} per target'); ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(f"{OUT}/fig06_mol_size_per_target.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"saved: {OUT}/fig06_mol_size_per_target.png")
