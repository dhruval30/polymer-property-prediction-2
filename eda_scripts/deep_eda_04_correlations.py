"""Cross-target correlations (matrix completion viability check)."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr

DATA = "/Users/dhruval/Documents/polymer-property-prediction-2/ppp-round-2"
tr = pd.read_csv(f"{DATA}/train.csv")

# Pivot to wide format: rows=SMILES, cols=target_type
# Average duplicate (smiles, target_type) pairs (only 3-4 dupes anyway)
wide = tr.groupby(['smiles','target_type'])['target'].mean().unstack()
print(f"wide shape: {wide.shape}")
print(f"per-target non-null counts:")
print(wide.notna().sum())

targets = ['eea','egb','egc','ei','eps','nc','tg']
print("\n=== pairwise Pearson correlation on co-measured molecules ===")
print(f"  {' ':>6s}", ' '.join(f"{t:>7s}" for t in targets))
for t1 in targets:
    row = [f"  {t1:>6s}"]
    for t2 in targets:
        if t1 == t2:
            n = wide[t1].notna().sum()
            row.append(f"  {'+1.00':>6s}({n:>3d})")
            continue
        m = wide[[t1,t2]].dropna()
        if len(m) < 5:
            row.append(f"  {'n<5':>5s}({len(m):>2d})")
        else:
            res = pearsonr(m[t1].values, m[t2].values)
            r = float(res.statistic if hasattr(res,'statistic') else res[0])
            row.append(f"  {r:>+.2f}({len(m):>3d})")
    print(''.join(row))

print("\n=== pairwise Spearman rank correlation on co-measured molecules ===")
print(f"  {' ':>6s}", ' '.join(f"{t:>7s}" for t in targets))
for t1 in targets:
    row = [f"  {t1:>6s}"]
    for t2 in targets:
        if t1 == t2:
            n = wide[t1].notna().sum()
            row.append(f"  {'+1.00':>6s}({n:>3d})")
            continue
        m = wide[[t1,t2]].dropna()
        if len(m) < 5:
            row.append(f"  {'n<5':>5s}({len(m):>2d})")
        else:
            sr = spearmanr(m[t1].values, m[t2].values)
            r = float(sr.statistic if hasattr(sr, 'statistic') else sr.correlation)
            row.append(f"  {r:>+.2f}({len(m):>3d})")
    print(''.join(row))

# Key: what CAN we predict from what for matrix completion?
print("\n=== For each target: best 'predictor target' (highest |Pearson|), and R² for a simple linear predictor ===")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

for t1 in targets:
    best = None
    best_r2 = -1
    for t2 in targets:
        if t1 == t2: continue
        m = wide[[t1,t2]].dropna()
        if len(m) < 10: continue
        # fit t1 ~ t2
        lr = LinearRegression().fit(m[[t2]], m[t1])
        pred = lr.predict(m[[t2]])
        r2 = r2_score(m[t1], pred)
        if r2 > best_r2:
            best_r2 = r2; best = (t2, len(m))
    if best is None:
        print(f"  {t1}: no co-measured pairs (n<10 with every other target)")
    else:
        print(f"  {t1}: best single-target predictor = {best[0]} (n_common={best[1]}), R²_train = {best_r2:.3f}")

# What if we use ALL other targets? multivariate linear
print("\n=== For each target: R² from linear regression using ALL other targets (train R²) ===")
from sklearn.linear_model import Ridge
for t1 in targets:
    others = [t for t in targets if t != t1]
    m = wide[[t1]+others].dropna()
    if len(m) < 10:
        print(f"  {t1}: n<10 co-measured with all other 6 -> {len(m)}")
        continue
    lr = Ridge(alpha=1.0).fit(m[others], m[t1])
    pred = lr.predict(m[others])
    r2 = r2_score(m[t1], pred)
    print(f"  {t1}: n_common_with_all_6={len(m)}, Ridge R²(train) = {r2:.3f}")

# And what if we use "any known other-target values" via mean-impute? Realistic scenario.
print("\n=== Realistic: for each target, train a model using the OTHER targets with NaN-mean-impute ===")
from sklearn.impute import SimpleImputer
for t1 in targets:
    others = [t for t in targets if t != t1]
    # rows where t1 is measured
    m = wide[wide[t1].notna()][[t1] + others].copy()
    X = m[others].values
    y = m[t1].values
    # any-other-known: how many of these rows have at least 1 other measurement
    n_any = (~pd.isna(X)).any(axis=1).sum()
    imputer = SimpleImputer(strategy='mean')
    X_imp = imputer.fit_transform(X)
    if len(y) < 10:
        continue
    lr = Ridge(alpha=1.0).fit(X_imp, y)
    pred = lr.predict(X_imp)
    r2 = r2_score(y, pred)
    print(f"  {t1}: n={len(y)}, rows-with-any-other-known={n_any} ({100*n_any/len(y):.1f}%), Ridge R²(train)={r2:.3f}")
