"""
R2-1 Part C statistics: is STEP's margin over DAISY still there on the
similarity-disjoint split?

Reads results/sim_split_eval/per_sample_predictions_sim.csv (identical row order
for every model, written by EVAL_SIM_SPLIT.py) and runs

  1. correlated DeLong, STEP vs DAISY_sim, on the identical sample set
  2. a cluster bootstrap that resamples similarity CLUSTERS, not rows -- the
     clusters are the independent unit on this split by construction
  3. the same CDR3b-membership confound check as section A4, on this split

The two DeLong helpers are copied verbatim from revision/evidence/r1_5_delong.py
(validated there against scipy's Mann-Whitney U and a permutation null) so this
script stands alone; r1_5_delong.py executes on import and cannot be imported.

Writes revision/evidence/r2_1_partC_stats.json.
"""
import json

import numpy as np
import pandas as pd
import Levenshtein
from scipy import stats
from sklearn.metrics import roc_auc_score

CSV = 'results/sim_split_eval/per_sample_predictions_sim.csv'
SPLIT = 'data/splits_600_epitope_sim'
OUT = 'revision/evidence/r2_1_partC_stats.json'
N_BOOT = 10000
SEED = 42
MAX_DIST, MIN_IDENT = 2, 0.80


# ============================================================ DeLong (copied)
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_structural(y, preds):
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    y = np.asarray(y).astype(int)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01, ddof=1) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10, ddof=1) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    return aucs, np.atleast_2d(s01) / m + np.atleast_2d(s10) / n


def delong_pair(y, p1, p2):
    aucs, cov = delong_structural(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(var)) if var > 0 else np.nan
    z = diff / se if se == se else np.nan
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), delta=diff, se=se,
                z=float(z), p=float(2 * stats.norm.sf(abs(z))),
                lo=diff - 1.96 * se, hi=diff + 1.96 * se,
                r=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))


rep = {}
df = pd.read_csv(CSV)
y = df['label'].values.astype(int)
epi = df['antigen.epitope'].values
print("=" * 78)
print(f"R2-1 Part C statistics | {CSV}")
print(f"n = {len(df)}  ({y.sum()} pos / {(y == 0).sum()} neg)  "
      f"{len(np.unique(epi))} epitopes")

# ------------------------------------------------- cluster map for this split
train_epi = sorted(set(pd.read_csv(f'{SPLIT}/train.tsv', sep='\t')['antigen.epitope']))
test_epi = sorted(set(epi))


def linked(a, b):
    d = Levenshtein.distance(a, b)
    return d <= MAX_DIST or (1.0 - d / max(len(a), len(b))) >= MIN_IDENT


# single-linkage over the TEST epitopes only: that is the resampling unit
parent = list(range(len(test_epi)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


by_len = {}
for i, e in enumerate(test_epi):
    by_len.setdefault(len(e), []).append(i)
for li in sorted(by_len):
    for lj in sorted(by_len):
        if lj < li or lj - li > 3:
            continue
        same = (li == lj)
        for i in by_len[li]:
            for j in by_len[lj]:
                if same and j <= i:
                    continue
                if linked(test_epi[i], test_epi[j]):
                    a, b = find(i), find(j)
                    if a != b:
                        parent[b] = a
e2c = {test_epi[i]: find(i) for i in range(len(test_epi))}
clu = np.array([e2c[e] for e in epi])
uc = np.unique(clu)
print(f"test epitopes group into {len(uc)} similarity clusters "
      f"(bootstrap resamples these)")
rep['n_test_clusters'] = int(len(uc))

MODELS = ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble',
          'DAISY_sim', 'DAISY_v1weights']
P = {m: df[m].values.astype(float) for m in MODELS}
rep['auc'] = {m: float(roc_auc_score(y, P[m])) for m in MODELS}

# ------------------------------------------------- 1. DeLong vs DAISY_sim
print("\n" + "=" * 78)
print("  1. CORRELATED DeLONG vs DAISY_sim (the fair comparator)")
print("=" * 78)
print(f"  {'STEP predictor':<16}{'dAUC':>9}{'SE':>9}{'Z':>8}{'P':>12}"
      f"{'95% CI':>22}{'r':>7}")
dl = {}
for m in ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble']:
    r = delong_pair(y, P[m], P['DAISY_sim'])
    dl[m] = r
    print(f"  {m:<16}{r['delta']:>+9.4f}{r['se']:>9.4f}{r['z']:>8.3f}"
          f"{r['p']:>12.4g}   [{r['lo']:+.4f}, {r['hi']:+.4f}]{r['r']:>7.3f}")
rep['delong_vs_daisy_sim'] = dl

# ------------------------------------------------- 2. cluster bootstrap
print("\n" + "=" * 78)
print(f"  2. CLUSTER BOOTSTRAP ({N_BOOT:,} resamples of the {len(uc)} test clusters)")
print("=" * 78)
idx_by_c = [np.flatnonzero(clu == c) for c in uc]
rng = np.random.RandomState(SEED)
pairs = [('STEP_ensemble', 'DAISY_sim'), ('STEP_seed42', 'DAISY_sim')]
boot = {f"{a}_vs_{b}": [] for a, b in pairs}
boot_auc = {m: [] for m in MODELS}
kept = 0
for _ in range(N_BOOT):
    pick = rng.randint(0, len(uc), len(uc))
    ii = np.concatenate([idx_by_c[k] for k in pick])
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    kept += 1
    a_ = {m: roc_auc_score(yy, P[m][ii]) for m in MODELS}
    for m in MODELS:
        boot_auc[m].append(a_[m])
    for a, b in pairs:
        boot[f"{a}_vs_{b}"].append(a_[a] - a_[b])
print(f"  usable resamples: {kept:,}/{N_BOOT:,}")
print(f"\n  {'model':<17}{'AUC':>8}{'cluster-bootstrap 95% CI':>30}")
bci = {}
for m in MODELS:
    v = np.array(boot_auc[m])
    lo, hi = np.percentile(v, [2.5, 97.5])
    bci[m] = dict(auc=rep['auc'][m], lo=float(lo), hi=float(hi), sd=float(v.std(ddof=1)))
    print(f"  {m:<17}{rep['auc'][m]:>8.4f}{f'[{lo:.4f}, {hi:.4f}]':>30}")
rep['cluster_bootstrap_auc'] = bci
print(f"\n  {'difference':<28}{'delta':>9}{'95% CI':>22}{'P (2-sided)':>13}")
bdiff = {}
for k, v in boot.items():
    v = np.array(v)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    p = max(p, 1.0 / len(v))
    bdiff[k] = dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), p=float(p))
    print(f"  {k:<28}{v.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>13.4g}")
rep['cluster_bootstrap_diff'] = bdiff

# ------------------------------------------------- 3. the A4 confound, here
print("\n" + "=" * 78)
print("  3. CDR3b-MEMBERSHIP CONFOUND ON THIS SPLIT")
print("=" * 78)
seen = set(pd.read_csv(f'{SPLIT}/train.tsv', sep='\t').cdr3)
cdr3_seen = df.cdr3.isin(seen).values
triv = (~cdr3_seen).astype(float)
print(f"  model-free rule 'CDR3b absent from train': ROC-AUC "
      f"{roc_auc_score(y, triv):.4f}  (unseen in {cdr3_seen.mean() == 0 or (~cdr3_seen).mean():.1%} of rows)")
rep['trivial_auc'] = float(roc_auc_score(y, triv))
for s in (True, False):
    c = df[cdr3_seen == s]
    yy = y[cdr3_seen == s]
    print(f"  CDR3b seen = {str(s):<5} rows {len(c):>6,}  pos {yy.sum():>6,}  "
          f"pos rate {yy.mean():>6.1%}"
          f"{'   <-- degenerate' if yy.min() == yy.max() else ''}")
sub = cdr3_seen
print(f"\n  restricted to the non-degenerate stratum (CDR3b seen in train, "
      f"n={sub.sum():,}: {y[sub].sum()} pos / {(y[sub] == 0).sum()} neg):")
strat = {}
for m in MODELS:
    strat[m] = float(roc_auc_score(y[sub], P[m][sub]))
    print(f"    {m:<17} ROC-AUC {strat[m]:.4f}")
rep['cdr3_seen_stratum'] = dict(n=int(sub.sum()), pos=int(y[sub].sum()), auc=strat)
r = delong_pair(y[sub], P['STEP_ensemble'][sub], P['DAISY_sim'][sub])
print(f"    DeLong STEP_ensemble - DAISY_sim = {r['delta']:+.4f}  "
      f"Z {r['z']:.3f}  P {r['p']:.4g}")
rep['cdr3_seen_stratum']['delong_ens_vs_daisy'] = r

json.dump(rep, open(OUT, 'w'), indent=2)
print(f"\nsaved {OUT}")
