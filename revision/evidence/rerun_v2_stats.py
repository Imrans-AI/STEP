"""R2-3 rerun: paired DeLong + bootstrap CIs, and the DAISY/ERGO train-overlap sensitivity.

Reads results/baseline_rerun_v2/test_predictions_v2.npz only. Writes JSON next to it.
Provisional for R1-5, which owns the full statistics.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

NPZ = 'results/baseline_rerun_v2/test_predictions_v2.npz'
OUT = 'revision/evidence/rerun_v2_stats.json'
BASELINES = ['DAISY', 'ERGO-LSTM', 'ERGO-AE', 'TEINet', 'ATM-TCR']


# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def delong_cov(y, preds):
    """Fast DeLong (Sun & Xu 2014). preds: (k, n). Returns aucs, covariance matrix."""
    pos = preds[:, y == 1]; neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]; k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_test(y, p1, p2):
    aucs, S = delong_cov(y, np.vstack([p1, p2]))
    L = np.array([[1, -1]])
    var = float(L @ S @ L.T)
    if var <= 0:
        return float(aucs[0] - aucs[1]), np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0] - aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def boot_ci(y, p1, p2, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); d = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], p1[s]) - roc_auc_score(y[s], p2[s]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def auc_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); a = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        a.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


z = np.load(NPZ, allow_pickle=True)
y = z['y'].astype(int)
epi = z['epitope']; cdr3 = z['cdr3']
rep = {'n': int(len(y))}

print("=" * 78)
print("ROC-AUC with 95% bootstrap CI (2,000 resamples, seed 0)")
rows = []
for m in ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble'] + BASELINES:
    p = z[m]
    lo, hi = auc_ci(y, p)
    rows.append(dict(model=m, roc=roc_auc_score(y, p), lo=lo, hi=hi,
                     pr=average_precision_score(y, p)))
    print(f"  {m:14s} ROC {rows[-1]['roc']:.4f}  [{lo:.4f}, {hi:.4f}]   PR {rows[-1]['pr']:.4f}")
rep['auc'] = rows

print("\n" + "=" * 78)
print("Paired DeLong, STEP vs each baseline (STEP = per-seed, and the seed ensemble)")
dl = []
for step in ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble']:
    for b in BASELINES:
        diff, zz, p = delong_test(y, z[step], z[b])
        lo, hi = boot_ci(y, z[step], z[b])
        dl.append(dict(step=step, baseline=b, delta=diff, z=zz, p=p, ci_lo=lo, ci_hi=hi))
        star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'
        print(f"  {step:14s} - {b:10s}  d={diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"z={zz:+.2f}  P={p:.3e}  {star}")
rep['delong'] = dl

# ------------------------------------------------- train-overlap sensitivity
print("\n" + "=" * 78)
print("Train-overlap sensitivity")
print("STEP's v2 checkpoints are epitope-disjoint from v2 test by construction.")
print("The other five were trained elsewhere, so v2 test is NOT disjoint for them.")
v1tr = pd.read_csv('data/splits_600_epitope/train.tsv', sep='\t')
daisy = pd.read_csv('baselines/repos/DAISY/data_ours/tcr_peptide_train.csv')
v1_epi = set(v1tr['antigen.epitope']); daisy_epi = set(daisy['peptide'])
v1_pair = set(zip(v1tr.cdr3, v1tr['antigen.epitope']))
pairs = list(zip(cdr3, epi))

contaminated = np.array([(e in v1_epi) or (e in daisy_epi) or (pr in v1_pair)
                         for e, pr in zip(epi, pairs)])
clean = ~contaminated
print(f"  rows whose epitope or exact pair appears in v1 train or DAISY train: "
      f"{int(contaminated.sum())} of {len(y)}")
print(f"  clean subset: n = {int(clean.sum())}  "
      f"({int(y[clean].sum())} pos / {int((y[clean] == 0).sum())} neg)")
sens = []
if len(np.unique(y[clean])) == 2:
    print(f"  {'model':14s} {'full':>8s} {'clean':>8s} {'delta':>8s}")
    for m in ['STEP_seed42', 'STEP_ensemble'] + BASELINES:
        a_full = roc_auc_score(y, z[m]); a_cl = roc_auc_score(y[clean], z[m][clean])
        sens.append(dict(model=m, full=a_full, clean=a_cl, delta=a_cl - a_full))
        print(f"  {m:14s} {a_full:8.4f} {a_cl:8.4f} {a_cl - a_full:+8.4f}")
rep['overlap_sensitivity'] = dict(
    n_contaminated=int(contaminated.sum()), n_clean=int(clean.sum()), rows=sens)

json.dump(rep, open(OUT, 'w'), indent=2)
print(f"\nsaved {OUT}")
