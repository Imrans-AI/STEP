"""
R1-5: correlated DeLong tests + epitope-level cluster bootstrap.

Input : results/baseline_rerun_v2/per_sample_predictions_v2.csv
        (all six models, identical row order, from RERUN_BASELINES_V2.py)
Output: revision/evidence/r1_5_delong.json

Replaces the released PROPER_STATISTICAL_TESTS.py, which reads model-level AUC
summaries from results/ALL_6_MODELS_BALANCED_FINAL.csv, applies Hanley-McNeil
standard errors, and combines them as sqrt(se1^2 + se2^2) - i.e. assumes the two
ROC curves are independent, which they are not: both are scored on the same samples.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

CSV = 'results/baseline_rerun_v2/per_sample_predictions_v2.csv'
OUT = 'revision/evidence/r1_5_delong.json'
BASELINES = ['DAISY', 'ERGO-LSTM', 'ERGO-AE', 'TEINet', 'ATM-TCR']
STEPS = ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble']
PRIMARY = 'STEP_seed42'
N_BOOT = 10000
SEED = 42

rep = {}


# ======================================================== DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1          # average rank, 1-based, ties shared
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_structural(y, preds):
    """preds: (k, n). Returns AUCs and the k x k DeLong covariance matrix of the AUCs."""
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    y = np.asarray(y).astype(int)
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n                  # structural components, positives
    v10 = 1.0 - (tz[:, m:] - ty) / m            # structural components, negatives
    s01 = np.cov(v01, ddof=1) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10, ddof=1) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, cov


def delong_pair(y, p1, p2):
    aucs, cov = delong_structural(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), delta=diff,
                    se=np.nan, z=np.nan, p=np.nan, lo=np.nan, hi=np.nan,
                    r=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))
    se = float(np.sqrt(var))
    zsc = diff / se
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), delta=diff, se=se, z=float(zsc),
                p=float(2 * stats.norm.sf(abs(zsc))),
                lo=diff - 1.96 * se, hi=diff + 1.96 * se,
                r=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                se_indep=float(np.sqrt(cov[0, 0] + cov[1, 1])))


# ======================================================== data
df = pd.read_csv(CSV)
y = df['label'].values.astype(int)
epi = df['antigen.epitope'].values
P = {c: df[c].values.astype(float) for c in STEPS + BASELINES}
n = len(df)
ue = np.unique(epi)
print("=" * 80)
print(f"R1-5 | {CSV}")
print(f"n = {n}  ({y.sum()} pos / {(y == 0).sum()} neg)  epitopes = {len(ue)}")
sizes = pd.Series(epi).value_counts()
print(f"samples per epitope: median {int(sizes.median())}, "
      f"mean {sizes.mean():.1f}, max {int(sizes.max())}, "
      f"epitopes with 1 sample {int((sizes == 1).sum())}")
print(f"largest epitope = {sizes.iloc[0]} samples "
      f"({100 * sizes.iloc[0] / n:.1f}% of the test set)")
rep['n'] = int(n)
rep['n_epitopes'] = int(len(ue))
rep['epitope_sizes'] = dict(median=float(sizes.median()), mean=float(sizes.mean()),
                            max=int(sizes.max()), n_singleton=int((sizes == 1).sum()))

# ---- implementation check 1: midrank AUC must equal sklearn to machine precision
print("\n[check 1] DeLong midrank AUC vs sklearn.roc_auc_score")
worst = 0.0
for m_ in STEPS + BASELINES:
    a_d = delong_structural(y, P[m_][None, :])[0][0]
    a_s = roc_auc_score(y, P[m_])
    worst = max(worst, abs(a_d - a_s))
print(f"  max |DeLong AUC - sklearn AUC| over 9 models = {worst:.3e}")
assert worst < 1e-10, "midrank AUC disagrees with sklearn"
rep['check_auc_max_abs_diff'] = float(worst)

# ---- implementation check 2: DeLong SE of a single AUC vs ordinary bootstrap SE
print("\n[check 2] DeLong SE vs ordinary (sample-level) bootstrap SE, single AUCs")
rng = np.random.default_rng(0)
idx = np.arange(n)
bidx = rng.choice(idx, (2000, n), replace=True)
chk = []
for m_ in [PRIMARY, 'DAISY', 'TEINet']:
    se_dl = float(np.sqrt(delong_structural(y, P[m_][None, :])[1][0, 0]))
    bs = np.array([roc_auc_score(y[s], P[m_][s]) for s in bidx
                   if len(np.unique(y[s])) > 1])
    chk.append(dict(model=m_, se_delong=se_dl, se_boot=float(bs.std(ddof=1))))
    print(f"  {m_:14s} DeLong SE {se_dl:.5f}   bootstrap SE {bs.std(ddof=1):.5f}")
rep['check_se'] = chk


# ======================================================== 1. correlated DeLong
print("\n" + "=" * 80)
print("1. CORRELATED DeLong (the manuscript's stated method)")
print("=" * 80)
dl_rows = []
for s in STEPS:
    for b in BASELINES:
        r = delong_pair(y, P[s], P[b])
        r.update(step=s, baseline=b)
        dl_rows.append(r)
print(f"\nPrimary predictor: {PRIMARY}\n")
print(f"{'comparison':<26} {'dAUC':>8} {'SE':>8} {'Z':>7} {'P':>11}  95% CI")
for r in [x for x in dl_rows if x['step'] == PRIMARY]:
    print(f"  STEP vs {r['baseline']:<16} {r['delta']:+8.4f} {r['se']:8.5f} "
          f"{r['z']:+7.2f} {r['p']:11.3e}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
print("\nCorrelation between the two curves' AUC estimates (the term the released")
print("script sets to zero), and what ignoring it does to the SE:")
print(f"{'comparison':<26} {'r':>7} {'SE correlated':>14} {'SE independent':>15} {'ratio':>7}")
for r in [x for x in dl_rows if x['step'] == PRIMARY]:
    print(f"  STEP vs {r['baseline']:<16} {r['r']:+7.3f} {r['se']:14.5f} "
          f"{r['se_indep']:15.5f} {r['se_indep'] / r['se']:7.3f}")
rep['delong'] = dl_rows


# ======================================================== 2. cluster bootstrap
print("\n" + "=" * 80)
print(f"2. EPITOPE-LEVEL CLUSTER BOOTSTRAP ({N_BOOT} resamples, seed {SEED})")
print("=" * 80)
print("Resamples the 576 epitopes with replacement and rebuilds the sample set from")
print("the drawn clusters, so correlation within an epitope is respected.")

by_epi = {e: np.flatnonzero(epi == e) for e in ue}
epi_arr = np.array(list(by_epi.keys()))
rng = np.random.default_rng(SEED)
draws = rng.integers(0, len(epi_arr), size=(N_BOOT, len(epi_arr)))

cb = {}
skipped = 0
deltas = {(s, b): [] for s in STEPS for b in BASELINES}
aucs_b = {m_: [] for m_ in STEPS + BASELINES}
for d in draws:
    rows = np.concatenate([by_epi[epi_arr[j]] for j in d])
    yy = y[rows]
    if yy.sum() == 0 or yy.sum() == len(yy):
        skipped += 1
        continue
    a = {m_: roc_auc_score(yy, P[m_][rows]) for m_ in STEPS + BASELINES}
    for m_ in a:
        aucs_b[m_].append(a[m_])
    for s in STEPS:
        for b in BASELINES:
            deltas[(s, b)].append(a[s] - a[b])
print(f"  usable resamples: {N_BOOT - skipped} / {N_BOOT}")
rep['cluster_boot'] = dict(n_boot=N_BOOT, n_used=N_BOOT - skipped, seed=SEED)

cb_rows = []
for s in STEPS:
    for b in BASELINES:
        dv = np.array(deltas[(s, b)])
        lo, hi = np.percentile(dv, [2.5, 97.5])
        # two-sided bootstrap p, floored at 1/B (cannot resolve below that)
        pb = 2 * min((dv <= 0).mean(), (dv >= 0).mean())
        floored = pb < 1.0 / len(dv)
        cb_rows.append(dict(step=s, baseline=b, delta_obs=float(roc_auc_score(y, P[s]) - roc_auc_score(y, P[b])),
                            boot_mean=float(dv.mean()), boot_se=float(dv.std(ddof=1)),
                            lo=float(lo), hi=float(hi),
                            p=float(max(pb, 1.0 / len(dv))), p_floored=bool(floored)))
print(f"\n{'comparison':<26} {'dAUC':>8} {'SE':>8} {'P':>12}  95% CI")
for r in [x for x in cb_rows if x['step'] == PRIMARY]:
    pstr = f"<{r['p']:.1e}" if r['p_floored'] else f"{r['p']:.3e}"
    print(f"  STEP vs {r['baseline']:<16} {r['delta_obs']:+8.4f} {r['boot_se']:8.5f} "
          f"{pstr:>12}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
rep['cluster'] = cb_rows

print("\nPer-model AUC, cluster-bootstrap 95% CI:")
ab = []
for m_ in STEPS + BASELINES:
    v = np.array(aucs_b[m_])
    lo, hi = np.percentile(v, [2.5, 97.5])
    ab.append(dict(model=m_, auc=float(roc_auc_score(y, P[m_])), lo=float(lo),
                   hi=float(hi), se=float(v.std(ddof=1))))
    print(f"  {m_:14s} {ab[-1]['auc']:.4f}  [{lo:.4f}, {hi:.4f}]  SE {v.std(ddof=1):.5f}")
rep['cluster_auc'] = ab


# ======================================================== 3. side by side
print("\n" + "=" * 80)
print(f"3. SIDE BY SIDE  (STEP = {PRIMARY})")
print("=" * 80)
print(f"{'baseline':<12} {'dAUC':>8} | {'DeLong SE':>10} {'DeLong P':>11} | "
      f"{'clust SE':>9} {'clust P':>11} | {'SE infl':>8}")
side = []
for b in BASELINES:
    d = [x for x in dl_rows if x['step'] == PRIMARY and x['baseline'] == b][0]
    c = [x for x in cb_rows if x['step'] == PRIMARY and x['baseline'] == b][0]
    infl = c['boot_se'] / d['se']
    side.append(dict(baseline=b, delta=d['delta'], delong_se=d['se'], delong_p=d['p'],
                     delong_lo=d['lo'], delong_hi=d['hi'],
                     cluster_se=c['boot_se'], cluster_p=c['p'],
                     cluster_lo=c['lo'], cluster_hi=c['hi'], se_inflation=float(infl)))
    cp = f"<{c['p']:.1e}" if c['p_floored'] else f"{c['p']:.3e}"
    print(f"  {b:<10} {d['delta']:+8.4f} | {d['se']:10.5f} {d['p']:11.3e} | "
          f"{c['boot_se']:9.5f} {cp:>11} | {infl:8.2f}x")
rep['side_by_side'] = side

# released-script reproduction, for the record
print("\n" + "=" * 80)
print("4. What the released script does, on these same new numbers")
print("=" * 80)


def hm_se(a, npos, nneg):
    q1 = a / (2 - a); q2 = 2 * a ** 2 / (1 + a)
    return np.sqrt((a * (1 - a) + (npos - 1) * (q1 - a ** 2) +
                    (nneg - 1) * (q2 - a ** 2)) / (npos * nneg))


npos, nneg = int(y.sum()), int((y == 0).sum())
hm = []
a_s = roc_auc_score(y, P[PRIMARY])
for b in BASELINES:
    a_b = roc_auc_score(y, P[b])
    se = np.sqrt(hm_se(a_s, npos, nneg) ** 2 + hm_se(a_b, npos, nneg) ** 2)
    zz = (a_s - a_b) / se
    pp = float(2 * stats.norm.sf(abs(zz)))
    d = [x for x in dl_rows if x['step'] == PRIMARY and x['baseline'] == b][0]
    hm.append(dict(baseline=b, hm_se=float(se), hm_p=pp, delong_se=d['se'], delong_p=d['p']))
    print(f"  STEP vs {b:<10} Hanley-McNeil SE {se:.5f} P {pp:.3e}   "
          f"| DeLong SE {d['se']:.5f} P {d['p']:.3e}")
rep['hanley_mcneil'] = hm

json.dump(rep, open(OUT, 'w'), indent=2)
print(f"\nsaved {OUT}")
