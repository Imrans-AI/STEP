"""
R2-1 Part A (read-only): how 'unseen' are the v2 test epitopes, and does STEP's
per-epitope performance depend on distance to the nearest training epitope?

Reads the v2 splits and results/baseline_rerun_v2/per_sample_predictions_v2.csv.
Writes revision/evidence/r2_1_partA.json + a per-epitope CSV. Changes nothing else.
"""
import json
import numpy as np
import pandas as pd
import Levenshtein
from sklearn.metrics import roc_auc_score

SP = 'data/splits_600_epitope_v2'
PRED = 'results/baseline_rerun_v2/per_sample_predictions_v2.csv'
OUTJ = 'revision/evidence/r2_1_partA.json'
OUTC = 'revision/evidence/r2_1_partA_per_epitope.csv'
MODELS = ['STEP_seed42', 'STEP_ensemble', 'DAISY']

CLINICAL = {
    'NLVPMVATV': 'CMV pp65', 'NIVPMVATV': 'CMV pp65 variant', 'NMVPMVATV': 'CMV pp65 variant',
    'LLVPMVATV': 'CMV pp65 variant', 'VLVPMVATV': 'CMV pp65 variant',
    'NLLPMVATV': 'CMV pp65 variant', 'NCVPMVATV': 'CMV pp65 variant',
    'QASGNHAAGILTM': 'EBV EBNA3C', 'FLKEQGGL': 'HIV gag',
}

train = pd.read_csv(f'{SP}/train.tsv', sep='\t')
pred = pd.read_csv(PRED)
train_epi = sorted(set(train['antigen.epitope']))
test_epi = sorted(set(pred['antigen.epitope']))
print("=" * 78)
print(f"train epitopes {len(train_epi)} | test epitopes {len(test_epi)} | "
      f"exact overlap {len(set(train_epi) & set(test_epi))}")

# ---------------------------------------------------------------- distances
def nearest(e, pool):
    # exhaustive. An earlier version short-circuited at d == 1, which is only the
    # true minimum when no exact match exists. That holds for the test epitopes
    # (exact overlap is 0 by construction) but NOT for the cancer sets, where it
    # reported distance 1 for epitopes that are in the training set. Do not
    # reintroduce the break.
    best, who = 10**9, None
    for t in pool:
        if abs(len(e) - len(t)) >= best:
            continue
        d = Levenshtein.distance(e, t)
        if d < best:
            best, who = d, t
            if d == 0:
                break
    return best, who


rows = []
for e in test_epi:
    d, who = nearest(e, train_epi)
    ident = 1.0 - d / max(len(e), len(who))
    rows.append(dict(epitope=e, min_dist=d, nearest_train=who, length=len(e),
                     identity=ident))
D = pd.DataFrame(rows)

print("\n--- distance of each test epitope to its nearest TRAINING epitope ---")
bins = {'1': (D.min_dist == 1), '2': (D.min_dist == 2), '3': (D.min_dist == 3),
        '>=4': (D.min_dist >= 4)}
dist_tab = []
for k, m in bins.items():
    dist_tab.append(dict(bin=k, n_epitopes=int(m.sum()),
                         pct=round(100 * m.mean(), 1)))
    print(f"  distance {k:>3}: {int(m.sum()):4d} epitopes ({100 * m.mean():5.1f}%)")
print(f"  distance 0 (exact): {int((D.min_dist == 0).sum())}")
print(f"  >=80% identity to a training epitope: {int((D.identity >= 0.8).sum())} "
      f"({100 * (D.identity >= 0.8).mean():.1f}%)")

# ---------------------------------------------------------------- per-epitope AUC
g = pred.groupby('antigen.epitope')
per = []
for e, sub in g:
    n, npos = len(sub), int(sub.label.sum())
    rec = dict(epitope=e, n=n, n_pos=npos, n_neg=n - npos)
    ok = n >= 5 and 0 < npos < n
    rec['evaluable'] = ok
    for m in MODELS:
        rec[m] = roc_auc_score(sub.label, sub[m]) if ok else np.nan
    per.append(rec)
P = pd.DataFrame(per).merge(D, on='epitope', how='left')
ev = P[P.evaluable]
print(f"\nevaluable epitopes (n>=5 and both classes): {len(ev)} of {len(P)}")
print(f"  they carry {int(ev.n.sum())} of {len(pred)} test rows")

print("\n--- per-epitope ROC-AUC stratified by distance to nearest training epitope ---")
print(f"{'bin':>5} {'n_epi':>6} {'n_rows':>7} " + ''.join(f"{m:>16}" for m in MODELS))
strat = []
for k, _ in bins.items():
    s = ev[ev.min_dist == 1] if k == '1' else \
        ev[ev.min_dist == 2] if k == '2' else \
        ev[ev.min_dist == 3] if k == '3' else ev[ev.min_dist >= 4]
    if len(s) == 0:
        continue
    r = dict(bin=k, n_epitopes=int(len(s)), n_rows=int(s.n.sum()))
    line = f"{k:>5} {len(s):6d} {int(s.n.sum()):7d} "
    for m in MODELS:
        r[f'{m}_mean'] = float(s[m].mean()); r[f'{m}_sd'] = float(s[m].std(ddof=1))
        r[f'{m}_median'] = float(s[m].median())
        line += f"  {s[m].mean():.3f}+-{s[m].std(ddof=1):.3f}"
    strat.append(r)
    print(line)

# pooled (sample-level) AUC within each distance bin, which weights by row count
print("\n--- pooled sample-level ROC-AUC within each distance bin ---")
pm = pred.merge(D[['epitope', 'min_dist']], left_on='antigen.epitope', right_on='epitope')
pooled = []
print(f"{'bin':>5} {'n_rows':>7} {'pos':>6} " + ''.join(f"{m:>14}" for m in MODELS))
for k in bins:
    s = pm[pm.min_dist == 1] if k == '1' else pm[pm.min_dist == 2] if k == '2' else \
        pm[pm.min_dist == 3] if k == '3' else pm[pm.min_dist >= 4]
    if len(s) == 0 or s.label.nunique() < 2:
        continue
    r = dict(bin=k, n_rows=int(len(s)), n_pos=int(s.label.sum()))
    line = f"{k:>5} {len(s):7d} {int(s.label.sum()):6d} "
    for m in MODELS:
        r[m] = float(roc_auc_score(s.label, s[m]))
        line += f"  {r[m]:12.4f}"
    pooled.append(r)
    print(line)

# ---------------------------------------------------------------- named sets
print("\n--- clinical epitopes ---")
print(f"{'epitope':<16}{'note':<20}{'dist':>5}{'nearest train':>16}{'n':>5}{'pos':>5}"
      + ''.join(f"{m:>14}" for m in MODELS))
clin = []
for e, note in CLINICAL.items():
    r = P[P.epitope == e]
    if len(r) == 0:
        print(f"  {e:<14}{note:<20}{'-':>5}{'(not in v2 test)':>16}")
        clin.append(dict(epitope=e, note=note, in_test=False))
        continue
    r = r.iloc[0]
    d = dict(epitope=e, note=note, in_test=True, min_dist=int(r.min_dist),
             nearest=r.nearest_train, n=int(r.n), n_pos=int(r.n_pos),
             evaluable=bool(r.evaluable))
    line = f"  {e:<14}{note:<20}{int(r.min_dist):5d}{r.nearest_train:>16}{int(r.n):5d}{int(r.n_pos):5d}"
    for m in MODELS:
        d[m] = None if np.isnan(r[m]) else float(r[m])
        line += f"{'  n/a' if np.isnan(r[m]) else f'{r[m]:14.3f}'}"
    clin.append(d)
    print(line)

print("\n--- melanoma epitopes (data/cancer_specific/melanoma_test.tsv) ---")
mel = pd.read_csv('data/cancer_specific/melanoma_test.tsv', sep='\t')
mel_epi = sorted(set(mel['antigen.epitope']))
mrows = []
for e in mel_epi:
    d, who = nearest(e, train_epi)
    inq = e in set(test_epi)
    mrows.append(dict(epitope=e, min_dist=int(d), nearest_train=who,
                      identity=round(1 - d / max(len(e), len(who)), 3),
                      in_v2_test=bool(inq), n_mel=int((mel['antigen.epitope'] == e).sum())))
M = pd.DataFrame(mrows).sort_values('min_dist')
print(f"  {len(M)} melanoma epitopes; distance to nearest v2 TRAINING epitope:")
for k, m in {'0': M.min_dist == 0, '1': M.min_dist == 1, '2': M.min_dist == 2,
             '3': M.min_dist == 3, '>=4': M.min_dist >= 4}.items():
    print(f"    distance {k:>3}: {int(m.sum()):3d}")
print(M.to_string(index=False))

P.to_csv(OUTC, index=False)
json.dump(dict(n_train_epi=len(train_epi), n_test_epi=len(test_epi),
               exact_overlap=len(set(train_epi) & set(test_epi)),
               distance_table=dist_tab,
               n_ge80_identity=int((D.identity >= 0.8).sum()),
               n_evaluable=int(len(ev)),
               stratified=strat, pooled=pooled, clinical=clin,
               melanoma=M.to_dict('records')),
          open(OUTJ, 'w'), indent=2, default=str)
print(f"\nsaved {OUTJ}\nsaved {OUTC}")
