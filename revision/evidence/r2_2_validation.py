"""
R2-2: what the validation set is, and what fitting Youden's J on test instead of
validation actually costs.

Read-only. Nothing outside revision/evidence/ is written. No model is loaded:
the test scores were produced by the corrected-prior rerun and are read from
results/matrix_v2/test_predictions.npz.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = '/home/muhammad-imran/PhD_data/TCR_Module_1'
rep = {}
sep = '=' * 78

# ------------------------------------------------- 1. the split, v1 and v2
print(sep); print('  1. THE VALIDATION SET'); print(sep)
for tag, D in [('v1 (as submitted)', 'data/splits_600_epitope/'),
               ('v2 (whitespace-repaired)', 'data/splits_600_epitope_v2/')]:
    d = {n: pd.read_csv(f'{ROOT}/{D}{n}.tsv', sep='\t')
         for n in ('train', 'val', 'test', 'test_balanced')}
    print(f'\n  {tag}')
    for n, df in d.items():
        print(f'    {n:<14} rows {len(df):>7,}  pos {int(df.label.sum()):>6,}  '
              f'pos% {100 * df.label.mean():>6.3f}  epitopes '
              f'{df["antigen.epitope"].nunique():>5}')
    E = lambda x: set(x['antigen.epitope'])
    P = lambda x: set(zip(x.cdr3, x['antigen.epitope']))
    ov = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test'),
                 ('val', 'test_balanced')]:
        ov[f'{a}|{b}'] = dict(epitopes=len(E(d[a]) & E(d[b])),
                              pairs=len(P(d[a]) & P(d[b])),
                              cdr3=len(set(d[a].cdr3) & set(d[b].cdr3)))
        print(f'    {a} n {b}: epitopes {ov[f"{a}|{b}"]["epitopes"]}, '
              f'exact pairs {ov[f"{a}|{b}"]["pairs"]}, '
              f'shared CDR3b {ov[f"{a}|{b}"]["cdr3"]:,}')
    print(f'    val epitopes subset of test epitopes: '
          f'{E(d["val"]).issubset(E(d["test"]))}')
    print(f'    union of the three epitope sets: '
          f'{len(E(d["train"]) | E(d["val"]) | E(d["test"]))}')
    rep[tag] = dict(
        counts={n: dict(rows=len(x), pos=int(x.label.sum()),
                        pos_frac=float(x.label.mean()),
                        epitopes=int(x['antigen.epitope'].nunique()))
                for n, x in d.items()},
        overlaps=ov)

# --------------------------------------- 2. threshold fitted on val vs test
print('\n' + sep)
print('  2. YOUDEN THRESHOLD: FITTED ON VALIDATION vs FITTED ON TEST')
print(sep)
z = np.load(f'{ROOT}/results/matrix_v2/test_predictions.npz')
y = z['y']
val = pd.read_csv(f'{ROOT}/results/matrix_v2/threshold_from_val.csv')


def at(tau, s):
    p = (s >= tau).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tau=float(tau), f1=f1, precision=prec, recall=rec,
                specificity=tn / (tn + fp) if tn + fp else 0.0)


def youden_on(s, labels):
    o = np.argsort(-s)
    sl, ss = labels[o], s[o]
    P, N = sl.sum(), (1 - sl).sum()
    tp = np.cumsum(sl); fp = np.cumsum(1 - sl)
    j = tp / P - fp / N
    return float(ss[int(np.argmax(j))])


rows = []
for _, r in val[val.prior == 'corrected'].iterrows():
    s = z[f'corrected_s{int(r.seed)}']
    assert abs(roc_auc_score(y, s) - r.roc_auc) < 1e-9, 'scores do not match the csv'
    v = at(r.tau_from_val, s)
    t = at(youden_on(s, y), s)
    rows.append(dict(seed=int(r.seed), roc=float(r.roc_auc),
                     tau_val=v['tau'], **{f'val_{k}': v[k] for k in
                                          ('f1', 'precision', 'recall', 'specificity')},
                     tau_test=t['tau'], **{f'test_{k}': t[k] for k in
                                           ('f1', 'precision', 'recall', 'specificity')}))
    # the csv's own metrics must reproduce
    for k in ('f1', 'precision', 'recall', 'specificity'):
        assert abs(v[k] - r[k]) < 1e-6, (k, v[k], r[k])

df = pd.DataFrame(rows)
print('\n  threshold fitted on VALIDATION (honest), metrics on test:')
print('    seed   ROC      tau      F1     prec    recall    spec')
for _, r in df.iterrows():
    print(f'    {r.seed:>4}  {r.roc:.4f}  {r.tau_val:.4f}  {r.val_f1:.4f}  '
          f'{r.val_precision:.4f}  {r.val_recall:.4f}  {r.val_specificity:.4f}')
print(f'    mean        {df.tau_val.mean():.4f}  {df.val_f1.mean():.4f}  '
      f'{df.val_precision.mean():.4f}  {df.val_recall.mean():.4f}  '
      f'{df.val_specificity.mean():.4f}')
print(f'    sd          {df.tau_val.std(ddof=1):.4f}  {df.val_f1.std(ddof=1):.4f}  '
      f'{df.val_precision.std(ddof=1):.4f}  {df.val_recall.std(ddof=1):.4f}  '
      f'{df.val_specificity.std(ddof=1):.4f}')

print('\n  threshold fitted on TEST (what the submitted paper did), same models:')
print('    seed   ROC      tau      F1     prec    recall    spec')
for _, r in df.iterrows():
    print(f'    {r.seed:>4}  {r.roc:.4f}  {r.tau_test:.4f}  {r.test_f1:.4f}  '
          f'{r.test_precision:.4f}  {r.test_recall:.4f}  {r.test_specificity:.4f}')
print(f'    mean        {df.tau_test.mean():.4f}  {df.test_f1.mean():.4f}  '
      f'{df.test_precision.mean():.4f}  {df.test_recall.mean():.4f}  '
      f'{df.test_specificity.mean():.4f}')

print(f'\n  optimism from fitting on test, mean F1: '
      f'{df.test_f1.mean() - df.val_f1.mean():+.4f} '
      f'({df.test_f1.mean():.4f} vs {df.val_f1.mean():.4f})')
print(f'  per seed: ' + ', '.join(f'{a:+.4f}' for a in (df.test_f1 - df.val_f1)))
print(f'  tau spread, validation-fitted: '
      f'{df.tau_val.min():.4f} to {df.tau_val.max():.4f} '
      f'({df.tau_val.max() / df.tau_val.min():.1f}x)')
print(f'  tau spread, test-fitted:       '
      f'{df.tau_test.min():.4f} to {df.tau_test.max():.4f} '
      f'({df.tau_test.max() / df.tau_test.min():.1f}x)')
rep['threshold'] = df.to_dict('records')

# ------------------------------------------- 3. the 0.7547 / 0.7549 near-match
print('\n' + sep)
print('  3. best-val 0.7547 vs test 0.7549')
print(sep)
h = json.load(open(f'{ROOT}/models/checkpoints/step_600epi_20260402_164333/history.json'))
va = h.get('val_auc') or h.get('val_aucs') or h.get('val_roc_auc')
print(f'  val ROC-AUC by epoch ({len(va)} epochs):')
print('    ' + ', '.join(f'{v:.4f}' for v in va))
print(f'  max {max(va):.6f} at epoch {int(np.argmax(va)) + 1}; '
      f'mean {np.mean(va):.4f}, sd {np.std(va, ddof=1):.4f}, '
      f'range {min(va):.4f}-{max(va):.4f}')
rep['history'] = dict(val_auc=list(map(float, va)), best=float(max(va)),
                      best_epoch=int(np.argmax(va)) + 1,
                      mean=float(np.mean(va)), sd=float(np.std(va, ddof=1)))

json.dump(rep, open(f'{ROOT}/revision/evidence/r2_2_validation.json', 'w'),
          indent=2, default=str)
print('\nsaved revision/evidence/r2_2_validation.json')
