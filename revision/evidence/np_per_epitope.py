"""
Numbers pass, part E: per-epitope statistics on the v2 balanced test set.

No inference needed: results/matrix_v2/test_predictions.npz already holds the
five configurations x three seeds on test_balanced, and its row order is
verified here against data/splits_600_epitope_v2/test_balanced.tsv.

Replaces the submitted figures, which were computed on the v1 split: 340
epitopes with >= 5 samples, mean ROC 0.773 +/- 0.248, median 0.833, and 42.6 /
54.1 / 66.5 % at or above 0.9 / 0.8 / 0.7.

Read-only. Writes only under revision/evidence/numbers_pass/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/numbers_pass'
MIN_N = 5

tb = pd.read_csv(BASE / 'data/splits_600_epitope_v2/test_balanced.tsv', sep='\t')
z = np.load(BASE / 'results/matrix_v2/test_predictions.npz')
y = z['y'].astype(int)
assert len(tb) == len(y) and (tb.label.values == y).all(), 'row order mismatch'
epi = tb['antigen.epitope'].astype(str).values
print(f'rows {len(y)}  epitopes {len(set(epi))}  positives {y.sum()}')

groups = {e: np.where(epi == e)[0] for e in sorted(set(epi))}
sizes = np.array([len(v) for v in groups.values()])
print(f'samples per epitope: median {np.median(sizes):.0f}, mean {sizes.mean():.1f}, '
      f'max {sizes.max()}, singletons {(sizes == 1).sum()}')

evaluable = [e for e, ix in groups.items()
             if len(ix) >= MIN_N and len(np.unique(y[ix])) == 2]
print(f'\nepitopes with >= {MIN_N} samples AND both classes: {len(evaluable)} '
      f'(submitted figure, v1: 340)')
only_n = [e for e, ix in groups.items() if len(ix) >= MIN_N]
print(f'  of which fail only the both-classes test: {len(only_n) - len(evaluable)} '
      f'(>= {MIN_N} samples: {len(only_n)})')

CONFIGS = ['corrected', 'none', 'length_only', 'shuffled', 'uniform']
rows, per_epi = [], []
for cfg in CONFIGS:
    cols = {s: z[f'{cfg}_s{s}'] for s in (42, 43, 44)}
    cols['ens'] = np.mean([cols[s] for s in (42, 43, 44)], axis=0)
    for tag, p in cols.items():
        vals = np.array([roc_auc_score(y[groups[e]], p[groups[e]]) for e in evaluable])
        rows.append(dict(config=cfg, run=tag, n_epitopes=len(vals),
                         mean=vals.mean(), sd=vals.std(ddof=1),
                         median=float(np.median(vals)),
                         frac_ge_90=float((vals >= 0.9).mean()),
                         frac_ge_80=float((vals >= 0.8).mean()),
                         frac_ge_70=float((vals >= 0.7).mean()),
                         frac_le_50=float((vals <= 0.5).mean())))
        if cfg == 'corrected':
            per_epi.append(pd.DataFrame(dict(epitope=evaluable, run=tag, roc=vals,
                                             n=[len(groups[e]) for e in evaluable])))

df = pd.DataFrame(rows)
print('\n=== per-epitope ROC-AUC, %d evaluable epitopes ===' % len(evaluable))
print(f"{'config':<12} {'run':>4} {'mean':>7} {'SD':>7} {'median':>7} "
      f"{'>=0.9':>7} {'>=0.8':>7} {'>=0.7':>7} {'<=0.5':>7}")
for _, r in df.iterrows():
    print(f"{r.config:<12} {r.run:>4} {r['mean']:>7.4f} {r.sd:>7.4f} {r['median']:>7.4f} "
          f"{100*r.frac_ge_90:>6.1f}% {100*r.frac_ge_80:>6.1f}% "
          f"{100*r.frac_ge_70:>6.1f}% {100*r.frac_le_50:>6.1f}%")

g = df[(df.config == 'corrected')]
print('\nSTEP (corrected), mean over the three seeds: '
      f"{g[g.run != 'ens']['mean'].mean():.4f} "
      f"+/- {g[g.run != 'ens']['mean'].std(ddof=1):.4f} (between-seed SD of the mean)")

df.to_csv(OUT / 'per_epitope_v2.csv', index=False)
pd.concat(per_epi).to_csv(OUT / 'per_epitope_v2_corrected_values.csv', index=False)
json.dump(dict(n_rows=int(len(y)), n_epitopes=int(len(groups)),
               n_evaluable=int(len(evaluable)),
               n_min5=int(len(only_n)), submitted_n=340,
               samples_per_epitope=dict(median=float(np.median(sizes)),
                                        mean=float(sizes.mean()),
                                        max=int(sizes.max()),
                                        singletons=int((sizes == 1).sum()))),
          open(OUT / 'per_epitope_v2_meta.json', 'w'), indent=2)
print('\nsaved', OUT / 'per_epitope_v2.csv')
