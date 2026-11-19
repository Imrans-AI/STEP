"""
Numbers pass, part B: rebuild Table 1's unseen-epitope column, and the Fig 4c,d
ablation panel data.

Read-only on results/ and models/. Writes only under
revision/evidence/numbers_pass/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/numbers_pass'
OUT.mkdir(parents=True, exist_ok=True)
rep = {}

# ------------------------------------------------------------------ Table 1
P = pd.read_csv(BASE / 'results/baseline_rerun_v2/per_sample_predictions_v2.csv')
y = P['label'].values.astype(int)
epi = P['antigen.epitope'].values
print(f'rows {len(P)}  pos {y.sum()}  epitopes {len(set(epi))}')
assert len(P) == 3314 and y.sum() == 1657

MODELS = ['STEP_seed42', 'STEP_seed43', 'STEP_seed44', 'STEP_ensemble',
          'DAISY', 'ERGO-LSTM', 'ERGO-AE', 'TEINet', 'ATM-TCR']
OLD = {'STEP': (0.755, 0.736), 'DAISY': (0.701, 0.658), 'ERGO-LSTM': (0.589, 0.583),
       'ERGO-AE': (0.570, 0.564), 'TEINet': (0.493, 0.518), 'ATM-TCR': (0.479, 0.486)}


def cluster_boot(fn, n=10000, seed=42):
    """resample the 576 epitopes, not the 3,314 rows"""
    rng = np.random.default_rng(seed)
    groups = {e: np.where(epi == e)[0] for e in set(epi)}
    keys = list(groups)
    out = []
    for _ in range(n):
        drawn = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in drawn])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(fn(idx))
    return np.array(out)


print('\n=== Table 1, unseen-epitope column: old vs new ===')
print(f"{'model':<14} {'ROC old':>8} {'ROC new':>8} {'Δ':>8} | "
      f"{'PR old':>7} {'PR new':>8} {'Δ':>8} | 95% CI (epitope cluster)")
t1 = []
for m in MODELS:
    s = P[m].values.astype(float)
    roc, pr = roc_auc_score(y, s), average_precision_score(y, s)
    b = cluster_boot(lambda i, s=s: roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    key = 'STEP' if m.startswith('STEP') else m
    o = OLD.get(key, (np.nan, np.nan))
    t1.append(dict(model=m, roc_new=roc, pr_new=pr, roc_old=o[0], pr_old=o[1],
                   ci_lo=lo, ci_hi=hi, above_chance=bool(lo > 0.5)))
    print(f"{m:<14} {o[0]:>8.3f} {roc:>8.4f} {roc-o[0]:>+8.4f} | "
          f"{o[1]:>7.3f} {pr:>8.4f} {pr-o[1]:>+8.4f} | "
          f"[{lo:.4f}, {hi:.4f}]{'' if lo > 0.5 else '   <-- includes 0.5'}")

st = [r for r in t1 if r['model'].startswith('STEP_seed')]
print(f"\nSTEP per-seed mean ROC {np.mean([r['roc_new'] for r in st]):.4f} "
      f"± {np.std([r['roc_new'] for r in st], ddof=1):.4f}; "
      f"PR {np.mean([r['pr_new'] for r in st]):.4f} "
      f"± {np.std([r['pr_new'] for r in st], ddof=1):.4f}")

# ATM-TCR vs ERGO-LSTM: are they orderable?
a, e = P['ATM-TCR'].values.astype(float), P['ERGO-LSTM'].values.astype(float)
d = cluster_boot(lambda i: roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], e[i]))
lo, hi = np.percentile(d, [2.5, 97.5])
pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
print(f"\nATM-TCR − ERGO-LSTM = {roc_auc_score(y,a)-roc_auc_score(y,e):+.4f}  "
      f"95% CI [{lo:+.4f}, {hi:+.4f}]  P = {max(pval,1e-4):.3f}  "
      f"-> orderable: {not (lo < 0 < hi)}")
rep['table1'] = t1
rep['atm_vs_ergolstm'] = dict(delta=float(roc_auc_score(y, a) - roc_auc_score(y, e)),
                              ci=[float(lo), float(hi)], p=float(max(pval, 1e-4)))

# ---------------------------------------------------------------- Fig 4c,d
print('\n=== Fig 4c,d: five retrained configurations, three seeds ===')
ps = pd.read_csv(BASE / 'results/matrix_v2/per_seed_metrics.csv')
fig = []
for v, g in ps.groupby('prior'):
    fig.append(dict(config=v, n=len(g),
                    roc_mean=g.roc_auc.mean(), roc_sd=g.roc_auc.std(ddof=1),
                    pr_mean=g.pr_auc.mean(), pr_sd=g.pr_auc.std(ddof=1),
                    roc_values=list(g.roc_auc), pr_values=list(g.pr_auc)))
order = ['corrected', 'none', 'length_only', 'shuffled', 'uniform']
fig.sort(key=lambda r: order.index(r['config']))
for r in fig:
    print(f"  {r['config']:<12} ROC {r['roc_mean']:.4f} ± {r['roc_sd']:.4f}   "
          f"PR {r['pr_mean']:.4f} ± {r['pr_sd']:.4f}   "
          f"seeds {', '.join(f'{x:.4f}' for x in r['roc_values'])}")
spread_roc = max(r['roc_mean'] for r in fig) - min(r['roc_mean'] for r in fig)
pooled_sd = np.sqrt(np.mean([r['roc_sd']**2 for r in fig]))
print(f"\n  spread of config means (ROC): {spread_roc:.4f}; "
      f"pooled within-config SD: {pooled_sd:.4f}  -> spread/SD = "
      f"{spread_roc/pooled_sd:.2f}")
rep['fig4'] = fig
rep['fig4_spread'] = dict(roc_spread=float(spread_roc), pooled_sd=float(pooled_sd))

json.dump(rep, open(OUT / 'table1_fig4.json', 'w'), indent=2, default=str)
pd.DataFrame(t1).to_csv(OUT / 'table1_rebuilt.csv', index=False)
pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)}
              for r in fig]).to_csv(OUT / 'fig4_ablation_panels.csv', index=False)
print('\nsaved', OUT)
