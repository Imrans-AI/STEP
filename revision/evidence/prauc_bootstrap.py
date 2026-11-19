"""Paired bootstrap on PR-AUC (average precision) for corrected vs each control.

Reads results/matrix_v2/test_predictions.npz only. No model, no checkpoint, no
training. Mirrors the ROC methodology already in EVAL_MATRIX_v2.py:
  * per seed: 2000 paired resamples, rng seed = the run seed (42/43/44)
  * across seeds: mean of the three paired deltas, t-CI with df = 2
  * seed ensemble: mean predicted probability over the three seeds, 4000
    paired resamples, rng seed 0
Resamples that lose a class are skipped (same rule as boot_diff).
"""
import json
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'results/matrix_v2/test_predictions.npz'
SEEDS = [42, 43, 44]
CONTROLS = ['none', 'length_only', 'uniform', 'shuffled']
N_BOOT = 2000
N_BOOT_ENS = 4000

d = np.load(NPZ)
y = d['y'].astype(int)
P = {k: d[k].astype(np.float64) for k in d.files if k != 'y'}

print(f"n={len(y)}  pos={int(y.sum())}  neg={int((1 - y).sum())}  "
      f"prevalence={y.mean():.4f}  (PR-AUC baseline = prevalence)")


def boot_diff_pr(y, pa, pb, n, rng_seed):
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(len(y))
    dd = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        dd.append(average_precision_score(y[s], pa[s]) -
                  average_precision_score(y[s], pb[s]))
    dd = np.asarray(dd)
    return dd.mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5), len(dd)


out = {'per_seed': [], 'across_seeds': [], 'ensemble_pr': {}, 'ensemble_diff': []}

print("\n=== per-seed PR-AUC ===")
for cfg in ['corrected'] + CONTROLS:
    vals = [average_precision_score(y, P[f'{cfg}_s{s}']) for s in SEEDS]
    print(f"  {cfg:12s} " + "  ".join(f"{v:.4f}" for v in vals) +
          f"   mean {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
    out['ensemble_pr'][cfg + '_per_seed'] = vals

print("\n=== corrected − control, paired bootstrap on PR-AUC, per seed ===")
deltas = {c: [] for c in CONTROLS}
for c in CONTROLS:
    for s in SEEDS:
        m, lo, hi, nk = boot_diff_pr(y, P[f'corrected_s{s}'], P[f'{c}_s{s}'],
                                     N_BOOT, s)
        point = (average_precision_score(y, P[f'corrected_s{s}']) -
                 average_precision_score(y, P[f'{c}_s{s}']))
        deltas[c].append(point)
        star = '*' if (lo > 0 or hi < 0) else ''
        out['per_seed'].append({'comparison': f'corrected - {c}', 'seed': s,
                                'delta_observed': point, 'delta_boot_mean': m,
                                'ci_lo': lo, 'ci_hi': hi,
                                'excludes_zero': bool(lo > 0 or hi < 0),
                                'n_resamples_kept': nk})
        print(f"  corrected - {c:12s} seed {s}: obs Δ={point:+.4f}  "
              f"boot mean={m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {star}")

print("\n=== across seeds (mean of the three paired deltas, t-CI df=2) ===")
for c in CONTROLS:
    a = np.array(deltas[c])
    mean, sd = a.mean(), a.std(ddof=1)
    half = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
    signs = ' '.join('+' if x > 0 else '-' for x in a)
    out['across_seeds'].append({'comparison': f'corrected - {c}', 'mean': mean,
                                'sd': sd, 'ci_lo': mean - half,
                                'ci_hi': mean + half, 'signs': signs,
                                'per_seed': a.tolist()})
    print(f"  corrected - {c:12s} Δ={mean:+.4f}  SD={sd:.4f}  "
          f"95% CI [{mean - half:+.4f}, {mean + half:+.4f}]  signs {signs}")

print("\n=== seed ensemble (mean probability over seeds 42/43/44) ===")
ENS = {c: np.mean([P[f'{c}_s{s}'] for s in SEEDS], axis=0)
       for c in ['corrected'] + CONTROLS}
for c in ['corrected'] + CONTROLS:
    v = average_precision_score(y, ENS[c])
    out['ensemble_pr'][c] = v
    print(f"  {c:12s} ensemble PR-AUC {v:.4f}")

print("\n=== corrected − control on the ensemble (4000 resamples) ===")
for c in CONTROLS:
    m, lo, hi, nk = boot_diff_pr(y, ENS['corrected'], ENS[c], N_BOOT_ENS, 0)
    point = (average_precision_score(y, ENS['corrected']) -
             average_precision_score(y, ENS[c]))
    star = '*' if (lo > 0 or hi < 0) else ''
    out['ensemble_diff'].append({'comparison': f'corrected - {c}',
                                 'delta_observed': point, 'delta_boot_mean': m,
                                 'ci_lo': lo, 'ci_hi': hi,
                                 'excludes_zero': bool(lo > 0 or hi < 0),
                                 'n_resamples_kept': nk})
    print(f"  corrected - {c:12s} obs Δ={point:+.4f}  boot mean={m:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] {star}")

print(json.dumps(out, indent=1, default=float), file=open(
    '/tmp/claude-1000/-home-muhammad-imran/9b1c1845-cbfd-4c8f-b061-5ef985a424ed/scratchpad/prauc_bootstrap.json', 'w'))
print("\nwrote prauc_bootstrap.json")
