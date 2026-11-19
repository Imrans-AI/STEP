"""
Evaluate the v2 training matrix — R1-2, R2-4, R2-7, R2-2.

Reads every completed run in models/checkpoints_v2/, scores
data/splits_600_epitope_v2/test_balanced.tsv, and reports:
  * per-seed and mean +/- SD ROC-AUC and PR-AUC per prior variant
  * corrected vs none and corrected vs length_only differences with 95% CIs
  * Youden's J threshold REFIT ON val.tsv for the corrected runs, with test F1,
    precision, recall and specificity at that val-derived threshold (R2-2: the
    submitted work fitted the threshold on test labels)

Writes results/matrix_v2/ as CSV + JSON. Never reads test labels for any fitting
decision; the threshold comes from val only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)
from torch.utils.data import DataLoader

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel
from dataset_physchem_v2 import PhysChemV2Dataset
from prior_variants import get_prior, VARIANTS

SPLITS = BASE / 'data' / 'splits_600_epitope_v2'
CKPT = BASE / 'models' / 'checkpoints_v2'
OUT = BASE / 'results' / 'matrix_v2'
SEEDS = [42, 43, 44]
N_BOOT = 2000


def predict(ckpt_dir, prior, seed, ds, loader, device):
    W, use_mask = get_prior(prior, seed)
    model = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                      priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=device,
                    weights_only=False)['model_state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for phys, gf, y, cl, el in loader:
            phys, gf = phys.to(device), gf.to(device)
            cl, el = cl.to(device), el.to(device)
            o = model(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else model(phys, gf)
            preds.append(o.cpu().numpy())
    return np.concatenate(preds)


def boot_diff(y, pa, pb, n=N_BOOT, seed=0):
    """paired bootstrap of ROC-AUC difference a - b on the same samples"""
    rng = np.random.default_rng(seed)
    d = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = PhysChemV2Dataset(SPLITS / 'test_balanced.tsv', verbose=False)
    val_ds = PhysChemV2Dataset(SPLITS / 'val.tsv', verbose=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)
    y_test = test_ds.labels.astype(int)
    y_val = val_ds.labels.astype(int)
    print(f"test_balanced n={len(y_test)} pos={y_test.sum()} | val n={len(y_val)} pos={y_val.sum()}")

    rows, preds_store, val_store = [], {}, {}
    for v in VARIANTS:
        for s in SEEDS:
            d = CKPT / f"{v}_seed{s}"
            if not (d / 'results.json').exists():
                print(f"  missing: {v}_seed{s}")
                continue
            p = predict(d, v, s, test_ds, test_loader, device)
            pv = predict(d, v, s, val_ds, val_loader, device)
            preds_store[(v, s)] = p
            val_store[(v, s)] = pv
            res = json.load(open(d / 'results.json'))
            rows.append({'prior': v, 'seed': s,
                         'roc_auc': roc_auc_score(y_test, p),
                         'pr_auc': average_precision_score(y_test, p),
                         'val_auc_best': res['best_val_auc'],
                         'epochs_run': res['epochs_run']})
            print(f"  {v:12s} seed {s}: ROC {rows[-1]['roc_auc']:.4f}  "
                  f"PR {rows[-1]['pr_auc']:.4f}  (val {res['best_val_auc']:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per_seed_metrics.csv', index=False)
    np.savez(OUT / 'test_predictions.npz',
             y=y_test, **{f"{v}_s{s}": p for (v, s), p in preds_store.items()})

    print("\n=== per configuration (test_balanced) ===")
    summ = []
    for v in VARIANTS:
        g = df[df.prior == v]
        if g.empty:
            continue
        summ.append({'prior': v, 'n_seeds': len(g),
                     'roc_mean': g.roc_auc.mean(), 'roc_sd': g.roc_auc.std(ddof=1),
                     'pr_mean': g.pr_auc.mean(), 'pr_sd': g.pr_auc.std(ddof=1),
                     'roc_values': ', '.join(f"{x:.4f}" for x in g.roc_auc)})
        print(f"  {v:12s} ROC {g.roc_auc.mean():.4f} ± {g.roc_auc.std(ddof=1):.4f}   "
              f"PR {g.pr_auc.mean():.4f} ± {g.pr_auc.std(ddof=1):.4f}   "
              f"[{', '.join(f'{x:.4f}' for x in g.roc_auc)}]")
    pd.DataFrame(summ).to_csv(OUT / 'summary.csv', index=False)

    print("\n=== corrected vs controls (paired bootstrap on test_balanced) ===")
    diffs = []
    for other in ['none', 'length_only', 'shuffled', 'uniform']:
        for s in SEEDS:
            if ('corrected', s) in preds_store and (other, s) in preds_store:
                m, lo, hi = boot_diff(y_test, preds_store[('corrected', s)],
                                      preds_store[(other, s)], seed=s)
                diffs.append({'comparison': f'corrected - {other}', 'seed': s,
                              'delta_roc': m, 'ci_lo': lo, 'ci_hi': hi,
                              'excludes_zero': bool(lo > 0 or hi < 0)})
                print(f"  corrected - {other:12s} seed {s}: Δ={m:+.4f} "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]"
                      f"{'  *' if (lo > 0 or hi < 0) else ''}")
    if diffs:
        pd.DataFrame(diffs).to_csv(OUT / 'differences.csv', index=False)

    print("\n=== R2-2: Youden's J refit on val.tsv, applied to test ===")
    thr_rows = []
    for v in ['corrected']:
        for s in SEEDS:
            if (v, s) not in val_store:
                continue
            fpr, tpr, thr = roc_curve(y_val, val_store[(v, s)])
            tau = float(thr[np.argmax(tpr - fpr)])
            p = preds_store[(v, s)]
            yhat = (p >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()
            thr_rows.append({'prior': v, 'seed': s, 'tau_from_val': tau,
                             'f1': f1_score(y_test, yhat),
                             'precision': precision_score(y_test, yhat, zero_division=0),
                             'recall': recall_score(y_test, yhat),
                             'specificity': tn / (tn + fp) if (tn + fp) else float('nan'),
                             'roc_auc': roc_auc_score(y_test, p)})
            r = thr_rows[-1]
            print(f"  {v} seed {s}: τ(val)={tau:.4f}  F1={r['f1']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"Spec={r['specificity']:.4f}")
    if thr_rows:
        t = pd.DataFrame(thr_rows)
        t.to_csv(OUT / 'threshold_from_val.csv', index=False)
        print(f"\n  mean over seeds: τ={t.tau_from_val.mean():.4f}±{t.tau_from_val.std(ddof=1):.4f}  "
              f"F1={t.f1.mean():.4f}±{t.f1.std(ddof=1):.4f}  "
              f"Prec={t.precision.mean():.4f}±{t.precision.std(ddof=1):.4f}  "
              f"Rec={t.recall.mean():.4f}±{t.recall.std(ddof=1):.4f}  "
              f"Spec={t.specificity.mean():.4f}±{t.specificity.std(ddof=1):.4f}")

    print(f"\nwrote {OUT}/")


if __name__ == '__main__':
    main()
