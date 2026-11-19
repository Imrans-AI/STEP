"""
Numbers pass, part A: Youden's J refit on val.tsv for ALL FIVE prior
configurations, not just `corrected`.

EVAL_MATRIX_v2.py already computes validation predictions for every
configuration, but its threshold block is hard-coded to `for v in ['corrected']`
(line 137), so results/matrix_v2/threshold_from_val.csv covers one of the five.
Table 2's "No structure" row needs the same treatment, so the loop is widened
here.

Nothing under results/ or models/ is written; output goes to
revision/evidence/numbers_pass/. No training — inference only, on checkpoints
that already exist.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel                       # noqa: E402
from dataset_physchem_v2 import PhysChemV2Dataset      # noqa: E402
from prior_variants import get_prior, VARIANTS         # noqa: E402

SPLITS = BASE / 'data' / 'splits_600_epitope_v2'
CKPT = BASE / 'models' / 'checkpoints_v2'
OUT = BASE / 'revision' / 'evidence' / 'numbers_pass'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]


def predict(ckpt_dir, prior, seed, loader, device):
    _, use_mask = get_prior(prior, seed)
    model = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                      priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=device,
                    weights_only=False)['model_state_dict']
    model.load_state_dict(sd, strict=True)          # assertion, not convenience
    model.eval()
    out = []
    with torch.no_grad():
        for phys, gf, y, cl, el in loader:
            phys, gf = phys.to(device), gf.to(device)
            cl, el = cl.to(device), el.to(device)
            o = model(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else model(phys, gf)
            out.append(o.cpu().numpy())
    return np.concatenate(out)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)
    test_ds = PhysChemV2Dataset(SPLITS / 'test_balanced.tsv', verbose=False)
    val_ds = PhysChemV2Dataset(SPLITS / 'val.tsv', verbose=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)
    y_test = test_ds.labels.astype(int)
    y_val = val_ds.labels.astype(int)
    print(f'test n={len(y_test)} pos={y_test.sum()} | val n={len(y_val)} '
          f'pos={y_val.sum()} ({100*y_val.mean():.2f}%)', flush=True)

    rows = []
    for v in VARIANTS:
        for s in SEEDS:
            d = CKPT / f'{v}_seed{s}'
            if not (d / 'results.json').exists():
                print('  missing', v, s, flush=True)
                continue
            pt = predict(d, v, s, test_loader, device)
            pv = predict(d, v, s, val_loader, device)
            fpr, tpr, thr = roc_curve(y_val, pv)
            tau = float(thr[np.argmax(tpr - fpr)])
            yhat = (pt >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0, 1]).ravel()
            # the same threshold fitted on TEST, for the optimism column
            fpr2, tpr2, thr2 = roc_curve(y_test, pt)
            tau_t = float(thr2[np.argmax(tpr2 - fpr2)])
            yh2 = (pt >= tau_t).astype(int)
            rows.append(dict(
                prior=v, seed=s,
                roc_auc=roc_auc_score(y_test, pt),
                pr_auc=average_precision_score(y_test, pt),
                val_roc_auc=roc_auc_score(y_val, pv),
                tau_from_val=tau,
                accuracy=accuracy_score(y_test, yhat),
                f1=f1_score(y_test, yhat),
                precision=precision_score(y_test, yhat, zero_division=0),
                recall=recall_score(y_test, yhat),
                specificity=tn / (tn + fp) if (tn + fp) else float('nan'),
                tau_from_test=tau_t,
                f1_at_test_tau=f1_score(y_test, yh2),
                best_val_auc=json.load(open(d / 'results.json'))['best_val_auc']))
            r = rows[-1]
            print(f"  {v:12s} s{s}: ROC {r['roc_auc']:.4f} PR {r['pr_auc']:.4f} "
                  f"tau(val) {tau:.4f} F1 {r['f1']:.4f} acc {r['accuracy']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'thresholds_all_variants.csv', index=False)

    agg = df.groupby('prior').agg(['mean', 'std'])
    print('\n=== mean +/- SD over three seeds ===', flush=True)
    for v in VARIANTS:
        g = df[df.prior == v]
        print(f"  {v:12s} ROC {g.roc_auc.mean():.4f}+/-{g.roc_auc.std(ddof=1):.4f}  "
              f"PR {g.pr_auc.mean():.4f}+/-{g.pr_auc.std(ddof=1):.4f}  "
              f"tau {g.tau_from_val.mean():.4f}+/-{g.tau_from_val.std(ddof=1):.4f}  "
              f"F1 {g.f1.mean():.4f}+/-{g.f1.std(ddof=1):.4f}  "
              f"Acc {g.accuracy.mean():.4f}+/-{g.accuracy.std(ddof=1):.4f}  "
              f"Prec {g.precision.mean():.4f}+/-{g.precision.std(ddof=1):.4f}  "
              f"Rec {g.recall.mean():.4f}+/-{g.recall.std(ddof=1):.4f}  "
              f"Spec {g.specificity.mean():.4f}+/-{g.specificity.std(ddof=1):.4f}",
              flush=True)
    agg.to_csv(OUT / 'thresholds_all_variants_agg.csv')

    # reproduce the published corrected rows exactly
    ref = pd.read_csv(BASE / 'results/matrix_v2/threshold_from_val.csv')
    mine = df[df.prior == 'corrected'].set_index('seed')
    ok = True
    for _, r in ref.iterrows():
        m = mine.loc[int(r.seed)]
        for k in ('tau_from_val', 'f1', 'precision', 'recall', 'specificity', 'roc_auc'):
            if abs(float(m[k]) - float(r[k])) > 1e-9:
                ok = False
                print(f'  MISMATCH {k} seed {int(r.seed)}: {m[k]} vs {r[k]}')
    print('\nreproduces results/matrix_v2/threshold_from_val.csv exactly:', ok,
          flush=True)
    print('saved', OUT / 'thresholds_all_variants.csv', flush=True)


if __name__ == '__main__':
    main()
