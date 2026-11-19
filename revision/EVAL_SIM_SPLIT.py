#!/usr/bin/env python
"""
R2-1 Part C: evaluate on the similarity-disjoint balanced test set.

STEP  - models/checkpoints_sim/corrected_seed{42,43,44}, corrected v2 prior, strict load.
DAISY - model/DAISY_sim.pt if it exists (retrained on the sim split, the fair comparator);
        the v1 DAISY_ours.pt is also scored, clearly labelled, because it was trained on a
        different epitope set and its test epitopes are NOT similarity-disjoint from it.

Writes results/sim_split_eval/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))
os.chdir(ROOT)

SPLIT = 'data/splits_600_epitope_sim'
TEST = f'{SPLIT}/test_balanced.tsv'
OUTDIR = 'results/sim_split_eval'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

for s in (SEED,):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

df = pd.read_csv(TEST, sep='\t').reset_index(drop=True)
Y = df['label'].values.astype(int)
N = len(df)
print("=" * 78)
print(f"R2-1 Part C | {TEST}")
print(f"n = {N}  ({Y.sum()} pos / {(Y == 0).sum()} neg)  "
      f"{df['antigen.epitope'].nunique()} epitopes")
print("=" * 78)

PRED, ROWS = {}, []


def rep(name, p, extra=''):
    p = np.asarray(p, float).ravel()
    r = dict(model=name, n=int(p.size), roc=float(roc_auc_score(Y, p)),
             pr=float(average_precision_score(Y, p)), mean=float(p.mean()),
             sd=float(p.std(ddof=1)), distinct=int(np.unique(np.round(p, 6)).size))
    ROWS.append(r); PRED[name] = p
    print(f"  {name:22s} ROC {r['roc']:.4f}  PR {r['pr']:.4f}  "
          f"(pred SD {r['sd']:.4f}, {r['distinct']} distinct) {extra}")
    return r


# ----------------------------------------------------------------- STEP
print("\nSTEP (corrected prior, trained on the similarity-disjoint split)")
from step_model import STEPModel
from dataset_physchem_v2 import PhysChemV2Dataset
from torch.utils.data import DataLoader

ds = PhysChemV2Dataset(TEST, verbose=False)
assert len(ds) == N
dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
cols = []
for seed in (42, 43, 44):
    ck = torch.load(f'models/checkpoints_sim/corrected_seed{seed}/best_model.pt',
                    map_location='cpu', weights_only=False)
    m = STEPModel(use_structure_priors=True, priors_file='structure_priors_v2.npz')
    msd, sd_ = m.state_dict(), ck['model_state_dict']
    assert set(msd) == set(sd_), "key mismatch"
    m.load_state_dict(sd_)            # strict
    m.to(DEVICE).eval()
    out, yl = [], []
    with torch.no_grad():
        for ph, gf, lb, _c, _e in dl:
            out.extend(m(ph.to(DEVICE), gf.to(DEVICE)).cpu().numpy().ravel())
            yl.extend(lb.numpy().ravel())
    assert np.array_equal(np.asarray(yl, int), Y)
    p = np.asarray(out, float)
    cols.append(p)
    rep(f'STEP_seed{seed}', p, f"[val {ck.get('val_auc', float('nan')):.4f}, "
                               f"epoch {ck.get('epoch', '?')}, 163/163 keys]")
    del m
ens = np.mean(cols, axis=0)
rep('STEP_ensemble', ens)
roc3 = [r['roc'] for r in ROWS if r['model'].startswith('STEP_seed')]
pr3 = [r['pr'] for r in ROWS if r['model'].startswith('STEP_seed')]
print(f"  --> STEP mean +- SD over 3 seeds:  ROC {np.mean(roc3):.4f} +- {np.std(roc3, ddof=1):.4f}"
      f"   PR {np.mean(pr3):.4f} +- {np.std(pr3, ddof=1):.4f}")

# ----------------------------------------------------------------- DAISY
print("\nDAISY")
sys.path.insert(0, 'baselines/repos/DAISY')
from DAISY import get_interaction_map, get_global_attributes_diff, resnet_danet

tcrs = df['cdr3'].tolist(); peps = df['antigen.epitope'].tolist()
maps = torch.tensor(get_interaction_map(tcrs, peps), dtype=torch.float32)
diffs = torch.tensor(get_global_attributes_diff(tcrs, peps), dtype=torch.float32)


def run_daisy(path):
    m = resnet_danet()
    sd_ = torch.load(path, map_location='cpu')
    assert set(m.state_dict()) == set(sd_), f"key mismatch for {path}"
    m.load_state_dict(sd_)
    m.to(DEVICE).eval()
    o = []
    with torch.no_grad():
        for i in range(0, N, 256):
            z = m(maps[i:i + 256].to(DEVICE), diffs[i:i + 256].to(DEVICE))
            o.extend(torch.softmax(z, 1)[:, 1].cpu().numpy() if z.dim() > 1
                     else torch.sigmoid(z).cpu().numpy().ravel())
    del m
    return np.asarray(o, float)


sim_ck = 'baselines/repos/DAISY/model/DAISY_sim.pt'
if os.path.exists(sim_ck):
    rep('DAISY_sim', run_daisy(sim_ck), '[retrained on the sim split - fair comparator]')
else:
    print("  DAISY_sim.pt not found - retrain not run")
rep('DAISY_v1weights', run_daisy('baselines/repos/DAISY/model/DAISY_ours.pt'),
    '[v1 weights: NOT similarity-disjoint from this test set]')

# ----------------------------------------------------------------- save
os.makedirs(OUTDIR, exist_ok=True)
o = pd.DataFrame({'cdr3': tcrs, 'antigen.epitope': peps, 'label': Y})
for k, v in PRED.items():
    o[k] = v
o.to_csv(f'{OUTDIR}/per_sample_predictions_sim.csv', index=False)
np.savez_compressed(f'{OUTDIR}/test_predictions_sim.npz', y=Y,
                    cdr3=np.array(tcrs), epitope=np.array(peps), **PRED)
pd.DataFrame(ROWS).to_csv(f'{OUTDIR}/summary_sim.csv', index=False)
json.dump(dict(test=TEST, n=N, rows=ROWS,
               step_mean_roc=float(np.mean(roc3)), step_sd_roc=float(np.std(roc3, ddof=1)),
               step_mean_pr=float(np.mean(pr3)), step_sd_pr=float(np.std(pr3, ddof=1))),
          open(f'{OUTDIR}/eval_sim.json', 'w'), indent=2)
print("\n" + pd.DataFrame(ROWS)[['model', 'roc', 'pr', 'sd', 'distinct']].to_string(index=False))
print(f"\nsaved {OUTDIR}/")
