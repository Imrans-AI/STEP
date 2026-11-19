"""
Numbers pass, part F: B3 calibration, B4 melanoma antigen epitopes, B5 clinical
viral epitopes, B6 the two Score-CAM pairs.

B3-B5 read predictions already on disk. Only B6 runs forward passes, on two
pairs. Nothing is trained; nothing under data/, models/ or results/ is written.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
sys.path.insert(0, str(BASE / 'models'))
from step_model import STEPModel                           # noqa: E402
from dataset_physchem_only import compute_global_features  # noqa: E402
from prior_variants import get_prior                       # noqa: E402

OUT = BASE / 'revision/evidence/numbers_pass'
SEEDS = [42, 43, 44]
rep = {}

tb = pd.read_csv(BASE / 'data/splits_600_epitope_v2/test_balanced.tsv', sep='\t')
z = np.load(BASE / 'results/matrix_v2/test_predictions.npz')
y = z['y'].astype(int)
assert (tb.label.values == y).all()
epi = tb['antigen.epitope'].astype(str).values
P = {s: z[f'corrected_s{s}'] for s in SEEDS}
P['ens'] = np.mean([P[s] for s in SEEDS], axis=0)

# ------------------------------------------------------------------ B3
print('=' * 76); print('  B3  CALIBRATION on the v2 balanced unseen-epitope set'); print('=' * 76)


def ece(p, t, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(t[m].mean() - p[m].mean())
    return e


rows = []
for tag, p in P.items():
    rows.append(dict(run=tag, roc=roc_auc_score(y, p), ece=ece(p, y),
                     brier=brier_score_loss(y, p), mean_pred=float(p.mean())))
    r = rows[-1]
    print(f"  {str(tag):>4}: ROC {r['roc']:.4f}  ECE {r['ece']:.4f}  "
          f"Brier {r['brier']:.4f}  mean predicted {r['mean_pred']:.4f}")
g = [r for r in rows if r['run'] != 'ens']
print(f"  per-seed mean: ECE {np.mean([r['ece'] for r in g]):.4f} "
      f"+/- {np.std([r['ece'] for r in g], ddof=1):.4f}   "
      f"Brier {np.mean([r['brier'] for r in g]):.4f} "
      f"+/- {np.std([r['brier'] for r in g], ddof=1):.4f}")
print('  submitted, v1: unseen-epitope ECE 0.33; seen-pair ECE 0.026, Brier 0.039,')
print('  ROC 0.990. The seen-pair figures have NO v2 counterpart (no such partition).')
rep['B3'] = rows

# ------------------------------------------------------------------ B5
print('\n' + '=' * 76); print('  B5  CLINICAL VIRAL EPITOPES'); print('=' * 76)
CLIN = ['NCVPMVATV', 'LLVPMVATV', 'VLVPMVATV', 'NMVPMVATV', 'QASGNHAAGILTM', 'FLKEQGGL']
clin = []
for e in CLIN:
    ix = np.where(epi == e)[0]
    n, pos = len(ix), int(y[ix].sum()) if len(ix) else 0
    row = dict(epitope=e, n=n, pos=pos, neg=n - pos)
    if n and len(np.unique(y[ix])) == 2:
        for tag, p in P.items():
            row[f'roc_{tag}'] = roc_auc_score(y[ix], p[ix])
    clin.append(row)
    ev = 'evaluable' if (n and len(np.unique(y[ix])) == 2) else (
        'single-class' if n else 'absent from the v2 test set')
    print(f"  {e:<14} n={n:>3} pos={pos:>3} neg={n-pos:>3}  {ev}"
          + (f"  ROC seeds " + ', '.join(f"{row[f'roc_{s}']:.3f}" for s in SEEDS)
             + f"  ens {row['roc_ens']:.3f}" if 'roc_ens' in row else ''))
n_ev = sum(1 for r in clin if 'roc_ens' in r)
print(f'  evaluable: {n_ev} of 6 (submitted: 5 of 6, each at ROC-AUC 1.000)')
if n_ev:
    allv = [r['roc_ens'] for r in clin if 'roc_ens' in r]
    print(f'  ensemble ROC range {min(allv):.3f}-{max(allv):.3f}; '
          f'all equal 1.000: {all(abs(v - 1) < 1e-12 for v in allv)}')
rep['B5'] = clin

# ------------------------------------------------------------------ B4
print('\n' + '=' * 76); print('  B4  MELANOMA ANTIGEN EPITOPES'); print('=' * 76)
mel = pd.read_csv(BASE / 'data/cancer_specific/melanoma_test.tsv', sep='\t')
zc = np.load(OUT / 'cancer_ablation_preds.npz')
ym = mel.label.values.astype(int)
me = mel['antigen.epitope'].astype(str).values
Pm = {s: zc[f'melanoma_corrected_s{s}'] for s in SEEDS}
Pm['ens'] = np.mean([Pm[s] for s in SEEDS], axis=0)
TARGETS = [('AAGIGILTV', 'MART-1', 0.983), ('IMDQVPFSV', 'gp100', 0.719),
           ('EAAGIGILTV', 'MART-1', 0.769), ('ELAGIGILTV', 'MART-1', 0.627)]
b4 = []
for e, antigen, old in TARGETS:
    ix = np.where(me == e)[0]
    n, pos = len(ix), int(ym[ix].sum()) if len(ix) else 0
    row = dict(epitope=e, antigen=antigen, submitted=old, n=n, pos=pos)
    if n and len(np.unique(ym[ix])) == 2:
        for tag, p in Pm.items():
            row[f'roc_{tag}'] = roc_auc_score(ym[ix], p[ix])
        vals = [row[f'roc_{s}'] for s in SEEDS]
        row['mean'], row['sd'] = float(np.mean(vals)), float(np.std(vals, ddof=1))
        print(f"  {e:<12} {antigen:<8} n={n:>4} pos={pos:>4}  submitted {old:.3f} -> "
              f"{row['mean']:.4f} +/- {row['sd']:.4f}  (ens {row['roc_ens']:.4f})")
    else:
        print(f"  {e:<12} {antigen:<8} n={n:>4} pos={pos:>4}  not evaluable")
    b4.append(row)
rep['B4'] = b4

# ------------------------------------------------------------------ B6
print('\n' + '=' * 76); print('  B6  SCORE-CAM PAIRS, rescored under v2'); print('=' * 76)
NAMES = ['HYDROPHOBICITY', 'CHARGE', 'FLEXIBILITY', 'REFRACTIVITY', 'AREA_LOSS']
_t = ast.parse(open(BASE / 'data/compute_physchem.py').read())
TABLES = {n.targets[0].id: ast.literal_eval(n.value) for n in _t.body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id in NAMES}
PROPS = [TABLES[n] for n in NAMES]

PAIRS = [('NLVPMVATV', 'CASSKTQLWETQYF', '3GSN', 1.000),
         ('EAAGIGILTV', 'CAWSETGLGTGELFF', '4QOK', 0.969)]


def pmap(c, e):
    m = np.zeros((5, 20, 15), dtype=np.float32)
    for k, tbl in enumerate(PROPS):
        for i, a in enumerate(c[:20]):
            for j, b in enumerate(e[:15]):
                if a in tbl and b in tbl:
                    m[k, i, j] = abs(tbl[a] - tbl[b])
    return m


device = 'cuda' if torch.cuda.is_available() else 'cpu'
phys = torch.from_numpy(np.stack([pmap(c, e) for e, c, _, _ in PAIRS])).float().to(device)
gf = torch.from_numpy(np.stack([compute_global_features(c, e)
                                for e, c, _, _ in PAIRS])).float().to(device)
cl = torch.tensor([min(len(c), 20) for _, c, _, _ in PAIRS]).to(device)
el = torch.tensor([min(len(e), 15) for e, _, _, _ in PAIRS]).to(device)
b6 = []
preds = {}
for s in SEEDS:
    _, use_mask = get_prior('corrected', s)
    m = STEPModel(use_structure_priors=True, use_length_mask=use_mask,
                  priors_file='structure_priors_v2.npz').to(device)
    sd = torch.load(BASE / f'models/checkpoints_v2/corrected_seed{s}/best_model.pt',
                    map_location=device, weights_only=False)['model_state_dict']
    m.load_state_dict(sd, strict=True)
    m.eval()
    with torch.no_grad():
        o = (m(phys, gf, cdr3_len=cl, epi_len=el) if use_mask else m(phys, gf))
    preds[s] = o.cpu().numpy()
ens = np.mean([preds[s] for s in SEEDS], axis=0)
for i, (e, c, pdb, old) in enumerate(PAIRS):
    vals = [float(preds[s][i]) for s in SEEDS]
    b6.append(dict(epitope=e, cdr3=c, pdb=pdb, submitted=old, seeds=vals,
                   mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)),
                   ens=float(ens[i])))
    print(f"  {e:<12} {pdb}  submitted P(bind) {old:.3f} -> "
          f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}   "
          f"seeds {', '.join(f'{v:.4f}' for v in vals)}   ens {ens[i]:.4f}")
rep['B6'] = b6

json.dump(rep, open(OUT / 'b3456.json', 'w'), indent=2, default=str)
print('\nsaved', OUT / 'b3456.json')
