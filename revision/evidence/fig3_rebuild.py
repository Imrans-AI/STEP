"""
Figure 3, rebuilt from the corrected predictions.

Panels a,b,e from the v2 per-epitope values; c,d from
results/baseline_rerun_v2/per_sample_predictions_v2.csv; f,g,h from the v2
cancer scoring in numbers_pass/cancer_ablation_preds.npz. Panel i (the melanoma
UMAP) needs 38,400-dimensional penultimate embeddings and is handled separately
- see FIGURES_REGENERATED.md.

Read-only on results/. Writes under revision/evidence/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'
NP = BASE / 'revision/evidence/numbers_pass'
INK, MUTED, GRID = '#1A1A1A', '#6B7280', '#9AA7B1'
STEP_C, DAISY_C, ERGO_C = '#2C6E8F', '#C2410C', '#7C7FB8'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'savefig.dpi': 600, 'figure.facecolor': 'white'})

pe = pd.read_csv(NP / 'per_epitope_v2_corrected_values.csv')
ens = pe[pe.run == 'ens'].reset_index(drop=True)
P = pd.read_csv(BASE / 'results/baseline_rerun_v2/per_sample_predictions_v2.csv')
y = P.label.values.astype(int)
zc = np.load(NP / 'cancer_ablation_preds.npz')
tr = pd.read_csv(BASE / 'data/splits_600_epitope_v2/train.tsv', sep='\t')
tpairs = set(zip(tr.cdr3.astype(str), tr['antigen.epitope'].astype(str)))

fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.0))
A = axes.ravel()


def tidy(ax, letter):
    ax.grid(color=GRID, alpha=0.28, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=8.5, colors=INK)
    ax.text(-0.17, 1.10, letter, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', color=INK)


# a -- per-epitope distribution
v = ens.roc.values
A[0].hist(v, bins=np.linspace(0, 1, 26), color=STEP_C, edgecolor='white', lw=0.6)
A[0].axvline(v.mean(), color=DAISY_C, ls='--', lw=1.7, label=f'mean {v.mean():.3f}')
A[0].axvline(np.median(v), color=DAISY_C, ls=':', lw=1.7,
             label=f'median {np.median(v):.3f}')
A[0].axvline(0.5, color=MUTED, lw=1.1, label='chance')
A[0].set_xlabel('Per-epitope ROC-AUC', fontsize=9, color=INK)
A[0].set_ylabel(f'Epitopes (n = {len(v)})', fontsize=9, color=INK)
A[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left')
tidy(A[0], 'a')

# b -- ROC vs sample count
A[1].scatter(ens.n, v, s=16, color=STEP_C, alpha=0.55, edgecolor='none')
A[1].axhline(v.mean(), color=DAISY_C, ls='--', lw=1.5)
A[1].axhline(0.5, color=MUTED, lw=1.1)
A[1].set_xlabel('Balanced samples per epitope', fontsize=9, color=INK)
A[1].set_ylabel('Per-epitope ROC-AUC', fontsize=9, color=INK)
tidy(A[1], 'b')

# c -- PR curves
for col, lab, c in (('STEP_ensemble', 'STEP', STEP_C), ('DAISY', 'DAISY', DAISY_C),
                    ('ERGO-LSTM', 'ERGO-LSTM', ERGO_C)):
    s = P[col].values.astype(float)
    pr, rc, _ = precision_recall_curve(y, s)
    A[2].plot(rc, pr, lw=1.9, color=c,
              label=f'{lab} (AP {average_precision_score(y, s):.3f})')
A[2].axhline(y.mean(), color=MUTED, ls=':', lw=1.1)
A[2].set_xlabel('Recall', fontsize=9, color=INK)
A[2].set_ylabel('Precision', fontsize=9, color=INK)
A[2].legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower left')
tidy(A[2], 'c')

# d -- AUPR, all six
models = ['STEP_ensemble', 'DAISY', 'ERGO-AE', 'ATM-TCR', 'ERGO-LSTM', 'TEINet']
labels = ['STEP', 'DAISY', 'ERGO-AE', 'ATM-TCR', 'ERGO-LSTM', 'TEINet']
ap = [average_precision_score(y, P[m].values.astype(float)) for m in models]
o = np.argsort(ap)
A[3].barh(range(6), [ap[k] for k in o], height=0.6,
          color=[STEP_C if labels[k] == 'STEP' else GRID for k in o])
A[3].set_yticks(range(6))
A[3].set_yticklabels([labels[k] for k in o], fontsize=8.5, color=INK)
for i2, k in enumerate(o):
    A[3].text(ap[k] + 0.012, i2, f'{ap[k]:.3f}', va='center', fontsize=8,
              color=INK)
A[3].axvline(0.5, color=MUTED, ls=':', lw=1.1)
A[3].set_xlim(0, 0.88)
A[3].set_xlabel('PR-AUC', fontsize=9, color=INK)
tidy(A[3], 'd')

# e -- ROC by epitope length
lens = ens.epitope.astype(str).str.len()
groups = sorted(lens.unique())
data = [v[lens.values == L] for L in groups]
keep = [k for k, g in enumerate(data) if len(g) >= 3]
parts = A[4].violinplot([data[k] for k in keep], positions=range(len(keep)),
                        showextrema=False, widths=0.8)
for b in parts['bodies']:
    b.set_facecolor(STEP_C); b.set_alpha(0.35); b.set_edgecolor('none')
for i2, k in enumerate(keep):
    A[4].scatter(np.full(len(data[k]), i2) + np.random.default_rng(0).normal(
        0, 0.05, len(data[k])), data[k], s=7, color=STEP_C, alpha=0.6,
        edgecolor='none')
    A[4].plot([i2 - 0.18, i2 + 0.18], [np.median(data[k])] * 2, color=DAISY_C, lw=2)
A[4].set_xticks(range(len(keep)))
# the per-group n goes above the violin, not into the tick label: stacking it
# under the axis made the labels of adjacent groups overlap.
A[4].set_xticklabels([str(groups[k]) for k in keep], fontsize=8.5)
for i2, k in enumerate(keep):
    A[4].text(i2, 1.045, f'n={len(data[k])}', ha='center', va='bottom',
              fontsize=7, color=MUTED)
A[4].set_ylim(-0.05, 1.14)
A[4].axhline(0.5, color=MUTED, lw=1.1)
A[4].set_xlabel('Epitope length (aa)', fontsize=9, color=INK)
A[4].set_ylabel('Per-epitope ROC-AUC', fontsize=9, color=INK)
tidy(A[4], 'e')

# f -- cancer, full and overlap-free
rows = []
for tag, f in (('Melanoma', 'data/cancer_specific/melanoma_test.tsv'),
               ('Lung', 'data/cancer_specific/lung_test.tsv')):
    dd = pd.read_csv(BASE / f, sep='\t')
    yy = dd.label.values.astype(int)
    free = np.array([(c, e) not in tpairs for c, e in
                     zip(dd.cdr3.astype(str), dd['antigen.epitope'].astype(str))])
    full = [roc_auc_score(yy, zc[f'{tag.lower()}_corrected_s{s}']) for s in (42, 43, 44)]
    clean = [roc_auc_score(yy[free], zc[f'{tag.lower()}_corrected_s{s}'][free])
             for s in (42, 43, 44)]
    rows.append((tag, np.mean(full), np.std(full, ddof=1),
                 np.mean(clean), np.std(clean, ddof=1), int(free.sum())))
x = np.arange(2)
A[5].bar(x - 0.19, [r[1] for r in rows], 0.36, yerr=[r[2] for r in rows],
         color=STEP_C, capsize=4, label='full set')
A[5].bar(x + 0.19, [r[3] for r in rows], 0.36, yerr=[r[4] for r in rows],
         color=STEP_C, alpha=0.4, hatch='///', edgecolor='white',
         capsize=4, label='overlap-free')
A[5].axhline(0.5, color=MUTED, lw=1.1)
A[5].set_xticks(x)
A[5].set_xticklabels([f'{r[0]}\n(clean n={r[5]})' for r in rows], fontsize=8.5)
A[5].set_ylabel('ROC-AUC', fontsize=9, color=INK)
A[5].set_ylim(0, 1.0)
A[5].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper right')
tidy(A[5], 'f')

# g -- melanoma per-epitope
mel = pd.read_csv(BASE / 'data/cancer_specific/melanoma_test.tsv', sep='\t')
ym = mel.label.values.astype(int)
me = mel['antigen.epitope'].astype(str).values
pm = np.mean([zc[f'melanoma_corrected_s{s}'] for s in (42, 43, 44)], axis=0)
g = []
for e in sorted(set(me)):
    ix = np.where(me == e)[0]
    if len(ix) >= 5 and len(np.unique(ym[ix])) == 2:
        g.append((e, roc_auc_score(ym[ix], pm[ix]), len(ix)))
g.sort(key=lambda r: -r[1])
g = g[:12]
A[6].barh(range(len(g)), [r[1] for r in g], height=0.62, color=STEP_C)
A[6].set_yticks(range(len(g)))
A[6].set_yticklabels([f'{r[0]} ({r[2]})' for r in g], fontsize=7.2, color=INK)
A[6].invert_yaxis()
A[6].axvline(0.5, color=MUTED, lw=1.1)
A[6].set_xlabel('Per-epitope ROC-AUC (melanoma)', fontsize=9, color=INK)
tidy(A[6], 'g')

# h -- cancer scatter
for tag, mk, c in (('melanoma', 'o', STEP_C), ('lung', 's', DAISY_C)):
    dd = pd.read_csv(BASE / ('data/cancer_specific/%s_test.tsv' % tag), sep='\t')
    yy = dd.label.values.astype(int)
    ee = dd['antigen.epitope'].astype(str).values
    pp = np.mean([zc[f'{tag}_corrected_s{s}'] for s in (42, 43, 44)], axis=0)
    xs, ys = [], []
    for e in sorted(set(ee)):
        ix = np.where(ee == e)[0]
        if len(ix) >= 5 and len(np.unique(yy[ix])) == 2:
            xs.append(len(ix)); ys.append(roc_auc_score(yy[ix], pp[ix]))
    A[7].scatter(xs, ys, s=28, marker=mk, color=c, alpha=0.7, edgecolor='none',
                 label=tag)
A[7].set_xscale('log')
A[7].axhline(0.5, color=MUTED, lw=1.1)
A[7].set_xlabel('Samples per epitope', fontsize=9, color=INK)
A[7].set_ylabel('Per-epitope ROC-AUC', fontsize=9, color=INK)
A[7].legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right')
tidy(A[7], 'h')

fig.tight_layout(w_pad=2.4, h_pad=3.0)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_3_rebuilt_v2.{ext}', bbox_inches='tight')
print('wrote', OUT / 'Figure_3_rebuilt_v2.png')
print('f:', [(r[0], round(r[1], 4), round(r[3], 4), r[5]) for r in rows])
print('g: top melanoma epitope', g[0][0], round(g[0][1], 3), '| n panels drawn: 8 of 9')
