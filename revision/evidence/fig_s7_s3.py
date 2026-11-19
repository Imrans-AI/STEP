"""
Figure S7 (SHAP) and Figure S3 (per-epitope distribution), regenerated.

S7: the channel labels in FINALIZE_SHAP_ANALYSIS.py line 31 are
['Charge','Hydrophobicity','Polarity','Size','Aromaticity'] - wrong names AND a
wrong order against the model's channel axis, which is hydrophobicity, charge,
flexibility, refractivity, area_loss. The stored SHAP array's per-channel means
reproduce results/shap_physchem_summary_CORRECTED.csv exactly in model order, so
only the labels were ever wrong; the values are reused unchanged.

S3: rebuilt on the v2 criterion - 342 epitopes with at least five samples AND
both classes present.

Read-only on results/. Writes under revision/evidence/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'
INK, MUTED, GRID, ACCENT = '#1A1A1A', '#6B7280', '#9AA7B1', '#2C6E8F'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'savefig.dpi': 600, 'savefig.bbox': 'tight',
                     'figure.facecolor': 'white'})

# ------------------------------------------------------------------ Fig S7
CH = ['hydrophobicity', 'charge', 'flexibility', 'refractivity', 'area loss']
WRONG = ['Charge', 'Hydrophobicity', 'Polarity', 'Size', 'Aromaticity']
shap = np.load(BASE / 'results/shap_values_physchem.npy')      # (400,5,20,15)
assert shap.shape[1] == 5
mean_abs = np.abs(shap).mean(axis=(0, 2, 3))
ref = pd.read_csv(BASE / 'results/shap_physchem_summary_CORRECTED.csv')
# the CSV is sorted by magnitude, the array axis is in MODEL order, so the
# comparison has to go through the names
MODEL_ORDER = ['hydrophobicity', 'charge', 'flexibility', 'refractivity',
               'area_loss']
ref_by_name = dict(zip(ref.Channel, ref.Mean_Abs_SHAP))
assert np.allclose(mean_abs, [ref_by_name[c] for c in MODEL_ORDER], atol=1e-6), \
    'stored array does not match the corrected summary'
print('channel means reproduce shap_physchem_summary_CORRECTED.csv')

order = np.argsort(-mean_abs)
fig = plt.figure(figsize=(11.5, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.25], wspace=0.28)

ax = fig.add_subplot(gs[0])
ypos = np.arange(5)[::-1]
ax.barh(ypos, mean_abs[order], height=0.58, color=ACCENT, edgecolor='none')
for yy, v in zip(ypos, mean_abs[order]):
    ax.text(v + mean_abs.max() * 0.025, yy, f'{v:.5f}', va='center',
            fontsize=8.5, color=INK)
ax.set_yticks(ypos)
ax.set_yticklabels([CH[k] for k in order], fontsize=9.5, color=INK)
ax.set_xlabel('mean |SHAP|', fontsize=9.5, color=INK)
ax.set_xlim(0, mean_abs.max() * 1.33)
ax.grid(axis='x', color=GRID, alpha=0.3, lw=0.6)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUTED)
ax.tick_params(labelsize=9, colors=INK, length=0)
ax.text(-0.30, 1.08, 'a', transform=ax.transAxes, fontsize=15,
        fontweight='bold', va='top', color=INK)

inner = gs[1].subgridspec(1, 5, wspace=0.18)
pos = np.abs(shap).mean(axis=0)                 # (5,20,15)
vmax = pos.max()
for k, ch in enumerate(order):
    a2 = fig.add_subplot(inner[k])
    im = a2.imshow(pos[ch], cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
    a2.set_title(CH[ch], fontsize=8.5, color=INK, pad=4)
    a2.set_xticks([0, 7, 14])
    a2.set_xticklabels(['1', '8', '15'], fontsize=7.5, color=INK)
    if k == 0:
        a2.set_yticks([0, 9, 19])
        a2.set_yticklabels(['1', '10', '20'], fontsize=7.5, color=INK)
        a2.set_ylabel('CDR3β position', fontsize=8.5, color=INK)
    else:
        a2.set_yticks([])
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_color(GRID)
    if k == 2:
        a2.set_xlabel('Epitope position', fontsize=8.5, color=INK)
    if k == 0:
        a2.text(-0.52, 1.20, 'b', transform=a2.transAxes, fontsize=15,
                fontweight='bold', va='top', color=INK)
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.022, pad=0.015)
cb.set_label('mean |SHAP| at position', fontsize=8, color=INK)
cb.ax.tick_params(labelsize=7.5, colors=INK)
cb.outline.set_edgecolor(GRID)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S7_SHAP_corrected_labels.{ext}', bbox_inches='tight')
plt.close(fig)
print('Fig S7 written; label order', [CH[k] for k in order])
print('  the labels it replaces were', WRONG)

# ------------------------------------------------------------------ Fig S3
pe = pd.read_csv(OUT.parent / 'numbers_pass/per_epitope_v2_corrected_values.csv')
ens = pe[pe.run == 'ens']
assert len(ens) == 342, len(ens)
v = ens.roc.values
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

ax = axes[0]
ax.hist(v, bins=np.linspace(0, 1, 26), color=ACCENT, edgecolor='white', lw=0.6)
ax.axvline(v.mean(), color='#C2410C', lw=1.8, ls='--',
           label=f'mean {v.mean():.3f}')
ax.axvline(np.median(v), color='#C2410C', lw=1.8, ls=':',
           label=f'median {np.median(v):.3f}')
ax.axvline(0.5, color=MUTED, lw=1.2, label='chance')
ax.set_xlabel('Per-epitope ROC-AUC', fontsize=9.5, color=INK)
ax.set_ylabel('Epitopes', fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='upper left')
ax.text(-0.14, 1.07, 'a', transform=ax.transAxes, fontsize=15,
        fontweight='bold', va='top', color=INK)

ax = axes[1]
ax.scatter(ens.n.values, v, s=18, color=ACCENT, alpha=0.6, edgecolor='none')
ax.axhline(v.mean(), color='#C2410C', lw=1.6, ls='--')
ax.axhline(0.5, color=MUTED, lw=1.2)
ax.set_xlabel('Balanced test samples for the epitope', fontsize=9.5, color=INK)
ax.set_ylabel('Per-epitope ROC-AUC', fontsize=9.5, color=INK)
ax.text(-0.14, 1.07, 'b', transform=ax.transAxes, fontsize=15,
        fontweight='bold', va='top', color=INK)

for ax in axes:
    ax.grid(color=GRID, alpha=0.28, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=9, colors=INK)
fig.text(0.5, -0.06, f'{len(v)} epitopes with at least five balanced test samples '
         'and both classes present; three-seed ensemble.', ha='center',
         fontsize=8.5, color=MUTED)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S3_per_epitope_v2.{ext}', bbox_inches='tight')
print(f'Fig S3 written; n={len(v)} mean {v.mean():.4f} median {np.median(v):.4f} '
      f'at-or-below chance {100*(v <= 0.5).mean():.1f}%')
