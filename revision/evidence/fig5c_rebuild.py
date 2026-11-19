"""
Fig 5 panel c, rebuilt from the v2 CAM for 4QOK / EAAGIGILTV.

Form: a matrix of one magnitude over two positional axes, so a heat map on a
single-hue sequential ramp, light to dark - never a rainbow. True sub-8 A
contacts are outlined rather than colour-coded, so geometry and attention are
two separate encodings and can disagree visibly. The three top-attention
positions on each axis are the only direct labels.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'
CDR3, EPI, OFFSET = 'CAWSETGLGTGELFF', 'EAAGIGILTV', 90
AA3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
       'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
       'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
       'W': 'TRP', 'Y': 'TYR'}
INK, MUTED, GRID = '#1A1A1A', '#6B7280', '#9AA7B1'

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'savefig.dpi': 600, 'savefig.bbox': 'tight',
                     'figure.facecolor': 'white'})

cam = np.load(OUT / 'EAAGIGILTV_4QOK_cam_v2.npy')[:len(CDR3), :len(EPI)]
dist = np.load(BASE / 'results/fig6_scorecam/EAAGIGILTV_4QOK_actual_dist.npy'
               )[:len(CDR3), :len(EPI)]
contact = dist < 8.0
etop = np.argsort(-cam.sum(0))[:3]
rtop = np.argsort(-cam.sum(1))[:3]

fig, ax = plt.subplots(figsize=(5.4, 5.8))
im = ax.imshow(cam, cmap='Blues', vmin=0, vmax=cam.max(), aspect='auto')
for i in range(len(CDR3)):
    for j in range(len(EPI)):
        if contact[i, j]:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor='#C2410C', lw=1.4, zorder=3))

ax.set_xticks(range(len(EPI)))
ax.set_xticklabels([f'{AA3[a]}\n{k+1}' for k, a in enumerate(EPI)], fontsize=7.5,
                   color=INK)
ax.set_yticks(range(len(CDR3)))
ax.set_yticklabels([f'{AA3[a]}-{k+1+OFFSET}' for k, a in enumerate(CDR3)],
                   fontsize=7.5, color=INK)
for k in etop:
    ax.get_xticklabels()[k].set_color('#C2410C')
    ax.get_xticklabels()[k].set_fontweight('bold')
for k in rtop:
    ax.get_yticklabels()[k].set_color('#C2410C')
    ax.get_yticklabels()[k].set_fontweight('bold')
ax.set_xlabel('Epitope position (EAAGIGILTV)', fontsize=9.5, color=INK)
ax.set_ylabel('CDR3β position (4QOK numbering)', fontsize=9.5, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_color(GRID)
ax.text(-0.26, 1.045, 'c', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left', color=INK)

cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Attention (max-normalised)', fontsize=8.5, color=INK)
cb.ax.tick_params(labelsize=8, colors=INK)
cb.outline.set_edgecolor(GRID)

h = [Rectangle((0, 0), 1, 1, fill=False, edgecolor='#C2410C', lw=1.4)]
ax.legend(h, ['Cα–Cα < 8 Å in 4QOK'], loc='upper center',
          bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5,
          labelcolor=INK)

for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_5c_rebuilt_4QOK.{ext}', bbox_inches='tight')

print('epitope top 3 :', ', '.join(f'{AA3[EPI[j]]}-{j+1}' for j in etop))
print('CDR3b  top 3 :', ', '.join(f'{AA3[CDR3[i]]}-{i+1+OFFSET}' for i in rtop))
print('top-3 epitope positions with a true <8 A contact:',
      [f'{AA3[EPI[j]]}-{j+1}' for j in etop if contact[:, j].any()])
print('top-3 CDR3b positions with a true <8 A contact:',
      [f'{AA3[CDR3[i]]}-{i+1+OFFSET}' for i in rtop if contact[i].any()])
print('contacts outlined:', int(contact.sum()), 'of', contact.size)
print('wrote', OUT / 'Figure_5c_rebuilt_4QOK.png')
