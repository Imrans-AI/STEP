"""
Attention-map panels for 3GSN (Fig 5a) and for 6RSY / 1OGA (Fig S2 a,b),
all on each structure's own CDR3b, with true Ca-Ca contacts outlined.
Same form as the rebuilt Fig 5c.
"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'
INK, MUTED, GRID, HOT = '#1A1A1A', '#6B7280', '#9AA7B1', '#C2410C'
AA3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY',
       'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
       'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
       'W': 'TRP', 'Y': 'TYR'}
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'savefig.dpi': 600, 'figure.facecolor': 'white'})

meta = json.load(open(OUT / 'cams_structure_cdr3.json'))
contacts = pickle.load(open(BASE / 'data/raw/stcrdab_cdr3_contacts.pkl', 'rb'))
by_pdb = {}
for e in contacts:
    by_pdb.setdefault(str(e['pdb']).lower(), []).append(e)


def panel(ax, pdb, letter):
    m = meta[pdb]
    cdr3, epi, nums = m['cdr3'], m['epitope'], m['resnums']
    cam = np.load(OUT / f'{pdb}_structure_cam_v2.npy')[:len(cdr3), :len(epi)]
    dm = np.array(by_pdb[pdb.lower()][0]['cdr3_epitope_distance'], dtype=float)
    cc = dm < 8.0
    im = ax.imshow(cam, cmap='Blues', vmin=0, vmax=cam.max(), aspect='auto')
    for i in range(len(cdr3)):
        for j in range(len(epi)):
            if cc[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=HOT, lw=1.3, zorder=3))
    eo = np.argsort(-cam.sum(0))[:3]
    ro = np.argsort(-cam.sum(1))[:3]
    ax.set_xticks(range(len(epi)))
    ax.set_xticklabels([f'{AA3[a]}\n{k+1}' for k, a in enumerate(epi)], fontsize=7)
    ax.set_yticks(range(len(cdr3)))
    ax.set_yticklabels([f'{AA3[a]}-{nums[k]}' for k, a in enumerate(cdr3)],
                       fontsize=7)
    for k in eo:
        ax.get_xticklabels()[k].set_color(HOT)
        ax.get_xticklabels()[k].set_fontweight('bold')
    for k in ro:
        ax.get_yticklabels()[k].set_color(HOT)
        ax.get_yticklabels()[k].set_fontweight('bold')
    ax.set_xlabel(f'Epitope position ({epi})', fontsize=9, color=INK)
    ax.set_ylabel(f'CDR3β position ({pdb} numbering)', fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.text(-0.30, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', color=INK)
    ax.set_title(f'{pdb} · P(bind) {m["p_mean"]:.3f} '
                 f'± {m["p_sd"]:.3f}', fontsize=8.5, color=MUTED, pad=6)
    return im


# ---- Fig 5a, on its own, to be composed into Figure 5
fig, ax = plt.subplots(figsize=(5.0, 5.4))
im = panel(ax, '3GSN', 'a')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Attention (max-normalised)', fontsize=8.5, color=INK)
cb.ax.tick_params(labelsize=8, colors=INK); cb.outline.set_edgecolor(GRID)
ax.legend([Rectangle((0, 0), 1, 1, fill=False, edgecolor=HOT, lw=1.3)],
          ['Cα–Cα < 8 Å'], loc='upper center',
          bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=8.5)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_5a_rebuilt_3GSN.{ext}', bbox_inches='tight')
plt.close(fig)
print('wrote Figure_5a_rebuilt_3GSN')

# ---- Fig S2, two panels
fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))
for ax, pdb, letter in zip(axes, ('6RSY', '1OGA'), 'ab'):
    im = panel(ax, pdb, letter)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label('Attention (max-normalised)', fontsize=8.5, color=INK)
cb.ax.tick_params(labelsize=8, colors=INK); cb.outline.set_edgecolor(GRID)
fig.legend([Rectangle((0, 0), 1, 1, fill=False, edgecolor=HOT, lw=1.3)],
           ['Cα–Cα < 8 Å in the crystal structure'],
           loc='lower center', bbox_to_anchor=(0.46, -0.04), frameon=False,
           fontsize=8.5)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S2_rebuilt.{ext}', bbox_inches='tight')
print('wrote Figure_S2_rebuilt')
for p in ('3GSN', '6RSY', '1OGA'):
    m = meta[p]
    print(f'  {p}: epitope {m["epitope_top3"]}  CDR3b {m["cdr3_top3"]}  '
          f'contacts {len(m["epitope_with_contact"])}/3 and '
          f'{len(m["cdr3_with_contact"])}/3')


# ---- Fig S2 as two separate single-panel files, because the supplementary
# holds panels a and b as two distinct images in one paragraph.
for pdb, letter in (('6RSY', 'a'), ('1OGA', 'b')):
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    im = panel(ax, pdb, letter)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('Attention (max-normalised)', fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK); cb.outline.set_edgecolor(GRID)
    ax.legend([Rectangle((0, 0), 1, 1, fill=False, edgecolor=HOT, lw=1.3)],
              ['C\u03b1\u2013C\u03b1 < 8 \u00c5'], loc='upper center',
              bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=8.5)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'Figure_S2{letter}_rebuilt_{pdb}.{ext}',
                    bbox_inches='tight')
    plt.close(fig)
    print(f'wrote Figure_S2{letter}_rebuilt_{pdb}')
