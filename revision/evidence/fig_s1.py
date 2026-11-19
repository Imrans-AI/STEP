"""
Figure S1, dataset flow, rebuilt.

Three defects in the submitted version, all hard-coded in BUILD_DATASET_FLOW.py:
the pooled positives read 99,323 over 1,181 epitopes, where the split files hold
99,643 over 1,281 (R2-C1); there is no validation block at all, which is why the
arithmetic appeared to need 19 phantom exclusions; and the counts are the
pre-repair split. Both stages are drawn here, so the corrected pool and the
repaired split that the Methods now describe are both visible.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path('/home/muhammad-imran/PhD_data/TCR_Module_1/revision/evidence/figures')
INK, MUTED, GRID = '#1A1A1A', '#6B7280', '#9AA7B1'
C_RAW, C_PROC, C_POS = '#E6EEF4', '#FBEDE3', '#E4EFE6'
C_TRAIN, C_VAL, C_TEST = '#CFE3D4', '#EBDCE6', '#D3E0EF'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'savefig.dpi': 600, 'figure.facecolor': 'white'})

fig, ax = plt.subplots(figsize=(11.2, 9.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')


def box(cx, cy, w, h, lines, fc, fs=9):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc=fc, ec=GRID, lw=0.9, zorder=2))
    n = len(lines)
    for k, (txt, bold) in enumerate(lines):
        ax.text(cx, cy + h / 2 - h * (k + 0.75) / (n + 0.2), txt, ha='center',
                va='center', fontsize=fs + (0.8 if bold else 0),
                fontweight='bold' if bold else 'normal', color=INK, zorder=3)


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=13, lw=1.1, color=MUTED,
                                 zorder=1, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, label, ha='left',
                va='center', fontsize=7.8, color=MUTED, style='italic')


# Four sources, not three: UniPMT was read by data/raw/integrate_unipmt.py into the
# multisource file the splits are built from, and was not named in the submitted figure.
for cx, name, n in ((13.5, 'VDJdb', '139,744 entries'),
                    (37.8, 'McPAS-TCR', '40,731 entries'),
                    (62.2, 'IEDB', '2,742 entries'),
                    (86.5, 'UniPMT', '8,256 positive pairs')):
    box(cx, 94, 22, 8, [(name, True), (n, False)], C_RAW, fs=8.0)
    arrow(cx, 90, 50, 86.5)

box(50, 82, 70, 8,
    [('Cross-database merge → filtering → deduplication', True),
     ('human, MHC-I; CDR3β 8–20 aa; epitopes 8–15 aa', False)],
    C_PROC, fs=8.5)
arrow(50, 78, 50, 73.5)

box(50, 68.5, 68, 10.5,
    [('99,643 unique positive pairs', True),
     ('1,281 epitopes  ·  93,654 unique CDR3β', False),
     ('by source: VDJdb 82,617  ·  UniPMT 8,256  ·  McPAS 6,585  ·  IEDB 2,185', False),
     ('(submitted figure: 99,323 over 1,181 epitopes, three sources)', False)],
    C_POS, fs=8.5)
arrow(50, 63.2, 50, 60)

box(50, 55.5, 62, 8,
    [('Whitespace repair and normalisation', True),
     ('4 contaminated spellings merged → 1,276 epitopes', False)],
    C_PROC, fs=8.5)
arrow(50, 51.5, 50, 47)

box(50, 42.5, 56, 7.5, [('Epitope-disjoint split', True),
                        ('600  |  100  |  576 epitopes', False)], C_PROC, fs=8.5)
for x in (17, 50, 83):
    arrow(50, 38.8, x, 34.5)

box(17, 28, 30, 11, [('TRAINING', True), ('130,067 samples', False),
                     ('95,571 pos + 34,496 neg', False), ('600 epitopes', False)],
    C_TRAIN, fs=8)
box(50, 28, 30, 11, [('VALIDATION', True), ('5,227 samples', False),
                     ('308 pos (5.9%)', False), ('100 epitopes', False)],
    C_VAL, fs=8)
box(83, 28, 30, 11, [('TEST', True), ('29,703 samples', False),
                     ('1,657 pos (5.6%)', False), ('576 epitopes', False)],
    C_TEST, fs=8)
ax.text(50, 20.6, 'checkpoint selection, learning-rate schedule and '
        'threshold fitting', ha='center', fontsize=7.6, color=MUTED,
        style='italic')
arrow(83, 22.4, 83, 18)
box(83, 13, 32, 8, [('Balanced test (primary)', True),
                    ('3,314 samples · 1,657/1,657 · 576 epi', False)],
    C_TEST, fs=8)

ax.add_patch(FancyBboxPatch((5, 1.5), 52, 9,
                            boxstyle='round,pad=0.3,rounding_size=1.0', fc='none',
                            ec=GRID, lw=0.9, linestyle=(0, (4, 3)), zorder=0))
ax.text(7, 9.2, 'Cancer-specific held-out evaluation', ha='left', fontsize=7.8,
        style='italic', color=MUTED)
box(19, 5.0, 24, 5.5, [('Melanoma 1,582', False)], C_TEST, fs=8)
box(45, 5.0, 24, 5.5, [('Lung 170', False)], C_TEST, fs=8)

for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S1_dataset_flow_v2.{ext}', dpi=600,
                bbox_inches='tight')
print('wrote', OUT / 'Figure_S1_dataset_flow_v2.png')
