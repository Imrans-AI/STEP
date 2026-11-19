"""
Numbers pass, part C: regenerate Fig 4c,d from the five retrained prior
configurations, three seeds each, mean +/- SD.

Form. The panel's job is a magnitude comparison across five configurations whose
headline is that they do not differ. A bar chart with error bars would hide n,
so each configuration shows its three seed values as open dots beside a mean
with an SD bar: the reader sees the sample, not an assertion about it. One
measure per panel, so one hue; the corrected prior is the entity of interest and
carries the accent, the four controls are neutral - colour follows the entity,
not its rank, and the controls keep their colour whatever their order.

No model is loaded. Reads results/matrix_v2/per_seed_metrics.csv only; writes
only under revision/evidence/numbers_pass/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/numbers_pass'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'axes.linewidth': 0.8, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

ACCENT, NEUTRAL, INK, MUTED = '#2C6E8F', '#9AA7B1', '#1A1A1A', '#6B7280'
ORDER = ['corrected', 'none', 'uniform', 'shuffled', 'length_only']
LABEL = {'corrected': 'Corrected\nprior', 'none': 'No\nprior', 'uniform': 'Uniform\nprior',
         'shuffled': 'Shuffled\nprior', 'length_only': 'Length-only\nmask'}

df = pd.read_csv(BASE / 'results/matrix_v2/per_seed_metrics.csv')
assert set(df.prior) == set(ORDER) and len(df) == 15, 'expected 5 configs x 3 seeds'

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
for ax, (col, name, panel) in zip(axes, [('roc_auc', 'ROC-AUC', 'c'),
                                         ('pr_auc', 'PR-AUC', 'd')]):
    means = [df[df.prior == p][col].mean() for p in ORDER]
    sds = [df[df.prior == p][col].std(ddof=1) for p in ORDER]
    x = np.arange(len(ORDER))

    # the band of +/-1 SD around the grand mean: the "within seed variance" claim,
    # drawn rather than asserted
    gm = float(np.mean(means))
    pooled = float(np.sqrt(np.mean(np.square(sds))))
    ax.axhspan(gm - pooled, gm + pooled, color=NEUTRAL, alpha=0.18, zorder=0,
               lw=0)
    ax.axhline(gm, color=MUTED, lw=0.8, ls=':', zorder=1)

    for i, p in enumerate(ORDER):
        c = ACCENT if p == 'corrected' else NEUTRAL
        vals = df[df.prior == p][col].values
        ax.errorbar(i, means[i], yerr=sds[i], fmt='none', ecolor=c,
                    elinewidth=2.0, capsize=5, capthick=2.0, zorder=3)
        ax.plot(i, means[i], 'o', ms=9, color=c, mec='white', mew=1.2, zorder=4)
        ax.plot(np.full_like(vals, i, dtype=float) + 0.17, vals, 'o', ms=5,
                mfc='white', mec=c, mew=1.3, zorder=4, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in ORDER], fontsize=8.5, color=INK)
    ax.set_ylabel(name, fontsize=10, color=INK)
    ax.set_xlim(-0.55, len(ORDER) - 0.25)
    lo = min(df[col]) - 0.012
    hi = max(df[col]) + 0.020   # headroom for the corner label
    ax.set_ylim(lo, hi)
    ax.tick_params(axis='y', labelsize=9, colors=INK, width=0.8)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color=NEUTRAL, alpha=0.30, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left', color=INK)
    # one selective label, naming the entity of interest. It sits in the
    # corner rather than beside the point: an offset annotation collided with
    # the grand-mean line in panel c and with the band edge in panel d.
    ax.text(0.015, 0.975, f'Corrected prior {means[0]:.3f} \u00b1 {sds[0]:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            color=ACCENT)

h = [plt.Line2D([], [], marker='o', ls='none', ms=8, color=MUTED, mec='white',
                mew=1.2, label='mean of 3 seeds (bar = ± 1 SD)'),
     plt.Line2D([], [], marker='o', ls='none', ms=5, mfc='white', mec=MUTED,
                mew=1.3, label='individual seed'),
     plt.Line2D([], [], color=NEUTRAL, lw=6, alpha=0.3,
                label='grand mean ± pooled within-configuration SD')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.08), labelcolor=INK)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_4cd_retrained_ablation.{ext}', dpi=600,
                bbox_inches='tight')
print('wrote', OUT / 'Figure_4cd_retrained_ablation.png')

rows = []
for p in ORDER:
    g = df[df.prior == p]
    rows.append(dict(panel_order=ORDER.index(p) + 1, config=p,
                     roc_mean=g.roc_auc.mean(), roc_sd=g.roc_auc.std(ddof=1),
                     pr_mean=g.pr_auc.mean(), pr_sd=g.pr_auc.std(ddof=1),
                     roc_seeds='; '.join(f'{v:.4f}' for v in g.roc_auc),
                     pr_seeds='; '.join(f'{v:.4f}' for v in g.pr_auc)))
pd.DataFrame(rows).to_csv(OUT / 'fig4cd_panel_values.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
