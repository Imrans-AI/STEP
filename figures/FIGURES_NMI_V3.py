#!/usr/bin/env python3
"""
FIGURES_NMI_V3.py — FINAL
Fig 3: (b) scatter ROC vs n_samples 225 epi, (e) overall STEP ROC curve
Fig 4: (e) Cleveland half-width, (f) NEW split violin binding vs non-binding
Run from ~/PhD_data/New_TCR/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import mannwhitneyu
import os

os.makedirs('results/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

NMI = {
    'ochre':'#D4A574','steel':'#7B9DBF','coral':'#CC7B7B','sage':'#8FB58F',
    'lavender':'#A68BB5','sand':'#C9B88C','slate':'#8A9AA5','rose':'#C98D8D',
    'teal':'#7BB5AD','amber':'#D4A24E',
    'bar_edge':'0.25','grid':'0.85','baseline':'#999999','text_dark':'0.15',
}
nmi_cmap = LinearSegmentedColormap.from_list(
    'nmi_heat', ['#FFFFFF','#FDDBC7','#F4A582','#D6604D','#B2182B','#67001F'])

def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', color=NMI['text_dark'])

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Load data ──────────────────────────────────────────────────────────────────
per_epi = pd.read_csv('results/per_epitope_performance_balanced.csv')
sub5    = per_epi[per_epi['N_Samples'] >= 5].copy()
preds   = pd.read_csv('results/STEP_predictions_balanced.csv')

# per-epitope ROC recomputed from canonical balanced predictions (>=5 samples, both classes) -> 340 epitopes
_CLIN = {'NCVPMVATV':'CMV','LLVPMVATV':'CMV','VLVPMVATV':'CMV','NMVPMVATV':'CMV','QASGNHAAGILTM':'EBV','FLKEQGGL':'HIV'}
_rows = []
for _ep, _g in preds.groupby('antigen.epitope'):
    if _g['label'].nunique() < 2:
        continue
    _rows.append({'Epitope': _ep, 'ROC_AUC': roc_auc_score(_g['label'], _g['prediction']), 'N_Samples': len(_g), 'Virus': _CLIN.get(_ep, 'Other')})
per_epi = pd.DataFrame(_rows)
sub5    = per_epi[per_epi['N_Samples'] >= 5].copy()
cancer  = pd.read_csv('results/cancer_validation_with_CI.csv')
clean   = pd.read_csv('results/cancer_validation_CLEAN_with_CI.csv')

CLINICAL = sub5[sub5['Virus'].isin(['CMV','EBV','HIV'])]['Epitope'].tolist()
clin_df  = sub5[sub5['Epitope'].isin(CLINICAL)]
nonclin  = sub5[~sub5['Epitope'].isin(CLINICAL)]
mean_auc   = sub5['ROC_AUC'].mean()
median_auc = sub5['ROC_AUC'].median()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  (6 panels: histogram, scatter-landscape, bar+scatter,
#            cumulative line, overall ROC, violin)
# ══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(14, 14))
gs3  = gridspec.GridSpec(3, 2, figure=fig3,
                         hspace=0.45, wspace=0.40,
                         height_ratios=[1, 1, 1])
ax_a = fig3.add_subplot(gs3[0,0])
ax_b = fig3.add_subplot(gs3[0,1])
ax_c = fig3.add_subplot(gs3[1,0])
ax_d = fig3.add_subplot(gs3[1,1])
ax_e = fig3.add_subplot(gs3[2,0])
ax_f = fig3.add_subplot(gs3[2,1])

for lbl, ax in zip('abcdef', [ax_a,ax_b,ax_c,ax_d,ax_e,ax_f]):
    panel_label(ax, lbl); despine(ax)

# ── a: Histogram ──────────────────────────────────────────────────────────────
bins = np.arange(0, 1.05, 0.05)
hc = []
for l in bins[:-1]:
    m = l+0.025
    if   m<0.5: hc.append(NMI['rose'])
    elif m<0.7: hc.append(NMI['sand'])
    elif m<0.8: hc.append(NMI['sage'])
    elif m<0.9: hc.append(NMI['steel'])
    else:       hc.append(NMI['teal'])
_, _, pa = ax_a.hist(sub5['ROC_AUC'], bins=bins,
                     edgecolor=NMI['bar_edge'], alpha=0.85, linewidth=0.8)
for p, c in zip(pa, hc): p.set_facecolor(c)
ax_a.axvline(mean_auc, color='0.2', linestyle='--', linewidth=1.2,
             label=f'Mean = {mean_auc:.3f}')
ax_a.axvline(median_auc, color='0.5', linestyle=':', linewidth=1.2,
             label=f'Median = {median_auc:.3f}')
ax_a.axvline(0.5, color=NMI['baseline'], linestyle='-', linewidth=0.8, alpha=0.5, label='Random')
ax_a.set_xlabel('ROC-AUC'); ax_a.set_ylabel('Number of epitopes')
ax_a.legend(fontsize=8, loc='upper left', frameon=False)
ax_a.set_xlim(0, 1.05)
# Stats annotation box (fills empty upper-left area)
_n_epi = len(sub5)
_p70 = 100 * (sub5['ROC_AUC'] >= 0.7).mean()
_p80 = 100 * (sub5['ROC_AUC'] >= 0.8).mean()
_p90 = 100 * (sub5['ROC_AUC'] >= 0.9).mean()
_stats_text = (f"$n$ = {_n_epi} epitopes\n"
               f"ROC-AUC ≥ 0.7: {_p70:.0f}%\n"
               f"ROC-AUC ≥ 0.8: {_p80:.0f}%\n"
               f"ROC-AUC ≥ 0.9: {_p90:.0f}%")
ax_a.text(0.02, 0.62, _stats_text, transform=ax_a.transAxes,
          fontsize=8, va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='0.75', linewidth=0.6, alpha=0.95))

# ── b: SCATTER — ROC-AUC vs sample size (ALL 225 epitopes) ───────────────────
virus_colors = {'CMV': NMI['coral'], 'EBV': NMI['amber'],
                'HIV': NMI['teal'], 'Other': NMI['steel']}
virus_markers = {'CMV': 'D', 'EBV': 's', 'HIV': '^', 'Other': 'o'}

for _, row in sub5.iterrows():
    v = row.get('Virus', 'Other')
    if v not in virus_colors: v = 'Other'
    col = virus_colors[v]
    marker = virus_markers[v]
    alpha = 0.9 if v in ['CMV','EBV','HIV'] else 0.5
    size = 60 if v in ['CMV','EBV','HIV'] else 30
    ax_b.scatter(row['N_Samples'], row['ROC_AUC'], s=size, color=col,
                 marker=marker, edgecolors=NMI['bar_edge'], linewidth=0.4,
                 alpha=alpha, zorder=3 if v in ['CMV','EBV','HIV'] else 2)

ax_b.axhline(0.5, color=NMI['baseline'], linestyle='--', linewidth=0.8, alpha=0.5)
ax_b.axhline(mean_auc, color='0.4', linestyle=':', linewidth=0.8, alpha=0.5)
ax_b.text(sub5['N_Samples'].max()*0.85, mean_auc+0.02,
          f'Mean = {mean_auc:.3f}', fontsize=7.5, color='0.4', fontstyle='italic')

ax_b.set_xlabel('Number of test samples')
ax_b.set_ylabel('ROC-AUC')
ax_b.set_ylim(-0.05, 1.08)
ax_b.grid(alpha=0.12, linestyle='--', linewidth=0.5)
ax_b.set_axisbelow(True)

ax_b.legend(handles=[
    Line2D([0],[0],marker='D',color='w',markerfacecolor=NMI['coral'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='CMV'),
    Line2D([0],[0],marker='s',color='w',markerfacecolor=NMI['amber'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='EBV'),
    Line2D([0],[0],marker='^',color='w',markerfacecolor=NMI['teal'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='HIV'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['steel'],
           markersize=6,markeredgecolor=NMI['bar_edge'],alpha=0.5,label='Other')],
    fontsize=7, loc='lower right', frameon=False)

# ── c: Clinical vs non-clinical ───────────────────────────────────────────────
_, pval = mannwhitneyu(clin_df['ROC_AUC'].values, nonclin['ROC_AUC'].values,
                       alternative='greater')
bm = [nonclin['ROC_AUC'].mean(), clin_df['ROC_AUC'].mean()]
bc = [NMI['steel'], NMI['coral']]
positions = [0.35, 0.65]
bars_c = ax_c.bar(positions, bm, color=bc, edgecolor=NMI['bar_edge'],
                  linewidth=0.8, width=0.22)
for pos, data, col in zip(positions,
                          [nonclin['ROC_AUC'].values, clin_df['ROC_AUC'].values], bc):
    x = np.random.RandomState(42).normal(pos, 0.03, size=len(data))
    ax_c.scatter(x, data, alpha=0.4, color=col, s=18,
                 edgecolor=NMI['bar_edge'], linewidth=0.3, zorder=3)
for bar, val in zip(bars_c, bm):
    ax_c.text(bar.get_x()+bar.get_width()/2, val+0.035,
              f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')
yt = 1.10
ax_c.annotate('', xy=(positions[1],yt), xytext=(positions[0],yt),
              arrowprops=dict(arrowstyle='-', linewidth=1, color='0.3'))
ax_c.text(np.mean(positions), yt+0.02, f'P = {pval:.3f}',
          ha='center', fontsize=9, fontweight='bold')
ax_c.set_xticks(positions)
ax_c.set_xticklabels([f'Non-clinical\n(n={len(nonclin)})',
                      f'Clinical viral\n(n={len(clin_df)})'], fontsize=9)
ax_c.set_ylabel('ROC-AUC')
ax_c.set_ylim(0, 1.22); ax_c.set_xlim(0.05, 0.95)
ax_c.axhline(0.5, color=NMI['baseline'], linestyle='--', linewidth=0.8, alpha=0.5)

# ── d: Cumulative line ────────────────────────────────────────────────────────
sorted_roc = np.sort(sub5['ROC_AUC'].values)[::-1]
pcts_cum = np.arange(1, len(sorted_roc)+1) / len(sorted_roc) * 100

ax_d.fill_between(sorted_roc, pcts_cum, alpha=0.12, color=NMI['steel'])
ax_d.plot(sorted_roc, pcts_cum, color=NMI['steel'], linewidth=2)

for thr in [0.5, 0.7, 0.8, 0.9, 0.95]:
    pct_at = (sub5['ROC_AUC'] >= thr).mean() * 100
    cnt = (sub5['ROC_AUC'] >= thr).sum()
    ax_d.scatter(thr, pct_at, color=NMI['coral'], s=50,
                 edgecolors=NMI['bar_edge'], linewidth=0.8, zorder=5)
    ha = 'right' if thr >= 0.9 else 'left'
    off = (-8, 6) if thr >= 0.9 else (6, 6)
    ax_d.annotate(f'{pct_at:.0f}% ({cnt})', xy=(thr, pct_at),
                  xytext=off, textcoords='offset points',
                  fontsize=7.5, fontweight='bold', ha=ha)

ax_d.set_xlabel('ROC-AUC threshold')
ax_d.set_ylabel('Epitopes exceeding threshold (%)')
ax_d.set_xlim(0.45, 1.0); ax_d.set_ylim(0, 105)
ax_d.grid(alpha=0.15, linestyle='--', linewidth=0.5)
ax_d.set_axisbelow(True)

# ── e: OVERALL STEP ROC CURVE (n=3,314 balanced test) ─────────────────────────
y_true_all = preds['label'].values
y_score_all = preds['prediction'].values
fpr_all, tpr_all, thresholds_all = roc_curve(y_true_all, y_score_all)
auc_all = roc_auc_score(y_true_all, y_score_all)

# Find Youden's J operating point
j_scores = tpr_all - fpr_all
j_idx = np.argmax(j_scores)
j_fpr, j_tpr = fpr_all[j_idx], tpr_all[j_idx]
j_thresh = thresholds_all[j_idx]

ax_e.plot([0,1],[0,1], color=NMI['baseline'], linestyle='--', linewidth=0.8,
          label='Random (AUC = 0.500)')
ax_e.fill_between(fpr_all, tpr_all, alpha=0.10, color=NMI['steel'])
ax_e.plot(fpr_all, tpr_all, color=NMI['steel'], linewidth=2.2,
          label=f'STEP (AUC = {auc_all:.3f})')

# Operating point
ax_e.scatter(j_fpr, j_tpr, s=80, color=NMI['coral'], edgecolors=NMI['bar_edge'],
             linewidth=1, zorder=5)
ax_e.annotate(f'Youden\'s J\n({j_fpr:.2f}, {j_tpr:.2f})\nτ = {j_thresh:.3f}',
              xy=(j_fpr, j_tpr), xytext=(j_fpr+0.12, j_tpr-0.15),
              fontsize=8, fontstyle='italic',
              )

ax_e.set_xlabel('False positive rate')
ax_e.set_ylabel('True positive rate')
ax_e.legend(fontsize=8, loc='lower right', frameon=False)
ax_e.set_xlim(-0.02, 1.02); ax_e.set_ylim(-0.02, 1.05)

# ── f: Violin by epitope length ──────────────────────────────────────────────
sub5['Length'] = sub5['Epitope'].str.len()
sub5['LenGroup'] = sub5['Length'].apply(
    lambda l: str(l) if sub5['Length'].value_counts().get(l,0) >= 5 else '13+')
groups_f = ['8','9','10','11','12','13+']
data_f = [sub5[sub5['LenGroup']==g]['ROC_AUC'].values for g in groups_f]
gf2, df2 = [], []
for g, d in zip(groups_f, data_f):
    if len(d) > 0: gf2.append(g); df2.append(d)
groups_f, data_f = gf2, df2

vcols = [NMI['rose'],NMI['sage'],NMI['steel'],NMI['sand'],NMI['teal'],NMI['lavender']]
parts = ax_f.violinplot(data_f, positions=range(len(groups_f)),
                        showmedians=False, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(vcols[i%len(vcols)])
    pc.set_edgecolor(NMI['bar_edge']); pc.set_alpha(0.65); pc.set_linewidth(0.8)

for i, d in enumerate(data_f):
    q25, med, q75 = np.percentile(d, [25,50,75])
    ax_f.vlines(i, q25, q75, color='0.2', linewidth=1.5, zorder=4)
    ax_f.scatter(i, med, color='white', s=25, edgecolors='0.2', linewidth=1, zorder=5)
    x = np.random.RandomState(42).normal(i, 0.06, size=len(d))
    ax_f.scatter(x, d, alpha=0.3, s=12, color='0.3', zorder=3)

ax_f.axhline(0.5, color=NMI['baseline'], linestyle='--', linewidth=0.8, alpha=0.5)
ax_f.set_xticks(range(len(groups_f)))
ax_f.set_xticklabels([f'{g} aa\n(n={len(d)})' for g,d in zip(groups_f,data_f)], fontsize=8)
ax_f.set_xlabel('Epitope length')
ax_f.set_ylabel('ROC-AUC')
ax_f.set_ylim(-0.05, 1.1)

fig3.savefig('results/figures/Figure_3_final.png', dpi=300,
             bbox_inches='tight', facecolor='white')
fig3.savefig('results/figures/Figure_3_final.pdf',
             bbox_inches='tight', facecolor='white')
print("Saved: Figure_3_final")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  (6 panels: grouped bars, heatmap strip, lung bars,
#            scatter, Cleveland dots, split violin)
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(14, 14))
gs4  = gridspec.GridSpec(3, 2, figure=fig4,
                         hspace=0.45, wspace=0.40,
                         height_ratios=[1, 1, 1])
a4 = fig4.add_subplot(gs4[0,0])
b4 = fig4.add_subplot(gs4[0,1])
c4 = fig4.add_subplot(gs4[1,0])
d4 = fig4.add_subplot(gs4[1,1])
e4 = fig4.add_subplot(gs4[2,0])
f4 = fig4.add_subplot(gs4[2,1])

for lbl, ax in zip('abcdef', [a4,b4,c4,d4,e4,f4]):
    panel_label(ax, lbl); despine(ax)

mel  = cancer[cancer['cancer']=='melanoma'].iloc[0]
lung_r = cancer[cancer['cancer']=='lung'].iloc[0]

# ── a: Grouped bars (overlap-inclusive + overlap-free) ────────────────────────
mel_c  = clean.iloc[0]   # Melanoma (clean)
lung_c = clean.iloc[1]   # Lung (clean)

groups = [
    ("Melanoma",    mel,    mel_c,  NMI['ochre']),
    ("Lung cancer", lung_r, lung_c, NMI['sage']),
]

bar_w = 0.18
gap_within = 0.04
group_centers = [0, 1.5]

for gi, (cname, full, clean_r, base_color) in enumerate(groups):
    cx = group_centers[gi]
    positions = [cx - 1.5*(bar_w+gap_within), cx - 0.5*(bar_w+gap_within),
                 cx + 0.5*(bar_w+gap_within), cx + 1.5*(bar_w+gap_within)]
    values  = [full['roc_auc'], clean_r['roc_auc'], full['pr_auc'], clean_r['pr_auc']]
    lows    = [full['roc_lo'],  clean_r['roc_lo'],  full['pr_lo'],  clean_r['pr_lo']]
    highs   = [full['roc_hi'],  clean_r['roc_hi'],  full['pr_hi'],  clean_r['pr_hi']]
    alphas  = [1.0, 1.0, 0.55, 0.55]
    hatches = ['', '//', '', '//']

    for pos, val, lo, hi, alpha, hatch in zip(positions, values, lows, highs, alphas, hatches):
        a4.bar(pos, val, bar_w, color=base_color, alpha=alpha,
               edgecolor=NMI['bar_edge'], linewidth=0.8, hatch=hatch,
               yerr=[[val-lo],[hi-val]], capsize=3, ecolor='0.3',
               error_kw={'linewidth':0.8})
        a4.text(pos, hi + 0.025, f'{val:.3f}', ha='center',
                fontsize=7.5, fontweight='bold')

a4.set_xticks(group_centers)
a4.set_xticklabels(['Melanoma', 'Lung cancer'], fontsize=10)
a4.set_ylabel('AUC'); a4.set_ylim(0, 1.18)
a4.axhline(0.5, color=NMI['baseline'], linestyle='--', linewidth=0.8, alpha=0.5)

legend_handles = [
    Patch(facecolor='lightgray', edgecolor=NMI['bar_edge'], label='ROC-AUC (full)'),
    Patch(facecolor='lightgray', edgecolor=NMI['bar_edge'], hatch='//', label='ROC-AUC (clean)'),
    Patch(facecolor='lightgray', edgecolor=NMI['bar_edge'], alpha=0.55, label='PR-AUC (full)'),
    Patch(facecolor='lightgray', edgecolor=NMI['bar_edge'], alpha=0.55, hatch='//', label='PR-AUC (clean)'),
]
a4.legend(handles=legend_handles, fontsize=6, loc='upper right',
          frameon=False, ncol=2)

mel_ep = [("EVDPIGHLY",1.000,"Other"),("AAGIGILTV",0.983,"MART-1"),
          ("GLCTLVAML",0.972,"Other"),("YLEPGPVTA",0.943,"Other"),
          ('SAYGEPRKL',0.880,'Other'),('KVDPIGHVY',0.870,'Other'),
          ('AARAVFLAL',0.821,'Other'),('EAAGIGILTV',0.769,'MART-1'),
          ('EEAAGIGILTVI',0.738,'MART-1'),('IMDQVPFSV',0.719,'gp100')]
mel_rev = mel_ep[::-1]
norm_h = Normalize(vmin=0.5, vmax=1.0)

for i, (name, roc, cat) in enumerate(mel_rev):
    color = nmi_cmap(norm_h(roc))
    b4.barh(i, roc, height=0.65, color=color, edgecolor=NMI['bar_edge'], linewidth=0.6)
    cat_col = NMI['coral'] if cat=='MART-1' else NMI['amber'] if cat=='gp100' else NMI['steel']
    b4.scatter(-0.03, i, color=cat_col, s=55, edgecolors=NMI['bar_edge'],
               linewidth=0.6, zorder=5, clip_on=False)
    b4.text(roc+0.008, i, f'{roc:.3f}', va='center', fontsize=7.5, fontweight='bold')

b4.set_yticks(range(len(mel_rev)))
b4.set_yticklabels([e[0] for e in mel_rev], fontsize=7.5)
b4.set_xlabel('ROC-AUC')
b4.set_xlim(-0.08, 1.08)
b4.axvline(0.5, color=NMI['baseline'], linestyle='--', alpha=0.3, linewidth=0.6)

sm_b = cm.ScalarMappable(cmap=nmi_cmap, norm=norm_h)
sm_b.set_array([])
cbar_b = plt.colorbar(sm_b, ax=b4, shrink=0.45, pad=0.02, aspect=15)
cbar_b.set_label('ROC-AUC', fontsize=8)
cbar_b.ax.tick_params(labelsize=7)

b4.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['coral'],
                          markersize=7,markeredgecolor=NMI['bar_edge'],label='MART-1'),
                   Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['amber'],
                          markersize=7,markeredgecolor=NMI['bar_edge'],label='gp100'),
                   Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['steel'],
                          markersize=7,markeredgecolor=NMI['bar_edge'],label='Other')],
          fontsize=7, loc='lower right', frameon=False)

# ── c: Lung per-epitope ──────────────────────────────────────────────────────
lung_ep = [('GADGVGKSAL',0.500,10),('GILGFVFTL',0.544,28),('VVGAVGVGK',0.556,11),
           ('YLAMPFATPME...',0.797,20),('AVGVGKSAL',0.938,10),
           ('FWIDLFETIG',0.980,21),('RFYKTLRAEQASQ',1.000,8),('GARGVGKSAL',1.000,7)]
cmap_c4 = [NMI['sage'] if any(k in e for k in ['GKSAL','VGVGK','GARGVG'])
           else NMI['teal'] for e,_,_ in lung_ep]
c4.barh(range(len(lung_ep)), [v for _,v,_ in lung_ep],
        color=cmap_c4, edgecolor=NMI['bar_edge'], linewidth=0.8)
c4.set_yticks(range(len(lung_ep)))
c4.set_yticklabels([e for e,_,_ in lung_ep], fontsize=7.5)
for i,(_,v,n) in enumerate(lung_ep):
    c4.text(v+0.01, i, f'{v:.3f} (n={n})', va='center', fontsize=7.5)
c4.legend(handles=[Patch(facecolor=NMI['sage'],edgecolor=NMI['bar_edge'],label='KRAS variants'),
                   Patch(facecolor=NMI['teal'],edgecolor=NMI['bar_edge'],label='Other lung')],
          fontsize=7, loc='lower right', frameon=False)
c4.set_xlabel('ROC-AUC')
c4.axvline(0.5, color=NMI['baseline'], linestyle='--', alpha=0.4, linewidth=0.6)
c4.set_xlim(0, 1.2)

# ── d: Scatter (clean, no adjustText arrows, no rug) ─────────────────────────
key_all = [
    ('AAGIGILTV','Mel.',0.983,35,'MART-1'),('EAAGIGILTV','Mel.',0.769,330,'MART-1'),
    ('EEAAGIGILTVI','Mel.',0.738,20,'MART-1'),('IMDQVPFSV','Mel.',0.719,26,'gp100'),
    ('YLEPGPVTA','Mel.',0.943,11,'Other'),('SAYGEPRKL','Mel.',0.880,10,'Other'),
    ('GARGVGKSAL','Lung',1.000,7,'KRAS'),('RFYKTLRAEQASQ','Lung',1.000,8,'Other'),
    ('FWIDLFETIG','Lung',0.980,21,'Other'),('AVGVGKSAL','Lung',0.938,10,'KRAS'),
    ('YLAMPFATPME','Lung',0.797,20,'Other'),('VVGAVGVGK','Lung',0.556,11,'Other'),
    ('GILGFVFTL','Lung',0.544,28,'Other'),('GADGVGKSAL','Lung',0.500,10,'KRAS'),
]

# Manual offset positions to avoid overlap
label_offsets = {
    'AAGIGILTV': (8, -10), 'EAAGIGILTV': (8, 5), 'IMDQVPFSV': (8, -8),
    'GARGVGKSAL': (8, -10), 'GILGFVFTL': (8, 5), 'GADGVGKSAL': (8, -8),
    'VVGAVGVGK': (8, -10),
}

for epi, canc, roc, n, cat in key_all:
    if 'Mel' in canc:
        col = NMI['coral'] if cat=='MART-1' else NMI['amber'] if cat=='gp100' else NMI['ochre']
        marker = 'o'
    else:
        col = NMI['sage'] if cat=='KRAS' else NMI['teal']
        marker = 's'
    d4.scatter(n, roc, s=90, color=col, marker=marker,
               edgecolors=NMI['bar_edge'], linewidth=0.7, zorder=3, alpha=0.8)
    # Only label select points (avoid clutter)
    if epi in label_offsets:
        off = label_offsets[epi]
        d4.annotate(epi[:12], xy=(n, roc), xytext=off,
                    textcoords='offset points', fontsize=6.5, alpha=0.7,
                    )

d4.axhline(0.5, color=NMI['baseline'], linestyle='--', linewidth=0.8, alpha=0.4)
d4.set_xlabel('Number of test samples')
d4.set_ylabel('ROC-AUC')
d4.set_ylim(0.40, 1.06)
d4.set_clip_on(True)
d4.set_xscale('log')
d4.grid(alpha=0.12, linestyle='--', linewidth=0.5)
d4.set_axisbelow(True)

d4.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['coral'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='MART-1 (Mel.)'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['amber'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='gp100 (Mel.)'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=NMI['ochre'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='Other (Mel.)'),
    Line2D([0],[0],marker='s',color='w',markerfacecolor=NMI['sage'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='KRAS (Lung)'),
    Line2D([0],[0],marker='s',color='w',markerfacecolor=NMI['teal'],
           markersize=7,markeredgecolor=NMI['bar_edge'],label='Other (Lung)')],
    fontsize=7, loc='lower right', frameon=False)

# ── e: UMAP — Melanoma embeddings ─────────────────────────────────────────────
mel_umap = np.load("results/umap/melanoma.npz")
u_mel, l_mel, p_mel = mel_umap["umap"], mel_umap["labels"], mel_umap["probs"]
bind_mel = l_mel == 1
e4.scatter(u_mel[~bind_mel,0], u_mel[~bind_mel,1], s=12, alpha=0.4,
           color=NMI["steel"], edgecolors="none", label="Non-binding")
e4.scatter(u_mel[bind_mel,0], u_mel[bind_mel,1], s=12, alpha=0.5,
           color=NMI["coral"], edgecolors="none", label="Binding")
e4.set_xlabel("UMAP 1"); e4.set_ylabel("UMAP 2")
e4.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=NMI["coral"],markersize=7,markeredgecolor=NMI["bar_edge"],label="Binding"), Line2D([0],[0],marker="o",color="w",markerfacecolor=NMI["steel"],markersize=7,markeredgecolor=NMI["bar_edge"],label="Non-binding")], fontsize=7, loc="upper right", frameon=False)

# ── f: UMAP — Lung cancer embeddings ─────────────────────────────────────────
lung_umap = np.load("results/umap/lung.npz")
u_lung, l_lung = lung_umap["umap"], lung_umap["labels"]
bind_lung = l_lung == 1
f4.scatter(u_lung[~bind_lung,0], u_lung[~bind_lung,1], s=20, alpha=0.3, zorder=2,
           color=NMI["steel"], edgecolors=NMI["bar_edge"], linewidth=0.3, label="Non-binding")
f4.scatter(u_lung[bind_lung,0], u_lung[bind_lung,1], s=45, alpha=0.8, zorder=4,
           color=NMI["coral"], edgecolors=NMI["bar_edge"], linewidth=0.3, label="Binding")
f4.set_xlabel("UMAP 1"); f4.set_ylabel("UMAP 2")
# Legend as manual text
f4.scatter([u_lung[:,0].max()+2], [u_lung[:,1].max()+0.6], s=45, color=NMI["coral"], edgecolors=NMI["bar_edge"], linewidth=0.3, clip_on=False)
f4.text(u_lung[:,0].max()+2.8, u_lung[:,1].max()+0.6, "Binding", fontsize=7, va="center")
f4.scatter([u_lung[:,0].max()+2], [u_lung[:,1].max()+0.2], s=45, color=NMI["steel"], edgecolors=NMI["bar_edge"], linewidth=0.3, clip_on=False)
f4.text(u_lung[:,0].max()+2.8, u_lung[:,1].max()+0.2, "Non-binding", fontsize=7, va="center")
fig4.savefig("results/figures/Figure_4_final.png", dpi=300, bbox_inches="tight", facecolor="white")
fig4.savefig('results/figures/Figure_4_final.pdf',
             bbox_inches='tight', facecolor='white')
print("Saved: Figure_4_final")
plt.close(fig4)

print("\nDone. Files:")
for f in sorted(os.listdir('results/figures')):
    sz = os.path.getsize(f'results/figures/{f}')/1024
    print(f"  {sz:>7.0f}KB  {f}")
