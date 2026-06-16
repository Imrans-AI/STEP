"""
BUILD_FIG5_DAISY_PATTERN.py
Final Fig 5 in DAISY pattern:
  a, b: SHAP beeswarm (global, physchem) - unchanged
  c:    Ablation radar on BALANCED test set (3 variants)
  d:    Ablation radar on MELANOMA test set (3 variants)

3 variants per radar: STEP (Full), STEP (No Structure), STEP (300 epi)
Same models, two test conditions - mirrors DAISY's Unseen-Epitope vs Unseen-Pair.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              accuracy_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from models.step_model import STEPModel
from models.dataset_physchem_only import PhysChemOnlyDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.linewidth': 1.2,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ---------- Variants to evaluate ----------
VARIANTS = [
    ('STEP (Full)',         'models/checkpoints/step_600epi_20260402_164333/best_model.pt', True,  '#2E86AB', '-',  'o'),
    ('STEP (No Structure)', 'models/checkpoints/step_600epi_20260402_164333/best_model.pt', False, '#C73E1D', ':',  'D'),
    ('STEP (300 epi)',      'models/checkpoints/step_300epi_20260402_171934/best_model.pt', True,  '#F18F01', '-.', '^'),
]

TEST_SETS = [
    ('balanced', 'data/splits_600_epitope/test_balanced.tsv'),
    ('melanoma', 'data/cancer_specific/melanoma_test.tsv'),
]


def evaluate(ckpt_path, use_struct, test_tsv):
    model = STEPModel(use_structure_priors=use_struct).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except RuntimeError:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    ds = PhysChemOnlyDataset(test_tsv)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    probs, labels = [], []
    with torch.no_grad():
        for physchem, glob, y in loader:
            physchem, glob = physchem.to(DEVICE), glob.to(DEVICE)
            logits = model(physchem, glob)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.extend(p); labels.extend(y.numpy().ravel())
    probs, labels = np.array(probs), np.array(labels)

    fpr, tpr, thr = roc_curve(labels, probs)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    preds = (probs >= best_thr).astype(int)

    return {
        'ROC_AUC':   roc_auc_score(labels, probs),
        'PR_AUC':    average_precision_score(labels, probs),
        'Accuracy':  accuracy_score(labels, preds),
        'F1':        f1_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall':    recall_score(labels, preds),
        'Threshold': best_thr,
    }


print("Evaluating all variants on both test sets...")
results = {}   # results[test_name][variant_name] = metrics dict
for test_name, tsv in TEST_SETS:
    print(f"\n=== {test_name.upper()} ({tsv}) ===")
    results[test_name] = {}
    for var_name, ckpt, use_struct, *_ in VARIANTS:
        print(f"  Scoring {var_name}...")
        m = evaluate(ckpt, use_struct, tsv)
        results[test_name][var_name] = m
        print(f"    ROC-AUC={m['ROC_AUC']:.4f}  F1={m['F1']:.4f}  thr={m['Threshold']:.3f}")

# Save numeric results
rows = []
for tname, vars_ in results.items():
    for vname, m in vars_.items():
        rows.append({'test_set': tname, 'variant': vname, **m})
pd.DataFrame(rows).to_csv('results/ablation_two_conditions.csv', index=False)
print("\nSaved: results/ablation_two_conditions.csv")

# ---------- Load SHAP for panels a, b ----------
shap_global = np.load('results/shap_values_global_v2.npy')
shap_physchem = np.load('results/shap_values_physchem.npy')

GLOBAL_NAMES = ['Isoelectric Point', 'Instability Index', 'Aliphatic Index',
                'Boman Index', 'Hydrophobic Moment', 'Molecular Weight', 'Autocorrelation']
PHYSCHEM_NAMES = ['Hydrophobicity', 'Charge', 'Flexibility', 'Refractivity', 'Mean Fractional Area Loss']

if shap_physchem.ndim == 4:
    shap_physchem_2d = shap_physchem.mean(axis=(2, 3))
else:
    shap_physchem_2d = shap_physchem

N_PLOT = min(300, shap_global.shape[0])
rng = np.random.default_rng(42)
idx = rng.choice(shap_global.shape[0], N_PLOT, replace=False)
sg = shap_global[idx]
sp = shap_physchem_2d[idx]


def daisy_shap_panel(ax, shap_vals, names):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_shap = shap_vals[:, order]
    sorted_mean = mean_abs[order]
    n_features = len(sorted_names)
    y_pos = np.arange(n_features)[::-1]

    raw_max = np.abs(sorted_shap).max() * 1.10
    ax.set_xlim(-raw_max, raw_max)

    for i, yp in enumerate(y_pos):
        m = sorted_mean[i]
        ax.barh(yp, 2 * m, left=-m, height=0.75,
                color='#E8DAEF', edgecolor='none', alpha=0.7, zorder=1)

    for i, yp in enumerate(y_pos):
        vals = sorted_shap[:, i]
        ranks = vals.argsort().argsort() / max(1, len(vals) - 1)
        colors = plt.cm.PRGn(ranks)
        y_jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(vals, [yp] * len(vals) + y_jitter,
                   c=colors, s=20, alpha=0.85,
                   edgecolor='black', linewidth=0.25, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.axvline(0, color='gray', linewidth=0.9, alpha=0.7, zorder=2)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.PRGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=20)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature value', fontsize=9, rotation=270, labelpad=15)
    cb.ax.tick_params(labelsize=8)


def radar_panel(ax, results_dict):
    metric_keys    = ['ROC_AUC', 'PR_AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
    metric_display = ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    for var_name, _, _, color, ls, mk in VARIANTS:
        m = results_dict[var_name]
        vals = [m[k] for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2.2, linestyle=ls, marker=mk,
                markersize=7, label=var_name, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.13, zorder=2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([''] * len(metric_keys))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8, color='gray')
    ax.grid(linestyle=':', linewidth=0.8, color='gray', alpha=0.7)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(0.8)

    label_radius = 1.18
    for ang, lbl in zip(angles[:-1], metric_display):
        x_norm = np.cos(np.pi/2 - ang)
        if abs(x_norm) < 0.15:
            ha = 'center'
        elif x_norm > 0:
            ha = 'left'
        else:
            ha = 'right'
        ax.text(ang, label_radius, lbl, ha=ha, va='center', fontsize=10)

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10), fontsize=9, frameon=False)


fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.50,
                       left=0.09, right=0.95, top=0.93, bottom=0.07)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0], projection='polar')
ax_d = fig.add_subplot(gs[1, 1], projection='polar')

daisy_shap_panel(ax_a, sg, GLOBAL_NAMES)
daisy_shap_panel(ax_b, sp, PHYSCHEM_NAMES)
radar_panel(ax_c, results['balanced'])
radar_panel(ax_d, results['melanoma'])

fig.text(0.07, 0.96, 'a', fontsize=22, fontweight='bold', va='top', ha='left')
fig.text(0.55, 0.96, 'b', fontsize=22, fontweight='bold', va='top', ha='left')
fig.text(0.07, 0.48, 'c', fontsize=22, fontweight='bold', va='top', ha='left')
fig.text(0.55, 0.48, 'd', fontsize=22, fontweight='bold', va='top', ha='left')

out_png = Path('results/Figure_5_SHAP_Analysis_DAISY_STYLE.png')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_png.with_suffix('.pdf')}")
