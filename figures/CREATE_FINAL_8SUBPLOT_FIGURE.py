"""
Create final DAISY-style 8-subplot comparison figure
with balanced clean Unseen-Epitope results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("  CREATING FINAL DAISY-STYLE FIGURE 2")
print("=" * 80)

# Load balanced results for Unseen-Epitope
balanced = pd.read_csv("results/ALL_6_MODELS_BALANCED_FINAL.csv")

# Load original 4-split results
roc_orig = pd.read_csv("results/COMPLETE_4SPLIT_ROC.csv")
pr_orig = pd.read_csv("results/COMPLETE_4SPLIT_PR.csv")

# Replace Unseen-Epitope with balanced clean results
models = ["STEP", "DAISY", "TEINet", "ERGO-LSTM", "ERGO-AE", "ATM-TCR"]

for model in models:
    bal_row = balanced[balanced["Model"] == model]
    if len(bal_row) > 0:
        roc_orig.loc[roc_orig["Split"] == "Unseen-Epitope", model] = bal_row["ROC-AUC"].values[0]
        pr_orig.loc[pr_orig["Split"] == "Unseen-Epitope", model] = bal_row["PR-AUC"].values[0]

print("\nUpdated ROC-AUC:")
print(roc_orig.to_string(index=False))

print("\nUpdated PR-AUC:")
print(pr_orig.to_string(index=False))

# Journal-style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

plt.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.92,
    bottom=0.12,
    wspace=0.35,
    hspace=0.42
)

# DAISY-like pastel colors
colors = [
    "#a8e6dc",  # STEP
    "#f8b99a",  # DAISY
    "#b8c5df",  # TEINet
    "#ecb5d4",  # ERGO-LSTM
    "#c4e69a",  # ERGO-AE
    "#f7e48a",  # ATM-TCR
]

splits = ["Seen-Pair", "Unseen-Tcr", "Unseen-Epitope", "Unseen-Pair"]
titles = ["Seen-Pair", "Unseen-TCR", "Unseen-Epitope", "Unseen-Pair"]

panel_letters_top = ["a", "b", "c", "d"]
panel_letters_bottom = ["e", "f", "g", "h"]

def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", labelsize=9, width=1.0, length=3)

def plot_panel(ax, values, title, ylabel, letter):
    bars = ax.bar(
        range(len(models)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.78,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=12, fontweight="normal", pad=7)

    # DAISY-style lowercase panel letter
    ax.text(
        -0.18,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    style_axis(ax, ylabel)

# Top row: ROC-AUC
for idx, split in enumerate(splits):
    ax = axes[0, idx]
    split_data = roc_orig[roc_orig["Split"] == split]

    if split_data.empty:
        raise ValueError(f"Missing split in ROC file: {split}")

    values = [float(split_data[model].values[0]) for model in models]

    plot_panel(
        ax=ax,
        values=values,
        title=titles[idx],
        ylabel="ROC-AUC",
        letter=panel_letters_top[idx],
    )

# Bottom row: PR-AUC
for idx, split in enumerate(splits):
    ax = axes[1, idx]
    split_data = pr_orig[pr_orig["Split"] == split]

    if split_data.empty:
        raise ValueError(f"Missing split in PR file: {split}")

    values = [float(split_data[model].values[0]) for model in models]

    plot_panel(
        ax=ax,
        values=values,
        title=titles[idx],
        ylabel="PR-AUC",
        letter=panel_letters_bottom[idx],
    )

# Save final figure
plt.savefig(
    "results/figures/Figure_2_final.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    "results/figures/Figure_2_final.pdf",
    bbox_inches="tight",
    facecolor="white",
)

print("\n✅ Figure saved:")
print("   - results/figures/Figure_2_final.png")
print("   - results/figures/Figure_2_final.pdf")

print("\n" + "=" * 80)
print("🎉 FINAL DAISY-STYLE FIGURE 2 COMPLETE")
print("=" * 80)