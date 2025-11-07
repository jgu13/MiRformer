import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
from typing import Optional

# Set global font to Gill Sans
gill_sans_font = font_manager.FontProperties(family='Gill Sans')

# === Load data ===
Performance_dir = os.path.join(os.path.expanduser("~/projects/mirLM"), "Performance/TargetScan_test")
df = pd.read_csv(os.path.join(Performance_dir, "model_comparison_metrics.csv"))

# === Color palette (fixed) ===
colors = {
    "MiLes": "#646EF2",       # blue
    "TargetScan": "#5BC999",  # greenish
    "REPRESS": "#63CFEF",     # cyan
    "miRAW": "#A267F2"        # purple
}

# === Metrics ===
metrics1 = ["Binding Accuracy", "Seed Exact Match Rate", "Seed F1 score"]
metrics2 = ["Hit at 5", "Hit at 3", "Hit at 0"]

# === Figure setup ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=500, gridspec_kw={'width_ratios': [5, 4]})
width = 0.2

# === Subplot 1: Binding / Seed metrics ===
x1 = np.arange(len(metrics1))
for i, model in enumerate(df["Model"]):
    axes[0].bar(x1 + i * width, df.loc[i, metrics1], width=width,
                label=model, color=colors[model])
    for j, val in enumerate(df.loc[i, metrics1]):
        axes[0].text(x1[j] + i * width, val + 0.01, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=10)

axes[0].set_xticks(x1 + width * 1.5)
axes[0].set_xticklabels(metrics1, fontsize=15)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].set_ylim(0, 1.1)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# === Subplot 2: Hit@N metrics (skip TargetScan) ===
df_hit = df[df["Model"] != "TargetScan"].reset_index(drop=True)
x2 = np.arange(len(metrics2))

for i, model in enumerate(df_hit["Model"]):
    axes[1].bar(x2 + i * width, df_hit.loc[i, metrics2], width=width,
                label=model, color=colors[model])
    for j, val in enumerate(df_hit.loc[i, metrics2]):
        axes[1].text(x2[j] + i * width, val + 0.005, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=10)

axes[1].set_xticks(x2 + width * 1.2)
axes[1].set_xticklabels(metrics2, fontsize=15)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].set_ylim(0, 0.8)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

# === Shared legend ===
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           frameon=False, fontsize=15)

# === Layout adjustments ===
fig.tight_layout(rect=[0, 0.1, 1, 1])
fig.subplots_adjust(wspace=0.25)

# === Save ===
save_path = os.path.join(Performance_dir, "model_comparison_combined.png")
plt.rcParams['font.family'] = gill_sans_font.get_name()
plt.rcParams['font.size'] = 18
plt.savefig(save_path, dpi=500)
plt.show()

print(f"Combined figure saved to: {save_path}")
