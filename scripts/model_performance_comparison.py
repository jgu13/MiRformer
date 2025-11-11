import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
from typing import List

# ======================
# Config & Load
# ======================
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()
plt.rcParams['font.size'] = 15

Performance_dir = os.path.join(os.path.expanduser("~/projects/mirLM"), "Performance/TargetScan_test")
os.makedirs(Performance_dir, exist_ok=True)
csv_path = os.path.join(Performance_dir, "model_comparison_metrics.csv")
df = pd.read_csv(csv_path)

# Fixed palette (+ alias for miTAE/miTAR)
colors = {
    "Ours": "#646EF2",
    "TargetScan": "#5BC999",
    "REPRESS": "#63CFEF",
    "miTAR": "#A267F2"
}

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 14

# ======================
# Helpers
# ======================
def _subset_df_in_order(df: pd.DataFrame, models_in_order: List[str]) -> pd.DataFrame:
    return (df[df["Model"].isin(models_in_order)]
            .set_index("Model")
            .reindex(models_in_order)
            .reset_index())

def _auto_ylim(values: np.ndarray, ceiling: float = 1.0, pad: float = 0.05):
    vmax = float(np.nanmax(values)) if values.size else 1.0
    top = min(max(ceiling, vmax + pad), max(1.05, vmax + pad))
    return (0.0, top)

def _bar_annot(ax, x, width, i, vals, ypad=0.01, fmt="{:.3f}"):
    for j, v in enumerate(vals):
        if np.isnan(v):
            continue
        ax.text(x[j] + i * width, v + ypad, fmt.format(v),
                ha='center', va='bottom', fontsize=10)

def _plot_axis(ax, df, models, metrics, title, ylim_ceiling=1.0):
    sub = _subset_df_in_order(df, models)
    x = np.arange(len(metrics))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    # bars
    for i, model in enumerate(models):
        vals = np.atleast_1d(sub.loc[sub["Model"] == model, metrics].values.squeeze().astype(float))
        ax.bar(x + i * width, vals, width=width, label=model, color=colors.get(model))
        _bar_annot(ax, x, width, i, vals, ypad=0.012)

    # axis cosmetics
    ax.set_xticks(x + (n_models - 1) * width / 2)
    ax.set_xticklabels(metrics, fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    ymin, ymax = _auto_ylim(sub[metrics].to_numpy(dtype=float), ceiling=ylim_ceiling, pad=0.05)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # axis labels (bigger font)
    ax.set_ylabel("Score", fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return ax.get_legend_handles_labels()


# ======================
# Build the combined figure (3 subplots)
# ======================
m1_models = ["Ours", "TargetScan", "REPRESS", "miTAR"]
m1_metrics = ["Binding Accuracy", "Seed F1 score"]

m2_models = ["Ours", "REPRESS", "miTAR"]
m2_metrics = ["Span AUROC"]

m3_models = ["Ours", "REPRESS"]
m3_metrics = ["Hit at 5", "Hit at 3", "Hit at 0"]

# Width ratios so the **visual bar width** matches across subplots:
# bar_visual_width ∝ axis_width / (len(metrics) * len(models))
# -> set axis_width ratios ∝ len(metrics) * len(models)
w1 = len(m1_metrics) * len(m1_models)
w2 = len(m2_metrics) * len(m2_models)
w3 = len(m3_metrics) * len(m3_models)

fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 6),
    dpi=500,
    gridspec_kw={"width_ratios": [w1, w2, w3]}
)

# 1) Ours vs TargetScan — Binding Accuracy, Seed F1
h1, l1 = _plot_axis(
    axes[0],
    df=df,
    models=m1_models,
    metrics=m1_metrics,
    title="Binding & Seed",
    ylim_ceiling=1.0
)

# 2) Ours vs REPRESS vs miTAE — Binding, Seed, Span AUROC
h2, l2 = _plot_axis(
    axes[1],
    df=df,
    models=m2_models,  # will fallback to miTAR if needed
    metrics=m2_metrics,
    title="Seed Only",
    ylim_ceiling=1.0
)

# 3) Ours vs REPRESS — Hit@5, Hit@3, Hit@0
h3, l3 = _plot_axis(
    axes[2],
    df=df,
    models=m3_models,
    metrics=m3_metrics,
    title="Cleavage Site (Hit@K)",
    ylim_ceiling=1.0
)

# Shared legend (collect unique labels in order of appearance)
handles, labels = [], []
for H, L in [(h1, l1), (h2, l2), (h3, l3)]:
    for h, l in zip(H, L):
        if l not in labels:
            handles.append(h); labels.append(l)

# --- Update the shared legend line to use the new constant ---
fig.legend(handles, labels, loc="lower center", ncol=len(labels),
           frameon=False, fontsize=LEGEND_FONTSIZE)


fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.subplots_adjust(wspace=0.3)

# Save
save_path = os.path.join(Performance_dir, "model_comparison_combined.png")
plt.savefig(save_path, dpi=500, bbox_inches="tight")
plt.close(fig)
print(f"Combined figure saved to: {save_path}")
