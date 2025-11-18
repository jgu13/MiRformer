"""
Create a 2x2 grid of pooling-method line plots from fixed CSV inputs.

The script expects the CSV files listed in `METRIC_FILES` to be located in the
current working directory. Each CSV must contain a `Step` column and one or
more metric columns; all are plotted against `Step`, and the figure is saved as
`Pooling_Method_comparison.png`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()

METRIC_FILES: Tuple[Tuple[str, str], ...] = (
    ("Binding Accuracy", "Pooling_Method_Binding_Accuracy.csv"),
    ("Exact Match Rate", "Pooling_Method_Exact_match_rate.csv"),
    ("F1 Score", "Pooling_Method_F1_score.csv"),
    ("Evaluation Loss", "Pooling_Method_Eval_loss.csv"),
)


def read_metric_csv(csv_path: Path) -> pd.DataFrame:
    """Return a dataframe with `Step` as the first column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f"`Step` column not found in {csv_path}")

    df = df.copy()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df = df.dropna(subset=["Step"])
    return df


def plot_metric(ax: plt.Axes, df: pd.DataFrame, title: str, colors: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, plt.Line2D]:
    """Plot all metric columns against `Step`."""
    if len(df.columns) <= 1:
        raise ValueError(f"No metric columns found for {title}")

    lines: Dict[str, plt.Line2D] = {}
    for column in df.columns:
        if column == "Step":
            continue
        line, = ax.plot(df["Step"], df[column], label=column, color=colors[column])
        lines[column] = line
    # set face color to whitesmoke
    ax.set_facecolor("whitesmoke")
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.set_xlabel("Step", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(title, fontsize=AXIS_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    return lines


metric_data: Dict[str, pd.DataFrame] = {}
for metric_name, file_name in METRIC_FILES:
    csv_path = Path(file_name)
    metric_data[metric_name] = read_metric_csv(csv_path)

# Determine a consistent color per metric column across all plots
metric_columns: List[str] = []
for df in metric_data.values():
    for column in df.columns:
        if column == "Step" or column in metric_columns:
            continue
        metric_columns.append(column)

cmap = plt.get_cmap("Set1", len(metric_columns))
metric_colors: Dict[str, Tuple[float, float, float, float]] = {
    column: cmap(idx % cmap.N) for idx, column in enumerate(metric_columns)
}

fig, axes = plt.subplots(2, 2, figsize=(30/2.54, 23.87/2.54), sharex=False)
axes_iter: Iterable[Tuple[str, plt.Axes]] = zip(
    metric_data.keys(), axes.flatten()
)

legend_handles: Dict[str, plt.Line2D] = {}
for metric_name, ax in axes_iter:
    lines = plot_metric(ax, metric_data[metric_name], metric_name, metric_colors)
    for name, line in lines.items():
        if name not in legend_handles:
            legend_handles[name] = line

fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(
    list(legend_handles.values()),
    list(legend_handles.keys()),
    loc="lower center",
    ncol=min(2, len(metric_columns)),
    fontsize=LEGEND_FONT_SIZE,
)
output_path = os.path.join(PROJ_HOME, "Performance/TargetScan_test/Pooling_Method_comparison.svg")
fig.savefig(output_path, dpi=500)
print(f"Saved plot to {output_path}")
