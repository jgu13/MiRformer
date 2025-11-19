import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.font_manager as font_manager
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE

# ======================
# Config & Fonts
# ======================
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()

test_500_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions_500_test.csv")
utr_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions.csv")
seed_type_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions_seed_type.csv")
distance_within_5_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions_distance_within_5.csv")
test_500_data = pd.read_csv(test_500_data_path)
utr_data = pd.read_csv(utr_data_path)
seed_type_data = pd.read_csv(seed_type_data_path)
distance_within_5_data = pd.read_csv(distance_within_5_data_path)

# calculate the seed length and distance to cleavage site
test_500_data['seed_length'] = test_500_data['pred end'] - test_500_data['pred start'] + 1
cleav = test_500_data['pred cleavage']
start = test_500_data['pred start']
end   = test_500_data['pred end']
test_500_data['distance_to_cleavage_site'] = np.minimum(np.abs(cleav - start), np.abs(cleav - end))

utr_data['seed_length'] = utr_data['pred end'] - utr_data['pred start'] + 1
cleav = utr_data['pred cleavage']
start = utr_data['pred start']
end   = utr_data['pred end']
utr_data['distance_to_cleavage_site'] = np.minimum(np.abs(cleav - start), np.abs(cleav - end))

order = [6, 7, 8]

def summarize(df, name, denom="all"):
    # Percent of seed lengths that are exactly 6/7/8
    counts = df['seed_length'].value_counts(dropna=True)
    if denom == "subset":
        # denominator = only rows with seed_length in {6,7,8}
        denom_val = counts.reindex(order, fill_value=0).sum()
    else:
        # denominator = all rows in df
        denom_val = len(df)

    pct_seed = (counts / denom_val).reindex(order, fill_value=0.0)
    pct_seed.index = pd.Index(pct_seed.index, name="seed_length")

    # Distance “within 5”: use <= 5 (your code used == 5, which is stricter)
    dist = df['distance_to_cleavage_site']
    pct_dist_within_5 = (dist <= 5).mean()  # ignores NaN automatically

    return pd.Series(
        {
            "pct_seed_len_6": pct_seed.get(6, 0.0),
            "pct_seed_len_7": pct_seed.get(7, 0.0),
            "pct_seed_len_8": pct_seed.get(8, 0.0),
            "pct_distance_within_5": pct_dist_within_5,
            "denominator": denom,
            "N_rows": len(df)
        },
        name=name
    )

# Summarize seed types and distance within 5
# summary_all = pd.concat(
#     [
#         summarize(test_500_data, "test_500", denom="all"),
#         summarize(utr_data, "utr", denom="all"),
#     ],
#     axis=1
# ).T

# summary_subset = pd.concat(
#     [
#         summarize(test_500_data, "test_500", denom="subset"),
#         summarize(utr_data, "utr", denom="subset"),
#     ],
#     axis=1
# ).T
# print("\nPercentages with denominator = ALL rows:")
# print(summary_all)

# print("\nPercentages with denominator = ONLY rows where seed_length in {6,7,8}:")
# print(summary_subset)

color_test_500 = 'hotpink'
color_utr = 'slateblue'

# plot the seed length as histogram
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=500)
ax.hist(test_500_data['seed_length'], bins=range(0,50), alpha=0.5, label='Seed Length', density=True, color=color_test_500)
ax.tick_params(labelsize=TICK_FONT_SIZE-2)  # set x-tick label font size
ax.set_xlabel('Seed Length', fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel('Ratio', fontsize=AXIS_FONT_SIZE-2)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_500_test.svg"), dpi=500)
print("Seed length distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_500_test.svg"))

# plot the seed length as histogram
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=500)
# draw y axis as percentage instead of count
ax.hist(utr_data['seed_length'], bins=range(0,50), alpha=0.5, label='Seed Length', density=True, color=color_utr)
ax.tick_params(labelsize=TICK_FONT_SIZE-2)  # set x-tick label font size
ax.set_xlabel('Seed Length', fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel('Ratio', fontsize=AXIS_FONT_SIZE-2)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_utr.svg"), dpi=500)
print("Seed length distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_utr.svg"))

# plot distance to cleavage site as histogram
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=500)
# draw y axis as percentage instead of count
ax.hist(test_500_data['distance_to_cleavage_site'], bins=range(0,50), alpha=0.5, label='Distance to Cleavage Site', density=True, color=color_test_500)
ax.tick_params(labelsize=TICK_FONT_SIZE-2)  # set x-tick label font size
ax.set_xlabel('Distance to Cleavage Site', fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel('Ratio', fontsize=AXIS_FONT_SIZE-2)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_500_test.svg"), dpi=500)
print("Distance to cleavage site distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_500_test.svg"))

# plot distance to cleavage site as histogram
fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=500)
# draw y axis as percentage instead of count
ax.hist(utr_data['distance_to_cleavage_site'], bins=range(0,50), alpha=0.5, label='Distance to Cleavage Site', density=True, color=color_utr)
ax.tick_params(labelsize=TICK_FONT_SIZE-2)  # set x-tick label font size
ax.set_xlabel('Distance to Cleavage Site', fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel('Ratio', fontsize=AXIS_FONT_SIZE-2)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_utr.svg"), dpi=500)
print("Distance to cleavage site distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_utr.svg"))

# Plot boxplot of distance to cleavage site
# filter seed length to only 6, 7 and 8
order = [6, 7, 8]
valid_data_test_500 = test_500_data[test_500_data['seed_length'].isin(order)].copy()
valid_data_utr = utr_data[utr_data['seed_length'].isin(order)].copy()

# Prepare data for boxplot in fixed order
data_to_plot_test_500 = [
    valid_data_test_500.loc[valid_data_test_500['seed_length'] == k, 'distance_to_cleavage_site'].dropna()
    for k in order
]
data_to_plot_utr = [
    valid_data_utr.loc[valid_data_utr['seed_length'] == k, 'distance_to_cleavage_site'].dropna()
    for k in order
]
# Plot
labels = [f"{k}-mer" for k in order]
n = len(order)

# X positions for grouped boxplots
x = np.arange(1, n + 1)           # centers: 1, 2, 3
offset = 0.18                     # half the gap between side-by-side boxes
pos_a = x - offset                # first dataset (e.g., test_500)
pos_b = x + offset                # second dataset (e.g., utr)
width = 0.32

fig, ax = plt.subplots(figsize=(11.33/2.54, 8.03/2.54), dpi=500)

bp_a = ax.boxplot(
    data_to_plot_test_500,
    positions=pos_a,
    widths=width,
    showfliers=False,
    patch_artist=True,
)
bp_b = ax.boxplot(
    data_to_plot_utr,
    positions=pos_b,
    widths=width,
    showfliers=False,
    patch_artist=True,
)

# (Optional) simple fill so they’re distinguishable
for patch in bp_a['boxes']:
    patch.set_facecolor(color_test_500)
for patch in bp_b['boxes']:
    patch.set_facecolor(color_utr)

# Center ticks under each pair
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE-2)
ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE-2)
ax.set_xlabel("Seed length", fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel("Distance to predicted\ncleavage site (nt)", fontsize=AXIS_FONT_SIZE-2)
# ax.set_title("Distance to cleavage vs. seed length (grouped)", fontsize=20)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Legend using proxy patches
handles = [
    Patch(facecolor=color_test_500, edgecolor='black', label='Test 500'),
    Patch(facecolor=color_utr, edgecolor='black', label="3'UTR"),
]
ax.legend(handles=handles, loc='best', fontsize=LEGEND_FONT_SIZE-2, frameon=False)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_boxplot.svg"), dpi=500)
print("Distance to cleavage site boxplot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_boxplot.svg"))

# plot bar plot of seed type
# Metrics / x-axis categories
metrics = ["6-mer %", "7-mer %", "8-mer %"]
datasets = seed_type_data["dataset"].tolist()
color_map = {
    "Test 500": color_test_500,
    "3'UTR": color_utr,
}

x = np.arange(len(metrics))   # positions for 6/7/8-mer
width = 0.45                  # bar width

fig, ax = plt.subplots(figsize=(13.77/2.54, 6.37/2.54), dpi=500)  # A4-width friendly

# Grouped bars
for i, ds in enumerate(datasets):
    vals = seed_type_data.loc[seed_type_data["dataset"] == ds, metrics].values.ravel().astype(float)

    # Center groups around x (6, 7, 8)
    x_positions = x + (i - (len(datasets) - 1) / 2) * width
    bar_color = color_map.get(ds, "grey")

    ax.bar(x_positions, vals, width=width, label=ds, color=bar_color)

    # Value labels above bars
    for xpos, v in zip(x_positions, vals):
        ax.annotate(
            f"{v:.1f}",
            xy=(xpos, v),
            xytext=(0, 2),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=TICK_FONT_SIZE-2
        )

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=TICK_FONT_SIZE-2)
ax.set_xlabel("Seed length", fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel("Percentage (%)", fontsize=AXIS_FONT_SIZE-2)
ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE-2)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE-2)
ax.margins(y=0.2)

fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_type_distribution.svg"), dpi=500)
print("Seed type distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_type_distribution.svg"))

# plot bar plot of distance within 5
datasets = distance_within_5_data["dataset"].tolist()
values = distance_within_5_data["distance within 5nt %"].astype(float).to_numpy()

x = np.arange(len(datasets))
width = 0.45

fig, ax = plt.subplots(figsize=(5.89 / 2.54, 8.03 / 2.54), dpi=500)

color_map = {
    "Test 500": color_test_500,
    "3'UTR": color_utr,
}

# Bar colors in same order as datasets
bar_colors = [color_map.get(ds, "gray") for ds in datasets]

bars = ax.bar(x, values, color=bar_colors, width=width)

# Value labels above bars
for xpos, v in zip(x, values):
    ax.annotate(
        f"{v:.1f}",
        xy=(xpos, v),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=TICK_FONT_SIZE-2,
    )

# Axes labels
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=TICK_FONT_SIZE-2)
ax.set_xlabel("Dataset", fontsize=AXIS_FONT_SIZE-2)
ax.set_ylabel("Distance within 5nt (%)", fontsize=AXIS_FONT_SIZE-2)
ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE-2)
# Light grid on y-axis
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.margins(y=0.2)

# Small margins to reduce whitespace
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_within_5nt_distribution.svg"), dpi=500)
print("Seed type distribution plot saved to: ", os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_within_5nt_distribution.svg"))
