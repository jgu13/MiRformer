import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
test_500_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions_500_test.csv")
utr_data_path = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/predicted_seeds_and_cleave_sites_predictions.csv")
test_500_data = pd.read_csv(test_500_data_path)
utr_data = pd.read_csv(utr_data_path)

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

# Examples (pick denominator="all" or "subset")
summary_all = pd.concat(
    [
        summarize(test_500_data, "test_500", denom="all"),
        summarize(utr_data, "utr", denom="all"),
    ],
    axis=1
).T

summary_subset = pd.concat(
    [
        summarize(test_500_data, "test_500", denom="subset"),
        summarize(utr_data, "utr", denom="subset"),
    ],
    axis=1
).T

print("\nPercentages with denominator = ALL rows:")
print(summary_all)

print("\nPercentages with denominator = ONLY rows where seed_length in {6,7,8}:")
print(summary_subset)

# color_test_500 = 'hotpink'
# color_utr = 'slateblue'

# # plot the seed length as histogram
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.hist(test_500_data['seed_length'], bins=range(0,50), alpha=0.5, label='Seed Length', density=True, color=color_test_500)
# ax.tick_params(labelsize=12)  # set x-tick label font size
# ax.set_xlabel('Seed Length', fontsize=15)
# ax.set_ylabel('Ratio', fontsize=15)
# # ax.set_title('Seed Length Distribution', fontsize=20)
# ax.legend(fontsize=12)
# fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_500_test.png"))

# # plot the seed length as histogram
# fig, ax = plt.subplots(figsize=(7, 7))
# # draw y axis as percentage instead of count
# ax.hist(utr_data['seed_length'], bins=range(0,50), alpha=0.5, label='Seed Length', density=True, color=color_utr)
# ax.tick_params(labelsize=12)  # set x-tick label font size
# ax.set_xlabel('Seed Length', fontsize=15)
# ax.set_ylabel('Ratio', fontsize=15)
# # ax.set_title('Seed Length Distribution', fontsize=20)
# ax.legend(fontsize=12)
# fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/seed_length_distribution_utr.png"))

# # plot distance to cleavage site as histogram
# fig, ax = plt.subplots(figsize=(7, 7))
# # draw y axis as percentage instead of count
# ax.hist(test_500_data['distance_to_cleavage_site'], bins=range(0,100), alpha=0.5, label='Distance to Cleavage Site', density=True, color=color_test_500)
# ax.tick_params(labelsize=12)  # set x-tick label font size
# ax.set_xlabel('Distance to Cleavage Site', fontsize=15)
# ax.set_ylabel('Ratio', fontsize=15)
# # ax.set_title('Distance to Cleavage Site Distribution', fontsize=20)
# ax.legend(fontsize=12)
# fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_500_test.png"))

# # plot distance to cleavage site as histogram
# fig, ax = plt.subplots(figsize=(7, 7))
# # draw y axis as percentage instead of count
# ax.hist(utr_data['distance_to_cleavage_site'], bins=range(0,100), alpha=0.5, label='Distance to Cleavage Site', density=True, color=color_utr)
# ax.tick_params(labelsize=12)  # set x-tick label font size
# ax.set_xlabel('Distance to Cleavage Site', fontsize=15)
# ax.set_ylabel('Ratio', fontsize=15)
# # ax.set_title('Distance to Cleavage Site Distribution', fontsize=20)
# ax.legend(fontsize=12)
# fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_distribution_utr.png"))

# # filter seed length to only 6, 7 and 8
# order = [6, 7, 8]
# valid_data_test_500 = test_500_data[test_500_data['seed_length'].isin(order)].copy()
# valid_data_utr = utr_data[utr_data['seed_length'].isin(order)].copy()

# # Prepare data for boxplot in fixed order
# data_to_plot_test_500 = [
#     valid_data_test_500.loc[valid_data_test_500['seed_length'] == k, 'distance_to_cleavage_site'].dropna()
#     for k in order
# ]
# data_to_plot_utr = [
#     valid_data_utr.loc[valid_data_utr['seed_length'] == k, 'distance_to_cleavage_site'].dropna()
#     for k in order
# ]
# # Plot
# labels = [f"{k}-mer" for k in order]
# n = len(order)

# # X positions for grouped boxplots
# x = np.arange(1, n + 1)           # centers: 1, 2, 3
# offset = 0.18                     # half the gap between side-by-side boxes
# pos_a = x - offset                # first dataset (e.g., test_500)
# pos_b = x + offset                # second dataset (e.g., utr)
# width = 0.32

# fig, ax = plt.subplots(figsize=(8, 6))

# bp_a = ax.boxplot(
#     data_to_plot_test_500,
#     positions=pos_a,
#     widths=width,
#     showfliers=False,
#     patch_artist=True,
# )
# bp_b = ax.boxplot(
#     data_to_plot_utr,
#     positions=pos_b,
#     widths=width,
#     showfliers=False,
#     patch_artist=True,
# )

# # (Optional) simple fill so they’re distinguishable
# for patch in bp_a['boxes']:
#     patch.set_facecolor(color_test_500)
# for patch in bp_b['boxes']:
#     patch.set_facecolor(color_utr)

# # Center ticks under each pair
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=12)

# ax.set_xlabel("Seed length", fontsize=15)
# ax.set_ylabel("Distance to predicted cleavage site (nt)", fontsize=15)
# # ax.set_title("Distance to cleavage vs. seed length (grouped)", fontsize=20)
# ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# # Legend using proxy patches
# handles = [
#     Patch(facecolor=color_test_500, label='test_500'),
#     Patch(facecolor=color_utr, edgecolor='black', label='UTR'),
# ]
# ax.legend(handles=handles, loc='best')
# fig.savefig(os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/distance_to_cleavage_site_boxplot.png"))