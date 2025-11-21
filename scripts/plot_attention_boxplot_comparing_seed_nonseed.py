import os
from re import M
import torch
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import transformer_model as tm
import matplotlib.patches as patches
from matplotlib import colors
from scipy.stats import mannwhitneyu
import mpmath as mp
import numpy as np
from Data_pipeline import CharacterTokenizer

from Global_parameters import PROJ_HOME, TICK_FONT_SIZE, AXIS_FONT_SIZE, LEGEND_FONT_SIZE

from plot_transformer_heatmap import load_model, encode_seq, predict

# plot boxplots showing the distribution of attention weights within seed region and outside the seed region
def plot_attention_weights_boxplot(
                 seed_weights_path,
                 non_seed_weights_path,
                 figsize=(3.4, 5),
                 save_attention_weights_boxplot_dir=None):

    seed_weights = torch.load(seed_weights_path)
    non_seed_weights = torch.load(non_seed_weights_path)
    # do mann whitney u test
    print("Mann-Whitney u test between seed and non-seed:")
    result = mannwhitneyu(seed_weights.flatten(), non_seed_weights.flatten(), alternative="two-sided", method="asymptotic")
    print(f"statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")
    p_value = result.pvalue

    # plot the boxplots
    fig, ax = plt.subplots(figsize=figsize)
    positions = [0.25, 0.65]
    ax.set_yscale("log")
    boxplots = ax.boxplot(
        [seed_weights.flatten(), non_seed_weights.flatten()],
        positions=positions,
        widths=0.4,
        patch_artist=True  # allows facecolor customization
    )
    # xticklabels = ["Global", "Mean Unchunked", "Norm by Query", "Norm by Key"]
    # ax.set_xticks(group_centers)          # 4 tick locations
    # ax.set_xticklabels(xticklabels)       # 4 labels
    ax.set_xticks([])
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    
    colors = ["#2a9d8f", "#e76f51"]
    for patch, color in zip(boxplots["boxes"], colors):
        patch.set(facecolor=color, alpha=0.8)
    ax.set_ylabel("Attention Weights", fontsize=AXIS_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_facecolor("whitesmoke")

    # annotate the p value
    # expand y-limits to leave vertical space for the text
    ymin, ymax = ax.get_ylim()
    y_text = ymax * 1.5          # factor > 1 adds headroom on log scale
    ax.set_ylim(ymin, y_text)

    ax.text(
        positions[0]+0.2, y_text,           # x in data coords, y above boxes
        f"p = {p_value:.4e}",
        ha="center", va="bottom",
        fontsize=TICK_FONT_SIZE,
        clip_on=False
    )

    if save_attention_weights_boxplot_dir is not None:
        file_path = os.path.join(save_attention_weights_boxplot_dir, f"attention_weights_boxplot_norm_by_key.svg")     
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights boxplot saved to {file_path}")
    else:
        file_path = os.path.join(os.getcwd(), f"attention_weights_boxplot_norm_by_key.svg")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights boxplot saved to {file_path}")
    return fig, ax

def plot_attention_weights_boxplots_comparison(
    miRNA_id,
    mRNA_id,
    global_seed_weights_path,
    global_non_seed_weights_path,
    mean_unchunked_seed_weights_path,
    mean_unchunked_non_seed_weights_path,
    norm_by_query_seed_weights_path,
    norm_by_query_non_seed_weights_path,
    norm_by_key_seed_weights_path,
    norm_by_key_non_seed_weights_path,
    figsize=(12, 6),
    save_attention_weights_boxplot_dir=None,
    p_values=None
):
    global_seed_weights = torch.load(global_seed_weights_path)
    global_non_seed_weights = torch.load(global_non_seed_weights_path)
    mean_unchunked_seed_weights = torch.load(mean_unchunked_seed_weights_path)
    mean_unchunked_non_seed_weights = torch.load(mean_unchunked_non_seed_weights_path)
    norm_by_query_seed_weights = torch.load(norm_by_query_seed_weights_path)
    norm_by_query_non_seed_weights = torch.load(norm_by_query_non_seed_weights_path)
    norm_by_key_seed_weights = torch.load(norm_by_key_seed_weights_path)
    norm_by_key_non_seed_weights = torch.load(norm_by_key_non_seed_weights_path)

    # plot the boxplots
    fig, ax = plt.subplots(figsize=figsize)
    # place the boxplots so that each two are grouped together
    # number of groups
    groups = 4
    # each group contains two distributions: seed vs non-seed
    group_centers = np.arange(groups) * 1.0
    offset = 0.2

    positions = []
    for c in group_centers:
        positions.extend([c - offset, c + offset])
    ax.set_yscale("log")
    boxplots = ax.boxplot(
        [global_seed_weights.flatten(), 
        global_non_seed_weights.flatten(), 
        mean_unchunked_seed_weights.flatten(), 
        mean_unchunked_non_seed_weights.flatten(), 
        norm_by_query_seed_weights.flatten(), 
        norm_by_query_non_seed_weights.flatten(), 
        norm_by_key_seed_weights.flatten(), 
        norm_by_key_non_seed_weights.flatten()],
        positions=positions,
        widths=0.4,
        patch_artist=True,  # allows facecolor customization
        # set boxline width to 2
        boxprops=dict(linewidth=2)
    )
    # xticklabels = ["Global", "Mean Unchunked", "Norm by Query", "Norm by Key"]
    # ax.set_xticklabels(xticklabels)
    ax.set_xticks([])
    colors = ["#2a9d8f", "#e76f51"] * 4
    for patch, color in zip(boxplots["boxes"], colors):
        patch.set(facecolor=color, alpha=0.8)
    ax.set_ylabel("Attention Weights", fontsize=AXIS_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    ax.set_facecolor("whitesmoke")

    # annotate the p values for each group
        # ---------- annotate p-values ----------
    if p_values is not None:
        # expand y-limits to leave vertical space for the text
        ymin, ymax = ax.get_ylim()
        y_text = ymax * 1.5          # factor > 1 adds headroom on log scale
        ax.set_ylim(ymin, y_text)

        p_keys = list(p_values.keys())
        for i in range(groups):
            p_val = p_values[p_keys[i]]
            ax.text(
                group_centers[i], y_text,           # x in data coords, y above boxes
                f"p = {p_val}",
                ha="center", va="bottom",
                fontsize=TICK_FONT_SIZE,
                clip_on=False
            )

    if save_attention_weights_boxplot_dir is not None:
        file_path = os.path.join(save_attention_weights_boxplot_dir, f"{miRNA_id}_{mRNA_id}_attention_weights_boxplot_comparison.svg")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights boxplot comparison saved to {file_path}")
    else:
        file_path = os.path.join(os.getcwd(), f"{miRNA_id}_{mRNA_id}_attention_weights_boxplot_comparison.svg")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights boxplot comparison saved to {file_path}")
    return fig, ax

def main():
    mirna_max_len   = 24
    mrna_max_len    = 520
    predict_span    = True
    predict_binding = True
    device          = "cuda:0" 
    args_dict = {"mirna_max_len": mirna_max_len,
                 "mrna_max_len": mrna_max_len,
                 "device": device,
                 "embed_dim": 1024,
                 "num_heads": 8,
                 "num_layers": 4,
                 "ff_dim": 4096,
                 "predict_span": predict_span,
                 "predict_binding": predict_binding,
                 "use_longformer":True}
    
    data_dir = os.path.join(PROJ_HOME, 'TargetScan_dataset')
    test_datapath = os.path.join(PROJ_HOME, data_dir, 
                                 "TargetScan_train_500_randomized_start.csv")
    test_data  = pd.read_csv(test_datapath, sep=',')
    positive_samples = test_data[test_data["label"] == 1]
    np.random.seed(10020)
    sample_ids = np.random.choice(len(positive_samples), 200, replace=False)
    mRNA_seqs  = positive_samples[["mRNA sequence"]].values
    miRNA_seqs = positive_samples[["miRNA sequence"]].values
    miRNA_IDs   = positive_samples[["miRNA ID"]].values
    mRNA_IDs    = positive_samples[["Transcript ID"]].values
    labels      = positive_samples[["label"]].values
    seed_starts = positive_samples[["seed start"]].values
    seed_ends   = positive_samples[["seed end"]].values
    
    ckpt_path = os.path.join(PROJ_HOME, "checkpoints/TargetScan/TwoTowerTransformer/Longformer/520/embed=1024d/norm_by_key/LSE/50k_best_composite_0.7424_0.9106_epoch18.pth")
    model = load_model(ckpt_path, **args_dict)

    seed_weights_list = []
    non_seed_weights_list = []
    for i in sample_ids:
        mRNA_seq  = mRNA_seqs[i][0]
        miRNA_seq = miRNA_seqs[i][0]
        mRNA_ID   = mRNA_IDs[i][0]
        miRNA_ID  = miRNA_IDs[i][0]
        label     = labels[i][0]
        seed_start = seed_starts[i][0]
        seed_end   = seed_ends[i][0]
        # replace U with T and reverse miRNA to 3' to 5' 
        miRNA_seq = miRNA_seq.replace("U", "T")[::-1]
        # create tokenizer
        tokenizer = CharacterTokenizer(
            characters=["A", "T", "C", "G", "N"],  # add RNA characters, N is uncertain
            model_max_length=model.mrna_max_len,
            padding_side="right",  # since HyenaDNA is causal, we pad on the left
        )
        encoded = encode_seq(
            model=model, 
            tokenizer=tokenizer,
            mRNA_seq=mRNA_seq,
            miRNA_seq=miRNA_seq,
        )
        # convert mRNA seq tokens to mRNA ids
        mRNA_tokens_squeezed = encoded["mRNA_seq"].squeeze(0)
        mRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in mRNA_tokens_squeezed]
        # convert miRNA seq tokens to mRNA ids
        miRNA_tokens_squeezed = encoded["miRNA_seq"].squeeze(0)
        miRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in miRNA_tokens_squeezed]

        model.to(args_dict["device"])

        print("Model Prediction ...")
        predict(
                model=model,
                mRNA_seq=encoded["mRNA_seq"],
                miRNA_seq=encoded["miRNA_seq"],
                mRNA_seq_mask=encoded["mRNA_seq_mask"],
                miRNA_seq_mask=encoded["miRNA_seq_mask"],
                seed_start=seed_start,
                seed_end=seed_end,
        )

        # save seed weights and non-seed weights to be easily loaded later
        attn_weights = model.predictor.cross_attn_layer.last_attention
        attn_weights = torch.amax(attn_weights[0], dim=0) # (mrna, mirna)
        # get the attention weights within seed region and outside the seed region
        seed_weights = attn_weights[seed_start:seed_end, :]
        # start = max(0, seed_start - 50)
        # end = min(len(mRNA_seq), seed_end + 50)
        non_seed_weights = torch.cat([attn_weights[0:seed_start, :], attn_weights[seed_end+1:, :]], dim=0)
        seed_weights_list.append(seed_weights.flatten().mean())
        non_seed_weights_list.append(non_seed_weights.flatten().mean())
        print(f"Seed weights and non-seed weights saved for {miRNA_ID}_{mRNA_ID}")

    # flatten all the seed weights
    seed_weights = torch.tensor(seed_weights_list)
    non_seed_weights = torch.tensor(non_seed_weights_list)
    save_plot_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test", "TwoTowerTransformer", str(mrna_max_len))
    seed_weights_path = os.path.join(save_plot_dir, f"norm_by_key_seed_weights.pt")
    non_seed_weights_path = os.path.join(save_plot_dir, f"norm_by_key_non_seed_weights.pt")
    torch.save(seed_weights, seed_weights_path)
    torch.save(non_seed_weights, non_seed_weights_path)
    print(f"Seed weights and non-seed weights saved to {seed_weights_path} and {non_seed_weights_path}")
    
    # plot boxplots showing the distribution of attention weights within seed region and outside the seed region
    save_boxplot_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test", "TwoTowerTransformer", str(520))
    os.makedirs(save_boxplot_dir, exist_ok=True)
    seed_weights_path = os.path.join(save_boxplot_dir, f"norm_by_key_seed_weights.pt")
    non_seed_weights_path = os.path.join(save_boxplot_dir, f"norm_by_key_non_seed_weights.pt")

    plot_attention_weights_boxplot(
                 seed_weights_path=seed_weights_path,
                 non_seed_weights_path=non_seed_weights_path,
                 save_attention_weights_boxplot_dir=save_boxplot_dir)

if __name__ == '__main__':
    main()
 