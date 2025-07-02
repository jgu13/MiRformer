import os
import time
import csv
import random
import gzip
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
predicted_targets_f = "Mouse_Predicted_Targets_Context_Scores.default_predictions.txt.zip"

# positive miRNA and mRNA pairs
# path = os.path.join(data_dir, predicted_targets_f)
# predicted_targets = pd.read_csv(path, sep='\t', compression="zip")
# print(predicted_targets["Site Type"].unique())
# print(predicted_targets.loc[(predicted_targets["miRNA"]=="hsa-miR-145-5p") & (predicted_targets["Transcript ID"]=="ENST00000532642.1")])
# filter for human (9606), chimpanzee (9598), mouse (10090)
# tax_ids = [10090]
# top_predicted_targets = predicted_targets[
#     (predicted_targets["Gene Tax ID"].isin(tax_ids)) &
#     (predicted_targets["context++ score percentile"] >= np.int64(80)) # top 20% likely pairs
#     ]
# # filter out non-canonical sites
# top_predicted_targets = top_predicted_targets.loc[~top_predicted_targets["Site Type"].isin([-2,-3])]

# positive_pairs = top_predicted_targets[[
#      "miRNA",
#      "Transcript ID",
#      "UTR_start",
#      "UTR_end"
# ]]
# positive_pairs.loc[:, "label"] = 1
# positive_pairs.columns = ["miRNA", "Transcript_ID", "UTR_start", "UTR_end", "label"]
# positive_pairs.to_csv(os.path.join(data_dir, "Positive_pairs_human.csv"), sep='\t', index=False)

# print("Total predicted mirna-transcript pairs = ", len(positive_pairs))

# # negative miRNA and mRNA pairs: select mRNA species that is not in positive pairs with the miRNA
# all_mrnas       = set(predicted_targets["Transcript ID"].unique())
# print("Start generating an equal number of negative samples.")
# with open(os.path.join(data_dir, "negative_pairs_human.csv"), "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["miRNA", "Transcript_ID", "UTR_start", "UTR_end", "label"], delimiter="\t")
#     writer.writeheader()
#     for mirna, group in positive_pairs.groupby("miRNA"):
#         pos_set       = set(group['Transcript_ID'].tolist())
#         n_pos         = len(group['Transcript_ID'].tolist())
#         neg_mrna_pool = list(all_mrnas - pos_set)
#         if len(neg_mrna_pool) < n_pos:
#             raise ValueError(ValueError(f"Warning: {mirna}: Pool of negative mrna ({len(neg_mrna_pool)}) is fewer than positive mrnas ({n_pos})!"))
#         chosen_neg = random.sample(neg_mrna_pool, k=n_pos)
#         for mrna in chosen_neg:
#             writer.writerow(
#                 {"miRNA":        mirna,
#                 "Transcript_ID": mrna,
#                 "UTR_start":     -1,
#                 "UTR_end":       -1,
#                 "label":         0}
#             )

# print("Finished generating negative samples")

# Add bottom 30% to negative samples
# tax_ids = [9606]
# bot_predicted_targets = predicted_targets[
#     (predicted_targets["Gene Tax ID"].isin(tax_ids)) &
#     (predicted_targets["context++ score percentile"] <= np.int64(30)) # bottom 30% likely pairs
#     ]

# negative_pairs = bot_predicted_targets[[
#     "miRNA",
#     "Transcript ID"
# ]].copy()

# print("Number of rows to add: ", len(negative_pairs))
# out_path = os.path.join(data_dir, "negative_pairs_human.csv")
# with open(out_path, "a", newline="") as f:
#     writer = csv.DictWriter(
#         f,
#         fieldnames=["miRNA", "Transcript_ID", "UTR_start", "UTR_end","label"],
#         delimiter="\t"
#     )
#     # don't write header, just append rows
#     for _, row in negative_pairs.iterrows():
#         writer.writerow({
#             "miRNA":         row["miRNA"],
#             "Transcript_ID": row["Transcript ID"],
#             "UTR_start":     -1,
#             "UTR_end":       -1,
#             "label":         0
#         })

# print(f"Samples are saved to {data_dir}")

# mRNAseq_path = os.path.join(data_dir, "mouse_3utr_sequences.fa.gz")
# mRNA_seq_dict = []
# with gzip.open(mRNAseq_path, "rt") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         # full UTR sequence
#         seq = str(record.seq)
#         # split off the coords, keep only the transcript ID
#         tran_id, _coords = record.id.split("::", 1)
#         # if you only need transcript → sequence mapping:
#         mRNA_seq_dict.append({
#             "Transcript ID": tran_id,
#             "mRNA sequence": seq
#         })
                
# mRNA_seq_df = pd.DataFrame(mRNA_seq_dict)
# print(mRNA_seq_df.head(n=10))
# mRNA_seq_df.to_csv(
#     os.path.join(os.path.join(data_dir, "mouse_mrna_seq.csv.gz")), 
#     sep='\t', 
#     index=False, 
#     compression='gzip')

# mRNA_df_path = os.path.join(data_dir, "mouse_3utr_sequences.txt.zip")
# utr_df = pd.read_csv(mRNA_df_path, sep='\t', compression='zip')
# # filter for mouse
# species_ids = [10090]
# utr_df = utr_df[utr_df['Species ID'].isin(species_ids)]
# def clean_seq(s):
#     return s.replace("-", "").upper().replace("U", "T")
# utr_df["UTR sequence"] = utr_df["UTR sequence"].apply(clean_seq)
# utr_df.columns = ['Transcript ID', 'Gene ID', 'Gene Symbol', 'Species ID', 'mRNA sequence']
# mrna_save_path = os.path.join(data_dir, "mouse_mrna_seq.csv.gz")
# utr_df.to_csv(mrna_save_path,
#               sep='\t',
#               index=False,
#               compression='gzip')

mRNA_df_path = os.path.join(data_dir, "human_utr_sequences.txt.zip")
utr_df = pd.read_csv(mRNA_df_path, sep='\t', compression='zip')
# filter for mouse
species_ids = [9606]
utr_df = utr_df[utr_df['Species ID'].isin(species_ids)]
def clean_seq(s):
    return s.replace("-", "").upper().replace("U", "T")
utr_df["UTR sequence"] = utr_df["UTR sequence"].apply(clean_seq)
utr_df.columns = ['Transcript ID', 'Gene ID', 'Gene Symbol', 'Species ID', 'mRNA sequence']
mrna_save_path = os.path.join(data_dir, "human_mrna_seq.csv.gz")
utr_df.to_csv(mrna_save_path,
              sep='\t',
              index=False,
              compression='gzip')