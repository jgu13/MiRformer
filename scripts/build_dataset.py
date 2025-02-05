import pandas as pd
import numpy as np
import os
import gzip
from Bio import SeqIO
import json
import random

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_path = os.path.join(PROJ_HOME, "data")

# ____________________________________
# get postive pairs of miRNA and mRNA
# ____________________________________
hsa_MTI_path = os.path.join(data_path, "hsa_MTI.csv.gz")
hsa_MTI = pd.read_csv(hsa_MTI_path, compression="gzip")
# Extract miRNA and Target Gene columns and drop duplicates
positive_pairs = hsa_MTI[["miRNA", "Target Gene"]].drop_duplicates()
# Convert to a set of tuples for faster lookup
positive_pairs_set = set([tuple(x) for x in positive_pairs.values])
print("Number of positive pairs = ", len(positive_pairs_set))

# missing_pairs = {
#     "miRNA id": [],
#     "mRNA symbol": [],
#     "miRNA sequence": [],
#     "mRNA sequence": [],
# }

# _____________________________
# Generate positive pairs
# _____________________________
mRNA_length = 1000
with open(os.path.join(data_path, "miRNA dictionary.json"), "r") as fp:
    miRNA_seq_dict = json.load(fp)
with open(os.path.join(data_path, f"mRNA {mRNA_length} dictionary.json"), "r") as fp:
    mRNA_seq_dict = json.load(fp)

positive_samples = []
print("Generate positive samples: ")
for miRNA_id, mRNA_symbol in positive_pairs_set:
    miRNA_seq_ls = miRNA_seq_dict.get(miRNA_id)
    mRNA_seq_ls = mRNA_seq_dict.get(mRNA_symbol)

    if miRNA_seq_ls and mRNA_seq_ls:
        for miRNA_seq in miRNA_seq_ls:
            for mRNA_seq in mRNA_seq_ls:
                # Append to positive samples
                positive_samples.append(
                    {
                        "miRNA": miRNA_id,
                        "mRNA": mRNA_symbol,
                        "miRNA sequence": miRNA_seq,
                        "mRNA sequence": mRNA_seq,
                        "label": 1,  # Positive label
                    }
                )


# ________________________
# Generate negative pairs
# ________________________
# Get all possible miRNA and mRNA IDs
all_miRNAs = list(miRNA_seq_dict.keys())
all_mRNAs = list(mRNA_seq_dict.keys())

# Calculate the number of negative samples desired
num_negative_samples = len(positive_samples)
# Initialize a set to keep track of negative pairs
negative_pairs_set = set()

# Randomly sample negative pairs
while len(negative_pairs_set) < num_negative_samples:
    miRNA = random.choice(all_miRNAs)
    mRNA = random.choice(all_mRNAs)
    pair = (miRNA, mRNA)
    if pair not in positive_pairs_set and pair not in negative_pairs_set:
        negative_pairs_set.add(pair)

# Create negative samples
negative_samples = []
print("Generate negative samples: ")
for miRNA_id, gene_symbol in negative_pairs_set:
    miRNA_seq_ls = miRNA_seq_dict.get(miRNA_id)
    mRNA_seq_ls = mRNA_seq_dict.get(gene_symbol)
    if miRNA_seq_ls and mRNA_seq_ls:
        negative_samples.append(
            {
                "miRNA": miRNA_id,
                "mRNA": gene_symbol,
                "miRNA sequence": miRNA_seq_ls[0],
                "mRNA sequence": mRNA_seq_ls[0],
                "label": 0,  # Negative label
            }
        )
    else:
        continue

# Convert samples to dataframes
positive_df = pd.DataFrame(positive_samples)
negative_df = pd.DataFrame(negative_samples)

# Combine the dataframes
training_set = pd.concat([positive_df, negative_df], ignore_index=True)

# save to csv
print("Save to csv file: ")
training_set.to_csv(
    os.path.join(data_path, f"training_{mRNA_length}.csv"), sep=",", index=False
)
