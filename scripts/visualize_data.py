import pandas as pd
import numpy as np
import os
import gzip
from Bio import SeqIO
import json
from itertools import islice

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_path = os.path.join(PROJ_HOME, "data")

hsa_MTI_path = os.path.join(data_path, "hsa_MTI.csv.gz")
hsa_MTI = pd.read_csv(hsa_MTI_path, compression="gzip")
# print("hsa_MTI")
# print(hsa_MTI.head())

mRNAseq_path = os.path.join(data_path, "human_3utr.fa.gz")
mirna_path = os.path.join(data_path, "mature.fa.gz")

# _______________________________
# Parse miRNA sequences
# _______________________________
# Initialize a dictionary to hold miRNA sequences
# miRNA_seq_dict = {}
# num_record_to_iter = 150
# # Parse the miRNA FASTA file
# with gzip.open(mirna_path, "rt") as handle:
#     for record in islice(SeqIO.parse(handle, "fasta"), num_record_to_iter):
#         # Extract miRNA ID from the record (e.g., 'hsa-miR-20a-5p')
#         miRNA_id = record.name.split()[0]  # In case there are spaces
#         if miRNA_id.startswith("hsa-miR"):  # only store miRNAseq of homo sapiens
#             # check for duplicates
#             if miRNA_id not in miRNA_seq_dict:
#                 miRNA_seq = str(record.seq)
#                 if miRNA_seq != "Sequenceunavailable":
#                     # Store the sequence
#                     miRNA_seq_dict[miRNA_id] = miRNA_seq.replace("U", "T")
#                 else:
#                     print(f"Sequence for miRNA [{miRNA_id}] is not available.")
#             else:
#                 print(
#                     f"Sequence for miRNA [{miRNA_id}] is duplicated. Keeping the existing sequence."
#                 )

# _______________________
# Parse mRNA sequences
# _______________________
# mRNA_seq_dict = {}

# num_record_to_iter = 5
# with gzip.open(mRNAseq_path, "rt") as handle:
#     for record in islice(SeqIO.parse(handle, "fasta"), num_record_to_iter):
#         # Extract gene symbol name for the record (e.g.: 'CTAGE4')
#         gene_symbol = record.id.split("|")[2]
#         if gene_symbol not in mRNA_seq_dict:
#             mRNA_seq = str(record.seq)
#             if mRNA_seq != "Sequenceunavailable":
#                 # Store the sequence
#                 mRNA_seq_dict[gene_symbol] = mRNA_seq
#             else:
#                 print(f"Transcript for gene [{gene_symbol}] is not available.")
#         else:
#             print(
#                 f"Transcript for gene [{gene_symbol}] is duplicated. Keeping the existing sequence."
#             )


# _______________________________________
# Parse mRNA sequence to specifc length
# _______________________________________
mRNA_length = 5000
mRNA_seq_dict = {}

# Store a 500nt version of mRNA for fast experiments
with gzip.open(mRNAseq_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        # Extract gene symbol name for the record (e.g.: 'CTAGE4')
        gene_symbol = record.id.split("|")[2]
        if gene_symbol not in mRNA_seq_dict:
            mRNA_seq = str(record.seq)
            if mRNA_seq != "Sequenceunavailable":
                # Store the sequence
                mRNA_seq_dict[gene_symbol] = mRNA_seq[-mRNA_length:] # get sequence from the 3' UTR end
            else:
                print(f"Transcript for gene [{gene_symbol}] is not available.")
        else:
            print(
                f"Transcript for gene [{gene_symbol}] is duplicated. Keeping the existing sequence."
            )

# with open(os.path.join(data_path, "miRNA dictionary.json"), "w") as fp:
#     json.dump(miRNA_seq_dict, fp, indent=4)

# with open(os.path.join(data_path, "mRNA dictionary.json"), "w") as fp:
#     json.dump(mRNA_seq_dict, fp, indent=4)

save_path = os.path.join(data_path, f"mRNA {mRNA_length} dictionary reverse.json")
with open(save_path, "w", encoding='utf-8') as fp:
    json.dump(mRNA_seq_dict, fp, indent=4)
print(f"Saved mRNA sequence of {mRNA_length} max length to {save_path}.")

def build_dmiso_dataset(MTI_path, mRNAseq_path, miRNAseq_path):
    """
    Build dataset for DMISO
    """
    # get mti dataframe
    hsa_mti = pd.read_csv(hsa_MTI_path, compression="gzip")
    hsa_mti_unique = hsa_mti[
        ["miRTarBase ID", "miRNA", "Target Gene"]
    ].drop_duplicates()

    # get miRNA name and sequences
    miRNA_seq_dict = {"miRNA": [], "miRNA seq": []}
    with gzip.open(mirna_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Extract miRNA ID from the record (e.g., 'hsa-miR-20a-5p')
            miRNA_id = record.name.split()[0]  # In case there are spaces
            if miRNA_id.startswith("hsa-miR"):  # only store miRNAseq of homo sapiens
                # check for duplicates
                if miRNA_id not in miRNA_seq_dict["miRNA"]:
                    miRNA_seq = str(record.seq)
                    if miRNA_seq != "Sequenceunavailable":
                        # Store the sequence
                        miRNA_seq_dict["miRNA"].append(miRNA_id)  # append miRNA name
                        miRNA_seq_dict["miRNA seq"].append(
                            miRNA_seq
                        )  # append miRNA sequence
                    else:
                        print(f"Sequence for miRNA [{miRNA_id}] is not available.")
    miRNA_df = pd.DataFrame.from_dict(miRNA_seq_dict)

    # get mRNA id, name and sequence
    mRNA_seq_dict = {"mRNA id": [], "Target Gene": [], "mRNA seq": []}
    with gzip.open(mRNAseq_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Extract gene id and symbol name for the record
            gene_id = "|".join(record.id.split("|")[:4])
            gene_symbol = record.id.split("|")[2]
            if gene_symbol not in mRNA_seq_dict["Target Gene"]:
                mRNA_seq = str(record.seq)
                if mRNA_seq != "Sequenceunavailable":
                    # Store the sequence
                    mRNA_seq_dict["mRNA id"].append(gene_id)
                    mRNA_seq_dict["Target Gene"].append(gene_symbol)
                    mRNA_seq_dict["mRNA seq"].append(mRNA_seq)
                else:
                    print(f"Transcript for gene [{gene_symbol}] is not available.")
            else:
                print(
                    f"Transcript for gene [{gene_symbol}] is duplicated. Keeping the existing sequence."
                )
    mRNA_df = pd.DataFrame.from_dict(mRNA_seq_dict)

    # combine hsa_mti and miRNA seq and mRNA
    df = hsa_mti_unique
    df = pd.merge(df, miRNA_df, on="miRNA", how="left")
    df = pd.merge(df, mRNA_df, on="Target Gene", how="left")
    df["miRNA id"] = df["miRNA"] + "|" + df["miRTarBase ID"]

    # generate equal size of negative samples by shuffling miRNA and mRNA pairs
    hsa_mti_shuffled = hsa_mti_unique.copy()
    hsa_mti_shuffled["Target Gene"] = np.random.permutation(
        hsa_mti_unique["Target Gene"].values
    )

    df2 = hsa_mti_shuffled
    df2 = pd.merge(df2, miRNA_df, on="miRNA", how="left")
    df2 = pd.merge(df2, mRNA_df, on="Target Gene", how="left")
    df2["miRNA id"] = df2["miRNA"] + "|" + df2["miRTarBase ID"]

    # combine positvie and negative pairs
    comb_df = pd.concat([df, df2], axis=0, ignore_index=True)
    # print(len(comb_df))
    comb_df[["miRNA id", "mRNA id", "miRNA seq", "mRNA seq"]].dropna().to_csv(
        "dmiso_pairs.txt.gz", sep="\t", header=False, index=False
    )


# build_dmiso_dataset(MTI_path=hsa_MTI_path, mRNAseq_path=mRNAseq_path, miRNAseq_path=mirna_path)
