# split the TargetScan_dataset/Merged_primates_test_500_randomized_start.csv using sckit_learn train_test_split into two files: 20k and the rest
# split the 20k into two files: 10k and 10k using sckit_learn train_test_split
# shuffle the pairs of miRNA and mRNA of one of the10k and concatenate them with the other 10k
# save as TargetScan_dataset/Merged_primates_finetune_20k.csv

import pandas as pd
import os
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
ts_data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
deg_data_dir = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data")
df = pd.read_csv(os.path.join(ts_data_dir, "TargetScan_train_500_randomized_start_random_samples.csv"))
# big_df, small_df = train_test_split(df, train_size=20000, test_size=511, random_state=42)
# df1, df2 = train_test_split(big_df, train_size=10000, test_size=10000, random_state=42)
# print("big_df: ", len(big_df))
# print("small_df: ", len(small_df))
# print("df1: ", len(df1))
# print("df2: ", len(df2))

# shuffled_mRNA = df1[["Transcript ID","mRNA sequence"]].sample(frac=1, random_state=42)
# shuffled_miRNA = df1[["miRNA ID","miRNA sequence"]].sample(frac=1, random_state=42)
# df1["mRNA sequence"] = shuffled_mRNA["mRNA sequence"]
# df1["Transcript_ID"] = shuffled_mRNA["Transcript ID"]
# df1["miRNA ID"] = shuffled_miRNA["miRNA ID"]
# df1["miRNA sequence"] = shuffled_miRNA["miRNA sequence"]
# df1["label"] = 0 # set label to 0
# df = pd.concat([df1, df2], axis=0, ignore_index=True)
# print(f"df: {len(df)}")
# df.to_csv(os.path.join(ts_data_dir, "Merged_primates_finetune.csv"), index=False)
# small_df.to_csv(os.path.join(ts_data_dir, "Merged_primates_finetune_test.csv"), index=False)

# sample 20,512 lines from miR_degradome_ago_clip_pairing_data/starBase_degradome_windows_500.tsv
# df = pd.read_csv(os.path.join(deg_data_dir, "starBase_degradome_windows_500.tsv"), sep="\t")
# df = df.sample(n=209000, random_state=42)
# df1, df2 = train_test_split(df, train_size=208000, test_size=1000, random_state=42)
# print("df1: ", len(df1))
# print("df2: ", len(df2))
# # save df1 as starbase_degradome_windows_finetune.tsv
# df1.to_csv(os.path.join(deg_data_dir, "starbase_degradome_windows_train.tsv"), index=False, sep="\t")
# # save df2 as starbase_degradome_windows_test.tsv
# df2.to_csv(os.path.join(deg_data_dir, "starbase_degradome_windows_validation.tsv"), index=False, sep="\t")