import pandas as pd
import os

# read miraw dataset 1 and 2
file1 = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "data_miRaw_noL_noMisMissing_remained_seed1122.txt")
df1 = pd.read_csv(file1,  sep="\t") 
file2 = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "data_miRaw_IndTest_noRepeats_3folds.txt")
df2 = pd.read_csv(file2,  sep="\t", names=["mir_id", "miRNA sequence", "gene id", "mRNA sequence", "label"])

# merge df1 and df2
df = pd.concat([df1, df2], axis=0, ignore_index=True)
df = df.drop_duplicates(subset=["miRNA sequence", "mRNA sequence"]) # remove duplicated samples
print(df.head())

# generate negative samples by shuffling mRNA seqs
negative_mRNA = df["mRNA sequence"].sample(frac=1, random_state=42).reset_index(drop=True)

# concatenate negative mRNA samples with the positive
full_mRNA = pd.concat([df["mRNA sequence"], negative_mRNA], axis=0).reset_index(drop=True)
full_miRNA = pd.concat([df["miRNA sequence"], df["miRNA sequence"]], axis=0).reset_index(drop=True)
labels = pd.Series([1] * len(df) + [0] * len(df), name='label').reset_index(drop=True)

# create new dataframe
final_df = pd.DataFrame({"miRNA sequence": full_miRNA,
                    "mRNA sequence": full_mRNA,
                    "label": labels})

# save full dataset
save_to_path = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "miRAW_dataset.csv")
print(f"Saving miRAW dataset to {save_to_path}")
final_df.to_csv(save_to_path, sep=",", index=False)