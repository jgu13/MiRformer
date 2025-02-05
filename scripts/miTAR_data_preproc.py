import pandas as pd
import os
import random

def generate_random_string(min_length, max_length):
    length = random.randint(min_length, max_length)
    letters = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(letters, k=length))

# read miraw dataset 1 and 2
file1 = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "data_miRaw_noL_noMisMissing_remained_seed1122.txt")
df1 = pd.read_csv(file1,  sep="\t") 
file2 = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "data_miRaw_IndTest_noRepeats_3folds.txt")
df2 = pd.read_csv(file2,  sep="\t", names=["mir_id", "miRNA sequence", "gene id", "mRNA sequence", "label"])

# merge df1 and df2
df = pd.concat([df1, df2], axis=0, ignore_index=True)
df = df.drop_duplicates(subset=["miRNA sequence", "mRNA sequence"]) # remove duplicated samples

# min_miRNA = 17, max_miRNA = 26
# generate negative samples by shuffling mRNA seqs
# negative_mRNA = df["mRNA sequence"].sample(frac=1, random_state=42).reset_index(drop=True)

# generate negative samples with randomly-ordered bases
negative_mRNA = [generate_random_string(min_length=40, max_length=40) for _ in range(len(df))]
negative_miRNA = [generate_random_string(min_length=17, max_length=26) for _ in range(len(df))]

# Convert lists to Pandas Series
negative_mRNA = pd.Series(negative_mRNA)
negative_miRNA = pd.Series(negative_miRNA)

# concatenate negative mRNA samples with the positive
full_mRNA = pd.concat([df["mRNA sequence"], negative_mRNA], axis=0).reset_index(drop=True)
full_miRNA = pd.concat([df["miRNA sequence"], negative_miRNA], axis=0).reset_index(drop=True)
labels = pd.Series([1] * len(df) + [0] * len(df), name='label').reset_index(drop=True)

# create new dataframe
final_df = pd.DataFrame({"miRNA sequence": full_miRNA,
                        "mRNA sequence": full_mRNA,
                        "label": labels})

# save full dataset
save_to_path = os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "miRAW_dataset_completely_random.csv")
print(f"Saving miRAW dataset to {save_to_path}")
final_df.to_csv(save_to_path, sep=",", index=False)