import os
import pandas as pd
from sklearn.model_selection import train_test_split

# PROJ_HOME = os.path.expanduser("~/projects/mirLM")
PROJ_HOME = "/Users/jiayaogu/Documents/Li Lab/mirLM---Micro-RNA-generation-with-mRNA-prompt/"
mRNA_length = 30

data_path_train = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_train_500_multiseeds.csv")
df_train = pd.read_csv(data_path_train, sep=",")
df_train_subset = df_train.sample(n=5000, random_state=42)

df_train_subset.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"TargetScan_train_500_multiseeds_random_samples.csv"
    ),
    sep=",",
    index=False
)

data_path_val = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_validation_500_multiseeds.csv")
df_val = pd.read_csv(data_path_val, sep=",")
df_val_subset = df_val.sample(n=500, random_state=42)

df_val_subset.to_csv(
    os.path.join(
        os.path.dirname(data_path_val), f"TargetScan_validation_500_multiseeds_random_samples.csv"
    ),
    sep=",",
    index=False
)