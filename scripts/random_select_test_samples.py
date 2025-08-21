import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
# PROJ_HOME = "/Users/jiayaogu/Documents/Li Lab/mirLM---Micro-RNA-generation-with-mRNA-prompt/"

data_path_train = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_train_500_randomized_start.csv")
df_train = pd.read_csv(data_path_train, sep=",")
df_train_subset = df_train.sample(n=2000, random_state=42)

df_train_subset.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"Positive_primates_train_500_randomized_start_random_samples.csv"
    ),
    sep=",",
    index=False
)

data_path_val = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_validation_500_randomized_start.csv")
df_val = pd.read_csv(data_path_val, sep=",")
df_val_subset = df_val.sample(n=50, random_state=42)

df_val_subset.to_csv(
    os.path.join(
        os.path.dirname(data_path_val), f"Positive_primates_validation_500_randomized_start_random_samples.csv"
    ),
    sep=",",
    index=False
)