import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
mRNA_length = 30

data_path = os.path.join(PROJ_HOME, "TargetScan_dataset", f"TargetScan_train_30_randomized_start.csv")
df = pd.read_csv(data_path, sep=",")
df = df.sample(n=512, random_state=42)

df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_train_{str(mRNA_length)}_random_start_samples.csv"
    ),
    sep=",",
    index=False
)

data_path = os.path.join(PROJ_HOME, "TargetScan_dataset", f"TargetScan_validation_30_randomized_start.csv")
df = pd.read_csv(data_path, sep=",")
df = df.sample(n=128, random_state=42)

df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_validation_{str(mRNA_length)}_random_start_samples.csv"
    ),
    sep=",",
    index=False
)