import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
mRNA_length = 30

data_path = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_train_500_randomized_start.csv")
df = pd.read_csv(data_path, sep=",")
df = df.sample(frac=0.1, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True)

train_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_train_500_randomized_start_random_samples.csv"
    ),
    sep=",",
    index=False
)

test_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_validation_500_randomized_start_random_samples.csv"
    ),
    sep=",",
    index=False
)