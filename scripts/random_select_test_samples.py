import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
mRNA_length = 30

data_path = os.path.join(PROJ_HOME, "TargetScan_dataset", f"positive_samples_30_randomized_start.csv")
df = pd.read_csv(data_path, sep=",")
df = df.sample(n=256, random_state=34)
train_df, test_df = train_test_split(df, test_size=0.2)

train_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_train_{str(mRNA_length)}_random_start_samples.csv"
    ),
    sep=",",
    index=False
)

# data_path = os.path.join(PROJ_HOME, "TargetScan_dataset", f"TargetScan_validation_30.csv")
# df = pd.read_csv(data_path, sep=",")
# df = df.sample(n=256, random_state=34)

test_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"TargetScan_validation_{str(mRNA_length)}_random_samples.csv"
    ),
    sep=",",
    index=False
)