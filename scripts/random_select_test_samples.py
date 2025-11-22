"""
Randomly select samples for train, validation and test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
PROJ_HOME = os.path.expanduser("~/projects/mirLM")

data_path_train = os.path.join(PROJ_HOME, "your_data_path.csv")
df = pd.read_csv(data_path_train, sep="\t")
df_train, df_rem = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
df_val, df_test = train_test_split(df_rem, test_size=0.2, random_state=42, shuffle=False)

df_train.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"your_data_path_train.csv"
    ),
    sep=",",
    index=False
)

df_val.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"your_data_path_validation.csv"
    ),
    sep=",",
    index=False
)

df_test.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"your_data_path_test.csv"
    ),
    sep=",",
    index=False
)