import os
import pandas as pd
from sklearn.model_selection import train_test_split

# dataset_name = "data_miRaw_noL_noMisMissing_remained_seed1122"
dataset_name = "selected_perfect_seed_match"
data_dir = os.path.expanduser("~/projects/mirLM/data")
D = pd.read_csv(os.path.join(data_dir, f"{dataset_name}.csv"), sep=',')

ds_train, ds_rem = train_test_split(
    D, test_size=0.3, random_state=34, shuffle=True
)

ds_val, ds_test = train_test_split(
    ds_rem, test_size=0.1, random_state=34, shuffle=False
)

ds_train.to_csv(os.path.join(data_dir, f"{dataset_name}_train.csv"), index=False)
ds_val.to_csv(os.path.join(data_dir, f"{dataset_name}_validation.csv"), index=False)
ds_test.to_csv(os.path.join(data_dir, f"{dataset_name}_test.csv"), index=False)