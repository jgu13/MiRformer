import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
# PROJ_HOME = "/Users/jiayaogu/Documents/Li Lab/mirLM---Micro-RNA-generation-with-mRNA-prompt/"
# PROJ_HOME = os.path.expanduser("/home/claris/projects/ctb-liyue/claris/projects/mirLM")

data_path_train = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data/starBase_degradome_UTR_windows_100.tsv")
df = pd.read_csv(data_path_train, sep="\t")
df_train, df_rem = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
df_val, df_test = train_test_split(df_rem, test_size=0.2, random_state=42, shuffle=False)
# df_train = pd.read_csv(data_path_train, sep=",")
# df_train_subset = df_train.sample(n=50000, random_state=42)

df_train.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"starBase_degradome_UTR_windows_100_train.csv"
    ),
    sep=",",
    index=False
)

df_val.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"starBase_degradome_UTR_windows_100_validation.csv"
    ),
    sep=",",
    index=False
)

df_test.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"starBase_degradome_UTR_windows_100_test.csv"
    ),
    sep=",",
    index=False
)

# data_path_val = os.path.join(PROJ_HOME, "TargetScan_dataset/Merged_primates_validation_500_randomized_start.csv")
# df_val = pd.read_csv(data_path_val, sep=",")
# df_val_subset = df_val.sample(n=500, random_state=42)

# df_val_subset.to_csv(
#     os.path.join(
#         os.path.dirname(data_path_val), f"Merged_primates_validation_500_randomized_start_random_samples.csv"
#     ),
#     sep=",",
#     index=False
# )