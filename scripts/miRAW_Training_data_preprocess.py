import os
import pandas as pd

data_dir = os.path.expanduser("~/projects/mirLM/data")
miraw_training_data = os.path.join(data_dir, "miRAW_Training_data.txt")

df = pd.read_csv(miraw_training_data, sep='\t')

# Remove trailing 'L' characters in miRNA
df['miRNA sequence'] = df['miRNA sequence'].str.rstrip('L')

# save dataframe
df.to_csv(os.path.join(data_dir, "miRAW_Training_data.csv"), sep='\t', index=False)