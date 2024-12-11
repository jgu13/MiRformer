import os
import pandas as pd

PROJ_HOME = os.path.expanduser('~/projects/mirLM')
mRNA_length = 2000
data_path = os.path.join(PROJ_HOME, 'data', f'training_{mRNA_length}.csv')

df = pd.read_csv(data_path, sep=",")
df = df.sample(n=256, random_state=34)

df.to_csv(os.path.join(os.path.dirname(data_path), f"training_{mRNA_length}_random_256_samples.csv"), sep=',')