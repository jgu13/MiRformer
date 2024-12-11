import pandas as pd
import os

file=os.path.join(os.path.expanduser('~/projects/mirLM'), "data", "data_miRaw_noL_noMisMissing_remained_seed1122.txt")
df = pd.read_csv(file,  sep="\t", names=["mir_id", "miRNA sequence", "gene id", "mRNA sequence", "label"])
df.to_csv(file, sep="\t", index=False)