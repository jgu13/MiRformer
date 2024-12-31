import os
import pandas as pd

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "data")
par_clip_df = pd.read_csv(os.path.join(data_dir, "DeepMirTar-PAR-CLIP.csv"), sep=",")

# generate negative pairs my randomly pairing mRNA with miRNA 
mRNA_shuffled = par_clip_df["mRNA Seq"].sample(frac=1, random_state=42).reset_index(drop=True)
full_mRNA = pd.concat([par_clip_df['mRNA Seq'], mRNA_shuffled], axis=0).reset_index(drop=True)
full_miRNA = pd.concat([par_clip_df["miRNA Seq (3'-->5')"], par_clip_df["miRNA Seq (3'-->5')"]], axis=0).reset_index(drop=True)
label = pd.Series([1] * len(par_clip_df) + [0] * len(par_clip_df), name='Label').reset_index(drop=True)
# Create the final dataframe
final_df = pd.DataFrame({'miRNA sequence': full_miRNA, 'mRNA sequence': full_mRNA, 'label': label})

# Display the final dataframe
print(final_df.tail())

final_df.to_csv(os.path.join(data_dir, "DeepMirTar-par-clip-selected.csv"), sep=',', index=False)
