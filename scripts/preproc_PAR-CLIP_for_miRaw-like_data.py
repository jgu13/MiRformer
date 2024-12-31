import os
import pandas as pd

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "data")
par_clip_df = pd.read_csv(os.path.join(data_dir, "DeepMirTar-PAR-CLIP.csv"), sep=",")

# generate mRNA sequences
# each mRNA sequence = 5-nt 5' + target_site + 5-nt 3'
positive_mRNA_seqs = []
for i in range(len(par_clip_df)):
    entry = par_clip_df.iloc[i]
    l = entry["End"] - entry["Start"] + 1
    diff = 40 - l
    if diff % 2 == 0:
        start_diff = diff // 2
        end_diff = diff // 2
    else:
        start_diff = diff // 2
        end_diff = diff // 2 + 1
    new_mRNA_seq = entry["mRNA Seq"][entry["Start"]-1-start_diff:entry["End"]-1+end_diff] # originally that target site is entry[start-1:end-1]
    positive_mRNA_seqs.append(new_mRNA_seq)
# print("Number of positive mRNA = ", len(positive_mRNA_seqs))
# generate negative pairs by randomly shuffle mRNA seq and taking the first 40-nt segment (close to 5' end)
mRNA_shuffled = par_clip_df["mRNA Seq"].sample(frac=1, random_state=42).reset_index(drop=True)
negative_mRNA_seqs = [mRNA[:40] for mRNA in mRNA_shuffled]
full_mRNA = positive_mRNA_seqs + negative_mRNA_seqs
# print("Number of negative mRNA = ", len(negative_mRNA_seqs))
full_miRNA = pd.concat([par_clip_df["miRNA Seq (3'-->5')"], par_clip_df["miRNA Seq (3'-->5')"]], axis=0).reset_index(drop=True)
# print("Number of miRNA = ", len(full_miRNA))
label = pd.Series([1] * len(par_clip_df) + [0] * len(par_clip_df), name='Label').reset_index(drop=True)
# # Create the final dataframe
final_df = pd.DataFrame({'miRNA sequence': full_miRNA, 
                        #  'target site': par_clip_df["Target Site"],
                         'mRNA sequence': full_mRNA, 
                         'label': label})

# Display the final dataframe
print(final_df.tail())

final_df.to_csv(os.path.join(data_dir, "DeepMirTar-par-clip-miraw-like.csv"), sep=',', index=False)
