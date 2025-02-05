import os
import random
import pandas as pd

def generate_random_string(min_length, max_length):
    length = random.randint(min_length, max_length)
    letters = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(letters, k=length))

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "data")
par_clip_df = pd.read_csv(os.path.join(data_dir, "DeepMirTar-PAR-CLIP.csv"), sep=",")

# # miRNA length = (min, max) = (21, 24)
# mRNA_lengths = [len(s) for s in par_clip_df["mRNA Seq"]]
# print("min mRNA length = ", min(mRNA_lengths))
# print("max mRNA length = ", max(mRNA_lengths))
# miRNA_lengths = [len(s) for s in par_clip_df["miRNA Seq (3'-->5')"]]
# print("min miRNA length = ", min(miRNA_lengths))
# print("max miRNA length = ", max(miRNA_lengths))

# generate mRNA sequences
# each mRNA sequence = 5-nt 5' + target_site + 5-nt 3'
positive_mRNA_seqs = []
for i in range(len(par_clip_df)):
    entry = par_clip_df.iloc[i]
    l = entry["End"] - entry["Start"] + 1
    diff = abs(40 - l)
    if diff % 2 == 0:
        start_diff = diff // 2
        end_diff = diff // 2
    else:
        start_diff = diff // 2
        end_diff = diff // 2 + 1
    start = entry["Start"]-1-start_diff # original target site is entry[start-1:end-1]
    end = entry["End"]+end_diff
    if start >= 0 and end < len(entry["mRNA Seq"]):
        new_mRNA_seq = entry["mRNA Seq"][start:end] 
        positive_mRNA_seqs.append(new_mRNA_seq)
    else: 
        start = 0 if start < 0 else start
        end = len(entry["mRNA Seq"] - 1) if end >= len(entry["mRNA Seq"]) else end
        new_mRNA_seq = entry["mRNA Seq"][start:end]
        positive_mRNA_seqs.append(new_mRNA_seq)

mRNA_min_len = min([len(s) for s in positive_mRNA_seqs])
mRNA_max_len = max([len(s) for s in positive_mRNA_seqs])
miRNA_min_len = 21
miRNA_max_len = 24
print(f"Min mRNA length = {mRNA_min_len}, max mRNA length = {mRNA_max_len}")

# generate negative mRNA seqs
negative_mRNA_seqs = [generate_random_string(min_length=mRNA_min_len, max_length=mRNA_max_len) for _ in range(len(par_clip_df))]
negative_miRNA_seqs = [generate_random_string(min_length=miRNA_min_len, max_length=miRNA_max_len) for _ in range(len(par_clip_df))]
full_mRNA = positive_mRNA_seqs + negative_mRNA_seqs
full_mRNA = pd.Series(full_mRNA)
print("Number of negative mRNA = ", len(negative_mRNA_seqs))
full_miRNA = pd.concat([par_clip_df["miRNA Seq (3'-->5')"], pd.Series(negative_miRNA_seqs)], axis=0).reset_index(drop=True)
print("Number of miRNA = ", len(full_miRNA))
label = pd.Series([1] * len(par_clip_df) + [0] * len(par_clip_df), name='Label').reset_index(drop=True)
# Create the final dataframe
final_df = pd.DataFrame({'miRNA sequence': full_miRNA, 
                         'mRNA sequence': full_mRNA, 
                         'label': label})

# Display the final dataframe
print(final_df.tail())

final_df.to_csv(os.path.join(data_dir, "DeepMirTar-par-clip-miraw-like_completely_random.csv"), sep=',', index=False)
