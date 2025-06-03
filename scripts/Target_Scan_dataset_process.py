import os
import time
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def get_complementary_seq(seq):
    d = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    return ''.join([d[c] for c in seq])

def check_complementarity(miRNA, mRNA, seg_length = 6):
    # check whether there are 6-consecutive complmentary segment between miRNA and mRNA
    miRNA = miRNA[::-1]
    miRNA = miRNA.replace('U', 'T')
    for start in range(0, len(mRNA)-seg_length+1):
        mRNA_seg = mRNA[start:start + seg_length]
        mRNA_seg_c = get_complementary_seq(mRNA_seg)
        if mRNA_seg_c in miRNA:
            return True
    return False

# mRNA = "AACTAACTTTAAACTGGATATATACTTTG"
# miRNA = "GUCCAGUUUUCCCAGGAAUCCCU"
# match = check_complementarity(miRNA, mRNA)
# print(match)

# positive miRNA and mRNA pairs
path = os.path.join(data_dir, "Predicted_Targets_Context_Scores.default_predictions.txt")
predicted_targets = pd.read_csv(path, sep='\t')
# filter for human
predicted_targets = predicted_targets[predicted_targets["Gene Tax ID"] == 9606]
positive_pairs    = predicted_targets[["Gene Symbol", "miRNA", "Gene ID", "Transcript ID", "UTR_start", "UTR_end"]]
# print("Total positive pairs = ", len(positive_pairs)) #228049
# positive_pairs    = positive_pairs.sample(n=2000, replace=False, random_state=42)
positive_pairs    = positive_pairs.to_dict(orient="records")
num_pospairs      = len(positive_pairs)

# # negative miRNA and mRNA pairs: select mRNA species that is not in positive pairs with the miRNA
all_genes      = predicted_targets["Gene Symbol"].unique()
all_miRNA      = predicted_targets["miRNA"].unique()
num_negpairs   = 2000
negative_pairs = []
print("Start getting negative pairs: ")
while len(negative_pairs) < num_negpairs:
    target_miRNA = np.random.choice(all_miRNA, replace=True)
    all_target_genes = predicted_targets.loc[predicted_targets["miRNA"] == target_miRNA, "Gene Symbol"].unique()
    neg_gene = np.random.choice(np.setdiff1d(all_genes, all_target_genes, assume_unique=True))
    # get gene_id
    neg_gene_id = predicted_targets.loc[(predicted_targets["Gene Symbol"] == neg_gene), "Gene ID"].iloc[0]
    neg_tran_id = predicted_targets.loc[(predicted_targets["Gene Symbol"] == neg_gene), "Transcript ID"].iloc[0]
    negative_pairs.append({"Gene Symbol":   neg_gene, 
                           "miRNA":         target_miRNA, 
                           "Gene ID":       neg_gene_id, 
                           "Transcript ID": neg_tran_id
                           })

# negative_pairs = pd.DataFrame(negative_pairs)
# positive_pairs.to_csv(os.path.join(data_dir, "Positive_pairs.csv"), sep='\t', index=False)
# negative_pairs.to_csv(os.path.join(data_dir, "Negative_pairs.csv"), sep='\t', index=False)

# # read positive pairs
# positive_pairs = pd.read_csv(os.path.join(data_dir, "Positive_pairs.csv"), sep='\t')
# negative_pairs = pd.read_csv(os.path.join(data_dir, "Negative_pairs.csv"), sep='\t')
# positive_pairs = positive_pairs.to_dict(orient = "records")
# negative_pairs = negative_pairs.to_dict(orient = "records")

# read utr sequence of mRNA
# mRNAseq_path = os.path.join(data_dir, "3utr.fasta")
# mRNA_seq_dict = []
# with open(mRNAseq_path, "rt") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         _, gene_id, _, tran_id, gene_name, _ = record.id.split('|')
#         mRNA_seq = str(record.seq) # full-length mRNA seq
#         if mRNA_seq != "Sequenceunavailable":
#             mRNA_seq_dict.append({"Gene ID": gene_id,
#                                   "Gene Symbol": gene_name,
#                                   "Transcript ID": tran_id,
#                                   "mRNA sequence": mRNA_seq
#                                   })
                
# mRNA_seq_df = pd.DataFrame(mRNA_seq_dict)
# mRNA_seq_df.to_csv(os.path.join(os.path.join(data_dir, "3utr.csv")), sep='\t', index=False)
# mRNA_df = pd.read_csv(os.path.join(data_dir, "3utr.csv"), sep='\t')

# # read mature sequence of miRNA
# path     = os.path.join(data_dir, "miR_Family_Info.txt")
# miRNA_df = pd.read_csv(path, sep='\t')
# # filter for human
# miRNA_df = miRNA_df[miRNA_df["Species ID"] == 9606]

# # print("Putting together positive samples: ")
# mRNA_len   = 30
# miRNA_len  = 24
# linker_len = 6
# seed_len   = 8
# diff       = (mRNA_len - seed_len) // 2
# start_time = time.time()
# # Find unmatched pairs
# unmatched  = []
# print(f"Diff = {diff}")

# randomize_start = True

# # assemble positive pairs
# positive_samples = []
# max_len = 0
# for positive_pair in positive_pairs:
#     target_gene    = positive_pair.get("Gene Symbol", "")
#     target_gene_id = positive_pair.get("Gene ID", "")
#     miRNA_ID       = positive_pair.get("miRNA", "")
#     tran_id        = positive_pair.get("Transcript ID", "")
#     if miRNA_ID: 
#         miRNA_seq = miRNA_df.loc[miRNA_df["MiRBase ID"] == miRNA_ID, "Mature sequence"]
#         miRNA_seq = miRNA_seq.values[0]
#     if target_gene_id and tran_id:
#         mRNA_seq        = mRNA_df.loc[(mRNA_df["Gene ID"] == target_gene_id) & (mRNA_df["Transcript ID"] == tran_id), "mRNA sequence"]
#         orig_seed_start = positive_pair["UTR_start"]
#         orig_seed_end   = positive_pair["UTR_end"]
#         seed_len = orig_seed_end - orig_seed_start + 1
#         print(f"orig_seed_start = {orig_seed_start}, orig_seed_end = {orig_seed_end}")
#         if (len(mRNA_seq) > 0) and (len(mRNA_seq.values[0]) > orig_seed_start) and (len(mRNA_seq.values[0]) > orig_seed_end):    
#             mRNA_seq = mRNA_seq.values[0]
#             # segment mRNA seq binding site
#             start    = max(orig_seed_start - diff - 1, 0            )
#             end      = min(orig_seed_end   + diff    , len(mRNA_seq)) # because slicing is end-exclusive
#             seed_rel_start = orig_seed_start - start
#             seed_rel_end   = seed_rel_start  + seed_len
#             if randomize_start:
#                 total_diff = mRNA_len - seed_len - 1 # max index for start
#                 start_diff = random.randint(0, total_diff)
#                 end_diff   = total_diff      - start_diff
#                 start      = orig_seed_start - start_diff
#                 end        = orig_seed_end   + end_diff
#                 print(f"start_diff = {start_diff}, end_diff = {end_diff}")
#                 # if we run off the left end, then clamp
#                 if start < 0:
#                     start = 0
#                     end   = mRNA_len # so that the length is always `mRNA_len`
#                 if end   > len(mRNA_seq):
#                     end   = len(mRNA_seq)
#                     start = len(mRNA_seq) - mRNA_len
#                 print(f"Start = {start}, End = {end}")
#                 seed_rel_start = orig_seed_start - start
#                 seed_rel_end   = seed_rel_start + seed_len
#             mRNA_seg = mRNA_seq[start:end]
#             print(f"Length of mRNA_seg = ", len(mRNA_seg))
#             if (start < end) and (len(mRNA_seg) >= 6): # segment length must be greater than 6
#                 max_len = max(max_len, len(mRNA_seg))
#                 positive_samples.append({"Gene ID":       target_gene_id, 
#                                         "Transcript ID":  tran_id,
#                                         "Gene Symbol":    target_gene,
#                                         "miRNA ID":       miRNA_ID,
#                                         "miRNA sequence": miRNA_seq,
#                                         "mRNA sequence":  mRNA_seg,
#                                         "seed start":     seed_rel_start,
#                                         "seed end":       seed_rel_end,
#                                         "label":          1
#                                         })
#             else:
#                 print(f"Cannot find valid segment for gene [{target_gene_id}] and mirRNA [{miRNA_ID}].")
#         else:
#             print(f"One of the following happened to gene [{target_gene_id}]:"
#                   f"1. No such gene is found OR "
#                   f"2. mRNA length <= seed start OR "
#                   f"3. mRNA length <= seed end")       
#     else:
#         print(f"Cannot find mRNA seq for gene id: [{target_gene_id}] and transcript id: [{tran_id}].", flush=True)
# print("Time taken to put together positive samples = ", (time.time() - start_time) / 60)

# positive_df = pd.DataFrame(positive_samples)
# print(f"Number of positive samples = {len(positive_df)}.")
# print(f"Maximum mRNA length = {max_len}.")
# # positive_df.to_csv(os.path.join(data_dir, f"positive_samples_{str(mRNA_len)}_randomized_start.csv"), sep="\t", index=False)


# assemble negative pairs
# print("Putting together negative samples: ")
# start_time = time.time()
# seg_len = mRNA_len
# print(f"Segment length = {seg_len}")

# negative_samples = []
# max_length = 0
# for negative_pair in negative_pairs:
#     target_gene    = negative_pair.get("Gene Symbol", "")
#     target_gene_id = negative_pair.get("Gene ID", "")
#     miRNA_ID       = negative_pair.get("miRNA", "")
#     tran_id        = negative_pair.get("Transcript ID", "")
#     if miRNA_ID: 
#         miRNA_seq  = miRNA_df.loc[miRNA_df["MiRBase ID"] == miRNA_ID, "Mature sequence"]
#         miRNA_seq  = miRNA_seq.values[0]
#     if target_gene_id and tran_id:
#         # target_gene_id = target_gene_id.split(".")[0]
#         # tran_id = tran_id.split(".")[0]
#         mRNA_seq = mRNA_df.loc[(mRNA_df["Gene ID"] == target_gene_id) & (mRNA_df["Transcript ID"] == tran_id), "mRNA sequence"]
#         if len(mRNA_seq) > 0:    
#             mRNA_seq = mRNA_seq.values[0]
#             # randomly select seg_len-nt mRNA segment
#             if len(mRNA_seq) < seg_len:
#                 mRNA_seg = mRNA_seq
#                 negative_samples.append({"Gene ID":   target_gene_id, 
#                                     "Transcript ID":  tran_id,
#                                     "Gene Symbol":    target_gene,
#                                     "miRNA ID":       miRNA_ID,
#                                     "miRNA sequence": miRNA_seq,
#                                     "mRNA sequence":  mRNA_seg,
#                                     "seed start":     -1,
#                                     "seed end":       -1,
#                                     "label":          0
#                                     })
#                 max_length = max(max_length, len(mRNA_seg))
#                 continue
#             else:
#                 start    = random.randint(0, len(mRNA_seq)-seg_len)
#                 end      = min(start + seg_len, len(mRNA_seq))
#                 mRNA_seg = mRNA_seq[start:end]
#                 match    = check_complementarity(miRNA_seq, mRNA_seg)
#                 if match:
#                     print("Found complementarity in negative samples, re-sampling mRNA segment...")
#                     possible_starts = len(mRNA_seq) - seg_len + 1
#                     n_samples_per_pair = 15
#                     max_tries       = min(n_samples_per_pair, possible_starts) # choose between N samples or max possible segments per sample
#                     tried_starts    = set()

#                     while len(tried_starts) < max_tries:
#                         start = random.randint(0, possible_starts - 1)
#                         if start in tried_starts:
#                             # Already tried this start; draw again
#                             continue
#                         tried_starts.add(start)
#                         end = start + seg_len
#                         mRNA_seg = mRNA_seq[start:end]
#                         if not check_complementarity(miRNA_seq, mRNA_seg):
#                             negative_samples.append({
#                                 "Gene ID":          target_gene_id,
#                                 "Transcript ID":    tran_id,
#                                 "Gene Symbol":      target_gene,
#                                 "miRNA ID":         miRNA_ID,
#                                 "miRNA sequence":   miRNA_seq,
#                                 "mRNA sequence":    mRNA_seg,
#                                 "seed start":       -1,
#                                 "seed end":         -1,
#                                 "label":            0
#                             })
#                 else:
#                     negative_samples.append({"Gene ID": target_gene_id, 
#                                     "Transcript ID":    tran_id,
#                                     "Gene Symbol":      target_gene,
#                                     "miRNA ID":         miRNA_ID,
#                                     "miRNA sequence":   miRNA_seq,
#                                     "mRNA sequence":    mRNA_seg,
#                                     "seed start":       -1,
#                                     "seed end":         -1,
#                                     "label":            0
#                                     })
#                 max_length = max(max_length, len(mRNA_seg))    
#         else:
#             print(f"Cannot find mRNA seq for gene id: [{target_gene_id}] and transcript id: [{tran_id}].", flush=True)

# negative_df = pd.DataFrame(negative_samples)
# print("Number of negative samples = ", len(negative_df))
# print("Maximum mRNA length = ", max_length)
# negative_df.to_csv(os.path.join(data_dir, f"negative_samples_{str(seg_len)}.csv"), sep="\t", index=False)
# print(f"Time taken for putting together negative samples = {(time.time() - start_time)/60} min")

# positive_samples = pd.read_csv(os.path.join(data_dir, f"positive_samples_{str(mRNA_len)}_randomized_start.csv"), sep='\t')
# negative_samples = pd.read_csv(os.path.join(data_dir, f"negative_samples_{str(mRNA_len)}.csv"), sep='\t')
# samples = pd.concat([positive_samples, negative_samples], axis=0)
# ds_train, ds_rem = train_test_split(samples, test_size=0.2, random_state=42, shuffle=True)
# ds_val, ds_test = train_test_split(ds_rem, test_size=0.2, shuffle=False)
# ds_train.to_csv(os.path.join(data_dir, f"TargetScan_train_{str(mRNA_len)}_randomized_start.csv"), sep=',', index=False)
# ds_val.to_csv(os.path.join(data_dir, f"TargetScan_validation_{str(mRNA_len)}_randomized_start.csv"), sep=',', index=False)
# ds_test.to_csv(os.path.join(data_dir, f"TargetScan_test_{str(mRNA_len)}_randomized_start.csv"), sep=',', index=False)
