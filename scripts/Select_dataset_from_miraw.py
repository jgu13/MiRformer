import os
import pandas as pd

complementary_base_pair = {'A':'T',
                           'T':'A',
                           'U':'A',
                           'G':'C',
                           'C':'G'}

def get_complementary_sequence(seq_A):
    seq_B = [complementary_base_pair[base] for base in seq_A]
    return ''.join(seq_B)

def check_complementary_subsequence(miRNA:str, 
                                    mRNA:str, 
                                    k=6):
    """
    Return True if there is ANY contiguous k-mer Watson-Crick match
    between miRNA and mRNA; otherwise, False.
    """
    mi_len = len(miRNA)
    if mi_len < k:
        return False  # miRNA too short to form a k-mer

    for i in range(mi_len - k + 1):
        # Extract a k-mer from miRNA
        mirna_sub = miRNA[i : i + k]
        
        # Convert it to its complement
        mirna_sub_complement = get_complementary_sequence(mirna_sub)
        
        # Check if complement is a substring of mRNA
        if mirna_sub_complement in mRNA:
            return True
    
    return False

def select_dataset_by_seedmatch(df:pd.DataFrame,
                                total_num_pair = None, 
                                positive_prop = None,
                                num_positive_samples = None):
    if positive_prop:
        total_positive_pairs = int(positive_prop * total_num_pair)
    elif num_positive_samples:
        total_positive_pairs = num_positive_samples

    total_negative_pairs = total_positive_pairs
    
    current_positive_pairs = []
    current_negative_pairs = []
    
    for row in df.itertuples(index=False):
        # miRNA_ID, gene_name, _, miRNA, mRNA, label = row
        miRNA_ID, miRNA, gene_name, mRNA, label = row
        miRNA = miRNA[::-1] # reverse miRNA
        matches = check_complementary_subsequence(miRNA=miRNA, mRNA=mRNA)
        
        if label == 1:
            if len(current_positive_pairs) == total_positive_pairs:
                continue
            if matches: 
                current_positive_pairs.append({"miRNA": miRNA_ID, "gene_name":gene_name,
                                               "miRNA sequence": miRNA[::-1], "mRNA sequence": mRNA,
                                               "label": 1})
        else: 
            if len(current_negative_pairs) == total_negative_pairs:
                continue
            if not matches:
                current_negative_pairs.append({"miRNA": miRNA_ID, "gene_name":gene_name, 
                                               "miRNA sequence": miRNA[::-1], "mRNA sequence": mRNA, # reverse miRNA back to 5' to 3'
                                               "label": 0})

        if len(current_positive_pairs) == total_positive_pairs and \
            len(current_negative_pairs) == total_negative_pairs:
                break
    
    positive_df = pd.DataFrame(current_positive_pairs)
    negative_df = pd.DataFrame(current_negative_pairs)
    
    DF = pd.concat([positive_df, negative_df])
    
    return DF

if __name__ == "__main__":
    data_dir = os.path.expanduser("~/projects/mirLM/data")
    miraw_training_data = os.path.join(data_dir, "data_miRaw_noL_noMissing_remained_seed1122_test.csv")
    df = pd.read_csv(miraw_training_data, sep=',')

    # get 50% of negative samples and an equal number of positive samples
    num_negative_samples = int(len(df[df['label'] == 0]) * 0.3)
    total_num_samples = num_negative_samples
    print("num negative_samples = ", num_negative_samples)
    
    new_DF = select_dataset_by_seedmatch(df, num_positive_samples=num_negative_samples)
    
    print("Total number of selected pairs = ", len(new_DF))
    
    save_path = os.path.expanduser("~/projects/mirLM/data/selected_perfect_seed_match_test.csv")
    new_DF.to_csv(save_path, sep="\t", index=False)
    print(f"saved dataset to {save_path}.")
    