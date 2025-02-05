import os
import pandas as pd
from search_seeds import search_seed


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
        miRNA_ID, gene_name, _, miRNA, mRNA, label = row
            
        seed_searcher = search_seed(miRNA=miRNA,
                                    mRNA=mRNA)
        
        matches = seed_searcher.find_seed_matches(rev=True)
        if label == 1:
            if len(current_positive_pairs) == total_positive_pairs:
                continue
            if matches: 
                current_positive_pairs.append({"miRNA": miRNA_ID, "gene_name":gene_name,
                                               "miRNA sequence": miRNA, "mRNA sequence": mRNA,
                                               "label": 1})
        else: 
            if len(current_negative_pairs) == total_negative_pairs:
                continue
            if not matches:
                current_negative_pairs.append({"miRNA": miRNA_ID, "gene_name":gene_name, 
                                               "miRNA sequence": miRNA, "mRNA sequence": mRNA,
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
    df = pd.read_csv(miraw_training_data, sep='\t')

    # get 50% of negative samples and an equal number of positive samples
    num_negative_samples = int(len(df[df['label'] == 0]) * 0.5)
    total_num_samples = num_negative_samples
    print("num negative_samples = ", num_negative_samples)
    
    new_DF = select_dataset_by_seedmatch(df, 
                                         num_positive_samples=num_negative_samples)
    
    print("Total number of selected pairs = ", len(new_DF))
    
    save_path = os.path.expanduser("~/projects/mirLM/data/selected_perfect_seed_match.csv")
    new_DF.to_csv(save_path, sep="\t")
    print(f"saved dataset to {save_path}.")
    