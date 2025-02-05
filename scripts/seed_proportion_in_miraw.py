import os
import pandas as pd
from search_seeds import search_seed


def calculate_seed_match_proportion(df):
    positive_seed_match_pairs = 0
    negative_seed_match_pairs = 0
    total_positive_pair = 0
    total_negative_pair = 0
    
    for row in df.itertuples(index=False):
        print(row)
        miRNA_ID, miRNA, gene_name, mRNA, label = row
            
        seed_searcher = search_seed(miRNA=miRNA,
                                    mRNA=mRNA)
        
        matches = seed_searcher.find_seed_matches(rev=True)
        if label == 1:
            total_positive_pair += 1
            print("positive pair:")
            print("Gene name: ", gene_name)
            print("miRNA", miRNA_ID) 
            if matches:
                positive_seed_match_pairs += 1
                for miRNA, mRNA, binding in matches:
                    print(miRNA)
                    print(mRNA)
                    print(binding)
        else:
            total_negative_pair += 1
            print("negative pair:")
            print("Gene name: ", gene_name)
            print("miRNA", miRNA_ID) 
            if matches:
                negative_seed_match_pairs += 1
                for miRNA, mRNA, binding in matches: 
                    print(miRNA)
                    print(mRNA)
                    print(binding)
    
    positive_seed_match_prop = positive_seed_match_pairs / total_positive_pair
    negative_seed_match_prop = negative_seed_match_pairs / total_negative_pair
    total_seed_match = positive_seed_match_pairs + negative_seed_match_pairs
    total_seed_match_prop = total_seed_match / (total_positive_pair + total_negative_pair)
        
    return positive_seed_match_prop, negative_seed_match_prop, total_seed_match_prop

if __name__ == '__main__':
    data_dir = os.path.expanduser("~/projects/mirLM/data")
    miraw_training_data = os.path.join(data_dir, "data_miRaw_noL_noMissing_remained_seed1122_test.csv")
    df = pd.read_csv(miraw_training_data, sep='\t')

    positive_seed_match_prop, negative_seed_match_prop, total_seed_match_prop = calculate_seed_match_proportion(df)
    print("Total seed match prop = {:.2f}".format(total_seed_match_prop))
    print("Positive seed match prop = {:.2f}".format(positive_seed_match_prop))
    print("Negative seed match prop = {:.2f}".format(negative_seed_match_prop))