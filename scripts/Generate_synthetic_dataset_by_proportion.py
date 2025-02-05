import random
import pandas as pd
from search_seeds import search_seed

complementary_base_pair = {'A':'U',
                           'T':'A',
                           'U':'A',
                           'G':'C',
                           'C':'G'}
                        #    'G':'U',
                        #    'G':'T'}

def generate_sequence(seq_len:int, 
                      use_U = False):
    '''
    Generate one `seq_len`-length rna sequence
    '''
    if use_U:
        seq = ''.join(random.choices(['A','U','C','G'], k=seq_len))
    else:
        seq = ''.join(random.choices(['A','T','C','G'], k=seq_len))
    return seq
    
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

def generate_num_pairs(num_pairs: int, 
                       miRNA_len = 26,
                       mRNA_len = 40,
                       insert_seed = True):
    miRNA_l = []
    mRNA_l = []
    for _ in range(num_pairs):
        if insert_seed:
            seed = generate_sequence(seq_len=8, use_U=True) # 8-mer seed in miRNA
            seed_match = get_complementary_sequence(seed) # seed match in mRNA
            # insert seed at index in range=[0, miRNA_len)
            s = random.randint(0, miRNA_len-len(seed))
            miRNA_seq = list(generate_sequence(seq_len=s, use_U=True)) + \
                        list(seed) + \
                        list(generate_sequence(seq_len=miRNA_len-s-len(seed), use_U=True))
            miRNA_l.append(''.join(miRNA_seq))
            # insert seed_match at index in range=[0,mRNA_len)
            s = random.randint(0, mRNA_len-len(seed_match))
            mRNA_seq = list(generate_sequence(seq_len=s, use_U=True)) + \
                        list(seed_match) + \
                        list(generate_sequence(seq_len=mRNA_len-s-len(seed_match), use_U=True))
            mRNA_l.append(''.join(mRNA_seq))
            
        else:
            miRNA_seq = generate_sequence(seq_len=miRNA_len, use_U=True)
            mRNA_seq = generate_sequence(seq_len=mRNA_len, use_U=True)
            # make sure there is no seed match by checking 
            result = check_complementary_subsequence(miRNA=miRNA_seq,
                                                    mRNA=mRNA_seq,
                                                    k=6)
            tries = 0
            max_tries = 1000
            while result and tries < max_tries:
                tries += 1
                print("Found matches in negative pairs, regenerating negative pairs...")
                miRNA_seq = generate_sequence(seq_len=miRNA_len, use_U=True)
                mRNA_seq = generate_sequence(seq_len=mRNA_len, use_U=True)
                result = check_complementary_subsequence(miRNA=miRNA_seq,
                                                         mRNA=mRNA_seq,
                                                         k=6)
            if tries == max_tries:
                print(f"Warning: could not find negative pair without seed match after {max_tries} tries.")
            miRNA_l.append(miRNA_seq)
            mRNA_l.append(mRNA_seq)  
    return miRNA_l, mRNA_l

def generate_synthetic_dataset(total_num_samples:int,
                               positive_prop:float
                               ):
    positive_num_pairs = int(total_num_samples * positive_prop)
    negative_num_pairs = total_num_samples - positive_num_pairs
    
    positive_miRNA_l, positive_mRNA_l = generate_num_pairs(num_pairs=positive_num_pairs, insert_seed=True)
    negative_miRNA_l, negative_mRNA_l = generate_num_pairs(num_pairs=negative_num_pairs, insert_seed=False)
    
    positive_labels = [1] * positive_num_pairs
    negative_labels = [0] * negative_num_pairs
    
    df = pd.DataFrame(
        {
         'miRNA sequence': positive_miRNA_l + negative_miRNA_l,
         'mRNA sequence': positive_mRNA_l + negative_mRNA_l,
         'label': positive_labels + negative_labels
         }
    )  
    
    return df

if __name__ == '__main__':
    total_num_samples = 10000
    positive_prop = 0.5
    positive_samples = []
    negative_samples = []
    
    df = generate_synthetic_dataset(
            total_num_samples=total_num_samples,
            positive_prop=positive_prop
            )
    
    import os
    data_dir = os.path.expanduser("~/projects/mirLM/data")
    save_path = os.path.join(data_dir, "perfect_seed_match.csv")
    df.to_csv(save_path, sep='\t')
    print(f"Dataset is saved to {save_path}")