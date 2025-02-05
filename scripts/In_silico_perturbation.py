import os
import random
import pandas as pd

def mutate_seed_single_position(mRNA_seq, seed_start, seed_end, mutation_rate=0.5):
    """
    Generate all single-nucleotide mutations for the specified
    seed region in mRNA_seq. Return a list of (mutant_seq, position, old_nt, new_nt).
    """
    valid_bases = ['A', 'T', 'C', 'G']
    
    # Convert mRNA_seq to list for easy manipulation
    mRNA_list = list(mRNA_seq)
    mutated_list = mRNA_list.copy()
    
    # Positions in the seed region
    seed_positions = list(range(seed_start, seed_end))
    seed_length = len(seed_positions)
    
    # Number of positions to mutate (integer division)
    num_to_mutate = int(seed_length * mutation_rate)
    
    # Randomly mutate half of the bases in seed region
    positions_to_mutate = random.sample(seed_positions, num_to_mutate)
    
    for pos in positions_to_mutate:
        original_nt = mRNA_list[pos]
        # base = random.choice([b for b in valid_bases if b != original_nt])
        mutated_list[pos] = 'C'
    mutated_seq = "".join(mutated_list)
    
    return mutated_seq

def insilico_mutagenesis_seed(mRNA_list):
    """
    mRNA_dict: a list of dicts, each containing
      {
        "mRNA_sequence": ...,
        "seed_start": ...,
        "seed_end": ...
      }
    Returns a results structure capturing the original score
    and mutated scores for each variant.
    """
    mutants = []
    
    for mRNA in mRNA_list:
        mRNA_seq = mRNA['mRNA sequence']
        seed_start = mRNA['seed_match_start']
        seed_end = mRNA['seed_match_end']
        # Generate single-point mutants in seed region
        mutant = mutate_seed_single_position(mRNA_seq, seed_start, seed_end, mutation_rate=1)
        mutants.append(mutant)
    return mutants

def insilico_mutagenesis_nonseed(mRNA_list):
    """
    mRNA_dict: a list of dicts, each containing
      {
        "mRNA_sequence": ...,
        "seed_start": ...,
        "seed_end": ...
      }
    Returns a results structure capturing the original score
    and mutated scores for each variant.
    """
    mutants = []
    
    for mRNA in mRNA_list:
        mRNA_seq = mRNA['mRNA sequence']
        seed_start = 0
        seed_end = mRNA["seed_match_start"]
        # Generate single-point mutants
        mutant = mutate_seed_single_position(mRNA_seq, seed_start, seed_end, mutation_rate=1)
        seed_start = mRNA["seed_match_end"] + 1
        seed_end = len(mutant)
        if seed_start < seed_end:
            mutant = mutate_seed_single_position(mutant, seed_start, seed_end, mutation_rate=1)
        mutants.append(mutant)
    return mutants   

def insilico_mutagenesis_miRNA_seed(miRNA_list):
    """
    miRNA_dict: a list of miRNA
    Returns a results structure capturing the original score
    and mutated scores for each variant.
    """
    mutants = []
    
    for miRNA in miRNA_list:
        seed_start = 1
        seed_end = 8
        # Generate single-point mutants in seed region
        mutant = mutate_seed_single_position(miRNA, seed_start, seed_end, mutation_rate=1)
        mutants.append(mutant)
    return mutants

def insilico_mutagenesis_miRNA_nonseed(miRNA_list):
    """
    miRNA_dict: a list of miRNA
    Returns a results structure capturing the original score
    and mutated scores for each variant.
    """
    mutants = []
    
    for miRNA in miRNA_list:
        seed_start = 0
        seed_end = 1
        mutant = mutate_seed_single_position(miRNA, seed_start, seed_end, mutation_rate=1)
        seed_start = 8
        seed_end = len(miRNA)
        # Generate single-point mutants in seed region
        mutant = mutate_seed_single_position(mutant, seed_start, seed_end, mutation_rate=1)
        mutants.append(mutant)
    return mutants

if __name__ == '__main__':
    ISM_datapath = os.path.expanduser("~/projects/mirLM/data/ISM_data.csv")
    ISM_df = pd.read_csv(ISM_datapath)
    mRNA_dict = ISM_df[["mRNA sequence", "seed_match_start", "seed_match_end"]].to_dict(orient="records")
    miRNA_dict = ISM_df["miRNA sequence"]
    mutants = insilico_mutagenesis_miRNA_nonseed(miRNA_dict)
    ISM_df["miRNA sequence"] = mutants
    ISM_df["label"] = [1] * len(ISM_df)
    print(ISM_df.head())
    ISM_mutant_datapath = os.path.expanduser("~/projects/mirLM/data/ISM_data_mutant_miRNA_nonseed.csv")
    ISM_df.to_csv(ISM_mutant_datapath, index=False)