import os
import pandas as pd


PROJ_HOME = os.path.expanduser("~/projects/mirLM")
data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

# ——— 1) Complement / reverse-complement helpers ——————————————————————
comp_table = str.maketrans("ACGT", "TGCA")
def rev_comp(seq: str) -> str:
    # complement, then reverse
    return seq.translate(comp_table)[::-1]

# ——— 2) Load your negative pairing dataset ————————————————————————
# Assumes you have columns “miRNA_seq” and “mRNA_seq” in your CSV
df = pd.read_csv(os.path.join(data_dir, "TargetScan_test_30_randomized_start.csv"))
neg_df = df.loc[df['label']==0]
neg_df = neg_df.sample(n=2000)
# neg_df.to_csv(os.path.join(data_dir, "negative_samples_30_miniset.csv"), index=False)

# ——— 3) Insert the complementary seed into each mRNA —————————————————
seed_len = 7  # from bases 2…8 of miRNA

def insert_seed(seq: str, seed: str, insert_pos: int) -> str:
    # inject seed, then trim/pad to length 30
    new = seq[:insert_pos] + seed + seq[insert_pos:]
    if len(new) >= 30:
        return new[:30]
    else:
        return new.ljust(30, "N")  # pad with Ns if too short

def make_seed_mrna(row):
    mirna = row["miRNA sequence"]
    mrna  = row["mRNA sequence"]
    mirna = mirna.replace("U", "T")
    # 2nd–8th bases (Python idx 1…7)
    raw_seed = mirna[1:1+seed_len]
    seed_rc  = rev_comp(raw_seed)
    # by default, center the seed in the 30-mer
    center_pos = (30 - seed_len) // 2
    return insert_seed(mrna, seed_rc, center_pos)

neg_df["mRNA sequence"] = neg_df.apply(make_seed_mrna, axis=1)
neg_df["label"] = 1

# ——— 4) Save the augmented dataset ——————————————————————————————
out_f = os.path.join(data_dir, "negative_samples_30_with_seed.csv")
neg_df.to_csv(out_f, index=False)
print(f"Wrote {len(neg_df)} rows with embedded seeds → {out_f}")

 