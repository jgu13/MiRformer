# fetch_transcripts_e102.py
import pandas as pd
import argparse, re, os

def cds(transcript_id, cds_seqs):
    # cds_seqs is a DataFrame with columns [transcript_id, sequence]
    mask = cds_seqs.iloc[:, 0] == transcript_id
    if mask.any():
        cds = cds_seqs[mask].iloc[0, 1]
    else:
        cds = ""
    return cds

def utr3(transcript_id, utr3_seqs):
    # utr3_seqs is a DataFrame with columns [transcript_id, sequence]
    mask = utr3_seqs.iloc[:, 0] == transcript_id
    if mask.any():
        utr3 = utr3_seqs[mask].iloc[0, 1]
    else:
        utr3 = ""
    return utr3

# fetch unique transcript ids from microT txt
def unique_transcripts_from_microt(df):
    """
    Get unique transcript ids and return a list of dictionary with item 
    {"transcript_id": transcript_id, "cds_utr": cds_utr}
    """
    df = df[["ensembl_transcript_id", "cds_utr"]].drop_duplicates()  # remove duplicated transcripts
    tx = df.to_dict(orient="records")
    return tx

def build_transcript_to_sequence_map(df, save_dir, cds_path, utr3_path):
    cds_seqs = pd.read_csv(cds_path, sep='\t', header=None)
    utr3_seqs = pd.read_csv(utr3_path, sep='\t', header=None)
    transcripts_ids = unique_transcripts_from_microt(df)
    print(f"removed duplicated transcripts")
    print(f"Number of unique transcripts: {len(transcripts_ids)}")
    
    d = {}
    print(f"Building transcripts to sequence map")
    i = 0
    for item in transcripts_ids:
        transcript_id = item["ensembl_transcript_id"]
        cds_utr = item["cds_utr"].lower()
        
        # Initialize transcript entry if not exists
        if transcript_id not in d:
            d[transcript_id] = {"cds": None, "utr3": None}
        
        try:
            if cds_utr == "cds":
                sequence = cds(transcript_id, cds_seqs)
            elif cds_utr == "utr3":
                sequence = utr3(transcript_id, utr3_seqs)
            else:
                print(f"Unknown cds_utr: {cds_utr}")
                sequence = ""

            if sequence == "":
                print(f"No sequence found for {transcript_id} ({cds_utr})")
            else:
                print(f"Fetched sequence for {transcript_id} ({cds_utr}): sequence length {len(sequence)}")
                d[transcript_id][cds_utr] = sequence
        except Exception as e:
            print(f"Warning: Failed to fetch sequence for {transcript_id} ({cds_utr}): {e}", flush=True)
            d[transcript_id][cds_utr] = ""
        i += 1
        if i % 1000 == 0:
            f_path = os.path.join(save_dir, f"mouse_transcripts_to_sequence_{i}.csv")
            df = pd.DataFrame.from_dict(d, orient="index") 
            df.index.name = "transcript_id"
            df.to_csv(f_path, sep='\t')
            print(f"Saved transcripts to sequence map to {f_path}")
    return d

# a helper function to get seed length, seed start and seed end
def get_seed_length(mre_type):
    m = re.search(r"(\d+)\s*mer", mre_type, flags=re.I)
    if m: return int(m.group(1))
    mt = mre_type.lower()
    if mt in {"7mer-m8","7mer-a1"}: return 7
    if "offset-6mer" in mt: return 6
    return 7  # default

# load interaction file
# get pairs of miRNA and mRNA that have interaction score > 0.85
# seed end = transcript end, seed start = transcript end - seed length
def load_interaction_file(interaction_path, mirna_path, cds_path, utr3_path, threshold=0.85, save_dir=None):
    df = pd.read_csv(interaction_path, sep='\t') # Time-limiting step
    
    # Validate required columns exist
    required_cols = ["interaction_score", "mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[df["interaction_score"] >= threshold]
    df = df[["mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]]

    # print(f"Filtered interactions by threshold >= {threshold}")
    transcripts_to_sequence = build_transcript_to_sequence_map(df, save_dir, cds_path, utr3_path)
    print(f"Built transcripts to sequence map")

    mirna_df = pd.read_csv(mirna_path, sep='\t')
    print(f"Loaded miRNA df with shape: {mirna_df.shape}")

    for index, row in df.iterrows():
        mirna_id, mre_type, cds_utr, ensembl_transcript_id, transcript_start, transcript_end = row
        cds_utr = cds_utr.lower()
        seed_length = get_seed_length(mre_type)
        # Ensure seed start doesn't go before transcript start
        seed_start = max(transcript_start, transcript_end - seed_length + 1) # seed length = seed_end - seed_start + 1
        seed_end = transcript_end
        df.loc[index, "seed start"] = int(seed_start + 1) # needs to be 1-based
        df.loc[index, "seed end"] = int(seed_end + 1) # needs to be 1-based
        print(f"Transcript id: {ensembl_transcript_id}, seed length: {seed_length}")
        print(f"Seed start: {seed_start}, seed end: {seed_end}")
        
        # Safe access to sequences with error handling
        try:
            mrna_sequence = transcripts_to_sequence.get(ensembl_transcript_id, {}).get(cds_utr, "")
            df.loc[index, "mrna_sequence"] = mrna_sequence
        except Exception as e:
            print(f"Warning: Failed to get mRNA sequence for {ensembl_transcript_id}: {e}", flush=True)
            df.loc[index, "mrna_sequence"] = ""
        
        try:
            mirna_matches = mirna_df[mirna_df["mirna_id"] == mirna_id]
            if len(mirna_matches) > 0:
                mirna_sequence = mirna_matches["sequence"].iloc[0]
                df.loc[index, "mirna_sequence"] = mirna_sequence
            else:
                print(f"No miRNA sequence found for {mirna_id}")
                df.loc[index, "mirna_sequence"] = ""
        except Exception as e:
            print(f"Warning: Failed to get miRNA sequence for {mirna_id}: {e}", flush=True)
            df.loc[index, "mirna_sequence"] = ""
    print(f"Processed {len(df)} interactions")
    return df

# main function to put together miRNA id, mRNA id, miRNA sequence, mRNA sequence, and seed start and end
def main(mirna_path, interaction_path, output_path, save_dir, cds_path, utr3_path):
    try:
        interaction = load_interaction_file(
            interaction_path=interaction_path, 
            mirna_path=mirna_path, 
            cds_path=cds_path, 
            utr3_path=utr3_path, 
            threshold=0.85, 
            save_dir=save_dir)
        interaction.to_csv(output_path, sep='\t', index=False)
        print(f"Data is saved to {output_path}")
        print(f"Processed {len(interaction)} interactions")
    except Exception as e:
        print(f"Error processing data: {e}", flush=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirna_path", type=str, required=True)
    parser.add_argument("--cds_path", type=str, required=True)
    parser.add_argument("--utr3_path", type=str, required=True)
    parser.add_argument("--interaction_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    save_dir = os.path.dirname(args.output_path)
    main(mirna_path=args.mirna_path, 
         cds_path=args.cds_path, 
         utr3_path=args.utr3_path, 
         interaction_path=args.interaction_path, 
         output_path=args.output_path, 
         save_dir=save_dir,
         )

