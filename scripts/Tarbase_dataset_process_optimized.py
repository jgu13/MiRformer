#!/usr/bin/env python3
# fetch_transcripts_e102.py - Optimized version for large files
import argparse, json, time, requests, re
import pandas as pd
import numpy as np

# fetch cds or utr sequences from ensembl using rest api
E = "https://e102.rest.ensembl.org"
H = {"Content-Type":"text/plain","User-Agent":"cdna/1.0"}

def cdna(transcript_id):
    try:
        r = requests.get(f"{E}/sequence/id/{transcript_id}?type=cdna", headers=H, timeout=30)
        r.raise_for_status()
        return r.text.strip().upper()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch cDNA for {transcript_id}: {e}")
        return ""

def cds_bounds(transcript_id):
    try:
        r = requests.get(f"{E}/lookup/id/{transcript_id}?expand=1", headers={"Accept":"application/json"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        tr = j.get("Translation") or j.get("translation")
        if not tr: 
            return None, None
        return int(tr["start"]), int(tr["end"])  # cDNA coords, 1-based inclusive
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Warning: Failed to fetch CDS bounds for {transcript_id}: {e}")
        return None, None

def cds(transcript_id):
    seq = cdna(transcript_id)
    cds_start, cds_end = cds_bounds(transcript_id)
    if not cds_start or not cds_end:
        return ""
    return seq[cds_start-1:cds_end]  # (cds_start .. cds_end), 0-based slice

def utr3(transcript_id):
    seq = cdna(transcript_id)
    _, cds_end = cds_bounds(transcript_id)
    if not cds_end or cds_end >= len(seq):  # no 3′UTR
        return ""
    return seq[cds_end:]  # (cds_end+1 .. end), 0-based slice

# fetch unique transcript ids from microT txt
def unique_transcripts_from_microt(path):
    """
    Get unique transcript ids and return a list of dictionary with item 
    {"transcript_id": transcript_id, "cds_utr": cds_utr}
    """
    df = pd.read_csv(path, sep='\t')
    df = df[["ensembl_transcript_id", "cds_utr"]].drop_duplicates()  # remove duplicated transcripts
    tx = df.to_dict(orient="records")
    return tx

def build_transcript_to_sequence_map_from_df(df):
    """Build transcript to sequence map from dataframe instead of file"""
    unique_transcripts = df[["ensembl_transcript_id", "cds_utr"]].drop_duplicates()
    d = {}
    
    print(f"Fetching sequences for {len(unique_transcripts)} unique transcript/region combinations...")
    for index, row in unique_transcripts.iterrows():
        transcript_id = row["ensembl_transcript_id"]
        cds_utr = row["cds_utr"].lower()
        
        # Initialize transcript entry if not exists
        if transcript_id not in d:
            d[transcript_id] = {"cds": None, "utr3": None}
        
        try:
            if cds_utr == "cds":
                sequence = cds(transcript_id)
            elif cds_utr == "utr3":
                sequence = utr3(transcript_id)
            else:
                sequence = ""
            d[transcript_id][cds_utr] = sequence
        except Exception as e:
            print(f"Warning: Failed to fetch sequence for {transcript_id} ({cds_utr}): {e}")
            d[transcript_id][cds_utr] = ""
    return d

# load mature miRNA fasta file
def load_mirna_fasta(path):
    d = {}
    name=None
    chunks=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                # header like: ">hsa-let-7a-2-3p MIMAT0000064 Homo sapiens ..."
                if name and chunks: 
                    d[name] = "".join(chunks).upper().replace("U","T")
                name = line[1:].split()[0]; chunks=[]
            else:
                chunks.append(line)
    if name and chunks: d[name] = "".join(chunks).upper().replace("U","T")
    return d

# a helper function to get seed length, seed start and seed end
def get_seed_length(mre_type):
    m = re.search(r"(\d+)\s*mer", mre_type, flags=re.I)
    if m: return int(m.group(1))
    mt = mre_type.lower()
    if mt in {"7mer-m8","7mer-a1"}: return 7
    if "offset-6mer" in mt: return 6
    return 7  # default

# Optimized load interaction file with chunked processing for large files
def load_interaction_file(path, mirna_fasta_path, threshold=0.85, chunk_size=100000):
    print(f"Loading large file {path} with chunked processing...")
    
    # First, get column names and validate
    sample_df = pd.read_csv(path, sep='\t', nrows=5)
    required_cols = ["interaction_score", "mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]
    missing_cols = [col for col in required_cols if col not in sample_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Process file in chunks to handle large files efficiently
    chunk_list = []
    total_rows = 0
    filtered_rows = 0
    
    print("Processing file in chunks...")
    for chunk_num, chunk in enumerate(pd.read_csv(path, sep='\t', chunksize=chunk_size)):
        total_rows += len(chunk)
        print(f"Processing chunk {chunk_num + 1}, rows so far: {total_rows}")
        
        # Filter by threshold early to reduce memory usage
        filtered_chunk = chunk[chunk["interaction_score"] >= threshold]
        if len(filtered_chunk) == 0:
            continue
            
        # Select only needed columns
        filtered_chunk = filtered_chunk[["mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]].copy()
        filtered_rows += len(filtered_chunk)
        chunk_list.append(filtered_chunk)
    
    print(f"Total rows processed: {total_rows}, filtered rows: {filtered_rows}")
    
    # Combine all chunks
    if chunk_list:
        df = pd.concat(chunk_list, ignore_index=True)
    else:
        # Create empty dataframe with correct columns if no data passes filter
        df = pd.DataFrame(columns=["mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"])
    
    print(f"Combined dataframe shape: {df.shape}")
    
    # Build transcript to sequence map (only for unique transcripts)
    print("Building transcript to sequence map...")
    transcripts_to_sequence = build_transcript_to_sequence_map_from_df(df)
    print(f"Built transcripts to sequence map with {len(transcripts_to_sequence)} unique transcripts")

    # Load miRNA fasta
    print("Loading miRNA fasta...")
    mirna_fasta = load_mirna_fasta(mirna_fasta_path)
    print(f"Loaded {len(mirna_fasta)} miRNA sequences")
    
    # Vectorized operations for better performance
    print("Processing sequences...")
    df["cds_utr"] = df["cds_utr"].str.lower()
    df["seed_length"] = df["mre_type"].apply(get_seed_length)
    df["seed start"] = df["transcript_end"] - df["seed_length"]
    df["seed end"] = df["transcript_end"]
    
    # Vectorized sequence lookup
    def get_mrna_sequence(row):
        transcript_id = row["ensembl_transcript_id"]
        cds_utr = row["cds_utr"]
        return transcripts_to_sequence.get(transcript_id, {}).get(cds_utr, "")
    
    def get_mirna_sequence(row):
        mirna_id = row["mirna"]
        return mirna_fasta.get(mirna_id, "")
    
    df["mrna_sequence"] = df.apply(get_mrna_sequence, axis=1)
    df["mirna_sequence"] = df.apply(get_mirna_sequence, axis=1)
    df["mirna_id"] = df["mirna"]
    
    print(f"Final dataset shape: {df.shape}")
    return df

# main function to put together miRNA id, mRNA id, miRNA sequence, mRNA sequence, and seed start and end
def main(mirna_fasta_path, interaction_path, output_path):
    try:
        interaction = load_interaction_file(interaction_path, mirna_fasta_path)
        interaction.to_csv(output_path, sep=',', index=False)
        print(f"Data is saved to {output_path}")
        print(f"Processed {len(interaction)} interactions")
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirna_fasta_path", type=str, required=True)
    parser.add_argument("--interaction_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.mirna_fasta_path, args.interaction_path, args.output_path)
