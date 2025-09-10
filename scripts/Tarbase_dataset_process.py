#!/usr/bin/env python3
# fetch_transcripts_e102.py
import pandas as pd
import argparse, json, time, requests, re
from requests.adapters import HTTPAdapter, Retry

# fetch cds or utr sequences from ensembl using rest api
E = "https://e102.rest.ensembl.org"
H_SEQ   = {"Accept": "text/plain", "User-Agent": "cdna/1.0"}           # for /sequence/*
H_JSON  = {"Accept": "application/json", "User-Agent": "cdna/1.0"}     # for /lookup/*

def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent":"cdna/1.0"})
    return s

SESSION = make_session()

def cdna(transcript_id):
    try:
        r = SESSION.get(f"{E}/sequence/id/{transcript_id}?type=cdna", headers=H_SEQ, timeout=30)
        r.raise_for_status()
        return r.text.strip().upper()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch cDNA for {transcript_id}: {e}", flush=True)
        return ""

def cds_bounds(transcript_id):
    try:
        r = SESSION.get(f"{E}/lookup/id/{transcript_id}?expand=1", headers=H_JSON, timeout=30)
        r.raise_for_status()
        j = r.json()
        tr = j.get("Translation") or j.get("translation")
        if not tr: 
            return None, None
        return int(tr["start"]), int(tr["end"])  # cDNA coords, 1-based inclusive
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Warning: Failed to fetch CDS bounds for {transcript_id}: {e}", flush=True)
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

def build_transcript_to_sequence_map(path):
    transcripts_ids = unique_transcripts_from_microt(path)
    d = {}
    for item in transcripts_ids:
        transcript_id = item["ensembl_transcript_id"]
        cds_utr = item["cds_utr"].lower()
        
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
            print(f"Warning: Failed to fetch sequence for {transcript_id} ({cds_utr}): {e}", flush=True)
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

# load interaction file
# get pairs of miRNA and mRNA that have interaction score > 0.85
# seed end = transcript end, seed start = transcript end - seed length
def load_interaction_file(path, mirna_fasta_path, threshold=0.85):
    df = pd.read_csv(path, sep='\t')
    
    # Validate required columns exist
    required_cols = ["interaction_score", "mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[df["interaction_score"] >= threshold]
    df = df[["mirna", "mre_type", "cds_utr", "ensembl_transcript_id", "transcript_start", "transcript_end"]]
    
    transcripts_to_sequence = build_transcript_to_sequence_map(path)
    mirna_fasta = load_mirna_fasta(mirna_fasta_path)
    
    for index, row in df.iterrows():
        mirna_id, mre_type, cds_utr, ensembl_transcript_id, transcript_start, transcript_end = row
        cds_utr = cds_utr.lower()
        seed_length = get_seed_length(mre_type)
        df.loc[index, "seed start"] = transcript_end - seed_length
        df.loc[index, "seed end"] = transcript_end
        
        # Safe access to sequences with error handling
        try:
            mrna_sequence = transcripts_to_sequence.get(ensembl_transcript_id, {}).get(cds_utr, "")
            df.loc[index, "mrna_sequence"] = mrna_sequence
        except Exception as e:
            print(f"Warning: Failed to get mRNA sequence for {ensembl_transcript_id}: {e}", flush=True)
            df.loc[index, "mrna_sequence"] = ""
        
        try:
            mirna_sequence = mirna_fasta.get(mirna_id, "")
            df.loc[index, "mirna_sequence"] = mirna_sequence
        except Exception as e:
            print(f"Warning: Failed to get miRNA sequence for {mirna_id}: {e}", flush=True)
            df.loc[index, "mirna_sequence"] = ""
        
        df.loc[index, "mirna_id"] = mirna_id
    return df

# main function to put together miRNA id, mRNA id, miRNA sequence, mRNA sequence, and seed start and end
def main(mirna_fasta_path, interaction_path, output_path):
    try:
        interaction = load_interaction_file(interaction_path, mirna_fasta_path)
        interaction.to_csv(output_path, sep=',', index=False)
        print(f"Data is saved to {output_path}")
        print(f"Processed {len(interaction)} interactions")
    except Exception as e:
        print(f"Error processing data: {e}", flush=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirna_fasta_path", type=str, required=True)
    parser.add_argument("--interaction_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.mirna_fasta_path, args.interaction_path, args.output_path)

