import argparse
import sys
import os
import pandas as pd
import random

def parse_args():
    ap = argparse.ArgumentParser("Augment paired_output.tsv with mRNA and miRNA sequences, extract 500nt windows")
    ap.add_argument("--pairs", required=True, help="paired_output.tsv (contains transcript, paired_seed_start, paired_seed_end, utr_pos_ts_1based)")
    ap.add_argument("--mrna-seq", required=True, help="human_mrna_seq.csv.gz (TargetScan mRNA sequences)")
    ap.add_argument("--mir-family", required=True, help="miR_Family_Info.txt (contains MiRBase ID, Mature sequence)")
    ap.add_argument("--strip-transcript-version", action="store_true", help="remove .version from ENST before matching")
    ap.add_argument("--miRNA-col", default="miRNAname", help="column name for miRNA name in paired table (e.g., hsa-let-7a-5p)")
    ap.add_argument("--window-size", type=int, default=500, help="window size for mRNA segment (default: 500)")
    ap.add_argument("--out", required=True, help="output TSV")
    return ap.parse_args()

def revcomp(seq: str) -> str:
    comp = str.maketrans("ACGTUacgtu","TGCAAtgcaa")
    return seq.translate(comp)[::-1]

def load_mir_seq_from_family_info(path):
    """
    Read from TargetScan miR_Family_Info.txt:
      MiRBase ID -> Mature sequence
    Only do exact name matching (e.g., hsa-let-7a-3p), no family/species mapping.
    """
    df = pd.read_csv(path, sep='\t')
    
    # filter for human
    species_ids = [9606]
    df          = df[df["Species ID"].isin(species_ids)]

    # tolerate different spellings
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("mirbase id") or cols.get("mirbase_id") or cols.get("mature id") or cols.get("mirna")
    seq_col = cols.get("mature sequence") or cols.get("mature_sequence") or cols.get("sequence")
    if not id_col or not seq_col:
        raise ValueError(f"[miR] missing 'MiRBase ID' and 'Mature sequence' columns in {path}. Actual columns: {df.columns.tolist()}")

    # direct mapping (exact matching)
    df[id_col] = df[id_col].astype(str).str.strip()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.replace("T","U").str.upper()  # use RNA alphabet
    mp = dict(zip(df[id_col], df[seq_col]))
    return mp

def load_mrna_sequences(path):
    """
    Read from human_mrna_seq.csv.gz:
      Transcript ID -> mRNA sequence
    """
    try:
        df = pd.read_csv(path, sep="\t", compression="gzip")
    except Exception as e:
        raise ValueError(f"[mRNA] Failed to load {path}: {e}")
    
    if "Transcript ID" not in df.columns or "mRNA sequence" not in df.columns:
        raise ValueError(f"[mRNA] missing 'Transcript ID' or 'mRNA sequence' columns. Actual columns: {df.columns.tolist()}")
    
    # Clean up sequences (convert to uppercase, replace T with U for RNA)
    df["Transcript ID"] = df["Transcript ID"].astype(str).str.strip()
    df["mRNA sequence"] = df["mRNA sequence"].astype(str).str.strip().str.upper().str.replace("T", "U")
    
    mrna_map = dict(zip(df["Transcript ID"], df["mRNA sequence"]))
    
    # Also create a version-stripped mapping
    mrna_map_novers = {k.split(".")[0]: v for k, v in mrna_map.items()}
    
    return mrna_map, mrna_map_novers

def extract_window_with_cleavage(mrna_seq, seed_start_1based, seed_end_1based, 
                                  cleavage_pos_1based, window_size=500):
    """
    Extract a window of specified size from mRNA sequence.
    
    The window must contain:
    - cleavage site
    - seed start and seed end
    
    Constraints:
    - Window start can be anywhere before the cleavage site
    - Window end can be anywhere after the seed end
    
    Args:
        mrna_seq: Full mRNA sequence
        seed_start_1based: Seed start position (1-based, TargetScan format)
        seed_end_1based: Seed end position (1-based, TargetScan format)
        cleavage_pos_1based: Cleavage site position (1-based, TargetScan format)
        window_size: Size of window to extract (default: 500)
    
    Returns:
        Dictionary with:
            - mrna_window: extracted mRNA segment
            - seed_start_0based: adjusted seed start (0-based relative to window)
            - seed_end_0based: adjusted seed end (0-based relative to window)
            - cleavage_pos_0based: adjusted cleavage position (0-based relative to window)
            - window_start_1based: start position of window in original sequence
            - success: True if successful, False otherwise
            - error: Error message if failed
    """
    seq_len = len(mrna_seq)
    
    # Convert to 0-based indices for easier calculation
    seed_start_0 = seed_start_1based - 1
    seed_end_0 = seed_end_1based - 1
    cleavage_pos_0 = cleavage_pos_1based - 1

    # return mrna sequence if the length of the sequence is shorter than window size
    if seq_len < window_size:
        return {
            "mrna_window": mrna_seq,
            "seed_start_0based": seed_start_0,
            "seed_end_0based": seed_end_0,
            "cleavage_pos_0based": cleavage_pos_0,
            "window_start_1based": 1,
            "success": True,
            "error": None
        }
    
    # Validate positions
    if not (0 <= seed_start_0 < seq_len and 0 <= seed_end_0 < seq_len and 0 <= cleavage_pos_0 < seq_len):
        return {
            "mrna_window": None,
            "seed_start_0based": None,
            "seed_end_0based": None,
            "cleavage_pos_0based": None,
            "window_start_1based": None,
            "success": False,
            "error": "Invalid positions"
        }
    
    # Find the minimum required span (from cleavage to seed_end)
    min_pos = min(seed_start_0, seed_end_0, cleavage_pos_0)
    max_pos = max(seed_start_0, seed_end_0, cleavage_pos_0)
    required_span = max_pos - min_pos + 1
    
    # Window must start at or before min_pos
    # Window must end at or after max_pos
    # Window start can be anywhere from [max(0, max_pos - window_size + 1), min_pos]
    earliest_start = 0
    latest_start = min_pos
    
    # Randomly choose window start position within valid range
    window_start_0 = random.randint(earliest_start, latest_start + 1)
    window_end_0 = window_start_0 + window_size
    
    # Make sure we don't exceed sequence length
    if window_end_0 > seq_len:
        window_end_0 = seq_len
        window_start_0 = max(0, window_end_0 - window_size)
    
    # Extract the window
    mrna_window = mrna_seq[window_start_0:window_end_0]
    
    # Calculate adjusted positions (0-based relative to window)
    adjusted_seed_start = seed_start_0 - window_start_0
    adjusted_seed_end = seed_end_0 - window_start_0
    adjusted_cleavage = cleavage_pos_0 - window_start_0
    
    return {
        "mrna_window": mrna_window,
        "seed_start_0based": adjusted_seed_start,
        "seed_end_0based": adjusted_seed_end,
        "cleavage_pos_0based": adjusted_cleavage,
        "window_start_1based": window_start_0 + 1,  # Convert back to 1-based
        "success": True,
        "error": None
    }

def main():
    args = parse_args()
    
    print(f"[INFO] Loading paired data from {args.pairs}")
    df = pd.read_csv(args.pairs, sep=None, engine="python")
    
    # Validate required columns
    need_cols = ["transcript", "paired_seed_start", "paired_seed_end", "utr_pos_ts_1based"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[pairs] missing required column '{c}' in {args.pairs}")
    
    mirname_col = args.miRNA_col
    if mirname_col not in df.columns:
        raise ValueError(f"[pairs] missing miRNA column '{mirname_col}' (use --miRNA-col)")
    
    # Load mRNA sequences
    print(f"[INFO] Loading mRNA sequences from {args.mrna_seq}")
    mrna_map, mrna_map_novers = load_mrna_sequences(args.mrna_seq)
    print(f"[INFO] Loaded {len(mrna_map)} mRNA sequences")
    
    # Load miRNA sequences
    print(f"[INFO] Loading miRNA sequences from {args.mir_family}")
    mir_map = load_mir_seq_from_family_info(args.mir_family)
    print(f"[INFO] Loaded {len(mir_map)} miRNA sequences")
    
    # Process each pair
    out_rows = []
    success_count = 0
    fail_count = 0
    
    print(f"[INFO] Processing {len(df)} paired records...")
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(df)} records...")
        
        # Extract basic information
        tx = row["transcript"]
        mirna_name = row[mirname_col]
        seed_start_1based = row["paired_seed_start"]
        seed_end_1based = row["paired_seed_end"]
        cleavage_pos_1based = row["utr_pos_ts_1based"]
        
        # Get mRNA sequence
        tx_key = str(tx).strip()
        if args.strip_transcript_version:
            tx_key_lookup = tx_key.split(".")[0]
        else:
            tx_key_lookup = tx_key
        
        mrna_seq = mrna_map.get(tx_key)
        if mrna_seq is None:
            if args.strip_transcript_version:
                mrna_seq = mrna_map_novers.get(tx_key_lookup)
        
        # Get miRNA sequence
        mir_seq = None
        if isinstance(mirna_name, str) and pd.notna(mirna_name):
            mir_seq = mir_map.get(mirna_name.strip())
        
        # Extract window
        if mrna_seq is None:
            fail_count += 1
            print(f"[INFO] Failed to find mRNA sequence for {tx_key}")
            print(f"[INFO] tx_key: {tx_key}")
            continue
        
        if mir_seq is None:
            fail_count += 1
            print(f"[INFO] Failed to find miRNA sequence for {mirna_name}")
            print(f"[INFO] miRNA name: {mirna_name}")
            continue
        
        if pd.isna(seed_start_1based) or pd.isna(seed_end_1based) or pd.isna(cleavage_pos_1based):
            fail_count += 1
            print(f"[INFO] Failed to find seed start, end, and cleavage site for {tx_key}")
            print(f"[INFO] seed start: {seed_start_1based}")
            print(f"[INFO] seed end: {seed_end_1based}")
            print(f"[INFO] cleavage site: {cleavage_pos_1based}")
            continue
        
        # Extract window with cleavage site
        result = extract_window_with_cleavage(
            mrna_seq,
            int(seed_start_1based),
            int(seed_end_1based),
            int(cleavage_pos_1based),
            window_size=args.window_size
        )
        
        if not result["success"]:
            fail_count += 1
            print(f"[INFO] Failed to extract window for {tx_key}")
            continue
        
        # Create output record
        out_row = {
            "transcript": tx,
            "mirna": mirna_name,
            "mrna_sequence": result["mrna_window"],
            "mirna_sequence": mir_seq,
            "seed_start": result["seed_start_0based"],
            "seed_end": result["seed_end_0based"],
            "cleavage_site": result["cleavage_pos_0based"]
        }
        
        out_rows.append(out_row)
        success_count += 1
    
    # Create output dataframe
    out_df = pd.DataFrame(out_rows)
    
    # Write output
    out_df.to_csv(args.out, sep=",", index=False)
    
    print(f"\n[SUMMARY]")
    print(f"  Total records: {len(df)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output written to: {args.out}")
    print(f"[OK] Done!")

if __name__ == "__main__":
    main()
