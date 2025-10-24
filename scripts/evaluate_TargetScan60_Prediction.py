# This script takes input file such as targetscan_60_output.txt
#  and evaluate the prediction performance.
# The binding label and ground target sites are given in the input file: TargetScan_dataset/TargetScan_validation_500_randomized_start.csv
# The script will need to first read the input files
# and for each row in the TargetScan_validation_500_randomized_start.csv, 
# it will need to find the corresponding row in the targetscan_60_output.txt by
# 1. matching the transcript ID and miRNA family name. miRNA family name is matched by reading the miR_Family_info_all.txt file 
# and find the miRNA family for the miRNA MiRBase ID.
# 2. once it finds the corresponding row in the output file, if the label is 1 in the 
# TargetScan_validation_500_randomized_start.csv, the number of correctly predicted positive sample + 1. Proceed to step 3. 
# If the label is 0 in the TargetScan_validation_500_randomized_start.csv, the number of correctly predicted negative sample + 0.
# If the label is 0 in the TargetScan_validation_500_randomized_start.csv, and it does not find the corresponding row in the targetscan_60_output.txt, the number of incorrectly predicted negative sample + 1.
# 3. If it finds the corresponding row in the output file, and the label is 1 in the TargetScan_validation_500_randomized_start.csv, it will need to check if the predicted
# seed start and end are the same as the ground truth seed start and end.
# If the 0-based seed start and end are the same as the ground truth seed start and end, the number of exact match + 1.
# If the 0-based seed start and end are not the same as the ground truth seed start and end, calculate the overlap between the predicted and ground truth seed.
# The overlap is calculated as the number of bases that are the same between the predicted and ground truth seed.
# This overlap is F1 score.
# Once all rows in the TargetScan_validation_500_randomized_start.csv are processed, the script will print the number of correctly predicted positive sample as accuracy, exact match rate and F1 score.
"""
Evaluate TargetScan Perl output against a paired validation CSV.

Inputs
------
--ts_sites: Path to TargetScan site table (e.g., targetscan_60_output.txt)
--pairs_csv: Path to your validation CSV:
    Required columns (case-insensitive, flexible):
      - Transcript ID
      - miRNA ID  (or MiRBase ID)
      - miRNA sequence
      - seed start    (0-based; your ground truth)
      - seed end      (0-based inclusive; your ground truth)
      - label         (1=positive, 0=negative)
--mir_fam: Path to downloaded miR_Family_info.txt (the *full* file that has
           columns like: miR family, Seed+m8, Species ID, MiRBase ID, ...)

Behavior
--------
- Map each row's MiRBase ID (or miRNA ID) to a TargetScan "miR family".
- Find all TS6 sites matching (Transcript ID, species, miR family).
- Label-level scoring:
    TP:  label=1 and at least one site found
    TN:  label=0 and no site found
    FP:  label=0 and at least one site found
    FN:  label=1 and no site found
- For label=1 with ≥1 site:
    * Convert TS UTR_start/UTR_end (assumed 1-based inclusive) -> 0-based
    * Normalize to a 7-nt seed span:
        - '8mer' or '7mer-1a' => drop downstream A1 (use first 7 nt)
        - '7mer-m8'           => use 7 nt directly
        - '6mer'              => treat as 6; try to extend +1 on the 3' side if possible
    * Pick the site with the **largest overlap** with ground truth (0-based)
    * Exact match if (start,end) match exactly after normalization
    * "F1" per your definition = overlap count (0..7)

Outputs
-------
Prints TP, TN, FP, FN, Accuracy, Exact-match rate, Mean F1 (overlap count).

Notes
-----
- We assume TS6 outputs UTR coordinates as 1-based inclusive (typical for TS6).
  If your TS table is already 0-based, set ASSUME_TS_ONE_BASED=False below.
"""

import argparse
import pandas as pd
import numpy as np
import sys
from typing import Optional, Tuple

ASSUME_TS_ONE_BASED = True  # set False if your TS 'UTR_start/end' are already 0-based

# Accept common variants of column names
PAIR_COLS_MAP = {
    "transcript id": "Transcript ID",
    "transcript_id": "Transcript ID",
    "mirbase id": "MiRBase ID",
    "mirbase_id": "MiRBase ID",
    "mirna id": "miRNA ID",
    "mirna_id": "miRNA ID",
    "mirna sequence": "miRNA sequence",
    "mirna_sequence": "miRNA sequence",
    "seed start": "seed start",
    "seed_start": "seed start",
    "seed end": "seed end",
    "seed_end": "seed end",
    "label": "label",
}

def normalize_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        k = c.strip().lower()
        if k in PAIR_COLS_MAP:
            rename[c] = PAIR_COLS_MAP[k]
    df = df.rename(columns=rename)
    required = ["Transcript ID", "miRNA sequence", "seed start", "seed end", "label"]
    # We need either MiRBase ID or miRNA ID to map to family
    if "MiRBase ID" not in df.columns and "miRNA ID" not in df.columns:
        raise SystemExit("pairs_csv must include either 'MiRBase ID' or 'miRNA ID'.")
    for r in required:
        if r not in df.columns:
            raise SystemExit(f"Missing required column in pairs_csv: {r}")
    return df

def load_family_map(mir_fam_path: str) -> pd.DataFrame:
    # Try to read as TSV; the downloaded file is tab-delimited with header
    fam = pd.read_csv(mir_fam_path, sep="\t", dtype=str)
    # Normalize likely header variants
    colmap = {}
    for c in fam.columns:
        k = c.strip().lower()
        if k in ("mir family", "miR family".lower(), "mir_family", "miR_family".lower()):
            colmap[c] = "miR family"
        elif k in ("seed+m8", "seed2-8", "seed_2_8", "seed"):
            colmap[c] = "Seed+m8"
        elif k in ("species id", "species_id"):
            colmap[c] = "Species ID"
        elif k in ("mirbase id", "mirbase_id"):
            colmap[c] = "MiRBase ID"
    fam = fam.rename(columns=colmap)
    needed = ["miR family", "Seed+m8", "Species ID", "MiRBase ID"]
    for n in needed:
        if n not in fam.columns:
            raise SystemExit(f"miR_Family_info file missing column: {n}")
    # T->U for seed (not strictly needed here, but keep tidy)
    fam["Seed+m8"] = fam["Seed+m8"].str.upper().str.replace("T","U", regex=False)
    return fam

def species_from_mirid(mir_id: str) -> Optional[int]:
    if not isinstance(mir_id, str):
        return None
    s = mir_id.strip().lower()
    if s.startswith("hsa-"):
        return 9606
    if s.startswith("mmu-"):
        return 10090
    return None

def normalize_ts_columns(ts: pd.DataFrame) -> pd.DataFrame:
    # Typical TS6 columns:
    # a_Gene_ID, miRNA_family_ID, species_ID, MSA_start, MSA_end,
    # UTR_start, UTR_end, Group_num, Site_type, ...
    rename = {}
    for c in ts.columns:
        k = c.strip().lower()
        if k in ("a_gene_id", "geneid", "gene_id"):
            rename[c] = "a_Gene_ID"
        elif k in ("mirna_family_id", "miRNA family ID".lower()):
            rename[c] = "miRNA_family_ID"
        elif k in ("species_id",):
            rename[c] = "species_ID"
        elif k in ("utr_start",):
            rename[c] = "UTR_start"
        elif k in ("utr_end",):
            rename[c] = "UTR_end"
        elif k in ("site_type", "site type"):
            rename[c] = "Site_type"
    ts = ts.rename(columns=rename)
    need = ["a_Gene_ID","miRNA_family_ID","species_ID","UTR_start","UTR_end","Site_type"]
    for n in need:
        if n not in ts.columns:
            raise SystemExit(f"TargetScan output missing column: {n}")
    # coerce types
    ts["a_Gene_ID"] = ts["a_Gene_ID"].astype(str)
    ts["miRNA_family_ID"] = ts["miRNA_family_ID"].astype(str)
    ts["species_ID"] = pd.to_numeric(ts["species_ID"], errors="coerce").astype("Int64")
    ts["UTR_start"] = pd.to_numeric(ts["UTR_start"], errors="coerce")
    ts["UTR_end"]   = pd.to_numeric(ts["UTR_end"], errors="coerce")
    ts["Site_type"] = ts["Site_type"].astype(str)
    return ts

def ts_site_to_seed_span_0based(row: pd.Series, seq_len: Optional[int]=None) -> Optional[Tuple[int,int]]:
    """
    Convert a TS site to a 0-based [start,end] *seed* span of length 7 when possible.
    Assumes TS UTR_start/UTR_end are 1-based inclusive unless ASSUME_TS_ONE_BASED=False.

    For 8mer or 7mer-1a, drop the trailing A1.
    For 7mer-m8, use the 7-nt span as given.
    For 6mer, use the 6-nt span and extend +1 on 3' side if possible (best effort),
    else compare as 6 nt.

    Returns (start0, end0) inclusive in 0-based coordinates, or None if coords invalid.
    """
    u0 = row["UTR_start"]
    u1 = row["UTR_end"]
    if np.isnan(u0) or np.isnan(u1):
        return None
    start = int(u0) - 1 if ASSUME_TS_ONE_BASED else int(u0)
    end   = int(u1) - 1 if ASSUME_TS_ONE_BASED else int(u1)

    stype = row["Site_type"].strip().lower()
    length = end - start + 1
    if length <= 0:
        return None

    # Normalize to seed (7 nt) where applicable
    if stype in ("8mer", "m8:1a", "m8:1a (8mer)"):
        # usually length 8, drop last base (A1)
        if length >= 8:
            return (start, start + 6)
        else:
            # fallback: take first 7 if possible
            if length >= 7:
                return (start, start + 6)
            return (start, end)  # as-is
    elif stype in ("7mer-1a", "1a", "7mer-1a (1a)"):
        # usually length 7 (6 seed + A1) -> drop A1
        if length >= 7:
            return (start, start + 6)
        if length == 6:
            # no A1 captured? expand by one if we know sequence length
            if seq_len is not None and end + 1 < seq_len:
                return (start, end + 1)
            return (start, end)
        return (start, end)
    elif stype in ("7mer-m8", "m8"):
        # 7 nt seed (2..8)
        if length >= 7:
            return (start, start + 6)
        # if shorter (shouldn't happen), best-effort extend
        if seq_len is not None and start + 6 < seq_len:
            return (start, start + 6)
        return (start, end)
    elif stype in ("6mer",):
        # 6 nt → try to extend by one to 7 if possible
        if seq_len is not None and end + 1 < seq_len:
            return (start, end + 1)
        return (start, end)
    else:
        # Unknown label; use given span
        return (start, end)

def overlap_len(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    """Inclusive overlap length between two [start,end] 0-based intervals."""
    if a is None or b is None:
        return 0
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0, e - s + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts_sites", required=True, help="TargetScan site table (TS6 output)")
    ap.add_argument("--pairs_csv", required=True, help="Validation CSV")
    ap.add_argument("--mir_fam", required=True, help="miR_Family_info.txt (full, with MiRBase ID)")
    ap.add_argument("--species", choices=["auto","human","mouse"], default="auto",
                    help="If 'auto', species inferred from miRNA prefix (hsa/mmu).")
    args = ap.parse_args()

    # Load inputs
    pairs = pd.read_csv(args.pairs_csv)
    pairs = normalize_pair_columns(pairs)
    fam   = load_family_map(args.mir_fam)
    ts    = pd.read_csv(args.ts_sites, sep="\t")
    ts    = normalize_ts_columns(ts)

    # Map each pair -> (species_id, ts_family)
    # Prefer "MiRBase ID"; fallback to "miRNA ID"
    if "MiRBase ID" in pairs.columns and pairs["MiRBase ID"].notna().any():
        key_col = "MiRBase ID"
    else:
        key_col = "miRNA ID"

    # Determine species per row
    if args.species == "human":
        pairs["species_id"] = 9606
    elif args.species == "mouse":
        pairs["species_id"] = 10090
    else:
        pairs["species_id"] = pairs[key_col].apply(species_from_mirid).fillna(9606).astype(int)

    # Build MiRBase ID -> family map (species-aware)
    fam_sub = fam[["MiRBase ID","miR family","Species ID"]].dropna().drop_duplicates()
    # Some files use space/comma in MiRBase ID cell (multiple IDs). Split if needed.
    fam_rows = []
    for _, r in fam_sub.iterrows():
        ids = str(r["MiRBase ID"]).split(",")
        for mid in ids:
            fam_rows.append((mid.strip(), r["miR family"], int(r["Species ID"])))
    fam_map = pd.DataFrame(fam_rows, columns=["MiRBase ID","miR family","Species ID"]).drop_duplicates()

    # Join family name to pairs by (MiRBase ID, species_id). If not found, try by miRNA ID text.
    pairs = pairs.merge(fam_map, how="left",
                        left_on=[key_col,"species_id"],
                        right_on=["MiRBase ID","Species ID"])\
                 .rename(columns={"miR family":"ts_family"})\
                 .drop(columns=["MiRBase ID_y","Species ID"], errors="ignore")\
                 .rename(columns={"MiRBase ID_x":"MiRBase ID"})

    # If still missing ts_family, try fuzzy join by miRNA ID text contained in MiRBase ID field (rare)
    missing = pairs["ts_family"].isna()
    if missing.any() and key_col == "miRNA ID":
        alt = fam_map.copy()
        alt["miRNA ID"] = alt["MiRBase ID"]
        pairs.loc[missing, :] = pairs.loc[missing, :].merge(
            alt[["miRNA ID","Species ID","miR family"]],
            how="left",
            left_on=["miRNA ID","species_id"],
            right_on=["miRNA ID","Species ID"]
        ).rename(columns={"miR family":"ts_family"}).drop(columns=["Species ID"])

    # Prepare counters
    TP = TN = FP = FN = 0
    exact_matches = 0
    f1_overlaps = []  # per-positive-with-site overlap counts (0..7)
    positives_with_site = 0

    # Normalize column names used below
    pairs["Transcript ID"] = pairs["Transcript ID"].astype(str)
    # ground truth seed start/end must be integers (0-based inclusive)
    pairs["seed start"] = pd.to_numeric(pairs["seed start"], errors="coerce")
    pairs["seed end"]   = pd.to_numeric(pairs["seed end"], errors="coerce")
    pairs["label"]      = pd.to_numeric(pairs["label"], errors="coerce").fillna(0).astype(int)

    # Index TS by (Transcript ID, species, family)
    ts["key"] = (ts["a_Gene_ID"].astype(str) + "||" +
                 ts["species_ID"].astype(str) + "||" +
                 ts["miRNA_family_ID"].astype(str))

    ts_groups = ts.groupby("key")

    total = pairs.shape[0]

    for idx, r in pairs.iterrows():
        tid = str(r["Transcript ID"])
        sp  = int(r["species_id"])
        fam_name = r.get("ts_family", None)
        label = int(r["label"])
        gt_seed = None
        if not np.isnan(r["seed start"]) and not np.isnan(r["seed end"]):
            gt_seed = (int(r["seed start"]), int(r["seed end"]))

        found_site = False
        best_overlap = 0
        exact_here = False

        if pd.notna(fam_name):
            key = f"{tid}||{sp}||{fam_name}"
            if key in ts_groups.groups:
                found_site = True
                sub = ts_groups.get_group(key)

                # evaluate each site; pick the one with max overlap vs ground truth seed
                for _, s in sub.iterrows():
                    pred_seed = (s["UTR_start"]-1, s["UTR_end"]-1) # just -1 from the prediced seed start and end
                    if gt_seed is not None and pred_seed is not None:
                        ov = overlap_len(gt_seed, pred_seed)
                        if ov > best_overlap:
                            best_overlap = ov
                            exact_here = (pred_seed == gt_seed)

        # Count confusion
        if label == 1 and found_site:
            TP += 1
        elif label == 1 and not found_site:
            FN += 1
        elif label == 0 and found_site:
            FP += 1
        elif label == 0 and not found_site:
            TN += 1

        # For positives with a site, compute exact and "F1" (overlap count)
        if label == 1 and found_site and gt_seed is not None:
            positives_with_site += 1
            if exact_here:
                exact_matches += 1
            # per spec, "F1 score" == 2 * (precision * recall) / (precision + recall)
            T = gt_seed[1] - gt_seed[0] + 1
            P = pred_seed[1] - pred_seed[0] + 1
            precision = best_overlap / P
            recall = best_overlap / T
            F1 = 2 * (precision * recall) / (precision + recall) if best_overlap > 0 else 0.0
            f1_overlaps.append(F1)

    accuracy = (TP + TN) / total if total else 0.0
    exact_rate = (exact_matches / positives_with_site) if positives_with_site else 0.0
    mean_f1 = (np.mean(f1_overlaps) if f1_overlaps else 0.0)

    print("===== Evaluation Summary =====")
    print(f"Total rows: {total}")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Exact match rate (among positives with a TS site): {exact_rate:.4f} "
          f"({exact_matches}/{positives_with_site})")
    print(f"Mean F1 among positives with a TS site: {mean_f1:.4f}")
    # Optional: also print recall/precision if useful
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    print(f"Recall (label=1 detected): {recall:.4f}")
    print(f"Precision (TS detection given label=1): {precision:.4f}")

if __name__ == "__main__":
    main()
