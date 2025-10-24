#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TargetScan 6.0 input files from a CSV of windowed mRNA sequences.

Inputs (CSV columns):
  Transcript ID, miRNA ID, mRNA sequence, miRNA sequence, seed start, seed end, label

Outputs (tab-delimited, no headers):
  - miR_Family_info_all.txt: <miRNA_family>  <7mer-seed-2..8>  <species_tax_id>
  - UTR_sequences_all.txt   : <GeneOrTranscriptID>  <species_tax_id>  <UTR_sequence_RNA>

Then run:
  perl targetscan_60.pl miR_Family_info_all.txt UTR_sequences_all.txt targetscan_60_output.txt
"""

import argparse
import pandas as pd
import re
from collections import defaultdict

# Minimal species mapping; extend if needed
SPECIES_TAX = {
    "ensm": 10090,   # mouse
    "enst": 9606,  # human
}

def infer_species_tax(tr_id: str) -> int:
    """
    Infer species tax ID from transcript ID prefix (e.g., ensm..., enst...).
    Defaults to None if no prefix is found.
    """
    m = re.match(r"^([a-z]{4})", tr_id.lower())
    # print(m.group(1))
    if m and m.group(1) in SPECIES_TAX:
        return SPECIES_TAX[m.group(1)]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Your dataset CSV")
    ap.add_argument("--out_utr", default="UTR_sequences_all.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    required = ["Transcript ID", "mRNA sequence"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # ---- Build UTR_sequences_all.txt ----
    # Format per line: <GeneOrTranscriptID> <species_tax_id> <UTR_sequence_RNA>
    # We’ll use Transcript ID as the first field (allowed by TS6 docs).
    utr_rows = []
    for _, r in df[["Transcript ID", "mRNA sequence"]].drop_duplicates().iterrows():
        tid = str(r["Transcript ID"]).strip()
        tax = infer_species_tax(str(r["Transcript ID"]))
        utr = str(r["mRNA sequence"]).upper()
        utr_rows.append((tid, tax, utr))

    # Deduplicate identical (tid, tax, seq) triplets
    seen_utr = set()
    utr_lines = []
    for tid, tax, utr in utr_rows:
        key = (tid, tax, utr)
        if key in seen_utr:
            continue
        seen_utr.add(key)
        utr_lines.append(f"{tid}\t{tax}\t{utr}")

    with open(args.out_utr, "w") as f:
        f.write("\n".join(utr_lines) + "\n")

    print(f"[OK] Wrote {len(utr_lines)} rows to {args.out_utr}")

if __name__ == "__main__":
    main()
