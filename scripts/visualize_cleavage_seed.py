#!/usr/bin/env python3
"""
visualize_cleavage_seed.py

Create:
  (1) per-transcript UTR plots with seed spans + cleavage markers (PDF and optional PNGs)
  (2) BED tracks for IGV/UCSC:
        - cleavage point features (from cleaveLocus)
        - seed site intervals (UTR -> genome via TS7 hg19 UTR blocks)

Inputs:
  --pairs: filtered paired_output.tsv  (expects columns: transcript, cleaveLocus, utr_pos_ts_1based,
                                        total_utr_len, paired_seed_start, paired_seed_end,
                                        paired_seed_type, miRNAname)
  --ts-gff: TSHuman_7_hg19_3UTRs.gff (TS7 hg19 UTR blocks; col9 = transcript id)
"""

import argparse, os, re, math
from collections import defaultdict, namedtuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

UTRBlock = namedtuple("UTRBlock", ["chrom","start","end","strand"])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="filtered paired_output.tsv")
    ap.add_argument("--ts-gff", required=True, help="TSHuman_7_hg19_3UTRs.gff (hg19)")
    ap.add_argument("--pdf", required=True, help="output multi-page PDF for plots")
    ap.add_argument("--png-dir", default=None, help="optional output dir for per-transcript PNGs")
    ap.add_argument("--bed-cleavage", required=True, help="output BED for cleavage points")
    ap.add_argument("--bed-seed", required=True, help="output BED for seed intervals (genomic)")
    ap.add_argument("--strip-transcript-version", action="store_true",
                    help="strip .version from transcript ids when matching GFF")
    return ap.parse_args()

def parse_cleavelocus(s):
    # formats like: "chr19:8027503:-"
    if not isinstance(s, str):
        return None, None, None
    m = re.match(r'^(chr[^:]+):(\d+):([+-])$', s.strip())
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)

def load_ts_utr_blocks(ts_gff, strip_version=False):
    blocks_by_enst = defaultdict(list)
    strand_by_enst = {}
    with open(ts_gff, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("browser") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feat, start, end, score, strand, frame, col9 = parts
            if feat.upper() != "UTR":
                continue
            try:
                start = int(start); end = int(end)
            except:
                continue
            enst = col9.strip()
            if strip_version:
                enst = enst.split(".")[0]
            blocks_by_enst[enst].append(UTRBlock(chrom, start, end, strand))
            strand_by_enst[enst] = strand
    # order blocks in transcript direction for each transcript
    for enst, blocks in blocks_by_enst.items():
        st = strand_by_enst[enst]
        if st == '+':
            blocks.sort(key=lambda b: b.start)
        else:
            blocks.sort(key=lambda b: b.start, reverse=True)
        blocks_by_enst[enst] = blocks
    return blocks_by_enst, strand_by_enst

def utr_to_genome(utr_pos_1based, blocks_ord, tx_strand):
    """ UTR 1-based -> genomic coordinate (hg19), using pre-ordered blocks in transcript direction. """
    remaining = int(utr_pos_1based) - 1
    for b in blocks_ord:
        blen = b.end - b.start + 1
        if remaining < blen:
            return (b.start + remaining) if tx_strand == '+' else (b.end - remaining)
        remaining -= blen
    return None

def plot_transcript(ax, utr_len, seed_spans, cleavages, title):
    """
    seed_spans: list of dicts with keys: start,end,label
    cleavages:  list of dicts with keys: pos,label
    """
    ax.set_title(title)
    ax.set_xlim(0, max(utr_len, 1))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    # baseline
    ax.hlines(0.5, 0, utr_len, linewidth=2)
    # seeds as boxes
    for s in seed_spans:
        x0 = max(0, s["start"]-1)
        w  = max(1, s["end"] - s["start"] + 1)
        ax.add_patch(plt.Rectangle((x0, 0.35), w, 0.3, fill=False, linewidth=1))
        # label slightly above
        ax.text(x0 + w/2, 0.7, s.get("label",""), ha="center", va="bottom", fontsize=8, rotation=0)
    # cleavage markers (lollipops)
    for c in cleavages:
        x = max(0, c["pos"]-1)
        ax.vlines(x, 0.5, 0.9, linewidth=1)
        ax.plot([x], [0.9], marker="o", markersize=4)
        ax.text(x-0.01*utr_len, 0.5, c.get("label",""), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xlabel("3'UTR coordinate (nt; 1-based)")

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.pdf), exist_ok=True)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    # Load pairs
    pairs = pd.read_csv(args.pairs, sep=None, engine="python")
    # Required cols check (be forgiving)
    need = ["transcript","cleaveLocus","utr_pos_ts_1based","total_utr_len",
            "paired_seed_start","paired_seed_end","miRNAname"]
    for c in need:
        if c not in pairs.columns:
            print(f"[WARN] Missing column '{c}' in {args.pairs}")
    # Keep only rows with both UTR pos and seed coords
    pairs = pairs.dropna(subset=["utr_pos_ts_1based","paired_seed_start","paired_seed_end"])

    # Normalize transcript id (optionally strip version for matching GFF)
    pairs["_tx_norm"] = pairs["transcript"].astype(str)
    if args.strip_transcript_version:
        pairs["_tx_norm"] = pairs["_tx_norm"].str.split(".").str[0]

    # Load TS UTR blocks
    blocks_by_tx, strand_by_tx = load_ts_utr_blocks(args.ts_gff, strip_version=args.strip_transcript_version)

    # Build BED writers
    bed_c = open(args.bed_cleavage, "w")
    bed_s = open(args.bed_seed, "w")

    # Group by transcript for plotting
    groups = pairs.groupby("_tx_norm", sort=False)

    with PdfPages(args.pdf) as pdf:
        for tx_norm, df in groups:
            if tx_norm not in blocks_by_tx:
                # skip transcripts missing from GFF (should be rare after your previous filtering)
                continue
            blocks = blocks_by_tx[tx_norm]
            tx_strand = strand_by_tx[tx_norm]
            utr_len = int(df["total_utr_len"].iloc[0]) if "total_utr_len" in df.columns and not pd.isna(df["total_utr_len"].iloc[0]) else \
                      sum(b.end - b.start + 1 for b in blocks)

            # Prepare plot data
            seed_spans = []
            cleavages  = []

            for r in df.itertuples(index=False):
                # UTR coordinates (1-based)
                seed_start = int(getattr(r, "paired_seed_start"))
                seed_end   = int(getattr(r, "paired_seed_end"))
                cleave_utr = int(getattr(r, "utr_pos_ts_1based"))
                mname_deg  = getattr(r, "miRNAname") if "miRNAname" in df.columns else ""
                mname_ts   = getattr(r, "paired_seed_miRNA") if "paired_seed_miRNA" in df.columns else ""

                # Add to plot lists
                seed_spans.append({"start": seed_start, "end": seed_end, "label": mname_ts})
                cleavages.append({"pos": cleave_utr, "label": "cleavage"})

                # Write BED for cleavage points (use genomic from cleaveLocus as-is)
                chrom, pos, strand = parse_cleavelocus(getattr(r, "cleaveLocus"))
                if chrom:
                    name = f"{tx_norm}|{mname_deg}|cleavage"
                    # BED is 0-based half-open; represent point as 1bp interval
                    bed_c.write(f"{chrom}\t{pos-1}\t{pos}\t{name}\t0\t{strand}\n")

                # Convert seed UTR interval -> genomic via blocks
                g_start = utr_to_genome(seed_start, blocks, tx_strand)
                g_end   = utr_to_genome(seed_end,   blocks, tx_strand)
                if g_start is not None and g_end is not None:
                    # ensure start <= end for BED
                    g0 = min(g_start, g_end)
                    g1 = max(g_start, g_end) + 1  # BED half-open
                    name = f"{tx_norm}|{mname_deg}|seed"
                    bed_s.write(f"{blocks[0].chrom}\t{g0-1}\t{g1-1}\t{name}\t0\t{tx_strand}\n")

            # Plot
            fig, ax = plt.subplots(figsize=(9, 2.4))
            title = f"{tx_norm}  (UTR len={utr_len} nt)"
            plot_transcript(ax, utr_len, seed_spans, cleavages, title)
            pdf.savefig(fig, bbox_inches="tight")
            if args.png_dir:
                fig.savefig(os.path.join(args.png_dir, f"{tx_norm}.png"), bbox_inches="tight", dpi=160)
            plt.close(fig)

    bed_c.close()
    bed_s.close()
    print(f"[OK] PDF plots -> {args.pdf}")
    if args.png_dir:
        print(f"[OK] PNGs -> {args.png_dir}")
    print(f"[OK] BED cleavage -> {args.bed_cleavage}")
    print(f"[OK] BED seed -> {args.bed_seed}")

if __name__ == "__main__":
    main()
