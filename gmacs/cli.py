#!/usr/bin/env python
"""
GMACS: GPU-accelerated MACS3
Command-line interface for peak calling from ATAC-seq fragments
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gmacs import gmacs, load_bedfile


def decompress_to_temp(file_path, processes):
    """Decompress a gzipped file to a temporary file using rapidgzip"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    subprocess.run(
        ['rapidgzip', '-d', '-c', '-P', str(processes), file_path],
        stdout=open(temp.name, 'w'),
        check=True
    )
    return temp.name


def main():
    """Main entry point for the GMACS CLI"""
    parser = argparse.ArgumentParser(
        description="GMACS: GPU-accelerated MACS3 for ATAC-seq peak calling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        "input",
        help="Path to ATAC fragments file (TSV or TSV.GZ format)"
    )
    parser.add_argument(
        "-o", "--output",
        default="gmacs_peaks",
        help="Output prefix for GMACS results (will create .tsv and .bed files)"
    )
    
    # Decompression options
    parser.add_argument(
        "--direct-load",
        action="store_true",
        default=False,
        help="Load .gz file directly using GPU (may hang for multi-GB files, use with caution!)"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=10,
        help="Number of CPU cores to use for decompression (when not using --direct-load)"
    )
    
    # Peak calling parameters
    parser.add_argument(
        "-q", "--qvalue",
        type=float,
        default=0.1,
        help="Q-value threshold for peak detection"
    )
    parser.add_argument(
        "--d-treat",
        type=int,
        default=150,
        help="Distance for extending alignments to compute peaks"
    )
    parser.add_argument(
        "--d-ctrl",
        type=int,
        default=10000,
        help="Distance for extending alignments for building null hypothesis"
    )
    parser.add_argument(
        "--genome-length",
        type=int,
        default=3088286401,
        help="Mappable genome length (default: human genome GRCh38)"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=30,
        help="Maximum gap for merging consecutive peaks"
    )
    parser.add_argument(
        "--peak-amp",
        type=int,
        default=150,
        help="Minimum peak amplitude (length) for filtering"
    )
    
    # Misc options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    start_time = dt.datetime.now()
    
    if args.verbose:
        print(f"GMACS: GPU-accelerated MACS3")
        print(f"Started at: {start_time}")
        print(f"Input file: {args.input}")
        print(f"Output prefix: {args.output}")
        print("-" * 60)
    
    # Load the data
    if args.direct_load:
        if args.verbose:
            print(f"{dt.datetime.now()}: Loading file directly (GPU-based)...")
            if args.input.endswith('.gz'):
                print("WARNING: Direct loading of .gz files may hang for multi-GB files!")
        intervals_by_chrom, num_reads = load_bedfile(args.input)
    else:
        if args.input.endswith('.gz'):
            if args.verbose:
                print(f"{dt.datetime.now()}: Decompressing file with {args.processes} processes...")
            try:
                temp_file = decompress_to_temp(args.input, args.processes)
                if args.verbose:
                    print(f"{dt.datetime.now()}: Loading decompressed file...")
                intervals_by_chrom, num_reads = load_bedfile(temp_file)
                os.remove(temp_file)
            except FileNotFoundError:
                print("Error: rapidgzip not found. Please install it or run with --direct-load", file=sys.stderr)
                sys.exit(1)
        else:
            if args.verbose:
                print(f"{dt.datetime.now()}: Loading file...")
            intervals_by_chrom, num_reads = load_bedfile(args.input)
    
    if args.verbose:
        print(f"{dt.datetime.now()}: Loaded {num_reads:,} reads")
        print(f"{dt.datetime.now()}: Calling peaks...")
    
    # Call peaks
    gmacs(
        intervals_by_chrom,
        num_reads,
        q_thresh=args.qvalue,
        d_treat=args.d_treat,
        d_ctrl=args.d_ctrl,
        genome_length=args.genome_length,
        max_gap=args.max_gap,
        peak_amp=args.peak_amp,
        out=args.output
    )
    
    end_time = dt.datetime.now()
    
    if args.verbose:
        print("-" * 60)
        print(f"Completed at: {end_time}")
        print(f"Total runtime: {end_time - start_time}")
        print(f"Output files:")
        print(f"  - {args.output}.tsv (detailed peak information)")
        print(f"  - {args.output}.bed (peak coordinates)")
    
    print(f"Peak calling completed successfully!")


if __name__ == "__main__":
    main()
