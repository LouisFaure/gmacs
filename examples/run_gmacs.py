import datetime as dt
from gmacs import gmacs, load_bedfile
from glob import glob
#import pybedtools
#import re
import subprocess
import argparse
import tempfile
import os


def decompress_to_temp(file_path,P):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    subprocess.run(['rapidgzip', '-d', '-c', '-P', P, file_path],
                   stdout=open(temp.name, 'w'), check=True)
    return temp.name

def main():
    start_time = dt.datetime.now()
    parser = argparse.ArgumentParser(description="Run GMACS on ATAC fragments file.")
    parser.add_argument("input", help="Path to atac_fragments.tsv.gz file")
    parser.add_argument("--decompress", action="store_true", default=True,
                        help="Decompress file to temp before loading (for .gz files)")
    parser.add_argument("--output", help="Output file for GMACS results", default="test_out")
    parser.add_argument("--processes", help="Number of CPU cores to use for decompression", default="10")
    args = parser.parse_args()


    if args.decompress:
        print(dt.datetime.now(), "decompressing")
        temp_file = decompress_to_temp(args.input, args.processes)
        print(dt.datetime.now(), "loading decompressed file")
        intervals_by_chrom, num_reads = load_bedfile(temp_file)
        os.remove(temp_file)
    else:
        print(dt.datetime.now(), "loading gzipped file")
        intervals_by_chrom, num_reads = load_bedfile(args.input)

    print(dt.datetime.now(), "done")
    gmacs(intervals_by_chrom, num_reads, out=args.output)
    end_time = dt.datetime.now()
    print(f"Total run time: {end_time - start_time}")

if __name__ == "__main__":
    main()