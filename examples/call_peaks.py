from pathlib import Path

from gmacs import gmacs, load_bedfile

intervals_by_chrom, num_reads = load_bedfile(Path("../MACS-3/Test_Data_2/1_.zst"))
gmacs(intervals_by_chrom, num_reads)
