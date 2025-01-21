from pathlib import Path

import numpy as np

from gmacs import call_peaks, load_bedfile

intervals_by_chrom, num_reads = load_bedfile(Path("../MACS-3/Test_Data_2/1_.zst"))
for chrom in intervals_by_chrom:
    starts = np.array(intervals_by_chrom[chrom]["start"])
    ends = np.array(intervals_by_chrom[chrom]["end"])
    call_peaks(starts, ends, num_reads)
