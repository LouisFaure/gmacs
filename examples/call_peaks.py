from pathlib import Path
from gmacs import gmacs, load_bedfile

result_df, num_reads = load_bedfile(Path("consensus_peaks_resized_500bp.bed"))
gmacs(result_df, num_reads)