import datetime as dt
from gmacs import gmacs, load_bedfile
from glob import glob
import pybedtools
import re
import subprocess

frags = glob("some_path_to_samples/*/cellranger/atac_fragments.tsv.gz")

for f in frags:
    sample=f.split('/')[7]
    print(sample)
    print(dt.datetime.now(), "loading")
    result_df, num_reads = load_bedfile(f)
    print(dt.datetime.now(), "done")
    gmacs(result_df, num_reads,out=sample)


beds =  [pybedtools.BedTool(f"{f.split('/')[7]}.bed") for f in frags]

bed = beds.pop(0)
bed_concatenated = bed.cat(*beds)

standard_chrom_regex = re.compile(r'^chr')
filtered_bed = bed_concatenated.filter(lambda interval: standard_chrom_regex.match(interval.chrom))
filtered_bed.saveas("merged_sorted_concatenated.bed")

awk_command = """
awk 'BEGIN{OFS="\\t"} {
    center=int(($2+$3)/2);
# Output chrom, center_start, center_end (1bp interval) and preserve other fields if needed
    printf "%s\\t%d\\t%d", $1, center, center+1;
# Preserve Name, Score, Strand (columns 4, 5, 6) - adjust if you have more/fewer
for (i=4; i<=NF; i++) { printf "\\t%s", $i };
    printf "\\n";
}' merged_sorted_concatenated.bed > peak_centers.bed
"""

result = subprocess.run(awk_command, shell=True, check=True, capture_output=True, text=True)
bed =  pybedtools.BedTool("peak_centers.bed")
bed = bed.slop(genome='hg38',l=250,r=249)

bed.saveas("consensus_peaks_resized_500bp.bed")
