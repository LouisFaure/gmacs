import cudf
import cupy as cp
#import gzip


def load_bedfile(gz_file):
    df = cudf.read_csv(
        gz_file,
        sep="\t",
        names=['chrom', 'start', 'end', 'barcode', 'count'],
        dtype={'chrom': 'str', 'start': 'int32', 'end': 'int32', 'barcode': 'str', 'count': 'int32'},
        comment='#',
    )
    num_reads = len(df)
    
    # Instead of using collect (which creates list columns that require CPU transfer),
    # store the sorted DataFrame and let gmacs extract per-chromosome data directly
    result_df = df.sort_values(['chrom', 'start'])[['chrom', 'start', 'end']]
    
    return result_df, num_reads
