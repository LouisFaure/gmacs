from pathlib import Path

import pandas as pd
import zstandard as zstd

dctx = zstd.ZstdDecompressor()


def load_bedfile(zst_file: Path):
    with open(zst_file, "rb") as f:
        with dctx.stream_reader(f) as reader:
            df = pd.read_csv(
                reader, sep="\t", names=["chr", "start", "end", "strand", "qual"]
            )
    num_reads = len(df)
    intervals_by_chrom = {
        key: group[["start", "end"]].to_dict(orient="list")
        for key, group in df.reset_index().groupby("chr")
    }
    return intervals_by_chrom, num_reads
