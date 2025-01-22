from pathlib import Path

import pandas as pd
import polars as ps
import zstandard as zstd

dctx = zstd.ZstdDecompressor()


def load_bedfile(zst_file):
    with open(zst_file, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            df = ps.read_csv(
                reader,
                separator="\t",
                new_columns=["chr", "start", "end", "strand", "qual"],
            )
    num_reads = len(df)
    result_df = df.group_by("chr").agg(
        [ps.col("start").implode(), ps.col("end").implode()]
    )

    # Convert to desired dictionary format
    result = {
        row["chr"]: {"start": row["start"], "end": row["end"]}
        for row in result_df.to_dicts()
    }
    return result, num_reads
