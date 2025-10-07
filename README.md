gmacs
===

An accelerated implementation of [MACS3](https://github.com/macs3-project/MACS)
ChIP-Seq peak calling algorithm using [cupy](https://cupy.dev/).

## Installation

```bash
pip install -e .
```

Developed for and tested on Ubuntu 22.04.5 with working NVIDIA drivers.

## Usage

### Command Line Interface

After installation, you can use the `gmacs` command directly:

```bash
# Basic usage (automatically decompresses .gz files with rapidgzip)
gmacs input_fragments.tsv.gz -o output_peaks

# With custom number of decompression cores
gmacs input_fragments.tsv.gz -p 20 -o output_peaks

# Direct GPU loading (use with caution for large .gz files - may hang!)
gmacs input_fragments.tsv.gz --direct-load -o output_peaks

# Custom peak calling parameters
gmacs input_fragments.tsv.gz \
    -o output_peaks \
    -q 0.05 \
    --d-treat 150 \
    --d-ctrl 10000 \
    --max-gap 30 \
    --peak-amp 150

# Verbose output
gmacs input_fragments.tsv.gz -v -o output_peaks
```

#### CLI Options

- `input`: Path to ATAC fragments file (TSV or TSV.GZ format)
- `-o, --output`: Output prefix for results (default: gmacs_peaks)
- `--direct-load`: Load .gz files directly using GPU (may hang for multi-GB files, use with caution!)
- `-p, --processes`: Number of CPU cores for decompression (default: 10, used when not using --direct-load)
- `-q, --qvalue`: Q-value threshold for peak detection (default: 0.1)
- `--d-treat`: Distance for extending alignments to compute peaks (default: 150)
- `--d-ctrl`: Distance for extending alignments for null hypothesis (default: 10000)
- `--genome-length`: Mappable genome length (default: 3088286401 for GRCh38)
- `--max-gap`: Maximum gap for merging consecutive peaks (default: 30)
- `--peak-amp`: Minimum peak amplitude for filtering (default: 150)
- `-v, --verbose`: Enable verbose output

**Note**: For .gz files, gmacs uses rapidgzip for decompression by default (recommended for large files). Use `--direct-load` only if you want GPU-based loading and have small files.

#### Output Files

- `<output>.tsv`: Detailed peak information including coordinates, scores, p-values, q-values
- `<output>.bed`: BED format file with peak coordinates

### Python API

You can also use gmacs as a Python library:

```python
from gmacs import gmacs, load_bedfile

# Load data
intervals_by_chrom, num_reads = load_bedfile("input_fragments.tsv")

# Call peaks
gmacs(
    intervals_by_chrom,
    num_reads,
    q_thresh=0.1,
    d_treat=150,
    d_ctrl=10000,
    genome_length=3088286401,
    out="output_peaks"
)
```

Example usage scripts can be found in `examples`.

More implementation details about gmacs can be found [here](https://substack.com/inbox/post/155316519)
