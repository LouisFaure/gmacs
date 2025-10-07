import datetime as dt
from collections import defaultdict
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import maximum_filter1d
from cupyx.scipy.special import pdtr, pdtrc

from .utils import load_bedfile

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def pileup(starts, ends, chr_length, offset):
    """
    For each point along the chromosome, compute the number of aligning
    reads.

    Pileup is the total number of reads aligning to the chromosome.
    inputs:
        - p_values: cupy vector of p-values at every point along the chromosome
    outputs:
        - q_values: cupy vector of q-values at every point along the chromosome
    """
    start_poss = cp.concatenate((starts - offset, ends - offset))
    end_poss = cp.concatenate((starts + offset, ends + offset))
    cov_vec = cp.zeros(chr_length, dtype=np.int32)
    start_poss = cp.clip(start_poss, 0, chr_length - 1)
    end_poss = cp.clip(end_poss, 0, chr_length - 1)
    cp.add.at(cov_vec, start_poss, 1)
    cp.add.at(cov_vec, end_poss, -1)
    cov_vec = cp.cumsum(cov_vec)

    return cov_vec


def compute_peaks(q_vals, thresh=0.1):
    """Filter peaks based q-values. Return the indices where q-value <= thresh

    inputs:
        - q_vals: cupy vector of q-values
        - thresh: Threshold parameter (default 0.1)
    outputs:
        - peaks: cupy vector of indices where q-values <= thresh
    """
    peaks = cp.where((q_vals >= thresh))[0]
    return peaks


def merge_consecutive(arr, max_gap=30):
    """Merge overlapping and consecutive peak calls.

    Returns a matrix with two columns encoding the start and ends of peaks.
    inputs:
        - arr: cupy vector of coordinates returned by compute_peaks.
        - max_gap: parameter to merge peaks within "max_gap". Default [30]
    outputs:
        - M: a matrix with two columns encoding the start and ends of peaks
    """
    boundaries = cp.where(cp.diff(arr) >= max_gap)[0] + 1
    split_indices = cp.concatenate(
        (cp.asarray([0]), boundaries, cp.asarray([len(arr)]))
    )
    starts = arr[split_indices[:-1]]
    ends = arr[split_indices[1:] - 1]
    M = cp.stack([starts, ends], axis=1)
    del split_indices, boundaries, starts, ends
    return M


def filter_peaks(peaks, peak_amplitude=150):
    """Remove short peaks. Removes peaks smaller than 150 bases.

    inputs:
        - peaks: cupy matrix of coordinates returned by merge_consecutive.
        - peak_amplitude: integer representing the peak length
    outputs:
        - merged_filt: a matrix with two columns encoding the start and ends of peaks
    """
    peak_lengths = peaks[:, 1] - peaks[:, 0]
    peak_ind = cp.where(peak_lengths >= peak_amplitude)[0]
    merged_filt = peaks[peak_ind]
    return merged_filt


def calculate_peak_summits(peaks, signal):
    """
    Function to compute peak summits. Peak summit is the highest point along the peak.
    The highest point in the peak corresponds to the point with the lowest q-value
    inputs:
        - peaks: a nx2 numpy matrix encoding the peak coordinates
        - signal: a vector based on which peaks are called. Here the signal corresponds to the q-values
    outputs:
        - min_values: q-values corresponding to peaks
        - arg_min_indices: indices of the peak summits along the chromosomes.
    """
    starts = peaks[:, 0]
    ends = peaks[:, 1]
    max_values = np.zeros(len(starts))
    arg_max_indices = np.zeros(len(starts), dtype=np.int64)
    for i in range(0, len(starts)):
        am = starts[i] + np.argmin(signal[starts[i] : ends[i]])
        arg_max_indices[i] = am
    return arg_max_indices


def extract_values(indexes, vec):
    return vec[indexes]


def fdr(unique_p_values, unique_p_counts):
    """
    Compute the FDR corrected values using the Benjamini-Hochberg procedure.

    Parameters:
    - unique_p_values (cp.ndarray): Vector of p-values (assumed unique).
    - unique_p_counts (cp.ndarray): Vector of counts corresponding to the p-values.

    Returns:
    - pq_table (dict): Mapping of p-values to their q-values.
    """
    # Step 1: Sort unique p-values and counts
    unique_p_values = -1 * unique_p_values  # Negate p-values for sorting
    sorted_indices = cp.argsort(unique_p_values)[::-1]
    sorted_unique_p_values = unique_p_values[sorted_indices]
    sorted_counts = unique_p_counts[sorted_indices]
    cumulative_k = cp.cumsum(sorted_counts) - sorted_counts + 1
    sorted_unique_p_values = cp.where(
        sorted_unique_p_values == -cp.inf, -cp.inf, sorted_unique_p_values
    )
    total_counts = cp.sum(unique_p_counts)

    f = cp.log10(total_counts)
    q_values = cp.asnumpy(sorted_unique_p_values + (cp.log10(cumulative_k) - f))

    q_values_np = np.zeros(len(q_values))
    preq_q = float("inf")
    for i in range(len(q_values)):
        q = max(min(preq_q, q_values[i]), 0)
        preq_q = q
        q_values_np[i] = q

    p_values_original = -1 * sorted_unique_p_values  # Convert back to original p-values
    pq_table = dict(zip(cp.asnumpy(p_values_original), q_values_np))

    return pq_table


def replace_with_dict(array, mapping):
    """
    Function to replace every element in the array with value in the dictionary. Utility function to replace p_values with q_values.
    inputs:
        - array: a cupy vector
        - mapping: dictionary to replace the elements of the array
    outputs:
        - result: a vector whose values are replaced with the values from mapping.
    """
    keys = cp.array(list(mapping.keys()))
    values = cp.array(list(mapping.values()))

    sorted_indices = cp.argsort(keys)
    sorted_keys = keys[sorted_indices]
    sorted_values = values[sorted_indices]

    idx = cp.searchsorted(sorted_keys, array)
    valid = (idx < len(sorted_keys)) & (sorted_keys[idx] == array)

    result = cp.where(valid, sorted_values[idx], array)

    return result

def make_pq_table(result_df, num_reads, d_treat=150, d_ctrl=10000, genome_length=3088286401):
    """
    This is a method to compute the pq-table. Across all chromosomes, we estimate significance values. Upon computing the frequencies of the
    p-values, we compute calculate their ranks, and correct the p-values with Benjamini Hochberg correction procedure. The function returns a
    dictionary of p-values as keys, and their correspoding q-values as it values. Since the alignment is done across all chromosomes, in one
    go, we compute do this procedure first, to prevent biasing towards any one chromosome.

    inputs:
        - result_df : Sorted cuDF DataFrame with 'chrom', 'start', 'end' columns
        - num_reads: number of reads
        - d_treat: distance for extending alignments to compute peaks
        - d_ctrl: distance for extending alignments for building the null hypothesis
        - genome_length: mappable genome length

    outputs:
        - pq_table: A dictionary of q-value for every p-value
            {-14.533:-11:566, ...}
        Note the p and q-values are log10 transformed.
    """
    unique_p_values = cp.asarray([])
    unique_p_counts = cp.asarray([])
    
    # Get unique chromosomes - must use .to_pandas() since chroms are strings (CuPy doesn't support strings)
    chroms = result_df['chrom'].unique().to_pandas()
    
    for chrom in chroms:
        # Filter for this chromosome - all operations stay on GPU
        chrom_df = result_df[result_df['chrom'] == chrom]
        # Extract columns as CuPy arrays directly - no CPU transfer for numeric data
        starts = chrom_df['start'].to_cupy()
        ends = chrom_df['end'].to_cupy()

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        chrom_length = cp.max(ends).item()

        scale = d_treat / d_ctrl
        lambda_bg = 2 * d_treat * num_reads / genome_length

        pileup_treat = pileup(starts, ends, chrom_length, int(d_treat / 2))
        pileup_ctrl = pileup(starts, ends, chrom_length, int(d_ctrl / 2))
        pileup_ctrl = cp.maximum(pileup_ctrl * scale, lambda_bg)
        p_values = compute_poisson_cdfs(pileup_treat, pileup_ctrl)
        p_values, p_counts = cp.unique(p_values, return_counts=True)
        all_values = cp.concatenate((unique_p_values, p_values))
        all_counts = cp.concatenate((unique_p_counts, p_counts))
        merged_values, inverse_indices = cp.unique(all_values, return_inverse=True)
        merged_counts = cp.zeros_like(merged_values, dtype=all_counts.dtype)
        cp.add.at(merged_counts, inverse_indices, all_counts)

        unique_p_values = merged_values
        unique_p_counts = merged_counts

        del (
            pileup_ctrl,
            pileup_treat,
            p_values,
            merged_values,
            inverse_indices,
            merged_counts,
            all_values,
            all_counts,
        )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
    pq_table = fdr(unique_p_values, unique_p_counts)
    return pq_table


def compute_poisson_cdfs(observations, lambdas):
    """
    Function to compute the upper tail of a Poisson distribution. At every point along the chromsome we estimate
    the expected number of reads based on a Poisson distribution. The expected number of reads is calculated as
    max(scale*depth, lambda_bg).
    lambda_bg = fragment_length * number_of_fragments/genome_length

    inputs:
        - observations: cupy vector of observations at every point along the chromosome
        - lambdas: cupy vector of the rates at every point along the chromosome.
    outputs:
        - p_values: cupy vector of p-values at every point along the chromosome
    """
    p_vals = pdtrc(observations, lambdas)
    return cp.log10(p_vals)


def call_peaks(
    starts,
    ends,
    pq_table,
    num_reads,
    q_thresh=0.1,
    d=150,
    d_ctrl=10000,
    genome_length=3088286401,
    max_gap=30,
    peak_amp=150,
):
    """
    Wrapper for the peak calling routines. This function computes the pileups from the data and the null hypothesis(pileup_ctrl).
    We compute the p_values, q values, identify and filter peaks.

    inputs:
        - starts: a list of start coordinates for the read alignments
        - ends: a list of end coordinates for the read alignments
        - num_reads: number of reads
        - q_thresh: threshold for q-values to identify peaks
        - d: distance for extending alignments to compute peaks
        - d_ctrl: distance for extending alignments for building the null hypothesis
        - genome_length: mappable genome length
        - max_gap: gaps for extending peaks
        - peak_amp: Amplitude for filtering peaks.
    outputs:
        - df_op: a dataframe with the peak information.
    """

    chrom_length = cp.max(ends).item()

    scale = d / d_ctrl
    lambda_bg = 2 * d * num_reads / genome_length
    q_thresh = -cp.log10(q_thresh)
    peak_amp = peak_amp - 1

    pileup_treat = pileup(starts, ends, chrom_length, int(d / 2))
    pileup_ctrl = pileup(starts, ends, chrom_length, int(d_ctrl / 2))
    pileup_ctrl = cp.maximum(pileup_ctrl * scale, lambda_bg)
    print(dt.datetime.now(), "Pileup")

    p_values = compute_poisson_cdfs(pileup_treat, pileup_ctrl)
    print(dt.datetime.now(), "Computed P Values")

    q_values = replace_with_dict(p_values, pq_table)

    p_values[p_values == -cp.inf] = -1000
    q_values[q_values == cp.inf] = 1000
    print(dt.datetime.now(), "Computed q Values")

    peaks = compute_peaks(q_values, q_thresh)
    print(dt.datetime.now(), "Calculated Peaks")

    if len(peaks) == 0:
        return pd.DataFrame()

    m = merge_consecutive(peaks, max_gap)
    filtered_peaks = filter_peaks(m, peak_amp)
    if len(filtered_peaks) == 0:
        return pd.DataFrame()

    print(
        dt.datetime.now(),
        "Merged Peaks",
        len(filtered_peaks),
        cp.max(filtered_peaks[:, 1] - filtered_peaks[:, 0]),
    )

    merged_peaks = cp.asnumpy(filtered_peaks)
    peak_summits_args = calculate_peak_summits(merged_peaks, cp.asnumpy(p_values))
    q_summit = cp.asnumpy(extract_values(peak_summits_args, q_values))
    p_summit = cp.asnumpy(extract_values(peak_summits_args, p_values))
    treat_summit = cp.asnumpy(extract_values(peak_summits_args, pileup_treat))
    ctrl_summit = cp.asnumpy(extract_values(peak_summits_args, pileup_ctrl))

    df_op = pd.DataFrame(
        data={
            "start": merged_peaks[:, 0],
            "end": merged_peaks[:, 1],
            "peak": peak_summits_args - merged_peaks[:, 0],
            "signal_value": (treat_summit + 1) / (ctrl_summit + 1),
            "p_value": -1 * p_summit,
            "q_value": q_summit,
            "pileup": treat_summit,
        }
    )
    df_op["name"] = "."
    df_op["score"] = np.minimum(np.array(df_op["q_value"] * 10, dtype=np.int64), 1000)
    df_op["strand"] = "."

    print(dt.datetime.now(), "Calculated peaks summits")

    del q_values, p_values, pileup_treat, m, peaks, pileup_ctrl, filtered_peaks

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    cp._default_memory_pool.free_all_blocks()

  
    return df_op


def gmacs(
    result_df,
    num_reads,
    q_thresh=0.1,
    d_treat=150,
    d_ctrl=10000,
    genome_length=3088286401,
    max_gap=30,
    peak_amp=150,
    out='called_peaks'
):
    """Function to run gmacs. It takes as input a DataFrame of read alignment coordinates, and computes the pq table and calls peaks.
    inputs:
        - result_df: Sorted cuDF DataFrame with 'chrom', 'start', 'end' columns
        - num_reads: number of reads
        - q_thresh: threshold for q-values to identify peaks
        - d_treat: distance for extending alignments to compute peaks
        - d_ctrl: distance for extending alignments for building the null hypothesis
        - genome_length: Length of the mappable genome
        - max_gap: gaps for extending peaks. Default [30]
        - peak_amp: Amplitude for filtering peaks. Default [150]
    outputs:
        - peaks: A dataframe of peaks

    """
    start = dt.datetime.now()
    pq_table = make_pq_table(
        result_df,
        num_reads,
        genome_length=genome_length,
        d_treat=d_treat,
        d_ctrl=d_ctrl,
    )
    end = dt.datetime.now()
    print(
        f"Calculated PQ Table....., {(end - start).total_seconds() / 60.0}, minutes..."
    )

    wf_start = dt.datetime.now()
    peaks = pd.DataFrame()

    # Get unique chromosomes - must use .to_pandas() since chroms are strings (CuPy doesn't support strings)
    chroms = result_df['chrom'].unique().to_pandas()
    
    for chrom in chroms:
        start = dt.datetime.now()
        # Filter for this chromosome - all operations stay on GPU
        chrom_df = result_df[result_df['chrom'] == chrom]
        # Extract columns as CuPy arrays directly - no CPU transfer for numeric data
        starts = chrom_df['start'].to_cupy()
        ends = chrom_df['end'].to_cupy()
        print(start, chrom)
        df_chr = call_peaks(
            starts,
            ends,
            pq_table,
            num_reads,
            max_gap=max_gap,
            q_thresh=q_thresh,
            peak_amp=peak_amp,
            genome_length=genome_length,
            d=d_treat,
            d_ctrl=d_ctrl,
        )
        df_chr["chrom"] = chrom
        peaks = pd.concat([peaks, df_chr], axis=0)
        end = dt.datetime.now()
        print(end, (end - start).total_seconds(), "Done\n")
    peaks = peaks[
        [
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signal_value",
            "p_value",
            "q_value",
            "peak",
        ]
    ]
    peaks['end']=peaks['end'].astype(int)
    peaks['start']=peaks['start'].astype(int)
    wf_end = dt.datetime.now()
    print(
        f"Total time elapsed..., {(wf_end - wf_start).total_seconds() / 60.0}, minutes"
    )
    peaks.to_csv(f'{out}.tsv',index=False,sep='\t')
    peaks.iloc[:,:3].to_csv(f'{out}.bed',index=False,sep='\t',header=False)
    #return peaks


if __name__ == "__main__":
    result_df, num_reads = load_bedfile(
        Path("../../MACS-3/Test_Data_2/1_.zst")
    )
    df_op = gmacs(result_df=result_df, num_reads=num_reads)
    print(df_op.head())
