import datetime as dt
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import maximum_filter1d
from cupyx.scipy.special import pdtr

from .utils import load_bedfile

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def compute_q_values(p_values):
    """Compute the FDR corrected qvalues.

    inputs:
        - p_values: cupy vector of p-values at every point along the chromosome
    outputs:
        - q_values: cupy vector of q-values at every point along the chromosome
    """
    n = len(p_values)
    sorted_indices = cp.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    ranks = cp.arange(1, n + 1)
    q_values = (sorted_p_values * n) / ranks
    q_values = q_values[::-1]
    q_values = np.minimum.accumulate(cp.asnumpy(q_values))[::-1]
    q_values = cp.asarray(q_values)
    original_order_q_values = cp.empty_like(q_values)
    original_order_q_values[sorted_indices] = q_values
    del q_values
    return original_order_q_values


def pileup(starts, ends, chr_length, offset):
    """For each point along the chromosome, compute the number of aligning
    reads.

    Pileup is the total number of reads aligning to the chromosome.
    inputs:
        - p_values: cupy vector of p-values at every point along the chromosome
    outputs:
        - q_values: cupy vector of q-values at every point along the chromosome
    """
    start_poss = np.concatenate((starts - offset, ends - offset))
    end_poss = np.concatenate((starts + offset, ends + offset))
    cov_vec = cp.zeros(chr_length, dtype=np.int32)
    start_poss = np.clip(start_poss, 0, chr_length - 1)
    end_poss = np.clip(end_poss, 0, chr_length - 1)
    cp.add.at(cov_vec, start_poss, 1)
    cp.add.at(cov_vec, end_poss, -1)
    cov_vec = cp.cumsum(cov_vec)

    return cov_vec


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
    p_values = 1 - pdtr(observations, lambdas)
    return p_values


def compute_peaks(q_vals, thresh=0.1):
    """Filter peaks based q-values. Return the indices where q-value <= thresh

    inputs:
        - q_vals: cupy vector of q-values
        - thresh: Threshold parameter (default 0.1)
    outputs:
        - peaks: cupy vector of indices where q-values <= thresh
    """
    peaks = cp.where(q_vals <= thresh)[0]
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
    boundaries = cp.where(cp.diff(arr) > max_gap)[0] + 1
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
    min_values = np.zeros(len(starts))
    arg_min_indices = np.zeros(len(starts))
    for i in range(0, len(starts)):
        am = starts[i] + np.argmin(signal[starts[i] : ends[i]])
        m = signal[am]
        min_values[i] = m
        arg_min_indices = am

    return min_values, arg_min_indices


def call_peaks(
    starts,
    ends,
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
        - peaks_summit_q_vals: q-values of peak summits
        - peak_summits_args: peak summits
        - merged_peaks: Matrix of peak coordinates
    """
    chrom_length = cp.max(ends)

    # These constants explained in Zheng et al. (MACS3 paper)
    scale = d / d_ctrl
    lambda_bg = 2 * d * num_reads / genome_length

    pileup_treat = pileup(starts, ends, chrom_length, int(d / 2))
    pileup_ctrl = pileup(starts, ends, chrom_length, int(d_ctrl / 2))
    pileup_ctrl = cp.maximum(pileup_ctrl * scale, lambda_bg)
    print(dt.datetime.now(), "Pileup")

    p_values = compute_poisson_cdfs(pileup_treat, pileup_ctrl)
    print(dt.datetime.now(), "Computed P Values")

    q_values = compute_q_values(p_values)
    print(dt.datetime.now(), "Computed q Values")

    peaks = compute_peaks(q_values, q_thresh)
    print(dt.datetime.now(), "Calculated Peaks")

    m = merge_consecutive(peaks, max_gap)
    filtered_peaks = filter_peaks(m, peak_amp)
    print(
        dt.datetime.now(),
        "Merged Peaks",
        len(filtered_peaks),
        cp.max(filtered_peaks[:, 1] - filtered_peaks[:, 0]),
    )

    merged_peaks = cp.asnumpy(filtered_peaks)
    peaks_summit_q_vals, peak_summits_args = calculate_peak_summits(
        merged_peaks, cp.asnumpy(q_values)
    )
    print(dt.datetime.now(), "Calulcated peaks summits")

    del q_values, p_values, pileup_treat, m, peaks, pileup_ctrl, filtered_peaks

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    cp._default_memory_pool.free_all_blocks()

    return peaks_summit_q_vals, peak_summits_args, merged_peaks


if __name__ == "__main__":
    intervals_by_chrom, num_reads = load_bedfile(Path("../MACS-3/Test_Data_2/1_.zst"))
    for chrom in intervals_by_chrom:
        starts = np.array(intervals_by_chrom[chrom]["start"])
        ends = np.array(intervals_by_chrom[chrom]["end"])
        call_peaks(starts, ends, num_reads)
