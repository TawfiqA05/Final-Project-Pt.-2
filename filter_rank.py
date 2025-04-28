#!/usr/bin/env python3
"""
Purpose:
    Filter counties by minimum EV count, rank by BEV/PHEV ratio & BEV count,
    and identify 'urgent' counties in the top 10th percentile of both metrics.

Expected Output:
    - Prints filtered DataFrame head.
    - Prints threshold values for ratio and BEV count.
"""

import numpy as np
from load_clean_data import load_clean_data
from compute_counts import compute_counts

def filter_and_rank(counts, min_evs=10):
    """
    Filter and rank counties.

    Steps:
      1. Select counties with total_evs >= min_evs.
      2. Rank by 'bev_phev_ratio' (descending) and 'BEV' (descending).
      3. Compute 90th percentile thresholds for ratio & BEV count.
      4. Mark 'urgent' = True if both ratio and BEV count >= thresholds.

    Returns:
      filtered: DataFrame including ratio_rank, bev_rank, urgent.
      thr_ratio: 90th percentile of bev_phev_ratio.
      thr_bev:   90th percentile of BEV count.
    """
    # Keep only sufficiently large counties
    filtered = counts[counts['total_evs'] >= min_evs].copy()

    # Rank metrics
    filtered['ratio_rank'] = filtered['bev_phev_ratio'].rank(method='min', ascending=False)
    filtered['bev_rank']   = filtered['BEV'].rank(method='min', ascending=False)

    # Determine top-10% thresholds
    thr_ratio = np.percentile(
        filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna(), 90
    )
    thr_bev = np.percentile(filtered['BEV'], 90)

    # Flag urgent counties
    filtered['urgent'] = (
        (filtered['bev_phev_ratio'] >= thr_ratio) &
        (filtered['BEV'] >= thr_bev)
    )

    return filtered, thr_ratio, thr_bev

if __name__ == '__main__':
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    counts = compute_counts(df)
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)
    print(filtered.head())
    print(f"Thresholds -> ratio: {thr_ratio:.2f}, BEV: {thr_bev:.0f}")