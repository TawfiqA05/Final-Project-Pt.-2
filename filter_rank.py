#!/usr/bin/env python3
"""
Purpose:
    Filter counties by minimum EV count, rank by BEV/PHEV ratio & BEV count,
    and identify 'urgent' counties in the top 10th percentile of both metrics.

Expected Output:
    - If no counties meet the minimum_evs threshold:
        Prints "No counties meet the minimum EV threshold."
    - Otherwise:
        Prints the head of the filtered DataFrame.
        Prints the 90th‐percentile thresholds for ratio & BEV count.
"""

import sys
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
      4. Mark 'urgent' = True if both ratio and BEV ≥ thresholds.

    Returns:
      filtered: DataFrame including ratio_rank, bev_rank, urgent.
      thr_ratio: 90th percentile of bev_phev_ratio.
      thr_bev:   90th percentile of BEV count.
    """
    # 1. Keep only sufficiently large counties
    filtered = counts[counts['total_evs'] >= min_evs].copy()

    if filtered.empty:
        return filtered, None, None

    # 2. Rank metrics (higher ratio & higher BEV count get rank 1)
    filtered['ratio_rank'] = filtered['bev_phev_ratio'].rank(method='min', ascending=False)
    filtered['bev_rank']   = filtered['BEV'].rank(method='min', ascending=False)

    # 3. Determine 90th percentile thresholds
    finite_ratios = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    thr_ratio = np.percentile(finite_ratios, 90) if not finite_ratios.empty else None
    thr_bev   = np.percentile(filtered['BEV'], 90)

    # 4. Flag urgent counties
    if thr_ratio is not None:
        filtered['urgent'] = (
            (filtered['bev_phev_ratio'] >= thr_ratio) &
            (filtered['BEV'] >= thr_bev)
        )
    else:
        filtered['urgent'] = False

    return filtered, thr_ratio, thr_bev

if __name__ == '__main__':
    # Load & clean data
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    # Compute per-county counts & ratios
    counts = compute_counts(df)
    # Filter, rank, and identify urgent counties
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)

    if filtered.empty:
        print("No counties meet the minimum EV threshold.")
        sys.exit(0)

    # Display results
    print("Filtered & Ranked Counties (sample):")
    print(filtered.head(), "\n")
    print(f"90th Percentile Thresholds -> Ratio: {thr_ratio:.2f}, BEV Count: {thr_bev:.0f}")
    print("\nCounties marked as 'urgent':")
    print(filtered.loc[filtered['urgent'], ['County', 'BEV', 'bev_phev_ratio']])