#!/usr/bin/env python3
"""
Purpose:
    Compute and print Pearson correlation between BEV/PHEV ratio and BEV count.

Expected Output:
    - Prints: "Pearson r = <r_value>, p-value = <p_value>"
"""

import numpy as np
from scipy import stats
from load_clean_data import load_clean_data
from compute_counts import compute_counts
from filter_rank import filter_and_rank

def compute_correlation(filtered):
    """
    Calculate Pearson's r for:
      x = bev_phev_ratio (finite values only)
      y = BEV count (for same indices)
    """
    x = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    y = filtered.loc[x.index, 'BEV']
    return stats.pearsonr(x, y)

if __name__ == '__main__':
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    counts = compute_counts(df)
    filtered, _, _ = filter_and_rank(counts)
    corr, pval = compute_correlation(filtered)
    print(f"Pearson r = {corr:.2f}, p-value = {pval:.3f}")
