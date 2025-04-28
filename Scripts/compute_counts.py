#!/usr/bin/env python3
"""
Purpose:
    Compute per-county counts of BEVs and PHEVs, total EVs, and BEV/PHEV ratio.

Expected Output:
    - Prints first few rows of a DataFrame with columns:
        County, BEV, PHEV, total_evs, bev_phev_ratio
"""

import numpy as np
import pandas as pd
from load_clean_data import load_clean_data

def compute_counts(df):
    """
    Compute counts and ratio.

    Steps:
      1. Group by County and Electric Vehicle Type; count entries.
      2. Ensure both 'BEV' and 'PHEV' columns exist.
      3. Compute 'total_evs' as BEV + PHEV.
      4. Compute 'bev_phev_ratio' = BEV / PHEV (inf if PHEV=0).

    Returns:
      DataFrame with columns: County, BEV, PHEV, total_evs, bev_phev_ratio.
    """
    counts = df.groupby(['County', 'Electric Vehicle Type']) \
               .size() \
               .unstack(fill_value=0)

    # Add missing columns if dataset lacks one type
    for col in ['BEV', 'PHEV']:
        if col not in counts.columns:
            counts[col] = 0

    counts['total_evs'] = counts['BEV'] + counts['PHEV']
    counts['bev_phev_ratio'] = counts.apply(
        lambda row: row['BEV'] / row['PHEV'] if row['PHEV'] > 0 else np.inf,
        axis=1
    )

    return counts.reset_index()

if __name__ == '__main__':
    # Load and compute counts
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    counts = compute_counts(df)
    print(counts.head())  # Display sample of results
