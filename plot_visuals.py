#!/usr/bin/env python3
"""
Purpose:
    Produce two visualizations:
      1. Bar chart of top counties by BEV/PHEV ratio.
      2. Scatterplot of ratio vs. BEV count, highlighting 'urgent' counties.

Expected Output:
    - Saves 'top_bev_phev_ratios.png' and 'ratio_vs_bev_load.png' in working directory.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from load_clean_data import load_clean_data
from compute_counts import compute_counts
from filter_rank import filter_and_rank

def plot_top_ratios(filtered, top_n=10):
    """
    Bar chart: county vs. BEV/PHEV ratio.
    """
    top = filtered.sort_values('bev_phev_ratio', ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(
        x='bev_phev_ratio', y='County', data=top, palette='viridis'
    )
    plt.title('Top BEV-to-PHEV Ratios by County')
    plt.xlabel('BEV / PHEV Ratio')
    plt.ylabel('County')
    plt.tight_layout()
    plt.savefig('top_bev_phev_ratios.png')
    plt.close()

def plot_ratio_vs_load(filtered):
    """
    Scatterplot: BEV/PHEV ratio vs. BEV count.
    Urgent counties in red; threshold lines at 90th percentiles.
    """
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x='bev_phev_ratio', y='BEV', hue='urgent', data=filtered,
        palette={True: 'red', False: 'blue'}, legend='brief'
    )
    # Draw percentile threshold lines
    plt.axvline(
        x=filtered['bev_phev_ratio'].quantile(0.90),
        linestyle='--', color='gray'
    )
    plt.axhline(
        y=filtered['BEV'].quantile(0.90),
        linestyle='--', color='gray'
    )
    plt.title('Infrastructure Gap: Ratio vs. BEV Load')
    plt.xlabel('BEV/PHEV Ratio')
    plt.ylabel('BEV Count')
    plt.tight_layout()
    plt.savefig('ratio_vs_bev_load.png')
    plt.close()

if __name__ == '__main__':
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    counts = compute_counts(df)
    filtered, _, _ = filter_and_rank(counts)
    plot_top_ratios(filtered)
    plot_ratio_vs_load(filtered)
    print("Plots saved: top_bev_phev_ratios.png, ratio_vs_bev_load.png")