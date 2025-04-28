#!/usr/bin/env python3
"""
Purpose:
    Run the full EV charger prioritization pipeline end-to-end.

Expected Output:
    - Two plot files in working directory:
        'top_bev_phev_ratios.png'
        'ratio_vs_bev_load.png'
    - Console prints:
        * Correlation statistic
        * Top vehicle model tokens
        * DataFrame of urgent counties
"""

from load_clean_data import load_clean_data
from compute_counts import compute_counts
from filter_rank import filter_and_rank
from plot_visuals import plot_top_ratios, plot_ratio_vs_load
from stats_insights import compute_correlation
from nlp_analysis import model_name_token_freq

def main():
    # Load and clean raw data
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)

    # Compute county-level counts and ratios
    counts = compute_counts(df)

    # Filter, rank, and flag urgent counties
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)

    # Create and save visualizations
    plot_top_ratios(filtered)
    plot_ratio_vs_load(filtered)

    # Compute and display correlation insight
    corr, pval = compute_correlation(filtered)
    print(f"Correlation (ratio vs BEV): r = {corr:.2f}, p = {pval:.3f}")

    # Compute and display NLP insight
    tokens = model_name_token_freq(df)
    print("Top vehicle model tokens:", tokens)

    # List counties requiring urgent charger deployment
    urgent = filtered[filtered['urgent']][['County', 'BEV', 'bev_phev_ratio']]
    print("Urgent counties for charger deployment:\n", urgent)

if __name__ == '__main__':
    main()