#!/usr/bin/env python3
"""
EV Charger Prioritization Pipeline

This script carries out the full workflow:
 1. Load & clean the raw EV dataset.
 2. Compute per-county BEV & PHEV counts, totals, and BEV/PHEV ratio.
 3. Filter to counties with ≥10 EVs, rank by ratio & BEV count, and flag top-10% as “urgent.”
 4. Produce and save two plots:
      • Bar chart of top BEV/PHEV ratios by county.
      • Scatterplot of ratio vs. BEV count, highlighting urgent counties.
 5. Compute Pearson correlation between ratio & BEV count.
 6. Tokenize all vehicle model names and report the 10 most common tokens.
 7. Print the list of urgent counties.

Usage:
    python3 ev_pipeline.py

Ensure that `cleaned_sample_ev_data (1).csv` is present in the same folder.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from scipy import stats

# Download NLTK data needed for tokenization (only on first run)
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize


def load_clean_data(filepath):
    """
    Load the CSV, drop rows missing key fields, and standardize county names.
    Exits with an error if file not found.
    """
    if not os.path.isfile(filepath):
        sys.exit(f"ERROR: File not found – {filepath}")
    df = pd.read_csv(filepath)

    # Drop rows with no county or no EV type
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])

    # Clean up county names: strip whitespace and title-case
    df['County'] = df['County'].str.strip().str.title()
    return df


def compute_counts(df):
    """
    Compute per-county counts:
      • BEV count
      • PHEV count
      • total EVs = BEV + PHEV
      • BEV/PHEV ratio (infinite if PHEV == 0)
    Returns a DataFrame with columns:
      County, BEV, PHEV, total_evs, bev_phev_ratio
    """
    # Count each EV type in each county
    counts = (
        df
        .groupby(['County', 'Electric Vehicle Type'])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both columns exist even if missing in data
    for col in ('BEV', 'PHEV'):
        if col not in counts.columns:
            counts[col] = 0

    # Total and ratio
    counts['total_evs'] = counts['BEV'] + counts['PHEV']
    counts['bev_phev_ratio'] = counts.apply(
        lambda row: row['BEV'] / row['PHEV'] if row['PHEV'] > 0 else np.inf,
        axis=1
    )
    return counts.reset_index()


def filter_and_rank(counts, min_evs=10):
    """
    Filter to counties with total_evs >= min_evs, then:
      1. Rank counties by bev_phev_ratio (desc) and BEV count (desc).
      2. Compute 90th-percentile thresholds for ratio and BEV.
      3. Flag as 'urgent' those meeting or exceeding both thresholds.
    Returns (filtered_df, threshold_ratio, threshold_bev).
    If no county meets min_evs, returns (empty_df, None, None).
    """
    # 1) Filter
    filtered = counts[counts['total_evs'] >= min_evs].copy()
    if filtered.empty:
        return filtered, None, None

    # 2) Ranking: rank 1 = highest
    filtered['ratio_rank'] = filtered['bev_phev_ratio'].rank(method='min', ascending=False)
    filtered['bev_rank']   = filtered['BEV'].rank(method='min', ascending=False)

    # 3) Thresholds at 90th percentile
    finite_ratios = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    thr_ratio = np.percentile(finite_ratios, 90) if not finite_ratios.empty else None
    thr_bev   = np.percentile(filtered['BEV'], 90)

    # 4) Mark urgent
    if thr_ratio is not None:
        filtered['urgent'] = (
            (filtered['bev_phev_ratio'] >= thr_ratio) &
            (filtered['BEV'] >= thr_bev)
        )
    else:
        # No finite ratios; none can be urgent by ratio
        filtered['urgent'] = False

    return filtered, thr_ratio, thr_bev


def plot_top_ratios(filtered, top_n=10):
    """
    Save a bar chart of the top N counties by BEV/PHEV ratio.
    Output file: top_bev_phev_ratios.png
    """
    top = filtered.sort_values('bev_phev_ratio', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
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
    Save a scatterplot of BEV/PHEV ratio vs. BEV count.
    Urgent counties in red, others in blue; 90th percentile lines drawn.
    Output file: ratio_vs_bev_load.png
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x='bev_phev_ratio', y='BEV', hue='urgent', data=filtered,
        palette={True: 'red', False: 'blue'}, legend='brief'
    )
    # Draw threshold lines
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


def compute_correlation(filtered):
    """
    Compute Pearson's r & p-value between BEV/PHEV ratio and BEV count,
    considering only finite ratio values.
    """
    x = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    y = filtered.loc[x.index, 'BEV']
    return stats.pearsonr(x, y)


def model_name_token_freq(df):
    """
    Tokenize every string in 'Vehicle Model' column,
    build frequency distribution, and return top 10 tokens.
    """
    if 'Vehicle Model' not in df.columns:
        return None
    all_text = ' '.join(df['Vehicle Model'].astype(str))
    tokens = word_tokenize(all_text)
    freq = nltk.FreqDist(tokens)
    return freq.most_common(10)


def main():
    # === 1) Load & Clean ===
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)

    # === 2) Compute Counts & Ratios ===
    counts = compute_counts(df)

    # === 3) Filter, Rank & Flag Urgent ===
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)

    # === 4) Visualizations ===
    plot_top_ratios(filtered)
    plot_ratio_vs_load(filtered)
    print("📊 Plots saved as 'top_bev_phev_ratios.png' and 'ratio_vs_bev_load.png'")

    # === 5) Correlation Insight ===
    if not filtered.empty and thr_ratio is not None:
        r, p = compute_correlation(filtered)
        print(f"🔗 Correlation (ratio vs BEV): r = {r:.2f}, p = {p:.3f}")
    else:
        print("⚠️ Not enough data to compute correlation.")

    # === 6) NLP Insight ===
    tokens = model_name_token_freq(df)
    print("📝 Top vehicle model tokens:", tokens)

    # === 7) Urgent Counties ===
    if filtered.empty:
        print("🚫 No counties meet the minimum EV threshold.")
    else:
        urgent = filtered.loc[filtered['urgent'], ['County', 'BEV', 'bev_phev_ratio']]
        print("\n🚨 Urgent counties for charger deployment:\n", urgent.to_string(index=False))


if __name__ == '__main__':
    main()