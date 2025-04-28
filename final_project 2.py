#!/usr/bin/env python3
"""
EV Charger Prioritization Pipeline

What’s supposed to happen:
 1. Load & clean the raw EV CSV.
 2. Count BEVs vs PHEVs per county, compute totals and BEV/PHEV ratio.
 3. Keep only counties with ≥10 EVs, rank by ratio and BEV count,
    then flag the top-10% in each metric as “urgent.”
 4. Plot:
      • Bar chart of top counties by BEV/PHEV ratio → top_bev_phev_ratios.png
      • Scatter of ratio vs BEV load highlighting urgent → ratio_vs_bev_load.png
 5. Compute Pearson’s r between ratio and BEV count.
 6. Tokenize all “Vehicle Model” names, list the 10 most common tokens.
 7. Print the list of urgent counties.

Predicted output:
 - Two PNGs saved to disk.
 - Console prints of correlation, top 10 model tokens, and urgent-county table.

Libraries used (≥6): os, sys, pandas, numpy, matplotlib, seaborn, scipy, nltk
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from scipy import stats

# 1. Ensure tokenizer data is present (first run only)
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize


# ──────────────────────────────────────────────────────────────────────────────
def load_clean_data(filepath):
    """
    Load the CSV, drop rows missing County or EV Type, and standardize county names.
    Exits with an error if the file is not found.
    """
    if not os.path.isfile(filepath):
        sys.exit(f"ERROR: File not found – {filepath}")
    df = pd.read_csv(filepath)

    # Drop any rows missing our two key columns
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])

    # Strip whitespace and title-case county names for consistency
    df['County'] = df['County'].str.strip().str.title()
    return df


# ──────────────────────────────────────────────────────────────────────────────
def compute_counts(df):
    """
    Group by County & EV Type to count BEVs and PHEVs, then compute:
      • total_evs = BEV + PHEV
      • bev_phev_ratio = BEV / PHEV (inf if PHEV == 0)
    Returns a DataFrame: County, BEV, PHEV, total_evs, bev_phev_ratio
    """
    counts = (
        df
        .groupby(['County', 'Electric Vehicle Type'])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure both columns exist
    for col in ('BEV', 'PHEV'):
        if col not in counts.columns:
            counts[col] = 0

    counts['total_evs'] = counts['BEV'] + counts['PHEV']
    counts['bev_phev_ratio'] = counts.apply(
        lambda row: row['BEV'] / row['PHEV'] if row['PHEV'] > 0 else np.inf,
        axis=1
    )
    return counts.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
def filter_and_rank(counts, min_evs=10):
    """
    1) Keep counties with total_evs ≥ min_evs.
    2) Rank by bev_phev_ratio and BEV count (highest=rank 1).
    3) Compute 90th percentile thresholds.
    4) Mark 'urgent' those ≥ both thresholds.
    Returns (filtered_df, thr_ratio, thr_bev).
    """
    # 1) Filter
    filtered = counts[counts['total_evs'] >= min_evs].copy()
    if filtered.empty:
        return filtered, None, None

    # 2) Rankings
    filtered['ratio_rank'] = filtered['bev_phev_ratio'].rank(method='min', ascending=False)
    filtered['bev_rank']   = filtered['BEV'].rank(method='min', ascending=False)

    # 3) 90th‐percentile thresholds
    finite = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    thr_ratio = np.percentile(finite, 90) if not finite.empty else None
    thr_bev   = np.percentile(filtered['BEV'], 90)

    # 4) Flag urgent
    if thr_ratio is not None:
        filtered['urgent'] = (
            (filtered['bev_phev_ratio'] >= thr_ratio) &
            (filtered['BEV'] >= thr_bev)
        )
    else:
        filtered['urgent'] = False

    return filtered, thr_ratio, thr_bev


# ──────────────────────────────────────────────────────────────────────────────
def plot_top_ratios(filtered, top_n=10):
    """
    Save a bar chart of the top N counties by BEV/PHEV ratio.
    Output: top_bev_phev_ratios.png
    """
    top = filtered.sort_values('bev_phev_ratio', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='bev_phev_ratio', y='County', data=top, palette='viridis')
    plt.title('Top BEV-to-PHEV Ratios by County')
    plt.xlabel('BEV / PHEV Ratio')
    plt.ylabel('County')
    plt.tight_layout()
    plt.savefig('top_bev_phev_ratios.png')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def plot_ratio_vs_load(filtered):
    """
    Save a scatterplot of BEV/PHEV ratio vs. BEV count.
    Urgent counties in red; threshold lines at 90th percentiles.
    Output: ratio_vs_bev_load.png
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x='bev_phev_ratio', y='BEV', hue='urgent', data=filtered,
        palette={True: 'red', False: 'blue'}, legend='brief'
    )
    # draw threshold lines
    plt.axvline(filtered['bev_phev_ratio'].quantile(0.90), linestyle='--', color='gray')
    plt.axhline(filtered['BEV'].quantile(0.90), linestyle='--', color='gray')
    plt.title('Infrastructure Gap: Ratio vs. BEV Load')
    plt.xlabel('BEV/PHEV Ratio')
    plt.ylabel('BEV Count')
    plt.tight_layout()
    plt.savefig('ratio_vs_bev_load.png')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def compute_correlation(filtered):
    """
    Compute Pearson’s r & p-value between bev_phev_ratio and BEV count.
    Only finite ratio values are used.
    """
    x = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    y = filtered.loc[x.index, 'BEV']
    return stats.pearsonr(x, y)


# ──────────────────────────────────────────────────────────────────────────────
def model_name_token_freq(df):
    """
    Concatenate all Vehicle Model strings,
    tokenize, and return the top 10 most common tokens.
    """
    if 'Vehicle Model' not in df.columns:
        return None
    text = ' '.join(df['Vehicle Model'].astype(str))
    tokens = word_tokenize(text)
    freq = nltk.FreqDist(tokens)
    return freq.most_common(10)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ** 1) Load & Clean **
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)

    # ** 2) Compute Counts & Ratios **
    counts = compute_counts(df)

    # ** 3) Filter, Rank & Flag Urgent **
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)

    # ** 4) Produce & Save Plots **
    plot_top_ratios(filtered)
    plot_ratio_vs_load(filtered)
    print("✅ Saved plots: top_bev_phev_ratios.png, ratio_vs_bev_load.png")

    # ** 5) Correlation Insight **
    if not filtered.empty and thr_ratio is not None:
        r, p = compute_correlation(filtered)
        print(f"🔗 Pearson r (ratio vs BEV): {r:.2f}, p = {p:.3f}")
    else:
        print("⚠️ Not enough data to compute correlation.")

    # ** 6) NLP Insight **
    tokens = model_name_token_freq(df)
    print("📝 Top vehicle model tokens:", tokens)

    # ** 7) List Urgent Counties **
    if filtered.empty:
        print("🚫 No counties meet the minimum EV threshold.")
    else:
        urgent = filtered.loc[filtered['urgent'], ['County', 'BEV', 'bev_phev_ratio']]
        print("\n🚨 Urgent counties for charger deployment:")
        print(urgent.to_string(index=False))


if __name__ == '__main__':
    main()