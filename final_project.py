# final_project.py
# Group: Tawfiq Abulail & Aidan Susnar
# Purpose: Identify high BEV-to-PHEV counties and prioritize charger deployment

import os  # standard module for path handling
import sys  # standard module for system operations
import re   # regular expressions for text cleaning

import numpy as np  # numeric operations (Lecture 5)
import pandas as pd  # data manipulation (Lecture 6)
import matplotlib.pyplot as plt  # plotting (Lecture 12)
import seaborn as sns  # statistical visualization (Lecture 13)
from scipy import stats  # statistical functions (Lecture 7)
from sklearn.preprocessing import StandardScaler  # scaling features (Lecture 10)
from sklearn.cluster import KMeans  # clustering (Lecture 10)
import nltk  # NLP toolkit (Lecture 14 & 15)

# For Tawfiq: Download punkt for tokenization (only first run)
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# ----------------------------------------
# 1. Data Loading & Cleaning
# ----------------------------------------

def load_clean_data(filepath):
    """
    Load the EV dataset and perform initial cleaning.
    Tawfiq: ensure path exists and handle errors.
    Aidan: drop missing or malformed records.
    """
    if not os.path.isfile(filepath):
        sys.exit(f"ERROR: File not found - {filepath}")
    df = pd.read_csv(filepath)

    # Filter out rows missing county or EV type
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])
    # Standardize county names
    df['County'] = df['County'].str.strip().str.title()
    return df

# ----------------------------------------
# 2. Ratio & Load Computation
# ----------------------------------------

def compute_counts(df):
    """
    Compute per-county counts of BEVs and PHEVs.
    """
    # Count by county and type
    counts = df.groupby(['County', 'Electric Vehicle Type']).size().unstack(fill_value=0)
    # Ensure both columns exist
    for col in ['BEV', 'PHEV']:
        if col not in counts.columns:
            counts[col] = 0
    counts['total_evs'] = counts['BEV'] + counts['PHEV']
    # Ratio BEV-to-PHEV, avoid division by zero
    counts['bev_phev_ratio'] = counts.apply(
        lambda row: row['BEV']/row['PHEV'] if row['PHEV']>0 else np.inf,
        axis=1
    )
    return counts.reset_index()

# ----------------------------------------
# 3. Filtering & Prioritization
# ----------------------------------------

def filter_and_rank(counts, min_evs=10):
    """
    Filter counties by minimum EV count, then rank by ratio and BEV load.
    """
    filtered = counts[counts['total_evs'] >= min_evs].copy()
    # Rank by ratio and BEV count
    filtered['ratio_rank'] = filtered['bev_phev_ratio'].rank(method='min', ascending=False)
    filtered['bev_rank'] = filtered['BEV'].rank(method='min', ascending=False)
    # Identify 'urgent' quadrant: top percentile on both metrics
    thr_ratio = np.percentile(filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna(), 90)
    thr_bev = np.percentile(filtered['BEV'], 90)
    filtered['urgent'] = ((filtered['bev_phev_ratio'] >= thr_ratio) & (filtered['BEV'] >= thr_bev))
    return filtered, thr_ratio, thr_bev

# ----------------------------------------
# 4. Visualizations
# ----------------------------------------

def plot_top_ratios(filtered, top_n=10):
    """
    Bar chart of top counties by BEV/PHEV ratio.
    """
    top = filtered.sort_values('bev_phev_ratio', ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x='bev_phev_ratio', y='County', data=top, palette='viridis')
    plt.title('Top BEV-to-PHEV Ratios by County')
    plt.xlabel('BEV / PHEV Ratio')
    plt.ylabel('County')
    plt.tight_layout()
    plt.savefig('top_bev_phev_ratios.png')
    plt.close()


def plot_ratio_vs_load(filtered):
    """
    Scatterplot of BEV/PHEV ratio vs. raw BEV count, highlighting urgent counties.
    """
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x='bev_phev_ratio', y='BEV', hue='urgent', data=filtered,
        palette={True:'red', False:'blue'}, legend='brief'
    )
    plt.axvline(x=filtered['bev_phev_ratio'].quantile(0.90), linestyle='--', color='gray')
    plt.axhline(y=filtered['BEV'].quantile(0.90), linestyle='--', color='gray')
    plt.title('Infrastructure Gap: Ratio vs. BEV Load')
    plt.xlabel('BEV/PHEV Ratio')
    plt.ylabel('BEV Count')
    plt.tight_layout()
    plt.savefig('ratio_vs_bev_load.png')
    plt.close()

# ----------------------------------------
# 5. Statistical Insights
# ----------------------------------------

def compute_correlation(filtered):
    """
    Use SciPy to compute Pearson correlation between ratio and BEV count.
    """
    x = filtered['bev_phev_ratio'].replace(np.inf, np.nan).dropna()
    y = filtered.loc[x.index, 'BEV']
    corr, pval = stats.pearsonr(x, y)
    return corr, pval

# ----------------------------------------
# 6. NLP Token Analysis
# ----------------------------------------

def model_name_token_freq(df):
    """
    Tokenize vehicle model names and compute frequency distribution (NLTK).
    """
    # Extract model names column if exists
    if 'Vehicle Model' not in df.columns:
        return None
    text = ' '.join(df['Vehicle Model'].astype(str).tolist())
    tokens = word_tokenize(text)
    freq = nltk.FreqDist(tokens)
    return freq.most_common(10)

# ----------------------------------------
# 7. Main Execution
# ----------------------------------------

def main():
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)
    counts = compute_counts(df)
    filtered, thr_ratio, thr_bev = filter_and_rank(counts)

    # Visuals
    plot_top_ratios(filtered)
    plot_ratio_vs_load(filtered)

    # Stats
    corr, pval = compute_correlation(filtered)
    print(f"Pearson correlation (ratio vs BEV): r={corr:.2f}, p={pval:.3f}")

    # NLP insight
    top_tokens = model_name_token_freq(df)
    print("Top vehicle model tokens:", top_tokens)

    # Output urgent counties
    urgent_counties = filtered[filtered['urgent']][['County','BEV','bev_phev_ratio']]
    print("Urgent counties for charger deployment:\n", urgent_counties)

if __name__ == '__main__':
    main()