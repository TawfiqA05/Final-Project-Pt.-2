#!/usr/bin/env python3
"""
EV Charger Prioritization – Complete Analysis

This script will:
  1. Load & clean the EV dataset.
  2. Compute BEV/PHEV counts & ratios by county.
  3. Filter to counties with ≥ MIN_EVS total EVs.
  4. Print:
       • Top TOP_N counties by BEV∶PHEV ratio
       • Among those, top TOP_N by raw BEV count
       • Base “urgent” thresholds at BASE_PCT
       • Urgent counties at each pct in PCTILES
  5. Run correlation tests (Pearson & Spearman).
  6. Run K-Means clustering.
  7. Plot clusters + urgency quadrants for each pct in PCTILES.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.cluster import KMeans

# ——— Parameters ———
INPUT_CSV   = "cleaned_sample_ev_data.csv"
MIN_EVS     = 10
TOP_N       = 10
BASE_PCT    = 90
PCTILES     = [60, 70, 85]
N_CLUSTERS  = 3
RANDOM_SEED = 0

# ——— 1. Load & Clean ———
def load_and_clean(fp):
    df = pd.read_csv(fp)
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])
    df['County'] = df['County'].str.strip().str.title()
    df['EVType'] = df['Electric Vehicle Type']\
                   .str.extract(r'\((BEV|PHEV)\)', expand=False)
    return df.dropna(subset=['EVType'])

# ——— 2. Compute Counts & Ratio ———
def compute_counts(df):
    counts = (df.groupby('County')['EVType']
                .value_counts()
                .unstack(fill_value=0))
    counts['total_evs']      = counts.sum(axis=1)
    counts['bev_phev_ratio'] = counts['BEV'] / counts['PHEV'].replace(0, np.nan)
    return counts

# ——— 3. Filter by MIN_EVS ———
def filter_min(counts, min_evs=MIN_EVS):
    return counts[counts['total_evs'] >= min_evs].copy()

# ——— 4. Print Top-10 & Base Urgency ———
def print_summary_and_base_urgency(filtered):
    # Top TOP_N by ratio
    top_ratio = filtered.sort_values('bev_phev_ratio', ascending=False).head(TOP_N)
    print(f"\nTop {TOP_N} counties by BEV∶PHEV ratio (≥ {MIN_EVS} EVs):")
    print(top_ratio[['BEV','PHEV','bev_phev_ratio']])

    # Among those, top TOP_N by BEV count
    top_bev = top_ratio.sort_values('BEV', ascending=False)
    print(f"\nAmong those, top {TOP_N} by BEV count:")
    print(top_bev[['BEV','PHEV','total_evs']])

    # Base “urgent” thresholds at BASE_PCT
    r_thr = np.nanpercentile(filtered['bev_phev_ratio'], BASE_PCT)
    b_thr = np.nanpercentile(filtered['BEV'], BASE_PCT)
    print(f"\nThresholds for “urgent” (≥ {BASE_PCT}th percentile):")
    print(f"  BEV∶PHEV ratio ≥ {r_thr:.2f}")
    print(f"  BEV count      ≥ {b_thr:.0f}")

# ——— 5. Print Urgency at Various Percentiles ———
def print_urgency_levels(filtered, pctiles):
    print("\nUrgent counties at selected percentiles:")
    for pct in pctiles:
        r_thr = np.nanpercentile(filtered['bev_phev_ratio'], pct)
        b_thr = np.nanpercentile(filtered['BEV'], pct)
        urgent = filtered[
            (filtered['bev_phev_ratio'] >= r_thr) &
            (filtered['BEV']             >= b_thr)
        ].index.tolist() or ["(none)"]
        print(f"  {pct}th pct → ratio ≥ {r_thr:.2f}, BEV ≥ {int(b_thr)} → {', '.join(urgent)}")

# ——— 6. Correlation Analysis ———
def analyze_correlation(filtered):
    x = filtered['bev_phev_ratio'].dropna()
    y = filtered.loc[x.index, 'BEV']
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    print("\nCorrelation Analysis:")
    print(f"  Pearson  r = {pr:.2f}, p = {pp:.3f}")
    print(f"  Spearman rho = {sr:.2f}, p = {sp:.3f}")

# ——— 7. K-Means Clustering ———
def cluster_analysis(filtered):
    X = filtered[['bev_phev_ratio','BEV']].fillna(0).values
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    labels = km.fit_predict(X)
    filtered = filtered.copy()
    filtered['cluster'] = labels
    print(f"\nK-Means Clustering (k={N_CLUSTERS}):")
    for i, c in enumerate(km.cluster_centers_):
        print(f"  Cluster {i} center → ratio={c[0]:.2f}, BEV={c[1]:.0f}")
    return filtered

# ——— 8. Plot Clusters & Urgency Quadrant ———
def plot_clusters_and_quadrant(filtered, pctile):
    sns.set_style("whitegrid")
    r_thr = np.nanpercentile(filtered['bev_phev_ratio'], pctile)
    b_thr = np.nanpercentile(filtered['BEV'], pctile)
    urgent = filtered[
        (filtered['bev_phev_ratio'] >= r_thr) &
        (filtered['BEV'] >= b_thr)
    ]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=filtered, x='bev_phev_ratio', y='BEV',
        hue='cluster',
        style=filtered.index.isin(urgent.index),
        markers={False: 'o', True: 's'},
        palette='tab10',
        legend='full'
    )
    plt.axvline(r_thr, linestyle='--', color='red', label=f'Ratio cutoff (≥ {pctile}th pct)')
    plt.axhline(b_thr, linestyle='--', color='blue', label=f'BEV cutoff (≥ {pctile}th pct)')
    plt.text(r_thr + 0.1, plt.ylim()[0], f'Ratio = {r_thr:.2f}', color='red', fontsize=10, rotation=90)
    plt.text(plt.xlim()[0], b_thr + 1, f'BEV = {b_thr:.0f}', color='blue', fontsize=10)

    # Add cluster centers to the legend
    cluster_centers = filtered.groupby('cluster')[['bev_phev_ratio', 'BEV']].mean()
    for cluster, (x, y) in cluster_centers.iterrows():
        plt.scatter([], [], label=f'Cluster {cluster}', color='black', alpha=0.6)

    plt.xlabel('BEV / PHEV Ratio')
    plt.ylabel('BEV Count')
    plt.title(f'Clusters & Urgency Quadrant (≥ {pctile}th percentile)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ——— 9. Plot Correlation Heatmap ———
def plot_correlation_heatmap(filtered):
    plt.figure(figsize=(8, 6))
    corr = filtered[['BEV', 'PHEV', 'bev_phev_ratio', 'total_evs']].corr()
    sns.heatmap(
        corr, annot=True, cmap='vlag', fmt=".2f", cbar=True,
        vmin=-1, vmax=1, annot_kws={"size": 10}
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
# ——— Main ———
if __name__ == "__main__":
    # Load and prep
    df     = load_and_clean("/Users/lambirghinibugatti/Downloads/B211/cleaned_sample_ev_data.csv")
    counts = compute_counts(df)
    filt   = filter_min(counts)

    # 4 & 5: summaries & urgency levels
    print_summary_and_base_urgency(filt)
    print_urgency_levels(filt, PCTILES)

    # 6: correlation
    analyze_correlation(filt)

    # 7: clustering
    filt = cluster_analysis(filt)

    # 8: plots for each percentile
    for pct in PCTILES:
        plot_clusters_and_quadrant(filt, pct)

    # Add this to the main section
if __name__ == "__main__":
    # Existing steps...
    for pct in PCTILES:
        plot_clusters_and_quadrant(filt, pct)

    # New heatmap plot
    plot_correlation_heatmap(filt)
