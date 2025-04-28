#!/usr/bin/env python3
"""
Purpose:
    Load the EV dataset from CSV, clean it by removing invalid rows, and standardize county names.

Expected Output:
    - On success: prints "Loaded N rows from <filepath>"
    - Returns a pandas DataFrame with no missing County or Electric Vehicle Type values,
      and properly formatted County names.
"""

import os
import sys
import pandas as pd

def load_clean_data(filepath):
    """
    Load and clean the EV dataset.

    Steps:
      1. Verify file exists; exit with error message if not.
      2. Read CSV into DataFrame.
      3. Drop rows missing 'County' or 'Electric Vehicle Type'.
      4. Standardize 'County' column to Title Case without extra spaces.

    Returns:
      Cleaned pandas DataFrame.
    """
    if not os.path.isfile(filepath):
        sys.exit(f"ERROR: File not found - {filepath}")
    df = pd.read_csv(filepath)

    # Drop rows where key fields are missing
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])

    # Standardize county names: strip whitespace, title-case
    df['County'] = df['County'].str.strip().str.title()

    return df

if __name__ == '__main__':
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")