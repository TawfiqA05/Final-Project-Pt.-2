#!/usr/bin/env python3
"""
Purpose:
    Load the EV dataset from CSV, clean it by removing invalid rows, standardize county names,
    and normalize the EV type to just "BEV" or "PHEV".

Expected Output:
    - On success: prints "Loaded N rows from <filepath>"
    - Returns a pandas DataFrame with:
        • County names title-cased
        • Electric Vehicle Type values only "BEV" or "PHEV"
"""

import os
import sys
import pandas as pd

def load_clean_data(filepath):
    """
    Load and clean the EV dataset.

    Steps:
      1. Verify file exists; exit with error if not.
      2. Read CSV into DataFrame.
      3. Drop rows missing 'County' or 'Electric Vehicle Type'.
      4. Standardize 'County' to Title Case without extra spaces.
      5. Extract the suffix "(BEV)" or "(PHEV)" where present, map to "BEV"/"PHEV".
      6. Drop any rows that didn’t match BEV/PHEV.

    Returns:
      Cleaned pandas DataFrame.
    """
    if not os.path.isfile(filepath):
        sys.exit(f"ERROR: File not found - {filepath}")
    df = pd.read_csv(filepath)

    # 3. Drop rows missing key fields
    df = df.dropna(subset=['County', 'Electric Vehicle Type'])

    # 4. Standardize county names
    df['County'] = df['County'].str.strip().str.title()

    # 5. Normalize EV type to just BEV or PHEV
    df['Electric Vehicle Type'] = (
        df['Electric Vehicle Type']
          .str.extract(r'\((BEV|PHEV)\)', expand=False)
    )
    # 6. Drop any rows that failed to extract
    df = df.dropna(subset=['Electric Vehicle Type'])

    return df

if __name__ == '__main__':
    filepath = 'cleaned_sample_ev_data (1).csv'
    df = load_clean_data(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")