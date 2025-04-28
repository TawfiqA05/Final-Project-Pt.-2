#!/usr/bin/env python3
"""
Purpose:
    Tokenize the 'Vehicle Model' column and output the 10 most common tokens.

Expected Output:
    - Prints: "Top model tokens: [(token1, count1), ..., (token10, count10)]"
"""

import nltk
from load_clean_data import load_clean_data

# Download tokenizer data (first run only)
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

def model_name_token_freq(df):
    """
    Concatenate all Vehicle Model strings, tokenize,
    then return top 10 frequent tokens.
    """
    if 'Vehicle Model' not in df.columns:
        return None
    # Join all model names into a single text
    text = ' '.join(df['Vehicle Model'].astype(str).tolist())
    tokens = word_tokenize(text)
    freq = nltk.FreqDist(tokens)
    return freq.most_common(10)

if __name__ == '__main__':
    df = load_clean_data('cleaned_sample_ev_data (1).csv')
    tokens = model_name_token_freq(df)
    print("Top model tokens:", tokens)
