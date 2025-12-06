"""
This script creates a final, enriched dataset by merging all previously extracted
features, performing data quality checks, and generating a data dictionary and
quality report.
"""
import sys
import os
import pandas as pd
import numpy as np
import re
import ast

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
BASE_DATA_PATH = "data/Results/data_with_events.csv"
# This price data file is optional and the script will proceed without it.
PRICE_DATA_PATH = "data/Bronze/price_data.csv" 

# Output paths
FINAL_DATASET_PATH = "data/Gold/enriched_dataset.csv"
DATA_DICT_PATH = "data/Gold/data_dictionary.txt"
QUALITY_REPORT_PATH = "data/Gold/quality_report.txt"

# --- Helper Functions ---
def safe_literal_eval(val):
    """Safely evaluate a string representation of a list/dict."""
    try:
        if isinstance(val, str) and val.strip().startswith(('[', '{')):
            return ast.literal_eval(val)
        elif isinstance(val, (list, dict)):
            return val
    except (ValueError, SyntaxError):
        pass
    return None # Return None for failed conversions

def get_sentiment_label(score):
    """Categorizes a sentiment score."""
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- Main Script ---
def main():
    print("--- Creating Final Enriched Dataset ---")
    
    # 1. Load Data
    print(f"Loading base dataset: {BASE_DATA_PATH}...")
    if not os.path.exists(BASE_DATA_PATH):
        print(f"FATAL: Base data file not found at '{BASE_DATA_PATH}'. Please run previous scripts.")
        return
    df = pd.read_csv(BASE_DATA_PATH)

    # Load optional price data
    price_df = None
    if os.path.exists(PRICE_DATA_PATH):
        print(f"Loading price data: {PRICE_DATA_PATH}...")
        price_df = pd.read_csv(PRICE_DATA_PATH)
    else:
        print(f"Warning: Optional price data file not found at '{PRICE_DATA_PATH}'. Proceeding without price info.")

    # Initial Stats
    initial_rows = len(df)
    report_lines = [f"Initial analysis of {BASE_DATA_PATH}\n" + "="*40]
    report_lines.append(f"Initial row count: {initial_rows}")

    # 2. Data Quality Checks & Cleaning (Initial Pass)
    print("Performing initial data cleaning and validation...")
    
    # Drop rows with missing critical data
    df.dropna(subset=['id', 'text'], inplace=True)
    rows_after_dropna = len(df)
    report_lines.append(f"Dropped {initial_rows - rows_after_dropna} rows with missing 'id' or 'text'.")

    # Drop duplicates
    df.drop_duplicates(subset=['id'], inplace=True)
    rows_after_dedup = len(df)
    report_lines.append(f"Dropped {rows_after_dropna - rows_after_dedup} duplicate rows based on 'id'.")

    # Format date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    date_nulls = df['date'].isnull().sum()
    if date_nulls > 0:
        report_lines.append(f"Found and marked {date_nulls} invalid date formats as NaT.")
        df.dropna(subset=['date'], inplace=True)
        report_lines.append("-> Rows with invalid dates have been dropped.")
    
    # Convert list/dict-like columns
    for col in ['cryptos_mentioned', 'exchanges_mentioned', 'influencers_mentioned', 'events_detected']:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)

    # 3. Feature Engineering
    print("Engineering new features...")
    df['text_length'] = df['text'].str.len()
    df['hashtag_count'] = df['text'].str.count(r'#\w+')
    df['emoji_count'] = df['text'].str.count(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
    
    # Sentiment features
    if 'sentiment' in df.columns:
        df.rename(columns={'sentiment': 'vader_compound_score'}, inplace=True)
        df['sentiment_label'] = df['vader_compound_score'].apply(get_sentiment_label)
        invalid_sentiment = df[~df['vader_compound_score'].between(-1, 1)].shape[0]
        report_lines.append(f"Validated 'vader_compound_score' is between -1 and 1. Found {invalid_sentiment} invalid scores.")
    else:
        report_lines.append("Warning: 'sentiment' column not found. Skipping sentiment feature engineering.")

    # Event features
    if 'events_detected' in df.columns and not df['events_detected'].isnull().all():
        df['event_types'] = df['events_detected'].apply(lambda events: [e['event_type'] for e in events if e] if isinstance(events, list) else [])
    else:
        df['event_types'] = pd.Series([[] for _ in range(len(df))])

    # 4. Merge Price Data (if available)
    if price_df is not None:
        print("Merging price data...")
        try:
            price_df['date'] = pd.to_datetime(price_df['date']).dt.date
            df['date_only'] = df['date'].dt.date
            
            # Explode dataframe to merge prices for each mentioned crypto
            df_exploded = df.explode('cryptos_mentioned').rename(columns={'cryptos_mentioned': 'crypto'})
            # Drop rows where crypto_mentioned was empty (resulting in NaN in 'crypto' after explode)
            df_exploded.dropna(subset=['crypto'], inplace=True) 
            df_merged = pd.merge(df_exploded, price_df, on=['date_only', 'crypto'], how='left')
            
            # The merged_df now contains one row per (post_id, crypto_mentioned) combination
            # This is the desired granularity for correlation analysis.
            df = df_merged.copy()
            df.drop(columns=['date_only'], inplace=True)
            report_lines.append("Successfully merged price data. Dataset is now at post-crypto granularity.")
        except Exception as e:
            report_lines.append(f"Error merging price data: {e}. Proceeding without it (original df retained).")
            # If merge fails, revert to original df structure (without price data)
            df.drop(columns=['date_only'], inplace=True, errors='ignore') # remove if it was added
            
    # 5. Final Schema & Summary
    final_schema = [
        'id', 'text', 'date', 'source', 'author', 'engagement',
        'sentiment_label', 'vader_compound_score',
        'crypto', 'exchanges_mentioned', 'influencers_mentioned', # 'crypto' is now the specific crypto for this row
        'events_detected', 'event_types',
        'price_open', 'price_close', 'price_change_pct', 'volume',
        'text_length', 'emoji_count', 'hashtag_count'
    ]
    # Add columns that might be missing, and reorder
    # Ensure 'crypto' is not a list in this context, it's the specific crypto for the row
    if 'cryptos_mentioned' in df.columns:
        df.drop(columns=['cryptos_mentioned'], inplace=True, errors='ignore')

    for col in final_schema:
        if col not in df.columns:
            df[col] = np.nan
    df = df[final_schema]

    report_lines.append("\n" + "Final Dataset Summary\n" + "="*40)
    report_lines.append(f"Final row count: {len(df)}")
    report_lines.append(f"Final column count: {len(df.columns)}")
    report_lines.append("\nColumn Completeness (%):")
    completeness = (df.notna().sum() / len(df)) * 100
    report_lines.append(completeness.to_string())

    # 6. Save Outputs
    print("Saving final outputs...")
    os.makedirs("data/Gold", exist_ok=True)
    df.to_csv(FINAL_DATASET_PATH, index=False)
    
    # Data Dictionary
    data_dict_content = {
        'id': 'Unique identifier for the post.',
        'text': 'The original text content of the post.',
        'date': 'The date and time the post was created.',
        'source': 'The platform the post originated from (e.g., Reddit).',
        'author': 'The author of the post.',
        'engagement': 'A measure of engagement (e.g., upvotes, likes).',
        'sentiment_label': 'Categorical sentiment (Positive, Negative, Neutral).',
        'vader_compound_score': 'Sentiment score from VADER (-1 to 1).',
        'crypto': 'The specific cryptocurrency mentioned in this post/row.', # Updated description
        'exchanges_mentioned': 'List of crypto exchanges mentioned.',
        'influencers_mentioned': 'List of key influencers mentioned.',
        'events_detected': 'List of detected event dictionaries (type, snippet, etc.).',
        'event_types': 'A simple list of event types found.',
        'price_open': 'Opening price of the mentioned crypto on the post date.',
        'price_close': 'Closing price of the mentioned crypto on the post date.',
        'price_change_pct': 'Percentage price change of the mentioned crypto on the post date.',
        'volume': 'Trading volume of the mentioned crypto on the post date.',
        'text_length': 'Number of characters in the post text.',
        'emoji_count': 'Number of emojis in the post text.',
        'hashtag_count': 'Number of hashtags in the post text.'
    }
    with open(DATA_DICT_PATH, 'w') as f:
        for col, desc in data_dict_content.items():
            f.write(f"{col}: {desc}\n")

    # Quality Report
    final_report = "\n".join(report_lines)
    with open(QUALITY_REPORT_PATH, 'w') as f:
        f.write(final_report)

    print("\n--- Data Quality Report ---")
    print(final_report)
    
    print(f"\nSuccessfully created enriched dataset at: {FINAL_DATASET_PATH}")
    print(f"Data dictionary saved to: {DATA_DICT_PATH}")
    print(f"Quality report saved to: {QUALITY_REPORT_PATH}")
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
