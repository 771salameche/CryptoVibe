"""
This script aggregates sentiment scores by day and crypto to create a time-series
dataset for trend analysis and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# --- Configuration ---
INPUT_CSV_PATH = "data/Results/data_with_influencers.csv"
OUTPUT_CSV_PATH = "data/Results/sentiment_timeseries.csv"
CHART_OUTPUT_PATH = "data/Visualizations/sentiment_timeline.png"

# --- Helper Functions ---
def safe_literal_eval(val):
    """Safely evaluate a string representation of a list."""
    try:
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        pass
    return []

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
    print("--- Sentiment Aggregation for Time-Series Analysis ---")

    # 1. Load Data
    print(f"Loading dataset: {INPUT_CSV_PATH}...")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"FATAL: Input file not found at '{INPUT_CSV_PATH}'. Please run previous scripts.")
        return
    df = pd.read_csv(INPUT_CSV_PATH)

    # 2. Data Preparation
    print("Preparing data for aggregation...")
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['cryptos_mentioned'] = df['cryptos_mentioned'].apply(safe_literal_eval)
    
    # Explode the dataframe for crypto-specific analysis
    df_exploded = df[df['cryptos_mentioned'].apply(len) > 0].explode('cryptos_mentioned')
    df_exploded.rename(columns={'cryptos_mentioned': 'crypto'}, inplace=True)
    
    # Add sentiment label for counting
    df_exploded['sentiment_label'] = df_exploded['sentiment'].apply(get_sentiment_label)

    # 3. Daily Aggregation
    print("Aggregating sentiment scores by day and crypto...")
    daily_sentiment = df_exploded.groupby(['date', 'crypto']).agg(
        sentiment_mean=('sentiment', 'mean'),
        sentiment_median=('sentiment', 'median'),
        sentiment_std=('sentiment', 'std'),
        post_count=('id', 'count')
    ).reset_index()

    # Count positive/negative/neutral posts
    sentiment_counts = df_exploded.groupby(['date', 'crypto', 'sentiment_label']).size().unstack(fill_value=0)
    sentiment_counts.rename(columns={
        'Positive': 'positive_count', 'Negative': 'negative_count', 'Neutral': 'neutral_count'
    }, inplace=True)
    
    # Merge counts back into the main aggregated dataframe
    daily_sentiment = pd.merge(daily_sentiment, sentiment_counts, on=['date', 'crypto'], how='left').fillna(0)

    # 4. Time-Series Analysis (Moving Averages, Trends, Spikes)
    print("Calculating moving averages and sentiment trends...")
    all_crypto_ts = []
    
    # Ensure all cryptos have a full date range to handle missing days
    full_date_range = pd.date_range(start=daily_sentiment['date'].min(), end=daily_sentiment['date'].max(), freq='D')
    
    for crypto in daily_sentiment['crypto'].unique():
        crypto_df = daily_sentiment[daily_sentiment['crypto'] == crypto].set_index('date')
        crypto_df = crypto_df.reindex(full_date_range.date).ffill().reset_index().rename(columns={'index':'date'})
        crypto_df['crypto'].fillna(crypto, inplace=True)
        
        # Moving Averages
        crypto_df['sentiment_ma7'] = crypto_df['sentiment_mean'].rolling(window=7, min_periods=1).mean()
        crypto_df['sentiment_ma30'] = crypto_df['sentiment_mean'].rolling(window=30, min_periods=1).mean()
        
        # Sentiment Trends
        crypto_df['sentiment_change'] = crypto_df['sentiment_mean'].diff()
        crypto_df['trend'] = np.select(
            [crypto_df['sentiment_change'] > 0.01, crypto_df['sentiment_change'] < -0.01],
            ['Improving', 'Declining'],
            default='Stable'
        )
        
        # Flag Sentiment Spikes
        mean_abs_dev = (crypto_df['sentiment_mean'] - crypto_df['sentiment_ma7']).abs().mean()
        crypto_df['is_spike'] = (crypto_df['sentiment_mean'] - crypto_df['sentiment_ma7']).abs() > (3 * mean_abs_dev)
        
        all_crypto_ts.append(crypto_df)

    final_ts_df = pd.concat(all_crypto_ts, ignore_index=True).sort_values(by=['crypto', 'date'])

    # 5. Save Output
    print(f"Saving time-series data to: {OUTPUT_CSV_PATH}...")
    final_ts_df.to_csv(OUTPUT_CSV_PATH, index=False)

    # 6. Summary Statistics
    print("\n--- Summary Statistics ---")
    for crypto in final_ts_df['crypto'].unique():
        print(f"\n--- {crypto} ---")
        crypto_df = final_ts_df[final_ts_df['crypto'] == crypto]
        print(f"Overall Average Sentiment: {crypto_df['sentiment_mean'].mean():.4f}")
        
        most_volatile_day = crypto_df.sort_values(by='sentiment_std', ascending=False).iloc[0]
        print(f"Most Volatile Day: {most_volatile_day['date']} (Std: {most_volatile_day['sentiment_std']:.4f})")
        
        highest_post_day = crypto_df.sort_values(by='post_count', ascending=False).iloc[0]
        print(f"Day with Most Posts: {highest_post_day['date']} (Count: {int(highest_post_day['post_count'])})")

    # 7. Visualization
    print(f"\nGenerating and saving timeline chart to: {CHART_OUTPUT_PATH}...")
    unique_cryptos = final_ts_df['crypto'].unique()
    num_cryptos = len(unique_cryptos)
    fig, axes = plt.subplots(num_cryptos, 1, figsize=(15, 6 * num_cryptos), sharex=True)
    if num_cryptos == 1:
        axes = [axes]

    for ax, crypto in zip(axes, unique_cryptos):
        crypto_df = final_ts_df[final_ts_df['crypto'] == crypto]
        
        # Plot main sentiment line and moving averages
        ax.plot(crypto_df['date'], crypto_df['sentiment_mean'], label='Daily Mean Sentiment', alpha=0.5)
        ax.plot(crypto_df['date'], crypto_df['sentiment_ma7'], label='7-Day Moving Avg.', linestyle='--')
        ax.plot(crypto_df['date'], crypto_df['sentiment_ma30'], label='30-Day Moving Avg.', linestyle=':', linewidth=2)
        
        # Highlight sentiment spikes
        spikes = crypto_df[crypto_df['is_spike']]
        ax.scatter(spikes['date'], spikes['sentiment_mean'], color='red', s=100, zorder=5, label='Sentiment Spike')
        
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_title(f'Sentiment Timeline for {crypto}', fontsize=16)
        ax.set_ylabel('Sentiment Score (VADER Compound)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('Date')
    plt.tight_layout()
    os.makedirs(os.path.dirname(CHART_OUTPUT_PATH), exist_ok=True)
    plt.savefig(CHART_OUTPUT_PATH)
    print("Chart saved successfully.")
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
