"""
This script analyzes the correlation between social media sentiment and
cryptocurrency prices, calculating Pearson and Spearman coefficients, checking for
statistical significance, and generating visualizations and a summary report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

# --- Configuration ---
SENTIMENT_DATA_PATH = "data/Results/sentiment_timeseries.csv"
PRICE_DATA_PATH = "data/Bronze/price_data.csv"

# Output paths
OUTPUT_DIR = "data/Visualizations/Correlation"
REPORT_PATH = "Correlation-Analysis/correlation_report.md"
HEATMAP_PATH = "data/Visualizations/correlation_heatmap.png"

# --- Main Analysis Functions ---
def load_and_merge_data():
    """Loads and merges sentiment and price datasets."""
    if not os.path.exists(SENTIMENT_DATA_PATH) or not os.path.exists(PRICE_DATA_PATH):
        print("Error: Required data files not found. Please run previous scripts.")
        return None
        
    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH, parse_dates=['date'])
    price_df = pd.read_csv(PRICE_DATA_PATH, parse_dates=['date'])
    
    # Merge the two dataframes
    df = pd.merge(sentiment_df, price_df, on=['date', 'crypto'], how='inner')
    
    # Calculate next-day price change for lagged correlation
    df = df.sort_values(by=['crypto', 'date'])
    df['next_day_price_change_pct'] = df.groupby('crypto')['price_change_pct'].shift(-1)
    
    return df

def analyze_crypto_correlation(df, crypto_symbol):
    """Performs correlation analysis for a single cryptocurrency."""
    print(f"\n--- Analyzing: {crypto_symbol} ---")
    crypto_df = df[df['crypto'] == crypto_symbol].dropna()
    
    if len(crypto_df) < 10:
        print("Not enough data points to perform meaningful correlation analysis.")
        return None, []
        
    # --- Calculations ---
    results = []
    # Same-day correlation
    p_r, p_p = pearsonr(crypto_df['sentiment_mean'], crypto_df['price_change_pct'])
    s_r, s_p = spearmanr(crypto_df['sentiment_mean'], crypto_df['price_change_pct'])
    results.append({'lag': 'Same-Day', 'type': 'Pearson', 'r': p_r, 'p_value': p_p})
    results.append({'lag': 'Same-Day', 'type': 'Spearman', 'r': s_r, 'p_value': s_p})

    # Next-day correlation
    p_r_lag, p_p_lag = pearsonr(crypto_df['sentiment_mean'], crypto_df['next_day_price_change_pct'])
    s_r_lag, s_p_lag = spearmanr(crypto_df['sentiment_mean'], crypto_df['next_day_price_change_pct'])
    results.append({'lag': 'Next-Day', 'type': 'Pearson', 'r': p_r_lag, 'p_value': p_p_lag})
    results.append({'lag': 'Next-Day', 'type': 'Spearman', 'r': s_r_lag, 'p_value': s_p_lag})

    # --- Visualizations ---
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='sentiment_mean', y='price_change_pct', data=crypto_df, ax=ax,
                scatter_kws={'alpha':0.5, 's':50})
    ax.set_title(f'Sentiment vs. Same-Day Price Change for {crypto_symbol}', fontsize=16)
    ax.set_xlabel('Mean Daily Sentiment')
    ax.set_ylabel('Price Change (%)')
    ax.axhline(0, color='grey', linestyle='--')
    ax.axvline(0, color='grey', linestyle='--')
    ax.grid(True)
    ax.text(0.05, 0.95, f"Pearson r = {p_r:.3f}\np-value = {p_p:.3f}", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.savefig(os.path.join(OUTPUT_DIR, f"correlation_scatter_{crypto_symbol}.png"))
    plt.close(fig)

    # Time Series Overlay
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.set_title(f'Sentiment and Price Change Timeline for {crypto_symbol}', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.plot(crypto_df['date'], crypto_df['sentiment_mean'], color='blue', label='Mean Sentiment')
    ax1.set_ylabel('Mean Sentiment Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(crypto_df['date'], crypto_df['price_change_pct'], color='green', alpha=0.6, label='Price Change %')
    ax2.set_ylabel('Price Change (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.savefig(os.path.join(OUTPUT_DIR, f"timeseries_overlay_{crypto_symbol}.png"))
    plt.close(fig)

    return crypto_df, results

def generate_report(all_results):
    """Generates and saves a markdown report of the findings."""
    report_df = pd.DataFrame(all_results)
    
    report_df['is_significant'] = report_df['p_value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    report_df['strength'] = report_df['r'].abs().apply(lambda r: 'Strong' if r > 0.3 else ('Moderate' if r > 0.1 else 'Weak'))
    
    def interpret(row):
        if row['is_significant'] == 'No':
            return "Not statistically significant."
        direction = 'positive' if row['r'] > 0 else 'negative'
        return f"{row['strength']} {direction} correlation."

    report_df['interpretation'] = report_df.apply(interpret, axis=1)
    
    # --- Key Insights ---
    insights = ["\n## Key Insights\n"]
    significant_results = report_df[report_df['is_significant'] == 'Yes']

    if not significant_results.empty:
        strongest = significant_results.loc[significant_results['r'].abs().idxmax()]
        insights.append(f"- **Strongest Relationship:** The strongest significant correlation was found for **{strongest['crypto']}** ({strongest['lag']}, {strongest['type']}) with r={strongest['r']:.3f} (p={strongest['p_value']:.3f}).")
    else:
        insights.append("- **Strongest Relationship:** No statistically significant (p < 0.05) correlations were found between sentiment and price changes.")

    # Same-Day vs. Next-Day analysis
    same_day_strongest = report_df[report_df['lag'] == 'Same-Day'].loc[report_df[report_df['lag'] == 'Same-Day']['r'].abs().idxmax()]
    next_day_strongest = report_df[report_df['lag'] == 'Next-Day'].loc[report_df[report_df['lag'] == 'Next-Day']['r'].abs().idxmax()]
    insights.append(f"- **Same-Day vs. Next-Day:** The strongest same-day correlation was {same_day_strongest['r']:.3f} (p={same_day_strongest['p_value']:.3f}) for {same_day_strongest['crypto']}, while the strongest next-day (predictive) correlation was {next_day_strongest['r']:.3f} (p={next_day_strongest['p_value']:.3f}) for {next_day_strongest['crypto']}.")

    # Predictive Power analysis
    predictive_results = report_df[(report_df['lag'] == 'Next-Day') & (report_df['is_significant'] == 'Yes')]
    if not predictive_results.empty:
        best_predictor = predictive_results.loc[predictive_results['r'].abs().idxmax()]
        insights.append(f"- **Predictive Power:** A significant next-day correlation was found for **{best_predictor['crypto']}** (r={best_predictor['r']:.3f}), suggesting sentiment may have some predictive power for this asset.")
    else:
        insights.append("- **Predictive Power:** No significant link was found between today's sentiment and tomorrow's price changes for any of the analyzed cryptocurrencies.")

    # --- Write Report ---
    with open(REPORT_PATH, 'w') as f:
        f.write("# Crypto Sentiment-Price Correlation Analysis\n\n")
        f.write("This report details the statistical correlation between daily social media sentiment and cryptocurrency price changes.\n\n")
        f.write(report_df.to_markdown(index=False))
        f.write("\n".join(insights))
    
    print("\n--- Correlation Report ---")
    print(report_df.to_string())
    print("\n".join(insights))
    print(f"\nFull report saved to {REPORT_PATH}")

def load_and_merge_data():
    """Loads and merges sentiment and price datasets."""
    if not os.path.exists(SENTIMENT_DATA_PATH) or not os.path.exists(PRICE_DATA_PATH):
        print("Error: Required data files not found. Please run previous scripts.")
        return None
        
    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH, parse_dates=['date'])
    price_df = pd.read_csv(PRICE_DATA_PATH, parse_dates=['date'])
    
    # --- FIX: Standardize crypto names to symbols ---
    # This mapping is based on the expected names from the sentiment aggregation script
    name_to_symbol_map = {
        'Bitcoin': 'BTC',
        'Ethereum': 'ETH',
        'Solana': 'SOL'
        # Add other mappings here if sentiment data contains more full names
    }
    # We will map the full names to symbols to match the price data
    sentiment_df['crypto'] = sentiment_df['crypto'].map(name_to_symbol_map).fillna(sentiment_df['crypto'])
    
    # Merge the two dataframes
    df = pd.merge(sentiment_df, price_df, on=['date', 'crypto'], how='inner')
    
    # Calculate next-day price change for lagged correlation
    df = df.sort_values(by=['crypto', 'date'])
    df['next_day_price_change_pct'] = df.groupby('crypto')['price_change_pct'].shift(-1)
    
    return df

def main():
    print("--- Correlation Analysis Script ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load and Merge
    merged_df = load_and_merge_data()

    if merged_df is None or merged_df.empty:
        print("Error: Merged dataframe is empty. The 'date' and 'crypto' columns may not have matching values, even after standardization.")
        return

    # 2. Per-Crypto Analysis
    all_results = []
    for crypto_symbol in merged_df['crypto'].unique():
        crypto_df, results = analyze_crypto_correlation(merged_df, crypto_symbol)
        if results:
            for r in results:
                r['crypto'] = crypto_symbol
            all_results.extend(results)

    # 3. Global Heatmap
    print("\nGenerating global correlation heatmap...")
    heatmap_cols = ['sentiment_mean', 'post_count', 'price_change_pct', 'next_day_price_change_pct', 'volume']
    corr_matrix = merged_df[heatmap_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Global Correlation Matrix of Key Variables', fontsize=16)
    plt.savefig(HEATMAP_PATH)
    plt.close()
    print(f"Heatmap saved to {HEATMAP_PATH}")

    # 4. Generate and Save Report
    if all_results:
        generate_report(all_results)
    else:
        print("Could not generate report as no correlations were calculated.")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
