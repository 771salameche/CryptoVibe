"""
This script performs a lag analysis to find the optimal time delay between
social media sentiment and cryptocurrency price changes. It calculates correlations
at various daily lags and visualizes the results to identify potential
predictive relationships.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SENTIMENT_TIMESERIES_PATH = "data/Results/sentiment_timeseries.csv"
PRICE_DATA_PATH = "data/Bronze/price_data.csv"

# Output paths
OUTPUT_DIR = "data/Visualizations/Lag_Analysis"
REPORT_PATH = "Correlation-Analysis/lag_analysis_report.txt"

# Lags to test (in days)
LAGS_TO_TEST = range(0, 6)  # 0 to 5 days

# --- Main Analysis Functions ---
def load_data():
    """Loads and merges the aggregated sentiment and price datasets."""
    if not os.path.exists(SENTIMENT_TIMESERIES_PATH) or not os.path.exists(PRICE_DATA_PATH):
        logger.fatal(f"FATAL: Required data files not found. Please ensure '{SENTIMENT_TIMESERIES_PATH}' and '{PRICE_DATA_PATH}' exist.")
        return None
    
    sentiment_df = pd.read_csv(SENTIMENT_TIMESERIES_PATH, parse_dates=['date'])
    price_df = pd.read_csv(PRICE_DATA_PATH, parse_dates=['date'])
    
    # Standardize crypto names to symbols for the merge
    name_to_symbol_map = {'Bitcoin': 'BTC', 'Ethereum': 'ETH', 'Solana': 'SOL'}
    sentiment_df['crypto'] = sentiment_df['crypto'].map(name_to_symbol_map).fillna(sentiment_df['crypto'])
    
    # Merge the two dataframes on date and crypto
    df = pd.merge(sentiment_df, price_df, on=['date', 'crypto'], how='inner')
    
    # Ensure we are only working with the main cryptos
    main_cryptos = ['BTC', 'ETH', 'SOL']
    df = df[df['crypto'].isin(main_cryptos)].copy()
    
    logger.info(f"Successfully loaded and merged sentiment and price time-series data. Shape: {df.shape}")
    return df

def perform_lag_analysis(df):
    """
    Calculates same-day and lagged correlations for each crypto.
    Returns a dataframe with all results.
    """
    all_results = []

    for crypto_symbol in df['crypto'].unique():
        logger.info(f"\n--- Analyzing Lags for: {crypto_symbol} ---")
        crypto_df = df[df['crypto'] == crypto_symbol].sort_values('date').reset_index(drop=True)
        
        # --- DEBUGGING STEP ---
        logger.debug(f"Data head for {crypto_symbol}:")
        logger.debug(crypto_df[['date', 'sentiment_mean', 'price_change_pct']].head().to_string())


        for lag in LAGS_TO_TEST:
            # Shift the price change column to test sentiment(t) vs price(t+lag)
            shifted_price = crypto_df['price_change_pct'].shift(-lag)
            
            # Combine into a temporary dataframe and drop NaNs for accurate correlation
            temp_df = pd.DataFrame({
                'sentiment': crypto_df['sentiment_mean'],
                'price_change_shifted': shifted_price
            }).dropna()
            
            logger.debug(f"  Lag {lag}: Found {len(temp_df)} valid data points after dropping NaNs.")

            if len(temp_df) < 2:
                logger.debug(f"  Lag {lag}: Skipping analysis due to insufficient data points ({len(temp_df)} < 2).")
                continue

            # Calculate correlations
            p_r, p_p = pearsonr(temp_df['sentiment'], temp_df['price_change_shifted'])
            s_r, s_p = spearmanr(temp_df['sentiment'], temp_df['price_change_shifted'])

            all_results.append({'crypto': crypto_symbol, 'lag_days': lag, 'pearson_r': p_r, 'pearson_p': p_p, 'spearman_r': s_r, 'spearman_p': s_p})
            
    return pd.DataFrame(all_results)

def generate_visualizations(results_df, full_df):
    """Generates and saves all required plots."""
    print("Generating visualizations...")

    # 1. Lag vs. Correlation Line Chart
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=results_df, x='lag_days', y='pearson_r', hue='crypto', marker='o')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title('Sentiment vs. Price Correlation by Lag Time', fontsize=16)
    plt.xlabel('Lag (Days) [Sentiment leads Price]')
    plt.ylabel('Pearson Correlation Coefficient (r)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Cryptocurrency')
    plt.savefig(os.path.join(OUTPUT_DIR, "lag_correlation_chart.png"))
    plt.close()

    # 2. Cross-Correlation Plots
    for crypto_symbol in full_df['crypto'].unique():
        crypto_df = full_df[full_df['crypto'] == crypto_symbol].set_index('date').sort_index()
        sentiment = crypto_df['sentiment_mean'].dropna()
        price = crypto_df['price_change_pct'].dropna()
        
        # Align series and fill missing values
        aligned_sentiment, aligned_price = sentiment.align(price, join='inner', fill_value=0)
        
        plt.figure(figsize=(12, 7))
        plt.xcorr(aligned_sentiment, aligned_price, usevlines=True, maxlags=10, normed=True, lw=2)
        plt.grid(True)
        plt.axhline(0, color='grey', linestyle='--')
        plt.title(f'Cross-Correlation: Sentiment & Price Change for {crypto_symbol}', fontsize=16)
        plt.xlabel('Lag (Days)')
        plt.ylabel('Cross-correlation')
        plt.savefig(os.path.join(OUTPUT_DIR, f"cross_correlation_{crypto_symbol}.png"))
        plt.close()

    print(f"Visualizations saved to '{OUTPUT_DIR}' directory.")


def generate_report(results_df):
    """Identifies optimal lags and generates a text report."""
    print("\n--- Lag Analysis Report ---")
    
    report_lines = ["# Lag Analysis Report\n\n"]
    report_lines.append("This report identifies the optimal time delay (lag) where social media sentiment shows the strongest correlation with cryptocurrency price changes.\n")
    
    summary_data = []

    for crypto_symbol in results_df['crypto'].unique():
        crypto_results = results_df[results_df['crypto'] == crypto_symbol].copy()
        crypto_results['is_significant'] = crypto_results['pearson_p'] < 0.05
        
        significant_results = crypto_results[crypto_results['is_significant']]
        
        if not significant_results.empty:
            optimal = significant_results.loc[significant_results['pearson_r'].abs().idxmax()]
            optimal_lag = int(optimal['lag_days'])
            optimal_r = optimal['pearson_r']
            optimal_p = optimal['pearson_p']
            interpretation = f"Significant correlation found at a {optimal_lag}-day lag."
        else:
            # If no significant results, find the lag with the highest (but not significant) correlation
            optimal = crypto_results.loc[crypto_results['pearson_r'].abs().idxmax()]
            optimal_lag = int(optimal['lag_days'])
            optimal_r = optimal['pearson_r']
            optimal_p = optimal['pearson_p']
            interpretation = "No statistically significant lag found."

        summary_data.append({
            'Crypto': crypto_symbol,
            'Optimal Lag (Days)': optimal_lag,
            'Pearson r': f"{optimal_r:.4f}",
            'P-value': f"{optimal_p:.4f}",
            'Significant?': "Yes" if not significant_results.empty else "No",
            'Interpretation': interpretation
        })

    summary_df = pd.DataFrame(summary_data)
    
    report_lines.append("\n## Summary of Findings\n")
    report_lines.append(summary_df.to_markdown(index=False))
    
    # --- Detailed Recommendations ---
    report_lines.append("\n\n## Recommendations & Insights\n")
    for item in summary_data:
        if item['Significant?'] == 'Yes':
            direction = "positive" if float(item['Pearson r']) > 0 else "negative"
            report_lines.append(f"- **{item['Crypto']}:** A statistically significant {direction} correlation exists when sentiment leads price by **{item['Optimal Lag (Days)']} day(s)**. This suggests that shifts in sentiment may be a useful predictor for this asset.")
        else:
            report_lines.append(f"- **{item['Crypto']}:** No significant predictive link was found between daily sentiment and future price changes within the tested lag period.")
    
    report_content = "\n".join(report_lines)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)
        
    print(report_content)
    print(f"\nFull report saved to {REPORT_PATH}")

def main():
    print("--- Lag Correlation Analysis Script ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    df = load_data()
    if df is None:
        return

    # 2. Perform Analysis
    results_df = perform_lag_analysis(df)
    if results_df.empty:
        print("Could not perform lag analysis. The resulting dataframe was empty.")
        return

    # 3. Generate Visualizations
    generate_visualizations(results_df, df)

    # 4. Generate Report
    generate_report(results_df)
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
