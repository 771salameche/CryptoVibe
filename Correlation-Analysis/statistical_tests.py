"""
This script performs advanced statistical tests to validate the correlation
findings between social media sentiment and cryptocurrency prices. It includes
hypothesis testing, Granger causality, bootstrapping for confidence intervals,
and Bonferroni correction for multiple comparisons.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import os
import warnings
import logging

# --- Configure Logging & Style ---
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8-whitegrid')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SENTIMENT_TIMESERIES_PATH = "data/Results/sentiment_timeseries.csv"
PRICE_DATA_PATH = "data/Bronze/price_data.csv"

# Output paths
OUTPUT_DIR = "data/Visualizations/Statistical_Tests"
REPORT_PATH = "Correlation-Analysis/statistical_tests_report.txt"

# Analysis Parameters
SIGNIFICANCE_LEVEL_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
GRANGER_MAX_LAG = 3

# --- Helper Functions ---
def load_and_prepare_data():
    """Loads, merges, and prepares the aggregated sentiment and price datasets."""
    if not os.path.exists(SENTIMENT_TIMESERIES_PATH) or not os.path.exists(PRICE_DATA_PATH):
        logger.fatal(f"Required data files not found. Ensure '{SENTIMENT_TIMESERIES_PATH}' and '{PRICE_DATA_PATH}' exist.")
        return None
    
    sentiment_df = pd.read_csv(SENTIMENT_TIMESERIES_PATH, parse_dates=['date'])
    price_df = pd.read_csv(PRICE_DATA_PATH, parse_dates=['date'])
    
    name_to_symbol_map = {'Bitcoin': 'BTC', 'Ethereum': 'ETH', 'Solana': 'SOL'}
    sentiment_df['crypto'] = sentiment_df['crypto'].map(name_to_symbol_map).fillna(sentiment_df['crypto'])
    
    df = pd.merge(sentiment_df, price_df, on=['date', 'crypto'], how='inner')
    
    main_cryptos = ['BTC', 'ETH', 'SOL']
    df = df[df['crypto'].isin(main_cryptos)].copy()
    
    logger.info(f"Successfully loaded and merged data. Shape: {df.shape}")
    return df

def bootstrap_ci(series1, series2, n_iterations=1000):
    """Calculates the 95% confidence interval for Pearson correlation using bootstrapping."""
    correlations = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(series1), len(series1), replace=True)
        sample1 = series1[indices]
        sample2 = series2[indices]
        
        if np.std(sample1) > 0 and np.std(sample2) > 0:
            correlations.append(pearsonr(sample1, sample2)[0])
            
    lower_bound = np.percentile(correlations, 2.5)
    upper_bound = np.percentile(correlations, 97.5)
    return lower_bound, upper_bound

def check_stationarity(series, series_name=""):
    """Performs the Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    p_value = result[1]
    if p_value > 0.05:
        logger.warning(f"Series '{series_name}' is likely non-stationary (p={p_value:.3f}). First-differencing will be applied for Granger causality.")
        return series.diff().dropna()
    else:
        logger.info(f"Series '{series_name}' is likely stationary (p={p_value:.3f}).")
        return series

def main():
    logger.info("--- Statistical Significance & Causality Testing ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    df = load_and_prepare_data()
    if df is None or df.empty:
        return

    # 2. Perform Analysis
    correlation_results = []
    granger_results = []

    for crypto in df['crypto'].unique():
        print(f"\n--- Analyzing: {crypto} ---")
        crypto_df = df[df['crypto'] == crypto].sort_values('date').dropna(subset=['sentiment_mean', 'price_change_pct'])

        if len(crypto_df) < 20: # Need more data for robust tests
            print(f"Skipping {crypto} due to insufficient data ({len(crypto_df)} points).")
            continue

        # --- Granger Causality ---
        print("Performing Granger Causality Test...")
        sentiment_series = check_stationarity(crypto_df['sentiment_mean'], f"{crypto} Sentiment")
        price_series = check_stationarity(crypto_df['price_change_pct'], f"{crypto} Price Change")
        granger_data = pd.DataFrame({'sentiment': sentiment_series, 'price': price_series}).dropna()
        
        gc_test = grangercausalitytests(granger_data[['price', 'sentiment']], maxlag=GRANGER_MAX_LAG, verbose=False)
        for lag in range(1, GRANGER_MAX_LAG + 1):
            p_value = gc_test[lag][0]['ssr_ftest'][1]
            granger_results.append({'crypto': crypto, 'lag_days': lag, 'p_value': p_value})

        # --- Correlation and Confidence Intervals ---
        sentiment = crypto_df['sentiment_mean'].values
        price_change = crypto_df['price_change_pct'].values
        
        p_r, p_p = pearsonr(sentiment, price_change)
        s_r, s_p = spearmanr(sentiment, price_change)
        r_squared = p_r**2
        ci_lower, ci_upper = bootstrap_ci(sentiment, price_change, n_iterations=BOOTSTRAP_ITERATIONS)
        
        correlation_results.append({
            'crypto': crypto, 'test': 'Pearson', 'r': p_r, 'p_value': p_p, 'r_squared': r_squared,
            'ci_95_lower': ci_lower, 'ci_95_upper': ci_upper
        })
        correlation_results.append({'crypto': crypto, 'test': 'Spearman', 'r': s_r, 'p_value': s_p})

    # 3. Process Results & Multiple Comparison Correction
    if not correlation_results:
        print("No correlation results were generated. Exiting.")
        return
        
    results_df = pd.DataFrame(correlation_results)
    num_tests = len(results_df)
    bonferroni_alpha = SIGNIFICANCE_LEVEL_ALPHA / num_tests
    
    results_df['is_significant'] = results_df['p_value'] < SIGNIFICANCE_LEVEL_ALPHA
    results_df['is_significant_bonferroni'] = results_df['p_value'] < bonferroni_alpha
    results_df['effect_size'] = np.abs(results_df['r']).apply(
        lambda r: 'Strong' if r >= 0.5 else ('Moderate' if r >= 0.3 else 'Weak')
    )
    
    # --- Generate Report ---
    print("\n--- Generating Report ---")
    with open(REPORT_PATH, 'w') as f:
        f.write("# Statistical Test Report: Sentiment vs. Price\n\n")
        f.write(f"Significance Level (alpha): {SIGNIFICANCE_LEVEL_ALPHA}\n")
        f.write(f"Bonferroni Corrected alpha: {bonferroni_alpha:.4f} (for {num_tests} tests)\n\n")

        f.write("## Correlation Analysis\n\n")
        f.write(results_df.to_markdown(index=False))

        f.write("\n\n## Granger Causality Test\n")
        f.write("Tests if past sentiment values help predict future price changes ('Sentiment Granger-causes Price').\n\n")
        granger_df = pd.DataFrame(granger_results)
        granger_df['significant'] = granger_df['p_value'] < SIGNIFICANCE_LEVEL_ALPHA
        f.write(granger_df.to_markdown(index=False))
        
        f.write("\n\n## Conclusions\n")
        for crypto in results_df['crypto'].unique():
            crypto_corr = results_df[(results_df['crypto'] == crypto) & (results_df['test'] == 'Pearson')]
            crypto_granger = granger_df[granger_df['crypto'] == crypto]
            if not crypto_corr.empty:
                if crypto_corr.iloc[0]['is_significant_bonferroni']:
                    f.write(f"- **{crypto}:** Found a **statistically significant** correlation after Bonferroni correction. The relationship is reliable.\n")
                elif crypto_corr.iloc[0]['is_significant']:
                    f.write(f"- **{crypto}:** Found a significant correlation, but it **did not survive Bonferroni correction**, suggesting it could be a chance finding.\n")
                else:
                    f.write(f"- **{crypto}:** No significant correlation found.\n")
            if not crypto_granger.empty and (crypto_granger['significant']).any():
                f.write(f"- **{crypto} (Granger):** Sentiment was found to significantly predict future price changes at a lag of {granger_df[granger_df['crypto']==crypto].loc[granger_df['significant'] == True]['lag_days'].tolist()} day(s).\n")

    print(f"Report saved to {REPORT_PATH}")

    # --- Generate Visualizations ---
    print("Generating visualizations...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    fig.suptitle('Statistical Significance Analysis', fontsize=20, y=0.95)

    # P-value plot
    sns.barplot(data=results_df, x='p_value', y='crypto', hue='test', ax=axes[0])
    axes[0].axvline(SIGNIFICANCE_LEVEL_ALPHA, color='r', linestyle='--', label=f'Alpha = {SIGNIFICANCE_LEVEL_ALPHA}')
    axes[0].axvline(bonferroni_alpha, color='purple', linestyle=':', label=f'Bonferroni Alpha = {bonferroni_alpha:.3f}')
    axes[0].set_title('P-value Comparison', fontsize=16)
    axes[0].legend()

    # Effect size plot
    sns.barplot(data=results_df[results_df['test'] == 'Pearson'], x='r_squared', y='crypto', ax=axes[1])
    axes[1].set_title('Effect Size (R-squared)', fontsize=16)
    axes[1].set_xlabel('Variance Explained (R²)')

    # Confidence Interval plot
    ci_df = results_df[results_df['test'] == 'Pearson'].copy()
    ci_df['ci_error'] = (ci_df['ci_95_upper'] - ci_df['ci_95_lower']) / 2
    axes[2].errorbar(x=ci_df['r'], y=ci_df['crypto'], xerr=ci_df['ci_error'], fmt='o', capsize=5)
    axes[2].axvline(0, color='r', linestyle='--')
    axes[2].set_title('95% Confidence Intervals for Pearson Correlation', fontsize=16)
    axes[2].set_xlabel('Pearson Correlation Coefficient (r)')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUTPUT_DIR, "significance_tests.png"))
    plt.close()

    print(f"Visualizations saved to {OUTPUT_DIR}")
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
