"""
This script fetches historical cryptocurrency price data from the Yahoo Finance API
using the yfinance library, calculates additional metrics, and saves the result
to a CSV file. It includes caching to avoid re-fetching data.
"""
import yfinance as yf # type: ignore
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
# Yahoo Finance tickers and their common symbols
CRYPTOS_TO_FETCH = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    'SOL-USD': 'SOL'
}
DAYS_TO_FETCH = 90

# Output and Caching
OUTPUT_DIR = "data/Bronze"
OUTPUT_FILENAME = "price_data.csv"
CACHE_DIR = ".cache/price_data_yf"
CACHE_EXPIRATION_HOURS = 24

# --- Caching Functions ---
def _get_cache_path(ticker):
    return os.path.join(CACHE_DIR, f"{ticker}_{DAYS_TO_FETCH}d.csv")

def _read_from_cache(path):
    if os.path.exists(path):
        file_mod_time = os.path.getmtime(path)
        if (time.time() - file_mod_time) / 3600 < CACHE_EXPIRATION_HOURS:
            return pd.read_csv(path, parse_dates=['Date'])
    return None

def _write_to_cache(path, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)

# --- Main Logic ---
def fetch_all_data():
    all_crypto_data = []
    
    print(f"Fetching data for {', '.join(CRYPTOS_TO_FETCH.values())} using Yahoo Finance...")
    
    pbar = tqdm(CRYPTOS_TO_FETCH.items(), desc="Fetching Cryptos")
    for ticker_name, symbol in pbar:
        pbar.set_postfix_str(f"{symbol}")
        
        cache_path = _get_cache_path(ticker_name)
        cached_df = _read_from_cache(cache_path)

        if cached_df is not None:
            print(f"Loading {symbol} data from cache...")
            df = cached_df
        else:
            print(f"Fetching {symbol} data from yfinance...")
            try:
                ticker = yf.Ticker(ticker_name)
                df = ticker.history(period=f"{DAYS_TO_FETCH}d", interval="1d")
                
                if df.empty:
                    print(f"Warning: No data returned for {symbol}.")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                _write_to_cache(cache_path, df)
                time.sleep(1) # Small delay to be polite to the API
            except Exception as e:
                print(f"Error: Could not fetch data for {symbol}: {e}")
                continue
        
        # Rename columns to match our schema
        df.rename(columns={
            'Date': 'date',
            'Open': 'price_open',
            'High': 'price_high',
            'Low': 'price_low',
            'Close': 'price_close',
            'Volume': 'volume'
        }, inplace=True)
        
        df['crypto'] = symbol
        
        # Calculate Additional Metrics
        df['price_change'] = df['price_close'] - df['price_open']
        df['price_change_pct'] = (df['price_change'] / df['price_open']) * 100
        df['intraday_volatility'] = ((df['price_high'] - df['price_low']) / df['price_open']) * 100
        
        all_crypto_data.append(df)

    if not all_crypto_data:
        print("Error: No data could be fetched for any cryptocurrency.")
        return None
        
    return pd.concat(all_crypto_data, ignore_index=True)

def main():
    print("--- Crypto Price Data Fetcher (Yahoo Finance) ---")
    final_df = fetch_all_data()

    if final_df is None or final_df.empty:
        print("Script finished with no data.")
        return

    # Ensure date column is just the date part
    final_df['date'] = pd.to_datetime(final_df['date']).dt.date

    # Reorder and select final columns
    # Note: market_cap is not available via yfinance history
    final_cols = [
        'date', 'crypto', 'price_open', 'price_high', 'price_low', 'price_close',
        'volume', 'price_change', 'price_change_pct', 'intraday_volatility'
    ]
    final_df = final_df[final_cols]
    final_df = final_df.sort_values(by=['crypto', 'date']).reset_index(drop=True)

    # --- Validation ---
    print("\n--- Data Validation ---")
    start_date = final_df['date'].min()
    end_date = final_df['date'].max()
    print(f"Date range covered: {start_date} to {end_date}")
    
    # Check for missing dates
    expected_dates = pd.date_range(start=start_date, end=end_date, freq='D').date
    for symbol in final_df['crypto'].unique():
        crypto_dates = final_df[final_df['crypto'] == symbol]['date'].unique()
        missing_dates = set(expected_dates) - set(crypto_dates)
        if missing_dates:
            print(f"Info: Found {len(missing_dates)} missing dates for {symbol} (likely weekends/holidays).")

    print("\nSample of final data:")
    print(final_df.head().to_string())

    # --- Save Output ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"\nSaving data to {output_path}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print("Save complete.")

    # --- Plotting for visual check ---
    print("Generating price charts for visual verification...")
    fig, axes = plt.subplots(len(CRYPTOS_TO_FETCH), 1, figsize=(12, 6 * len(CRYPTOS_TO_FETCH)), sharex=True)
    if len(CRYPTOS_TO_FETCH) == 1:
        axes = [axes]
        
    for ax, symbol in zip(axes, final_df['crypto'].unique()):
        crypto_df = final_df[final_df['crypto'] == symbol]
        ax.plot(crypto_df['date'], crypto_df['price_close'], label='Close Price')
        ax.set_title(f"{symbol} Closing Price (Last {DAYS_TO_FETCH} Days)")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        ax.legend()
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
