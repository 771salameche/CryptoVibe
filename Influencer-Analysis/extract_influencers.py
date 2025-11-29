"""
This script extracts and analyzes mentions of crypto influencers from social media posts.
It calculates mention frequency, associated sentiment, and co-occurrence with cryptocurrencies.
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Add project root to Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# VADER for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
INPUT_CSV_PATH = "data/Results/data_with_entities.csv"
OUTPUT_CSV_PATH = "data/Results/data_with_influencers.csv"
CHART_OUTPUT_PATH = "data/Visualizations/influencer_analysis.png"

def safe_literal_eval(val):
    """Safely evaluate a string representation of a list."""
    try:
        # Check if the value is a string that looks like a list
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return ast.literal_eval(val)
        # If it's already a list, return it
        elif isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        # Return an empty list if evaluation fails
        pass
    return []

def main():
    """Main script execution."""
    print("--- Influencer Mention Analysis Script ---")

    # 1. Load Data
    print(f"Loading dataset: {INPUT_CSV_PATH}...")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # Safely convert string representations of lists back to actual lists
    for col in ['influencers_mentioned', 'cryptos_mentioned']:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. It will be treated as empty.")
            df[col] = pd.Series([[] for _ in range(len(df))])
        else:
            df[col] = df[col].apply(safe_literal_eval)

    # 2. Calculate Sentiment
    print("Calculating sentiment scores for all posts...")
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    
    # 3. Filter for posts that mention at least one influencer
    df_influencers = df[df['influencers_mentioned'].apply(lambda x: len(x) > 0)].copy()
    
    if df_influencers.empty:
        print("No posts mentioning influencers were found. Exiting.")
        return

    # Explode the DataFrame to have one row per influencer mention
    df_exploded = df_influencers.explode('influencers_mentioned')

    # 4. Analysis
    print("Analyzing influencer data...")
    # Top 10 most mentioned influencers
    influencer_counts = df_exploded['influencers_mentioned'].value_counts().nlargest(10)
    
    # Average sentiment per influencer
    sentiment_per_influencer = df_exploded.groupby('influencers_mentioned')['sentiment'].mean().sort_values(ascending=False)
    
    # Crypto association per influencer
    crypto_associations = {}
    for influencer, group in df_exploded.groupby('influencers_mentioned'):
        all_cryptos = [crypto for sublist in group['cryptos_mentioned'] for crypto in sublist]
        crypto_counts = Counter(all_cryptos)
        total_mentions = sum(crypto_counts.values())
        if total_mentions > 0:
            crypto_associations[influencer] = {
                crypto: f"{(count / total_mentions) * 100:.1f}%"
                for crypto, count in crypto_counts.most_common(3)
            }

    # 5. Visualization
    print(f"Generating and saving analysis to: {CHART_OUTPUT_PATH}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Crypto Influencer Analysis', fontsize=20, y=0.95)

    # Bar chart: Top 10 Influencer Mentions
    sns.barplot(x=influencer_counts.values, y=influencer_counts.index, ax=ax1, palette='mako')
    ax1.set_title('Top 10 Most Mentioned Influencers', fontsize=16)
    ax1.set_xlabel('Mention Count', fontsize=12)
    ax1.set_ylabel('Influencer', fontsize=12)

    # Table: Influencer Sentiment & Crypto Associations
    ax2.axis('off')
    ax2.set_title('Sentiment and Top Crypto Associations', fontsize=16, pad=20)
    
    table_data = []
    for influencer in sentiment_per_influencer.index[:10]: # Limit table to top influencers
        sentiment = f"{sentiment_per_influencer.get(influencer, 0.0):.3f}"
        associations = crypto_associations.get(influencer, {'-': ''})
        assoc_str = ', '.join([f"{k}: {v}" for k, v in associations.items()])
        table_data.append([influencer, sentiment, assoc_str])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Influencer', 'Avg. Sentiment', 'Top Crypto Associations'],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.5]
    )
    # table.setAutoSetFontSize(False) # This method is deprecated in recent matplotlib versions
    table.scale(1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(CHART_OUTPUT_PATH), exist_ok=True)
    plt.savefig(CHART_OUTPUT_PATH)
    print("Chart saved successfully.")

    # 6. Output Examples & Save Data
    print("\n--- Example Posts Mentioning Influencers ---")
    print(df_influencers[['text', 'influencers_mentioned', 'sentiment']].head(10).to_string())

    print("\n--- Top Influencer-Crypto Pairings ---")
    influencer_crypto_pairs = df_exploded.explode('cryptos_mentioned').groupby(['influencers_mentioned', 'cryptos_mentioned']).size()
    print(influencer_crypto_pairs.nlargest(10).to_string())

    print(f"\nSaving data with sentiment to: {OUTPUT_CSV_PATH}...")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Save complete.")
    
    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()
