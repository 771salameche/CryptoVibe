"""This script detects crypto-related events in social media posts using a keyword-based
taxonomy, analyzes their impact, and generates aggregated timelines and visualizations."""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import re

# Add project root to Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from our event keyword taxonomy
from Event_Detection.event_keywords import EVENT_TAXONOMY

# --- Configuration ---
INPUT_CSV_PATH = "data/Results/data_with_influencers.csv"
OUTPUT_CSV_PATH = "data/Results/data_with_events.csv"
TIMELINE_CSV_PATH = "data/Results/events_timeline.csv"
CHART_OUTPUT_PATH = "data/Visualizations/events_analysis.png"
CONFIDENCE_THRESHOLD = 0.5

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

def get_snippet(text, keyword, window=50):
    """Extracts a text snippet around a found keyword."""
    try:
        match = re.search(re.escape(keyword), text, re.IGNORECASE)
        if not match:
            return "Snippet not available."
        start, end = match.span()
        start = max(0, start - window)
        end = min(len(text), end + window)
        return f"...{text[start:end]}..."
    except Exception:
        return "Snippet not available."

# --- Core Detection Logic ---
def detect_events(row):
    """
    Detects events in a single post (DataFrame row).
    Handles multiple events and associates them with mentioned cryptos.
    """
    text = str(row['text'])
    cryptos_mentioned = row['cryptos_mentioned']
    
    if not cryptos_mentioned:
        return []  # Skip posts that don't mention a specific crypto

    detected_events = []
    text_lower = text.lower()
    
    # Sort taxonomy by priority to handle overlapping keywords correctly if needed
    sorted_taxonomy = sorted(EVENT_TAXONOMY.items(), key=lambda item: item[1]['priority'], reverse=True)

    for category_code, details in sorted_taxonomy:
        for keyword in details['keywords']:
            # Use regex search for patterns, otherwise simple string search
            is_regex = bool(re.search(r'[\(\)\|\?\+\*\\]', keyword))
            match_found = re.search(keyword, text_lower) if is_regex else keyword in text_lower
            
            if match_found:
                # Associate the event with all cryptos mentioned in the post
                for crypto in cryptos_mentioned:
                    confidence = details['priority'] / 10.0
                    if confidence >= CONFIDENCE_THRESHOLD:
                        event = {
                            'date': row['date'],
                            'event_type': details['name'],
                            'crypto': crypto,
                            'snippet': get_snippet(text, match_found.group(0) if is_regex else keyword),
                            'confidence': confidence,
                            'sentiment': row.get('sentiment', 0.0),
                            'engagement': row.get('engagement', 0)
                        }
                        detected_events.append(event)
                # We found a keyword for this category, so we can break and move to the next category
                # This prevents finding multiple keywords of the same type (e.g. 'hacked' and 'stolen')
                break 

    return detected_events


def main():
    print("--- Crypto Event Detection System ---")

    # 1. Load Data
    print(f"Loading dataset: {INPUT_CSV_PATH}...")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'. Please run the previous scripts first.")
        return
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # Clean and prepare columns
    for col in ['cryptos_mentioned', 'influencers_mentioned']:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)
        else:
            df[col] = pd.Series([[] for _ in range(len(df))])

    # 2. Detect Events in each Post
    print("Detecting events in all posts...")
    df['events_detected'] = df.apply(detect_events, axis=1)
    
    # Save the main dataframe with the new events column
    print(f"Saving data with detected events to: {OUTPUT_CSV_PATH}...")
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    # 3. Aggregate Events for Timeline and Analysis
    print("Aggregating detected events...")
    all_events_list = [event for sublist in df['events_detected'] for event in sublist]
    if not all_events_list:
        print("No events detected with confidence >= {CONFIDENCE_THRESHOLD}. Exiting analysis.")
        return
        
    events_df = pd.DataFrame(all_events_list)
    events_df['date'] = pd.to_datetime(events_df['date']).dt.date

    # 4. Event Aggregation (Timeline)
    print(f"Creating event timeline and saving to: {TIMELINE_CSV_PATH}...")
    events_df['importance'] = events_df['engagement'].fillna(0) + 1 # Use engagement for importance
    
    timeline = events_df.groupby([pd.Grouper(key='date'), 'event_type', 'crypto']).agg(
        mention_count=('event_type', 'size'),
        avg_sentiment=('sentiment', 'mean'),
        total_importance=('importance', 'sum')
    ).reset_index().sort_values(by=['date', 'total_importance'], ascending=[False, False])
    
    timeline.to_csv(TIMELINE_CSV_PATH, index=False)

    # 5. Analysis & Statistics
    print("\n--- Event Analysis & Statistics ---")
    
    # Top 10 most impactful events (from timeline)
    print("Top 10 Most Impactful Events (by date and engagement):")
    print(timeline.head(10).to_string())
    
    # Sentiment by event type
    sentiment_by_event = events_df.groupby('event_type')['sentiment'].mean().sort_values()
    print("\nAverage Sentiment by Event Type:")
    print(sentiment_by_event.to_string())
    
    # 6. Visualizations
    print(f"\nGenerating and saving analysis charts to: {CHART_OUTPUT_PATH}...")
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle('Crypto Event Analysis', fontsize=24, y=0.95)

    # Plot 1: Event Type Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    event_type_counts = events_df['event_type'].value_counts()
    sns.barplot(x=event_type_counts.values, y=event_type_counts.index, ax=ax1, palette='viridis')
    ax1.set_title('Event Type Distribution', fontsize=16)
    ax1.set_xlabel('Total Mentions', fontsize=12)

    # Plot 2: Sentiment by Event Type
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x=sentiment_by_event.values, y=sentiment_by_event.index, ax=ax2, palette='coolwarm')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_title('Average Sentiment by Event Type', fontsize=16)
    ax2.set_xlabel('Average Compound Sentiment', fontsize=12)

    # Plot 3: Events Timeline
    ax3 = fig.add_subplot(gs[1, :])
    timeline_plot_data = events_df.groupby([events_df['date'], 'event_type']).size().unstack(fill_value=0)
    timeline_plot_data.plot(kind='line', ax=ax3, colormap='tab20', marker='o', linestyle='--')
    ax3.set_title('Events Timeline Over Time', fontsize=16)
    ax3.set_ylabel('Number of Event Mentions', fontsize=12)
    ax3.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 4: Crypto x Event Type Heatmap
    ax4 = fig.add_subplot(gs[2, :])
    heatmap_data = pd.crosstab(events_df['crypto'], events_df['event_type'])
    # Filter for top 15 cryptos to keep heatmap readable
    top_cryptos = events_df['crypto'].value_counts().nlargest(15).index
    heatmap_data = heatmap_data.loc[heatmap_data.index.isin(top_cryptos)]
    sns.heatmap(heatmap_data, ax=ax4, cmap='YlGnBu', annot=True, fmt='d')
    ax4.set_title('Crypto vs. Event Type Frequency (Top 15 Cryptos)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(CHART_OUTPUT_PATH), exist_ok=True)
    plt.savefig(CHART_OUTPUT_PATH)
    print("Charts saved successfully.")
    
    # 7. Output Examples
    print("\n--- Top 10 Detected Event Examples ---")
    top_events_examples = events_df.sort_values(by='importance', ascending=False).head(10)
    print(top_events_examples[['date', 'event_type', 'crypto', 'snippet', 'confidence', 'sentiment']].to_string())

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()
