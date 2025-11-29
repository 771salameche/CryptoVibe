"""
Named Entity Recognition (NER) script to extract cryptocurrency-related entities
from social media posts using spaCy and a custom crypto dictionary.
"""

import sys
import os

# Add project root to Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import spacy
from spacy.tokens import Span
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import itertools
import os
import re

# Import from our custom dictionary
# Assuming the script is run from the project root directory
from ner_resources.crypto_dictionary import (
    CRYPTOCURRENCIES,
    EXCHANGES,
    INFLUENCERS,
    CRYPTO_REGEX,
    EXCHANGE_REGEX,
    INFLUENCER_REGEX,
    _CRYPTO_REVERSE_MAP,
    _EXCHANGE_REVERSE_MAP,
    _INFLUENCER_REVERSE_MAP
)

# --- Configuration ---
INPUT_CSV_PATH = r"D:\CryptoVibe\CryptoVibe\data\Silver\cleaned_crypto_data.csv"
OUTPUT_CSV_PATH = r"D:\CryptoVibe\CryptoVibe\data\Results\data_with_entities.csv"
CHART_OUTPUT_PATH = r"D:\CryptoVibe\CryptoVibe\data\Visualizations\crypto_mentions_chart.png"
SPACY_MODEL = "en_core_web_sm"

# --- Main Functions ---

def load_spacy_model(model_name):
    """Loads the spaCy model, providing guidance on installation if not found."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found.")
        print(f"Please run: python -m spacy download {model_name}")
        return None
    return nlp

def extract_custom_entities(text):
    """
    Uses pre-compiled regex from the crypto dictionary to find custom entities.
    Returns canonical names and a set of all matched text variations.
    """
    # Find all matches for each category
    crypto_matches = CRYPTO_REGEX.findall(text)
    exchange_matches = EXCHANGE_REGEX.findall(text)
    influencer_matches = INFLUENCER_REGEX.findall(text)

    # Get unique canonical names
    cryptos = sorted(list({_CRYPTO_REVERSE_MAP.get(m.lower()) for m in crypto_matches}))
    exchanges = sorted(list({_EXCHANGE_REVERSE_MAP.get(m.lower()) for m in exchange_matches}))
    influencers = sorted(list({_INFLUENCER_REVERSE_MAP.get(m.lower()) for m in influencer_matches}))

    # Combine all matched strings to prevent spaCy from re-labeling them
    all_matched_text = set(m.lower() for m in crypto_matches + exchange_matches + influencer_matches)

    return cryptos, exchanges, influencers, all_matched_text

def extract_all_entities(text, nlp):
    """
    Extracts both custom crypto entities and standard spaCy entities from text.

    Returns a dictionary with lists of found entities.
    """
    if not isinstance(text, str):
        return {
            "cryptos_mentioned": [],
            "exchanges_mentioned": [],
            "influencers_mentioned": [],
            "all_spacy_entities": []
        }

    # 1. Extract custom entities using our regex-based dictionary
    cryptos, exchanges, influencers, custom_entity_texts = extract_custom_entities(text)

    # 2. Process text with spaCy
    doc = nlp(text)

    # 3. Extract standard spaCy entities, avoiding overlaps with custom ones
    spacy_entities = []
    for ent in doc.ents:
        # Only add entity if its text was not already captured by our custom dictionaries
        if ent.text.lower() not in custom_entity_texts:
            spacy_entities.append({"text": ent.text, "label": ent.label_})

    return {
        "cryptos_mentioned": cryptos,
        "exchanges_mentioned": exchanges,
        "influencers_mentioned": influencers,
        "all_spacy_entities": spacy_entities
    }

def main():
    """Main script execution."""
    print("---" * 10 + " Crypto NER Extraction Script " + "---" * 10)

    # 1. Load resources
    print(f"Loading spaCy model: {SPACY_MODEL}...")
    nlp = load_spacy_model(SPACY_MODEL)
    if nlp is None:
        return

    print(f"Loading dataset: {INPUT_CSV_PATH}...")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # Ensure 'text' column exists and handle potential NaN values
    if 'text' not in df.columns:
        print("Error: 'text' column not found in the CSV file.")
        return
    df['text'] = df['cleaned_text'].astype(str)

    # 2. Apply NER extraction to the dataset
    tqdm.pandas(desc="Extracting Entities")
    # Using a lambda to pass the nlp model to the apply function
    entity_results = df['text'].progress_apply(lambda text: extract_all_entities(text, nlp))

    # Expand the dictionary results into separate columns
    df_entities = entity_results.apply(pd.Series)
    df = pd.concat([df, df_entities], axis=1)

    # 3. Save the processed data
    print(f"Saving data with entities to: {OUTPUT_CSV_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Save complete.")

    # 4. Statistics & Validation
    print("\n---" * 10 + " Extraction Statistics " + "---" * 10)
    
    # Top 10 Cryptos
    all_cryptos = [crypto for sublist in df['cryptos_mentioned'] for crypto in sublist]
    crypto_counts = Counter(all_cryptos)
    print("\nTop 10 Most Mentioned Cryptos:")
    for crypto, count in crypto_counts.most_common(10):
        print(f"- {crypto}: {count}")

    # Top 5 Exchanges
    all_exchanges = [exchange for sublist in df['exchanges_mentioned'] for exchange in sublist]
    exchange_counts = Counter(all_exchanges)
    print("\nTop 5 Most Mentioned Exchanges:")
    for exchange, count in exchange_counts.most_common(5):
        print(f"- {exchange}: {count}")

    # Crypto Co-occurrences
    co_occurrences = Counter()
    df['cryptos_mentioned'].apply(
        lambda cryptos: co_occurrences.update(itertools.combinations(sorted(cryptos), 2))
    )
    print("\nTop 10 Crypto Co-occurrences:")
    for pair, count in co_occurrences.most_common(10):
        print(f"- {pair[0]} & {pair[1]}: {count}")

    # Display example extractions
    print("\n--- Example Extractions for Validation ---")
    print(df[['text', 'cryptos_mentioned', 'exchanges_mentioned']].head(10).to_string())

    # 5. Visualization
    print(f"\nGenerating and saving chart to: {CHART_OUTPUT_PATH}...")
    top_10_cryptos = crypto_counts.most_common(10)
    if top_10_cryptos:
        crypto_df = pd.DataFrame(top_10_cryptos, columns=['Crypto', 'Mentions'])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mentions', y='Crypto', data=crypto_df, palette='viridis')
        plt.title('Top 10 Most Mentioned Cryptocurrencies', fontsize=16)
        plt.xlabel('Mention Count', fontsize=12)
        plt.ylabel('Cryptocurrency', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(CHART_OUTPUT_PATH), exist_ok=True)
        plt.savefig(CHART_OUTPUT_PATH)
        print("Chart saved successfully.")
    else:
        print("No crypto mentions found, skipping chart generation.")

    print("\n---" * 10 + " Script Finished " + "---" * 10)

if __name__ == "__main__":
    main()
