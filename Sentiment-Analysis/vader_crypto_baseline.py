import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
INPUT_FILES = [
    r'D:\CryptoVibe\CryptoVibe\data\Processed\processed_train_data.csv',
    r'D:\CryptoVibe\CryptoVibe\data\Processed\processed_validation_data.csv',
    r'D:\CryptoVibe\CryptoVibe\data\Processed\processed_test_data.csv'
]
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Results'
VIS_DIR = r'D:\CryptoVibe\CryptoVibe\data\Visualizations'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'vader_crypto_sentiment_analysis_full.csv')

# --- Lexique Crypto Personnalisé ---
CRYPTO_LEXICON = {
    "moon": 3.0, "bullish": 2.5, "pump": 2.0, "hodl": 2.0, "lambo": 2.0,
    "diamond hands": 2.5, "to the moon": 3.0, "buy the dip": 1.5,
    "dump": -3.0, "crash": -3.0, "scam": -2.5, "rekt": -2.5, "bearish": -2.0,
    "paper hands": -2.0, "fud": -1.5, "rug pull": -3.0
}

def analyze_sentiment(text, analyzer):
    """Applique VADER et retourne les scores."""
    if not isinstance(text, str):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return analyzer.polarity_scores(text)

def classify_sentiment(compound_score):
    """Classe le sentiment basé sur le score composé."""
    if compound_score >= 0.05:
        return 'POSITIVE'
    elif compound_score <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def plot_score_distribution(df, score_column, title, filename):
    """Crée et sauvegarde un histogramme de la distribution des scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[score_column], bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()
    logging.info(f"Visualisation sauvegardée : {filename}")

def plot_class_distribution(df, class_column, title, filename):
    """Crée et sauvegarde un diagramme circulaire de la distribution des classes."""
    class_counts = df[class_column].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#99ff99'])
    plt.title(title)
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()
    logging.info(f"Visualisation sauvegardée : {filename}")

def main():
    logging.info("Démarrage de l'analyse de sentiment VADER avec lexique crypto sur l'ensemble des données...")

    # --- 1. Chargement des Données ---
    try:
        df_list = [pd.read_csv(file) for file in INPUT_FILES]
        df = pd.concat(df_list, ignore_index=True)
        logging.info(f"Données complètes chargées, {len(df)} lignes au total.")
    except FileNotFoundError as e:
        logging.error(f"Erreur: Fichier non trouvé - {e}. Arrêt.")
        return
    except Exception as e:
        logging.error(f"Erreur lors du chargement des fichiers: {e}. Arrêt.")
        return

    # --- 2. Analyse avec VADER Baseline ---
    logging.info("Analyse avec VADER baseline...")
    base_analyzer = SentimentIntensityAnalyzer()
    df['base_scores'] = df['cleaned_text'].apply(lambda x: analyze_sentiment(x, base_analyzer))
    df['base_compound'] = df['base_scores'].apply(lambda x: x['compound'])
    df['base_sentiment'] = df['base_compound'].apply(classify_sentiment)

    # --- 3. Analyse avec VADER Augmenté ---
    logging.info("Analyse avec VADER augmenté par le lexique crypto...")
    crypto_analyzer = SentimentIntensityAnalyzer()
    crypto_analyzer.lexicon.update(CRYPTO_LEXICON)
    df['crypto_scores'] = df['cleaned_text'].apply(lambda x: analyze_sentiment(x, crypto_analyzer))
    df['crypto_compound'] = df['crypto_scores'].apply(lambda x: x['compound'])
    df['crypto_sentiment'] = df['crypto_compound'].apply(classify_sentiment)

    # --- 4. Visualisations ---
    os.makedirs(VIS_DIR, exist_ok=True)
    plot_score_distribution(df, 'base_compound', 'Distribution des Scores VADER Baseline (Full Dataset)', 'base_score_dist_full.png')
    plot_class_distribution(df, 'base_sentiment', 'Distribution des Classes VADER Baseline (Full Dataset)', 'base_class_dist_full.png')
    plot_score_distribution(df, 'crypto_compound', 'Distribution des Scores VADER Augmenté (Full Dataset)', 'crypto_score_dist_full.png')
    plot_class_distribution(df, 'crypto_sentiment', 'Distribution des Classes VADER Augmenté (Full Dataset)', 'crypto_class_dist_full.png')

    # --- 5. Affichage des Top Posts ---
    logging.info("\n--- Top 5 Posts les plus Positifs (VADER Augmenté) ---")
    for _, row in df.nlargest(5, 'crypto_compound').iterrows():
        logging.info(f"  Score: {row['crypto_compound']:.2f} | Texte: {row['cleaned_text']}")

    logging.info("\n--- Top 5 Posts les plus Négatifs (VADER Augmenté) ---")
    for _, row in df.nsmallest(5, 'crypto_compound').iterrows():
        logging.info(f"  Score: {row['crypto_compound']:.2f} | Texte: {row['cleaned_text']}")

    # --- 6. Comparaison des Lexiques ---
    logging.info("\n--- Comparaison Baseline vs. VADER Augmenté ---")
    comparison_df = df[df['base_sentiment'] != df['crypto_sentiment']]
    logging.info(f"{len(comparison_df)} posts ont changé de sentiment après augmentation du lexique.")
    for _, row in comparison_df.head(5).iterrows():
        logging.info(f"  Texte: {row['cleaned_text']}")
        logging.info(f"    Sentiment Baseline : {row['base_sentiment']} ({row['base_compound']:.2f})")
        logging.info(f"    Sentiment Crypto   : {row['crypto_sentiment']} ({row['crypto_compound']:.2f})\n")

    # --- 7. Sauvegarde des Résultats ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Analyse complète sauvegardée à {OUTPUT_FILE}")

    logging.warning("Aucune évaluation de performance (Accuracy, etc.) n'a été effectuée car il n'y a pas de labels manuels de 'ground truth'.")

if __name__ == "__main__":
    main()
