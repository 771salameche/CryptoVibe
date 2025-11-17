import pandas as pd
import logging
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
INPUT_FILE = r'D:\CryptoVibe\CryptoVibe\data\Processed\processed_validation_data.csv'
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Results'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'vader_sentiment_validation.csv')

# --- Téléchargement des ressources NLTK pour VADER ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("Téléchargement du lexique VADER de NLTK...")
    nltk.download('vader_lexicon')

def get_vader_sentiment(text, analyzer):
    """
    Applique l'analyseur VADER pour obtenir les scores de sentiment.
    """
    if not isinstance(text, str):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return analyzer.polarity_scores(text)

def assign_sentiment_label(compound_score, threshold=0.05):
    """
    Convertit le score composé VADER en une étiquette de sentiment discrète.
    """
    if compound_score >= threshold:
        return 'Positive'
    elif compound_score <= -threshold:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    """
    Script principal pour l'implémentation de la baseline VADER.
    """
    logging.info("Démarrage de l'implémentation de la baseline VADER...")

    # --- 1. Chargement des Données ---
    try:
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"Données chargées depuis {INPUT_FILE}, {len(df)} lignes.")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier non trouvé à {INPUT_FILE}. Arrêt du script.")
        return
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier: {e}. Arrêt du script.")
        return

    # --- 2. Initialisation de VADER ---
    analyzer = SentimentIntensityAnalyzer()
    logging.info("Analyseur VADER initialisé.")

    # --- 3. Application de VADER et génération des scores ---
    logging.info("Application de VADER pour générer les scores de sentiment...")
    df['vader_scores'] = df['processed_text'].apply(lambda x: get_vader_sentiment(x, analyzer))

    # Extraction des scores individuels
    df['vader_neg'] = df['vader_scores'].apply(lambda x: x['neg'])
    df['vader_neu'] = df['vader_scores'].apply(lambda x: x['neu'])
    df['vader_pos'] = df['vader_scores'].apply(lambda x: x['pos'])
    df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])

    # Assignation de l'étiquette de sentiment discrète
    df['vader_sentiment'] = df['vader_compound'].apply(assign_sentiment_label)
    logging.info("Scores et étiquettes de sentiment VADER générés.")

    # --- 4. Sauvegarde des Résultats ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Résultats VADER sauvegardés à {OUTPUT_FILE}")
    except IOError as e:
        logging.error(f"Erreur lors de la sauvegarde des résultats : {e}")

    logging.info("Implémentation de la baseline VADER terminée.")
    logging.warning("Note: Les métriques d'évaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix) n'ont pas été calculées car aucune colonne de 'ground truth' sentiment n'a été fournie dans le jeu de données.")

if __name__ == "__main__":
    main()
