import pandas as pd
import spacy
import logging
import os
from spacy.lang.en.stop_words import STOP_WORDS

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
INPUT_FILE = r'D:\CryptoVibe\CryptoVibe\data\Gold\validation_annot.csv'
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_validation_annot.csv')

# --- Stopwords personnalisés pour la crypto ---
CRYPTO_STOPWORDS = [
    'coin', 'crypto', 'bitcoin', 'ethereum', 'btc', 'eth', 'altcoin', 'blockchain',
    'buy', 'sell', 'hodl', 'pump', 'dump', 'moon', 'lambo', 'fud', 'fomo',
    'whale', 'bull', 'bear', 'market', 'price', 'trading', 'investing'
]

# --- Chargement du modèle spaCy ---
try:
    nlp = spacy.load('en_core_web_sm')
    # Ajout des stopwords personnalisés à la liste de spaCy
    for word in CRYPTO_STOPWORDS:
        nlp.vocab[word].is_stop = True
except IOError:
    logging.error("Modèle spaCy 'en_core_web_sm' non trouvé. Exécutez 'python -m spacy download en_core_web_sm'")
    nlp = None

def preprocess_text(text):
    """
    Applique le pipeline de prétraitement (tokenisation, lemmatisation, stopwords)
    à une seule chaîne de texte.
    """
    if not nlp or not isinstance(text, str):
        return ""

    # Création d'un document spaCy
    doc = nlp(text)

    # Lemmatisation et suppression des stopwords et de la ponctuation
    lemmatized_tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]

    # Rejoint les tokens en une seule chaîne
    return " ".join(lemmatized_tokens)

def main():
    """
    Script principal pour prétraiter le fichier validation_annot.csv.
    """
    if not nlp:
        return

    logging.info("Démarrage du script de prétraitement pour validation_annot.csv...")

    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        logging.info(f"Chargement du fichier: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)

        # Applique le prétraitement sur la colonne 'text_content'
        logging.info(f"Prétraitement de la colonne 'text_content'...")
        df['processed_text'] = df['text_content'].apply(preprocess_text)

        # Sauvegarde du DataFrame traité
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Fichier traité sauvegardé à {OUTPUT_FILE}")

    except FileNotFoundError:
        logging.error(f"Erreur: Fichier non trouvé à {INPUT_FILE}. Arrêt.")
    except Exception as e:
        logging.error(f"Une erreur est survenue lors du traitement du fichier {INPUT_FILE}: {e}")

    logging.info("Script de prétraitement terminé.")

if __name__ == "__main__":
    main()
