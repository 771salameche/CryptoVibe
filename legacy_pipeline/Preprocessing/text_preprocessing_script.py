import pandas as pd
import spacy
import logging
import os
from spacy.lang.en.stop_words import STOP_WORDS

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
INPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Gold'
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Processed'
DATA_FILES = {
    'train': os.path.join(INPUT_DIR, 'train_data.csv'),
    'validation': os.path.join(INPUT_DIR, 'validation_data.csv'),
    'test': os.path.join(INPUT_DIR, 'test_data.csv')
}
PROCESSED_FILES = {
    'train': os.path.join(OUTPUT_DIR, 'processed_train_data.csv'),
    'validation': os.path.join(OUTPUT_DIR, 'processed_validation_data.csv'),
    'test': os.path.join(OUTPUT_DIR, 'processed_test_data.csv')
}

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
    Script principal pour charger, prétraiter et sauvegarder les données.
    """
    if not nlp:
        return

    logging.info("Démarrage du script de prétraitement de texte...")

    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for key, file_path in DATA_FILES.items():
        try:
            logging.info(f"Chargement du fichier {key}: {file_path}")
            df = pd.read_csv(file_path)

            # Applique le prétraitement sur la colonne 'cleaned_text'
            logging.info(f"Prétraitement de la colonne 'cleaned_text' pour l'ensemble {key}...")
            df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

            # Sauvegarde du DataFrame traité
            output_path = PROCESSED_FILES[key]
            df.to_csv(output_path, index=False)
            logging.info(f"Fichier traité sauvegardé à {output_path}")

        except FileNotFoundError:
            logging.error(f"Erreur: Fichier non trouvé à {file_path}. Passage au suivant.")
            continue
        except Exception as e:
            logging.error(f"Une erreur est survenue lors du traitement du fichier {file_path}: {e}")
            continue

    logging.info("Script de prétraitement de texte terminé.")

if __name__ == "__main__":
    main()
