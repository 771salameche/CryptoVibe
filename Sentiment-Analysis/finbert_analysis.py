import pandas as pd
import logging
import os
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'transformer_sentiment_analysis.csv')
CHUNK_SIZE = 10000  # Pour les grands datasets

# --- Modèles à utiliser ---
MODELS = {
    "finbert": "ProsusAI/finbert",
    "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment"
}

def plot_class_distribution_transformer(df, model_prefix):
    """Crée et sauvegarde un diagramme circulaire pour la distribution des classes d'un modèle."""
    class_col = f'{model_prefix}_label'
    if class_col not in df.columns:
        logging.warning(f"Colonne {class_col} non trouvée pour la visualisation.")
        return
        
    class_counts = df[class_col].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution des Sentiments - {model_prefix}')
    filename = f'{model_prefix}_class_dist.png'
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()
    logging.info(f"Visualisation sauvegardée : {filename}")

def analyze_sentiment_transformer(df, model_name, model_prefix, batch_size):
    """
    Analyse le sentiment en utilisant un modèle Transformer de HuggingFace.
    """
    logging.info(f"Chargement du modèle : {model_name}")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device, truncation=True)
    
    texts = df['cleaned_text'].fillna('').tolist()
    results = []
    
    logging.info(f"Début de l'inférence avec batch_size={batch_size}...")
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Inférence {model_prefix}"):
        batch = texts[i:i+batch_size]
        try:
            predictions = sentiment_pipeline(batch)
            results.extend(predictions)
        except Exception as e:
            logging.error(f"Erreur sur le batch {i}-{i+batch_size}: {e}")
            # Ajoute des résultats vides pour maintenir la correspondance de taille
            results.extend([{'label': 'ERROR', 'score': 0.0}] * len(batch))

    end_time = time.time()
    processing_time = end_time - start_time
    posts_per_second = len(texts) / processing_time if processing_time > 0 else 0
    logging.info(f"Inférence terminée pour {model_prefix}. Temps: {processing_time:.2f}s ({posts_per_second:.2f} posts/s)")

    # Ajout des résultats au DataFrame
    df[f'{model_prefix}_label'] = [res['label'].upper() for res in results]
    df[f'{model_prefix}_score'] = [res['score'] for res in results]
    
    # Pour FinBERT, les labels sont 'positive', 'negative', 'neutral'
    # Pour RoBERTa, ils sont 'LABEL_0' (négatif), 'LABEL_1' (neutre), 'LABEL_2' (positif)
    # Nous devons normaliser cela si nous voulons une comparaison directe
    if model_prefix == 'twitter_roberta':
        label_map = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
        df[f'{model_prefix}_label'] = df[f'{model_prefix}_label'].map(label_map)

    return df

def main():
    logging.info("Démarrage du script d'analyse de sentiment avec Transformers...")

    # --- 1. Chargement des Données ---
    try:
        df_list = [pd.read_csv(file) for file in INPUT_FILES]
        full_df = pd.concat(df_list, ignore_index=True)
        logging.info(f"Données complètes chargées, {len(full_df)} lignes.")
    except FileNotFoundError as e:
        logging.error(f"Erreur: Fichier non trouvé - {e}. Arrêt.")
        return

    # --- 2. Analyse avec chaque modèle ---
    batch_size = 16 if torch.cuda.is_available() else 8
    
    for prefix, model_name in MODELS.items():
        full_df = analyze_sentiment_transformer(full_df, model_name, prefix, batch_size)
        plot_class_distribution_transformer(full_df, prefix)
        
        # Afficher des exemples de prédictions
        logging.info(f"\n--- Exemples de prédictions pour {prefix} ---")
        sample_df = full_df.sample(5)
        for _, row in sample_df.iterrows():
            logging.info(f"  Texte: {row['cleaned_text']}")
            logging.info(f"  Prédiction {prefix}: {row[f'{prefix}_label']} (Score: {row[f'{prefix}_score']:.2f})\n")

    # --- 3. Comparaison (si les deux modèles ont été exécutés) ---
    if "finbert_label" in full_df.columns and "twitter_roberta_label" in full_df.columns:
        logging.info("\n--- Comparaison FinBERT vs. Twitter-RoBERTa ---")
        comparison_df = full_df[full_df['finbert_label'] != full_df['twitter_roberta_label']]
        logging.info(f"{len(comparison_df)} posts ont des prédictions différentes.")
        for _, row in comparison_df.head(5).iterrows():
            logging.info(f"  Texte: {row['cleaned_text']}")
            logging.info(f"    FinBERT         : {row['finbert_label']} (Score: {row['finbert_score']:.2f})")
            logging.info(f"    Twitter-RoBERTa : {row['twitter_roberta_label']} (Score: {row['twitter_roberta_score']:.2f})\n")

    # --- 4. Sauvegarde des Résultats ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Analyse complète sauvegardée à {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
