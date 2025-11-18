import pandas as pd
import logging
import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
LABELED_FILE_PROCESSED_CSV = r'D:\CryptoVibe\CryptoVibe\data\Processed\processed_validation_annot.csv'
VADER_RESULTS_FILE = r'D:\CryptoVibe\CryptoVibe\data\Results\vader_crypto_sentiment_analysis_full.csv'
TRANSFORMER_RESULTS_FILE = r'D:\CryptoVibe\CryptoVibe\data\Results\transformer_sentiment_analysis.csv'
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\Comparaison'
README_FILE = os.path.join(OUTPUT_DIR, 'README.md')

def load_and_merge_data():
    """Charge et fusionne les données étiquetées et les résultats des modèles."""
    try:
        # Charger les données étiquetées depuis le CSV prétraité
        df_labeled = pd.read_csv(LABELED_FILE_PROCESSED_CSV)
        logging.info(f"Données étiquetées chargées depuis {LABELED_FILE_PROCESSED_CSV}, {len(df_labeled)} lignes.")
        
        # Renommer les colonnes pour la fusion et la clarté
        df_labeled.rename(columns={'unified_id': 'id', 'classe': 'ground_truth'}, inplace=True)
        
        # Mapper les valeurs numériques de 'ground_truth' aux étiquettes de chaîne
        sentiment_mapping = {1: 'POSITIVE', 0: 'NEUTRAL', -1: 'NEGATIVE'}
        df_labeled['ground_truth'] = df_labeled['ground_truth'].map(sentiment_mapping)
        
        # Charger les résultats VADER
        df_vader = pd.read_csv(VADER_RESULTS_FILE)
        
        # Charger les résultats Transformer
        df_transformer = pd.read_csv(TRANSFORMER_RESULTS_FILE)
        
        # Garder uniquement les colonnes pertinentes pour la fusion
        df_vader = df_vader[['id', 'crypto_sentiment']]
        df_transformer = df_transformer[['id', 'finbert_label', 'cleaned_text']] # Garder cleaned_text ici
        
        # Fusionner les résultats
        df_merged = pd.merge(df_labeled, df_transformer, on='id', how='left') # Fusionner avec transformer en premier
        df_merged = pd.merge(df_merged, df_vader, on='id', how='left')
        
        # Renommer les colonnes pour plus de clarté (les prédictions VADER et FinBERT sont déjà en majuscules)
        df_merged.rename(columns={
            'crypto_sentiment': 'vader_pred',
            'finbert_label': 'finbert_pred'
        }, inplace=True)
        
        # Assurer que les labels sont en majuscules pour la comparaison
        df_merged['vader_pred'] = df_merged['vader_pred'].str.upper()
        df_merged['finbert_pred'] = df_merged['finbert_pred'].str.upper()
            
        logging.info(f"Données fusionnées, {len(df_merged)} lignes prêtes pour l'évaluation.")
        return df_merged.dropna(subset=['ground_truth', 'vader_pred', 'finbert_pred'])

    except FileNotFoundError as e:
        logging.error(f"Erreur de fichier non trouvé: {e}. Arrêt.")
        return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement/fusion des données: {e}. Arrêt.")
        return None

def evaluate_model(df, model_name):
    """Calcule les métriques d'évaluation pour un modèle."""
    y_true = df['ground_truth']
    y_pred = df[f'{model_name}_pred']
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    logging.info(f"\n--- Évaluation pour le modèle: {model_name} ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Rapport de classification:\n" + classification_report(y_true, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], yticklabels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité Terrain')
    cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Matrice de confusion sauvegardée à {cm_path}")
    
    return accuracy, report, cm_path

def create_readme_report(metrics):
    """Génère un rapport d'évaluation en Markdown."""
    with open(README_FILE, 'w') as f:
        f.write("# Rapport d'Évaluation des Modèles de Sentiment\n\n")
        f.write("Ce rapport compare les performances de VADER (avec lexique crypto) et FinBERT sur un ensemble de données de validation étiquetées manuellement.\n\n")
        
        f.write("## Tableau Comparatif des Métriques\n")
        f.write("| Modèle   | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-score (Weighted) |\n")
        f.write("|----------|----------|----------------------|-------------------|---------------------|\n")
        for model, data in metrics.items():
            f.write(f"| {model} | {data['accuracy']:.4f} | {data['report']['weighted avg']['precision']:.4f} | {data['report']['weighted avg']['recall']:.4f} | {data['report']['weighted avg']['f1-score']:.4f} |\n")
            
        f.write("\n## Matrices de Confusion\n")
        for model, data in metrics.items():
            f.write(f"### {model}\n")
            f.write(f"![Matrice de Confusion {model}](confusion_matrix_{model}.png)\n\n")
            
        f.write("## Analyse des Erreurs\n")
        # Vous pouvez ajouter ici une analyse plus détaillée des erreurs
        f.write("Une analyse plus approfondie des cas où les modèles se trompent peut être effectuée pour identifier les faiblesses (par exemple, sarcasme, contexte complexe).\n")

    logging.info(f"Rapport d'évaluation sauvegardé à {README_FILE}")

def main():
    df = load_and_merge_data()
    if df is None:
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_metrics = {}
    
    # Évaluation de VADER
    vader_accuracy, vader_report, _ = evaluate_model(df, 'vader')
    all_metrics['VADER'] = {'accuracy': vader_accuracy, 'report': vader_report}
    
    # Évaluation de FinBERT
    finbert_accuracy, finbert_report, _ = evaluate_model(df, 'finbert')
    all_metrics['FinBERT'] = {'accuracy': finbert_accuracy, 'report': finbert_report}
    
    # Création du rapport README
    create_readme_report(all_metrics)
    
    # Analyse des erreurs
    logging.info("\n--- Analyse des Erreurs ---")
    error_df = df[df['ground_truth'] != df['finbert_pred']]
    logging.info(f"FinBERT a fait {len(error_df)} erreurs. Exemples:")
    for _, row in error_df.head(5).iterrows():
        logging.info(f"  Texte: {row['cleaned_text']}")
        logging.info(f"    Vérité: {row['ground_truth']}, Prédiction FinBERT: {row['finbert_pred']}\n")

if __name__ == "__main__":
    main()
