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
        df_transformer = df_transformer[['id', 'finbert_label', 'twitter_roberta_label', 'cleaned_text']]
        
        # Fusionner les résultats
        df_merged = pd.merge(df_labeled, df_transformer, on='id', how='left')
        df_merged = pd.merge(df_merged, df_vader, on='id', how='left')
        
        # Renommer les colonnes pour plus de clarté
        df_merged.rename(columns={
            'crypto_sentiment': 'vader_pred',
            'finbert_label': 'finbert_pred',
            'twitter_roberta_label': 'roberta_pred'
        }, inplace=True)
        
        # Assurer que les labels sont en majuscules pour la comparaison
        for col in ['ground_truth', 'vader_pred', 'finbert_pred', 'roberta_pred']:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].str.upper()
            
        logging.info(f"Données fusionnées, {len(df_merged)} lignes prêtes pour l'évaluation.")
        return df_merged.dropna(subset=['ground_truth', 'vader_pred', 'finbert_pred', 'roberta_pred'])

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
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    logging.info(f"\n--- Évaluation pour le modèle: {model_name} ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Rapport de classification:\n" + classification_report(y_true, y_pred, zero_division=0))
    
    # Matrice de confusion
    labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
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
        f.write("Ce rapport compare les performances de VADER, FinBERT et RoBERTa sur un ensemble de données de validation étiquetées manuellement.\n\n")
        
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
        f.write("Une analyse plus approfondie des cas où les modèles se trompent peut être effectuée pour identifier les faiblesses.\n")

    logging.info(f"Rapport d'évaluation sauvegardé à {README_FILE}")

def main():
    df = load_and_merge_data()
    if df is None:
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_metrics = {}
    models_to_evaluate = ['vader', 'finbert', 'roberta']
    
    for model_name in models_to_evaluate:
        accuracy, report, _ = evaluate_model(df, model_name)
        all_metrics[model_name.upper()] = {'accuracy': accuracy, 'report': report}
    
    # Création du rapport README
    create_readme_report(all_metrics)
    
    # Analyse des erreurs pour le meilleur modèle (par exemple, FinBERT)
    logging.info("\n--- Analyse des Erreurs (FinBERT) ---")
    error_df = df[df['ground_truth'] != df['finbert_pred']]
    logging.info(f"FinBERT a fait {len(error_df)} erreurs. Exemples:")
    for _, row in error_df.head(5).iterrows():
        logging.info(f"  Texte: {row['cleaned_text']}")
        logging.info(f"    Vérité: {row['ground_truth']}, Prédiction FinBERT: {row['finbert_pred']}\n")

if __name__ == "__main__":
    main()
