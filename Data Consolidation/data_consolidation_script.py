import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_and_standardize_crypto_data(filepath):
    """Charge et standardise reddit_crypto_data.csv"""
    df = pd.read_csv(filepath)
    
    # Standardisation des colonnes
    standardized = pd.DataFrame({
        'id': df['id'],
        'text': df['text'],
        'date': pd.to_datetime(df['date']),
        'source': 'reddit',
        'author': df['author'],
        'engagement': df['score'],
        'crypto_mentioned': np.nan  # Pas d'info crypto dans ce fichier
    })
    
    return standardized

def load_and_standardize_posts_comments(filepath):
    """Charge et standardise reddit_posts_comment.csv"""
    df = pd.read_csv(filepath)
    
    # Standardisation des colonnes
    standardized = pd.DataFrame({
        'id': df['unified_id'],
        'text': df['text_content'],
        'date': pd.to_datetime(df['created_date']),
        'source': 'reddit',
        'author': df['author'],
        'engagement': np.nan,  # Pas d'info engagement dans ce fichier
        'crypto_mentioned': df['crypto_mentions']
    })
    
    return standardized

def clean_data(df):
    """Nettoie le dataset unifié"""
    logging.info("Starting data cleaning process")
    initial_count = len(df)
    
    # Supprimer les valeurs nulles dans le texte
    df = df.dropna(subset=['text'])
    removed = initial_count - len(df)
    logging.info(f"Removed {removed} entries with null text")
    
    # Convertir le texte en string et supprimer les espaces
    df['text'] = df['text'].astype(str).str.strip()
    
    # Supprimer les posts vides ou < 10 caractères
    count_before = len(df)
    df = df[df['text'].str.len() >= 10]
    removed = count_before - len(df)
    logging.info(f"Removed {removed} short posts (< 10 characters)")
    
    # Supprimer les duplicates exacts (même texte)
    count_before = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    removed = count_before - len(df)
    logging.info(f"Removed {removed} exact duplicates")
    
    # Trier par date (plus récent en premier)
    df = df.sort_values('date', ascending=False)
    
    # Réinitialiser l'index
    df = df.reset_index(drop=True)
    
    logging.info(f"Data cleaning completed: {len(df)} posts remaining")
    return df

def display_statistics(df):
    """Affiche des statistiques descriptives"""
    logging.info("=" * 60)
    logging.info("DATASET STATISTICS")
    logging.info("=" * 60)
    
    # Nombre total de posts
    logging.info(f"Total number of posts: {len(df):,}")
    
    # Distribution par source
    logging.info("Distribution by source:")
    source_dist = df['source'].value_counts()
    for source, count in source_dist.items():
        percentage = (count / len(df)) * 100
        logging.info(f"  {source.capitalize()}: {count:,} ({percentage:.1f}%)")
    
    # Plage de dates
    logging.info("Date range:")
    min_date = df['date'].min()
    max_date = df['date'].max()
    logging.info(f"  Earliest date: {min_date}")
    logging.info(f"  Latest date: {max_date}")
    logging.info(f"  Period covered: {(max_date - min_date).days} days")
    
    # Statistiques sur le texte
    logging.info("Content statistics:")
    text_lengths = df['text'].str.len()
    logging.info(f"  Average length: {text_lengths.mean():.0f} characters")
    logging.info(f"  Median length: {text_lengths.median():.0f} characters")
    logging.info(f"  Shortest: {text_lengths.min()} characters")
    logging.info(f"  Longest: {text_lengths.max()} characters")
    
    # Statistiques sur l'engagement (si disponible)
    engagement_data = df['engagement'].dropna()
    if len(engagement_data) > 0:
        logging.info("Engagement statistics:")
        logging.info(f"  Posts with engagement data: {len(engagement_data):,}")
        logging.info(f"  Average score: {engagement_data.mean():.2f}")
        logging.info(f"  Median score: {engagement_data.median():.2f}")
        logging.info(f"  Max score: {engagement_data.max():.0f}")
    
    # Cryptos mentionnées (si disponible)
    crypto_data = df['crypto_mentioned'].dropna()
    if len(crypto_data) > 0:
        logging.info("Crypto mentions:")
        logging.info(f"  Posts with crypto mentions: {len(crypto_data):,}")
    
    logging.info("=" * 60)

def plot_posts_per_day(df):
    """Crée un graphique des posts par jour"""
    logging.info("Generating posts per day plot")
    
    # Extraire la date (sans l'heure)
    df['date_only'] = df['date'].dt.date
    
    # Compter les posts par jour
    posts_per_day = df.groupby('date_only').size()
    
    # Créer le graphique
    plt.figure(figsize=(14, 6))
    plt.plot(posts_per_day.index, posts_per_day.values, marker='o', 
             linestyle='-', linewidth=2, markersize=4, color='#1f77b4')
    
    plt.title('Distribution des Posts par Jour', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nombre de Posts', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_file = 'posts_per_day.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved: {output_file}")
    plt.show()

def main():
    """Fonction principale d'exécution"""
    logging.info("Starting Reddit data merge process")
    
    try:
        # 1. Charger et standardiser les fichiers
        logging.info("Loading CSV files")
        df1 = load_and_standardize_crypto_data(r'D:\CryptoVibe\CryptoVibe\data\Bronze\reddit_crypto_data.csv')
        logging.info(f"Loaded reddit_crypto_data.csv: {len(df1):,} entries")
        
        df2 = load_and_standardize_posts_comments(r'D:\CryptoVibe\CryptoVibe\data\Bronze\reddit_posts_comment.csv')
        logging.info(f"Loaded reddit_posts_comment.csv: {len(df2):,} entries")
        
        # 2. Fusionner les datasets
        logging.info("Merging datasets")
        df_combined = pd.concat([df1, df2], ignore_index=True)
        logging.info(f"Total after merge: {len(df_combined):,} entries")
        
        # 3. Nettoyer les données
        df_clean = clean_data(df_combined)
        
        # 4. Afficher les statistiques
        display_statistics(df_clean)
        
        # 5. Créer le graphique
        plot_posts_per_day(df_clean)
        
        # 6. Sauvegarder le résultat
        output_file = r'D:\CryptoVibe\CryptoVibe\data\Bronze\consolidated_data.csv'
        df_clean.to_csv(output_file, index=False)
        logging.info(f"Consolidated dataset saved: {output_file}")
        logging.info(f"Columns: {', '.join(df_clean.columns)}")
        logging.info(f"Rows: {len(df_clean):,}")
        
        logging.info("Process completed successfully")
        
        return df_clean
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Ensure CSV files are in the same directory as this script")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    df_final = main()