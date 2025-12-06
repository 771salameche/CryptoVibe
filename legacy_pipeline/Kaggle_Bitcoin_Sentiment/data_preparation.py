import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================
# PARTIE 1: PRÉPARATION DES DONNÉES
# ============================================

def load_and_prepare_data(filepath, sample_size=100000):
    """
    Charge et prépare le dataset Kaggle
    
    Args:
        filepath: chemin vers le CSV
        sample_size: nombre de tweets à utiliser (pour réduire temps d'entraînement)
    """
    logging.info("Loading data...")
    df = pd.read_csv(filepath)
    
    logging.info(f"Original dataset size: {len(df):,}")
    logging.info(f"Columns: {list(df.columns)}")
    
    # Renommer si nécessaire
    if 'text' not in df.columns and 'tweet' in df.columns:
        df.rename(columns={'tweet': 'text'}, inplace=True)
    if 'tweet_text' in df.columns:
        df.rename(columns={'tweet_text': 'text'}, inplace=True)
    
    # Nettoyage
    df = df.dropna(subset=['text', 'Sentiment'])
    df = df[df['text'].str.len() > 10]  # Enlever tweets trop courts
    df = df[df['Sentiment'] != 'Neutral'] # Remove neutral class

    # Vérifier distribution des classes
    logging.info(f"Class distribution:\n{df['Sentiment'].value_counts()}")
    logging.info(f"Class distribution (%):\n{df['Sentiment'].value_counts(normalize=True) * 100}")
    
    # Échantillonnage stratifié
    if len(df) > sample_size:
        logging.info(f"Sampling {sample_size:,} tweets (stratified)...")
        # Calculate the fraction for each group
        frac = sample_size / len(df)
        df = df.groupby('Sentiment', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42)
        )
    
    # Mapper labels en format standard
    df['Sentiment'] = df['Sentiment'].str.lower()
    label_map = {
        'positive': 1,
        'negative': 0,
    }
    df['label'] = df['Sentiment'].map(label_map)
    
    logging.info(f"Final dataset size: {len(df):,}")
    
    return df

def create_splits(df, train_size=0.7, val_size=0.15):
    """Crée train/val/test splits"""
    
    # Train et temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        stratify=df['label'],
        random_state=42
    )
    
    # Val et test
    val_ratio = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df['label'],
        random_state=42
    )
    
    logging.info(f"Dataset splits:")
    logging.info(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logging.info(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    logging.info(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    DATA_FILEPATH = r'D:\CryptoVibe\CryptoVibe\data\Kaggle\mbsa.csv'
    OUTPUT_DIR = r'.'
    
    # Charger et préparer les données
    df = load_and_prepare_data(DATA_FILEPATH, sample_size=100000)
    
    # Créer les splits
    train_df, val_df, test_df = create_splits(df)
    
    # Sauvegarder les splits
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'validation_data.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False)
    
    logging.info(f"Data preparation complete. Files saved in: {os.path.abspath(OUTPUT_DIR)}")

