import pandas as pd
import logging
import os
from sklearn.model_selection import StratifiedShuffleSplit

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
INPUT_FILE = r'D:\CryptoVibe\CryptoVibe\data\Silver\cleaned_crypto_data.csv'
OUTPUT_DIR = r'D:\CryptoVibe\CryptoVibe\data\Gold'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_data.csv')
VALIDATION_FILE = os.path.join(OUTPUT_DIR, 'validation_data.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test_data.csv')

# Ratios pour le split
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
# Le ratio de test est implicitement 1 - TRAIN_RATIO - VALIDATION_RATIO

def split_data(df, label_column=None, n_splits=1, test_size=0.2):
    """
    Applique un 'Stratified Shuffle Split' sur le DataFrame.
    Retourne les index pour les ensembles d'entraînement et de test.
    """
    if label_column and df[label_column].nunique() > 1:
        # Utilise la stratification si une colonne de label est fournie
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        # 'y' pour la stratification doit être la colonne de labels
        return next(split.split(df, df[label_column]))
    else:
        # Si pas de label ou si le label est unique, effectue un split simple
        # Note: Ce n'est pas un 'shuffle split' mais un simple découpage d'index
        train_size = int(len(df) * (1 - test_size))
        return df.index[:train_size], df.index[train_size:]

def main():
    """
    Script principal pour charger, splitter et sauvegarder les données.
    """
    logging.info("Démarrage du script de splitting des données...")

    # --- 1. Chargement des Données ---
    try:
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"Données chargées depuis {INPUT_FILE}, {len(df)} lignes.")
    except FileNotFoundError:
        logging.error(f"Erreur: Fichier non trouvé à {INPUT_FILE}. Arrêt du script.")
        return

    # --- 2. Préparation des Données ---
    # Convertir la colonne 'date' en datetime et la trier
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    logging.info("Données triées par date.")

    # --- 3. Définition des Fenêtres Temporelles ---
    # Calcule les jours uniques et les points de split
    unique_days = df['date'].dt.date.unique()
    
    if len(unique_days) < 3:
        logging.error("Pas assez de jours uniques dans les données pour un split temporel significatif. Arrêt.")
        return

    train_end_day_index = int(len(unique_days) * TRAIN_RATIO)
    validation_end_day_index = train_end_day_index + int(len(unique_days) * VALIDATION_RATIO)

    train_end_date = unique_days[train_end_day_index]
    validation_end_date = unique_days[validation_end_day_index]

    logging.info(f"Fin de la période d'entraînement : {train_end_date}")
    logging.info(f"Fin de la période de validation : {validation_end_date}")

    # --- 4. Split Temporel ---
    # Crée les masques pour chaque ensemble de données
    train_mask = df['date'].dt.date <= train_end_date
    validation_mask = (df['date'].dt.date > train_end_date) & (df['date'].dt.date <= validation_end_date)
    test_mask = df['date'].dt.date > validation_end_date

    train_df = df[train_mask]
    validation_df = df[validation_mask]
    test_df = df[test_mask]

    logging.info(f"Taille de l'ensemble d'entraînement : {len(train_df)} lignes")
    logging.info(f"Taille de l'ensemble de validation : {len(validation_df)} lignes")
    logging.info(f"Taille de l'ensemble de test : {len(test_df)} lignes")

    # --- 5. Stratification (Optionnelle, si une colonne de label existe) ---
    # Si vous aviez une colonne 'sentiment', vous pourriez l'utiliser ici.
    # Exemple: train_indices, _ = split_data(train_df, label_column='sentiment', test_size=0.0)
    # Pour l'instant, nous gardons les blocs temporels complets.

    # --- 6. Sauvegarde des Fichiers ---
    try:
        # Crée le répertoire de sortie s'il n'existe pas
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        train_df.to_csv(TRAIN_FILE, index=False)
        logging.info(f"Ensemble d'entraînement sauvegardé à {TRAIN_FILE}")
        
        validation_df.to_csv(VALIDATION_FILE, index=False)
        logging.info(f"Ensemble de validation sauvegardé à {VALIDATION_FILE}")
        
        test_df.to_csv(TEST_FILE, index=False)
        logging.info(f"Ensemble de test sauvegardé à {TEST_FILE}")
        
    except IOError as e:
        logging.error(f"Erreur lors de la sauvegarde des fichiers : {e}")

    logging.info("Script de splitting terminé.")

if __name__ == "__main__":
    main()
