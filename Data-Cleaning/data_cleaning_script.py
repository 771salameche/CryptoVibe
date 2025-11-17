import pandas as pd
import re
import html
import logging
from langdetect import detect, LangDetectException

# --- Configuration du Logging ---
# Utilise logging au lieu de print pour les messages de statut
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Constantes ---
# Liste des emojis à conserver (regex)
CRYPTO_EMOJIS_TO_KEEP = '🚀💎🌙📈📉💰🐕'

# Regex compilé pour les caractères autorisés
# Garde a-z, 0-9, espaces, et nos emojis crypto
# Tout le reste sera supprimé
ALLOWED_CHARS_PATTERN = re.compile(r'[^a-z0-9\s' + re.escape(CRYPTO_EMOJIS_TO_KEEP) + r']')

# --- Fonctions de Nettoyage et Filtrage ---

def clean_text(text):
    """
    Applique une série de règles de nettoyage à une seule chaîne de texte.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convertir les entités HTML (ex: &amp; -> &)
    text = html.unescape(text)

    # 2. Supprimer les URLs
    text = re.sub(r'(https|http)://\S+|www\.\S+', '', text)

    # 3. Supprimer les mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # 4. Garder le texte des hashtags mais supprimer le '#' (ex: #bitcoin -> bitcoin)
    text = re.sub(r'#(\w+)', r'\1', text)

    # 5. Convertir tout le texte en minuscules pour la normalisation
    text = text.lower()

    # 6. Supprimer tous les caractères non autorisés (y compris les autres emojis)
    #    en utilisant le pattern compilé.
    text = ALLOWED_CHARS_PATTERN.sub('', text)

    # 7. Normaliser les espaces (remplacer les espaces multiples par un seul)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def detect_language(text):
    """
    Détecte la langue du texte. Gère les erreurs de langdetect.
    """
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown' # Marque comme inconnu si langdetect échoue

def calculate_non_alnum_ratio(text):
    """
    Calcule le ratio de caractères non-alphanumériques dans le texte original.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0
    # Compte tout ce qui n'est PAS une lettre ou un chiffre
    non_alnum_count = len(re.findall(r'[^a-zA-Z0-9]', text))
    return non_alnum_count / len(text)

def mark_spam(df, text_column, threshold=5):
    """
    Crée une colonne 'is_spam' si le texte apparaît plus de 'threshold' fois.
    """
    text_counts = df[text_column].value_counts()
    df['text_freq'] = df[text_column].map(text_counts)
    
    # Est spam si la fréquence dépasse le seuil OU si le texte est vide après nettoyage
    df['is_spam'] = (df['text_freq'] > threshold) | (df[text_column] == '')
    return df

# --- Script Principal ---

def main():
    """
    Script principal pour charger, nettoyer et filtrer les données.
    """
    logging.info("Démarrage du script de nettoyage...")

    # --- 1. Chargement des Données ---
    # Charge le fichier consolidated_data.csv
    try:
        df = pd.read_csv(r'D:\CryptoVibe\CryptoVibe\data\Bronze\consolidated_data.csv')
        df_original = df.copy() # Garder une copie pour la comparaison
        logging.info(f"Données brutes chargées : {len(df)} lignes.")
    except FileNotFoundError:
        logging.error("Erreur: consolidated_data.csv non trouvé. Assurez-vous que le chemin est correct.")
        return # Arrête le script si le fichier n'est pas trouvé
    except Exception as e:
        logging.error(f"Erreur lors du chargement de consolidated_data.csv: {e}")
        return

    # --- 2. Application des Filtres de Qualité (Pré-nettoyage) ---

    # Filtre 1: Langue (doit être 'en')
    logging.info("Application du filtre de langue...")
    df['lang'] = df['text'].apply(detect_language)
    df = df[df['lang'] == 'en'].copy()
    logging.info(f"Lignes restantes après filtre de langue : {len(df)}")

    # Filtre 2: Ratio non-alphanumérique (doit être <= 30%)
    logging.info("Application du filtre non-alphanumérique...")
    df['non_alnum_ratio'] = df['text'].apply(calculate_non_alnum_ratio)
    df = df[df['non_alnum_ratio'] <= 0.30].copy()
    logging.info(f"Lignes restantes après filtre non-alnum : {len(df)}")

    # Filtre 3: Texte tout en majuscules (sauf acronymes - règle simplifiée)
    # Nous filtrons les textes qui sont *entièrement* en majuscules
    logging.info("Application du filtre majuscules...")
    df['is_all_caps'] = df['text'].apply(lambda x: isinstance(x, str) and x.isupper())
    df = df[df['is_all_caps'] == False].copy()
    logging.info(f"Lignes restantes après filtre majuscules : {len(df)}")

    # --- 3. Nettoyage du Texte ---
    logging.info("Nettoyage du texte...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # --- 4. Application des Filtres de Qualité (Post-nettoyage) ---

    # Filtre 4: Spam (texte dupliqué > 5 fois)
    logging.info("Application du filtre spam...")
    df = mark_spam(df, 'cleaned_text', threshold=5)
    df = df[df['is_spam'] == False].copy()
    logging.info(f"Lignes restantes après filtre spam : {len(df)}")
    
    # Filtre 5: Longueur du texte (entre 10 et 500 caractères)
    logging.info("Application du filtre de longueur...")
    df['cleaned_text_len'] = df['cleaned_text'].str.len()
    df = df[
        (df['cleaned_text_len'] >= 10) &
        (df['cleaned_text_len'] <= 500)
    ].copy()
    logging.info(f"Nettoyage terminé. Lignes finales : {len(df)}")

    # --- 5. Affichage des Exemples Avant/Après ---
    
    logging.info("\n--- Exemples de Nettoyage Avant/Après ---")
    
    # On utilise les IDs des lignes qui ont survécu au nettoyage
    final_ids = df['id'].values
    
    # On sélectionne ces IDs dans le DataFrame *original*
    examples_df = df_original[df_original['id'].isin(final_ids)].copy()
    
    # On applique le 'cleaned_text' correspondant
    examples_df = examples_df.merge(
        df[['id', 'cleaned_text']], 
        on='id', 
        how='left'
    )

    for index, row in examples_df.iterrows():
        logging.info(f"\n[Exemple ID: {row['id']}]")
        logging.info(f"  AVANT : {row['text']}")
        logging.info(f"  APRÈS : {row['cleaned_text']}")

    # --- 6. Sauvegarde ---
    output_path = r'D:\CryptoVibe\CryptoVibe\data\Silver\cleaned_crypto_data.csv'
    final_df = df[['id', 'cleaned_text', 'date', 'source', 'author', 'engagement']]
    final_df.to_csv(output_path, index=False)
    logging.info(f"Fichier nettoyé sauvegardé : {output_path}")


if __name__ == "__main__":
    main()