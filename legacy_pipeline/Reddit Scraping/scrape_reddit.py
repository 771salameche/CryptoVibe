import praw
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm
import time

# --- Configuration du Logging ---
# Configure le logging pour enregistrer les infos et les erreurs dans un fichier
logging.basicConfig(
    filename='reddit_scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# --- Chargement des variables d'environnement ---
# Charge les clés d'API depuis le fichier .env
try:
    load_dotenv()
    CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.environ.get("REDDIT_USER_AGENT")

    if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
        logging.error("Variables d'environnement (CLIENT_ID, CLIENT_SECRET, USER_AGENT) non trouvées. Veuillez créer un fichier .env.")
        exit()

except ImportError:
    logging.error("Le package python-dotenv n'est pas installé. Veuillez l'installer avec 'pip install python-dotenv'")
    exit()
except Exception as e:
    logging.error(f"Erreur lors du chargement des variables d'environnement : {e}")
    exit()


def connect_to_reddit():
    """
    Initialise et retourne une instance PRAW authentifiée.
    PRAW gère automatiquement les rate limits de Reddit.
    """
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.read_only = True # Important pour un scraper
        logging.info("Connexion à l'API Reddit réussie (mode lecture seule).")
        return reddit
    except Exception as e:
        logging.critical(f"Impossible de se connecter à Reddit: {e}")
        return None

def scrape_subreddit(reddit_instance, subreddit_name, limit, min_score, top_comments_n=5):
    """
    Scrape les posts et les top commentaires d'un subreddit donné.
    
    Args:
        reddit_instance (praw.Reddit): L'instance PRAW connectée.
        subreddit_name (str): Le nom du subreddit (ex: 'bitcoin').
        limit (int): Nombre maximum de posts à récupérer.
        min_score (int): Score minimum requis pour un post (bonus).
        top_comments_n (int): Nombre de top commentaires à récupérer par post.

    Returns:
        list: Une liste de dictionnaires, chaque dict représentant un post ou un commentaire.
    """
    data = []
    subreddit = reddit_instance.subreddit(subreddit_name)
    logging.info(f"Début du scraping de r/{subreddit_name} (limite={limit}, score min={min_score})...")
    
    # Utilise .top('month') pour obtenir les posts des ~30 derniers jours
    # Utilise tqdm pour la barre de progression
    try:
        post_stream = subreddit.top(time_filter="month", limit=limit)
        
        for submission in tqdm(post_stream, desc=f"Scraping r/{subreddit_name}", total=limit):
            try:
                # --- Filtre Bonus ---
                if submission.score < min_score:
                    continue
                
                # Formatage de la date en ISO 8601
                post_date = datetime.utcfromtimestamp(submission.created_utc).isoformat()
                
                # Combinaison du titre et du selftext pour la colonne 'text'
                post_text = f"{submission.title} {submission.selftext}"
                
                # Gestion des auteurs supprimés
                author_name = submission.author.name if submission.author else "[deleted]"
                
                # Ajout des données du post
                data.append({
                    "id": f"t3_{submission.id}", # 't3_' est le "thing type" pour les submissions
                    "text": post_text.strip(),
                    "date": post_date,
                    "source": f"r/{subreddit_name}",
                    "author": author_name,
                    "score": submission.score,
                    "type": "post" # Ajout d'un type pour distinguer post/commentaire
                })

                # --- Récupération des Top Commentaires ---
                # .replace_more(limit=0) supprime les objets "MoreComments"
                submission.comments.replace_more(limit=0)
                
                # Trie les commentaires par score (top) et prend les N premiers
                top_comments = sorted(submission.comments, key=lambda c: c.score, reverse=True)
                
                for comment in top_comments[:top_comments_n]:
                    comment_date = datetime.utcfromtimestamp(comment.created_utc).isoformat()
                    comment_author = comment.author.name if comment.author else "[deleted]"
                    
                    data.append({
                        "id": f"t1_{comment.id}", # 't1_' est le "thing type" pour les commentaires
                        "text": comment.body.strip(),
                        "date": comment_date,
                        "source": f"r/{subreddit_name}",
                        "author": comment_author,
                        "score": comment.score,
                        "type": "comment"
                    })

            except praw.exceptions.APIException as e:
                logging.warning(f"Erreur API PRAW sur le post {submission.id}: {e}")
                time.sleep(5) # Pause en cas d'erreur API
            except Exception as e:
                logging.error(f"Erreur inconnue lors du traitement du post {submission.id}: {e}")

    except Exception as e:
        logging.error(f"Erreur majeure lors du scraping de r/{subreddit_name}: {e}")
        
    logging.info(f"Scraping de r/{subreddit_name} terminé. {len(data)} items collectés.")
    return data

def save_to_csv(data, filename="reddit_crypto_data.csv"):
    """
    Sauvegarde la liste de données dans un fichier CSV en utilisant pandas.
    """
    if not data:
        logging.warning("Aucune donnée à sauvegarder.")
        return

    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        logging.info(f"Données sauvegardées avec succès dans {filename} ({len(df)} lignes).")
    except Exception as e:
        logging.error(f"Échec de la sauvegarde CSV: {e}")

def main():
    """
    Fonction principale pour orchestrer le scraping.
    """
    logging.info("Lancement du script de scraping Reddit...")
    
    # --- Paramètres ---
    SUBREDDITS_TO_SCRAPE = ["cryptocurrency", "bitcoin", "ethereum"]
    POST_LIMIT_PER_SUB = 1000 # Limite max par subreddit
    MIN_POST_SCORE = 10      # (Bonus) Ignorer les posts avec moins de 10 upvotes
    TOP_COMMENTS_PER_POST = 5
    OUTPUT_FILENAME = "D:\\CryptoVibe\\CryptoVibe\\data\\Bronze\\reddit_crypto_data.csv"
    
    reddit = connect_to_reddit()
    if reddit is None:
        return # Arrête le script si la connexion échoue

    all_data = []
    
    for sub_name in SUBREDDITS_TO_SCRAPE:
        subreddit_data = scrape_subreddit(
            reddit_instance=reddit,
            subreddit_name=sub_name,
            limit=POST_LIMIT_PER_SUB,
            min_score=MIN_POST_SCORE,
            top_comments_n=TOP_COMMENTS_PER_POST
        )
        all_data.extend(subreddit_data)

    save_to_csv(all_data, OUTPUT_FILENAME)
    logging.info("Scraping terminé.")

if __name__ == "__main__":
    main()