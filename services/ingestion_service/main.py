import praw
import os
import json
import pika
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Chargement des variables d'environnement ---
load_dotenv() # Load from .env file at the root or within the service directory

CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_QUEUE = os.environ.get("RABBITMQ_QUEUE", "raw_reddit_posts")

if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    logging.error("Reddit API environment variables (CLIENT_ID, CLIENT_SECRET, USER_AGENT) not found.")
    exit()

def connect_to_reddit():
    """Initialise et retourne une instance PRAW authentifiée."""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.read_only = True
        logging.info("Connexion à l'API Reddit réussie (mode lecture seule).")
        return reddit
    except Exception as e:
        logging.critical(f"Impossible de se connecter à Reddit: {e}")
        return None

def connect_to_rabbitmq(host):
    """Connects to RabbitMQ."""
    max_retries = 10
    retry_delay = 5  # seconds
    for i in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            channel = connection.channel()
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            logging.info(f"Connected to RabbitMQ at {host}, queue '{RABBITMQ_QUEUE}' declared.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.warning(f"RabbitMQ connection failed ({e}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
    logging.critical("Failed to connect to RabbitMQ after multiple retries.")
    exit(1)

def publish_message(channel, message):
    """Publishes a message to RabbitMQ."""
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )
    logging.info(f"Published message to RabbitMQ: {message['id']}")

def scrape_subreddit_and_publish(reddit_instance, channel, subreddit_name, limit, min_score, top_comments_n=2):
    """
    Scrape les posts et les top commentaires d'un subreddit donné et publie sur RabbitMQ.
    """
    subreddit = reddit_instance.subreddit(subreddit_name)
    logging.info(f"Début du scraping de r/{subreddit_name} (limite={limit}, score min={min_score})...")
    
    scraped_count = 0
    try:
        # Use .top('day') for quicker testing
        for submission in subreddit.top(time_filter="day", limit=limit): 
            try:
                if submission.score < min_score:
                    continue
                
                post_date = datetime.utcfromtimestamp(submission.created_utc).isoformat()
                post_text = f"{submission.title} {submission.selftext}"
                author_name = submission.author.name if submission.author else "[deleted]"
                
                post_message = {
                    "id": f"t3_{submission.id}",
                    "text": post_text.strip(),
                    "date": post_date,
                    "source": f"r/{subreddit_name}",
                    "author": author_name,
                    "score": submission.score,
                    "type": "post"
                }
                publish_message(channel, post_message)
                scraped_count += 1

                submission.comments.replace_more(limit=0)
                top_comments = sorted(submission.comments, key=lambda c: c.score, reverse=True)
                
                for comment in top_comments[:top_comments_n]:
                    comment_date = datetime.utcfromtimestamp(comment.created_utc).isoformat()
                    comment_author = comment.author.name if comment.author else "[deleted]"
                    
                    comment_message = {
                        "id": f"t1_{comment.id}",
                        "text": comment.body.strip(),
                        "date": comment_date,
                        "source": f"r/{subreddit_name}",
                        "author": comment_author,
                        "score": comment.score,
                        "type": "comment",
                        "parent_id": f"t3_{submission.id}"
                    }
                    publish_message(channel, comment_message)
                    scraped_count += 1

            except praw.exceptions.APIException as e:
                logging.warning(f"PRAW API Error on post {submission.id}: {e}")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Unknown error processing post {submission.id}: {e}")

    except Exception as e:
        logging.error(f"Major error during scraping r/{subreddit_name}: {e}")
        
    logging.info(f"Scraping of r/{subreddit_name} finished. {scraped_count} items published.")
    return scraped_count

def main():
    logging.info("Starting Ingestion Service (Reddit Scraper)...")
    
SUBREDDITS_TO_SCRAPE = os.environ.get("SUBREDDITS_TO_SCRAPE", "cryptocurrency,bitcoin,ethtrader").split(",")
POST_LIMIT_PER_SUB = int(os.environ.get("POST_LIMIT_PER_SUB", "10"))
MIN_POST_SCORE = int(os.environ.get("MIN_POST_SCORE", "5"))
TOP_COMMENTS_PER_POST = int(os.environ.get("TOP_COMMENTS_PER_POST", "2"))
SCRAPE_INTERVAL_SECONDS = int(os.environ.get("SCRAPE_INTERVAL_SECONDS", "900"))  # 15 minutes default
    
    reddit_instance = connect_to_reddit()
    if reddit_instance is None:
        return

    connection, channel = connect_to_rabbitmq(RABBITMQ_HOST)

    try:
        while True:
            total_items_published = 0
            for sub_name in SUBREDDITS_TO_SCRAPE:
                total_items_published += scrape_subreddit_and_publish(
                    reddit_instance=reddit_instance,
                    channel=channel,
                    subreddit_name=sub_name,
                    limit=POST_LIMIT_PER_SUB,
                    min_score=MIN_POST_SCORE,
                    top_comments_n=TOP_COMMENTS_PER_POST
                )
            logging.info(f"Ingestion Service cycle finished. Total {total_items_published} items published to RabbitMQ.")
            logging.info(f"Sleeping for {SCRAPE_INTERVAL_SECONDS} seconds before next cycle.")
            time.sleep(SCRAPE_INTERVAL_SECONDS)
    finally:
        connection.close()
        logging.info("RabbitMQ connection closed.")

if __name__ == "__main__":
    main()
