import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set

import pika
import yfinance as yf
from transformers import pipeline, Pipeline

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Chargement des variables d'environnement ---
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RAW_POSTS_QUEUE = os.environ.get("RAW_POSTS_QUEUE", "raw_reddit_posts")  # Queue to consume from
PROCESSED_POSTS_QUEUE = os.environ.get("PROCESSED_POSTS_QUEUE", "processed_reddit_posts")  # Queue to publish to
PRICE_TICKER = os.environ.get("PRICE_TICKER", "BTC-USD")

# Cache for deduplication to avoid double-processing the same message id
processed_ids: Set[str] = set()

def connect_to_rabbitmq(host):
    """Connects to RabbitMQ."""
    max_retries = 10
    retry_delay = 5  # seconds
    for i in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            channel = connection.channel()
            
            # Declare both queues
            channel.queue_declare(queue=RAW_POSTS_QUEUE, durable=True)
            channel.queue_declare(queue=PROCESSED_POSTS_QUEUE, durable=True)

            logging.info(f"Processing Service: Connected to RabbitMQ at {host}.")
            logging.info(f"Processing Service: Consuming from '{RAW_POSTS_QUEUE}', publishing to '{PROCESSED_POSTS_QUEUE}'.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.warning(f"Processing Service: RabbitMQ connection failed ({e}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
    logging.critical("Processing Service: Failed to connect to RabbitMQ after multiple retries.")
    exit(1)

def load_finbert_pipeline() -> Pipeline:
    """
    Load the FinBERT sentiment analysis pipeline.
    Using a singleton to avoid reloading weights for every message.
    """
    logging.info("Processing Service: Loading FinBERT model...")
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
    )


# Instantiate once at module import to warm cache early
finbert_analyzer: Pipeline = load_finbert_pipeline()


def analyze_sentiment_finbert(text: str) -> Dict[str, Any]:
    """
    Run FinBERT sentiment analysis and return the dominant label and its score.
    """
    if not text:
        return {"label": "neutral", "score": 0.0}

    result = finbert_analyzer(text, truncation=True)
    if isinstance(result, list) and len(result) > 0:
        # pipeline returns list[dict] with 'label' and 'score'
        top = sorted(result, key=lambda r: r["score"], reverse=True)[0]
        return {"label": top["label"].lower(), "score": float(top["score"])}
    return {"label": "neutral", "score": 0.0}


def fetch_latest_price(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent price for the given ticker using yfinance.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # Get last known price from intraday data
        hist = ticker_obj.history(period="1d", interval="1m")
        if hist.empty:
            return None
        latest = hist.tail(1).iloc[0]
        return {
            "ticker": ticker,
            "price": float(latest["Close"]),
            "as_of": latest.name.to_pydatetime().isoformat(),
        }
    except Exception as e:
        logging.warning(f"Processing Service: Failed to fetch price for {ticker}: {e}")
        return None

def callback(ch, method, properties, body):
    """Callback function to process received messages."""
    try:
        raw_message = json.loads(body.decode('utf-8'))
        logging.info(f"Processing Service: Received message (ID: {raw_message.get('id', 'N/A')}): {raw_message.get('text', '')[:50]}...")

        msg_id = raw_message.get("id")
        if msg_id and msg_id in processed_ids:
            logging.info(f"Processing Service: Duplicate message detected (ID: {msg_id}), acking without processing.")
            ch.basic_ack(method.delivery_tag)
            return

        # Perform dummy sentiment analysis
        sentiment_result = analyze_sentiment_finbert(raw_message.get('text', ''))

        # Fetch latest price snapshot
        price_snapshot = fetch_latest_price(PRICE_TICKER)
        
        # Enrich the message with sentiment data
        processed_message = {
            **raw_message,
            "sentiment": sentiment_result,
            "processed_at": datetime.now().isoformat(),
            "price": price_snapshot,
        }

        # Publish the enriched message to the processed queue
        ch.basic_publish(
            exchange='',
            routing_key=PROCESSED_POSTS_QUEUE,
            body=json.dumps(processed_message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        logging.info(f"Processing Service: Published processed message (ID: {raw_message.get('id', 'N/A')}) to '{PROCESSED_POSTS_QUEUE}'.")

        # Acknowledge the raw message
        ch.basic_ack(method.delivery_tag)
        logging.info(f"Processing Service: Raw message (ID: {raw_message.get('id', 'N/A')}) acknowledged.")
        if msg_id:
            processed_ids.add(msg_id)

    except json.JSONDecodeError as e:
        logging.error(f"Processing Service: Failed to decode JSON message: {body}. Error: {e}")
        ch.basic_nack(method.delivery_tag, requeue=False) # Nack and don't requeue malformed messages
    except Exception as e:
        logging.error(f"Processing Service: An error occurred during message processing: {e}", exc_info=True)
        ch.basic_nack(method.delivery_tag, requeue=True) # Nack and requeue on other errors


def main():
    logging.info("Processing Service: Starting up...")
    
    connection, channel = connect_to_rabbitmq(RABBITMQ_HOST)

    try:
        # Set up a consumer for the raw posts queue
        channel.basic_consume(queue=RAW_POSTS_QUEUE, on_message_callback=callback, auto_ack=False)
        logging.info(f"Processing Service: Waiting for messages on '{RAW_POSTS_QUEUE}'. To exit press CTRL+C")
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("Processing Service: Detected KeyboardInterrupt. Shutting down.")
    except Exception as e:
        logging.critical(f"Processing Service: An unhandled critical error occurred: {e}", exc_info=True)
    finally:
        if connection:
            connection.close()
            logging.info("Processing Service: RabbitMQ connection closed.")

if __name__ == "__main__":
    main()
