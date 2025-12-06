from fastapi import FastAPI, BackgroundTasks
import logging
import os
import json
import pika
import threading
import time
from dotenv import load_dotenv

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Chargement des variables d'environnement ---
load_dotenv()
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
PROCESSED_POSTS_QUEUE = os.environ.get("PROCESSED_POSTS_QUEUE", "processed_reddit_posts")

app = FastAPI(
    title="CryptoVibe API Gateway",
    description="API for accessing processed cryptocurrency sentiment and price data.",
    version="0.1.0",
)

# In-memory store for processed data
processed_data_store = []
# Lock to ensure thread-safe access to processed_data_store
data_store_lock = threading.Lock()

def connect_to_rabbitmq(host):
    """Connects to RabbitMQ."""
    max_retries = 10
    retry_delay = 5  # seconds
    for i in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            channel = connection.channel()
            channel.queue_declare(queue=PROCESSED_POSTS_QUEUE, durable=True)
            logging.info(f"API Gateway: Connected to RabbitMQ at {host}, queue '{PROCESSED_POSTS_QUEUE}' declared.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.warning(f"API Gateway: RabbitMQ connection failed ({e}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
    logging.critical("API Gateway: Failed to connect to RabbitMQ after multiple retries.")
    exit(1)

def rabbitmq_consumer_callback(ch, method, properties, body):
    """Callback function to process received messages from RabbitMQ."""
    try:
        message = json.loads(body.decode('utf-8'))
        logging.info(f"API Gateway: Consumed processed message (ID: {message.get('id', 'N/A')}).")
        with data_store_lock:
            processed_data_store.append(message)
        ch.basic_ack(method.delivery_tag)
    except json.JSONDecodeError as e:
        logging.error(f"API Gateway: Failed to decode JSON message: {body}. Error: {e}")
        ch.basic_nack(method.delivery_tag, requeue=False) # Don't requeue malformed messages
    except Exception as e:
        logging.error(f"API Gateway: Error processing message: {e}", exc_info=True)
        ch.basic_nack(method.delivery_tag, requeue=True) # Requeue on other errors

def start_rabbitmq_consumer():
    """Starts the RabbitMQ consumer in a separate thread."""
    connection, channel = connect_to_rabbitmq(RABBITMQ_HOST)
    
    channel.basic_consume(queue=PROCESSED_POSTS_QUEUE, on_message_callback=rabbitmq_consumer_callback, auto_ack=False)
    logging.info(f"API Gateway: Starting RabbitMQ consumer for queue '{PROCESSED_POSTS_QUEUE}'...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("API Gateway: RabbitMQ consumer interrupted.")
    except Exception as e:
        logging.critical(f"API Gateway: RabbitMQ consumer failed: {e}", exc_info=True)
    finally:
        connection.close()
        logging.info("API Gateway: RabbitMQ connection closed.")

@app.on_event("startup")
async def startup_event():
    """Event handler for application startup."""
    logging.info("API Gateway: Startup event triggered.")
    # Start RabbitMQ consumer in a background thread
    consumer_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    consumer_thread.start()
    logging.info("API Gateway: RabbitMQ consumer thread started.")

@app.get("/")
async def root():
    return {"message": "CryptoVibe API Gateway is running!"}

@app.get("/health")
async def health_check():
    # In a real scenario, this would check connectivity to RabbitMQ, DB, etc.
    # For now, just indicate that the service is up and consumer thread started
    rabbitmq_connected = False
    try:
        # Attempt a quick connection test to RabbitMQ if the consumer thread is running
        temp_conn = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=0, port=5672, blocked_connection_timeout=5))
        temp_conn.close()
        rabbitmq_connected = True
    except Exception:
        rabbitmq_connected = False

    return {"status": "ok", "rabbitmq_host": RABBITMQ_HOST, "rabbitmq_connected": rabbitmq_connected, "processed_data_count": len(processed_data_store)}

@app.get("/sentiment/timeline")
async def get_sentiment_timeline():
    """
    Returns the accumulated processed sentiment data.
    """
    with data_store_lock:
        return {"data": list(processed_data_store)} # Return a copy