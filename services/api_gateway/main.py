from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import json
import pika
import threading
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
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
DB_PATH = Path(os.environ.get("DB_PATH", "/app/data/processed.db"))

app = FastAPI(
    title="CryptoVibe API Gateway",
    description="API for accessing processed cryptocurrency sentiment and price data.",
    version="0.1.0",
)

# --- Configuration CORS ---
origins = [
    "http://localhost:5173",  # Allow frontend origin
    "http://127.0.0.1:5173",  # Also allow for loopback
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_lock = threading.Lock()

def init_db():
    """Initialize SQLite database and table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_messages (
                id TEXT PRIMARY KEY,
                text TEXT,
                date TEXT,
                source TEXT,
                author TEXT,
                score REAL,
                type TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                processed_at TEXT,
                price_ticker TEXT,
                price REAL,
                price_as_of TEXT,
                raw_json TEXT
            )
            """
        )
        conn.commit()

def save_processed_message(message: Dict[str, Any]):
    """Persist a processed message into SQLite (idempotent)."""
    with db_lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO processed_messages (
                id, text, date, source, author, score, type,
                sentiment_label, sentiment_score, processed_at,
                price_ticker, price, price_as_of, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.get("id"),
                message.get("text"),
                message.get("date"),
                message.get("source"),
                message.get("author"),
                message.get("score"),
                message.get("type"),
                message.get("sentiment", {}).get("label"),
                message.get("sentiment", {}).get("score"),
                message.get("processed_at"),
                (message.get("price") or {}).get("ticker") if message.get("price") else None,
                (message.get("price") or {}).get("price") if message.get("price") else None,
                (message.get("price") or {}).get("as_of") if message.get("price") else None,
                json.dumps(message),
            ),
        )
        conn.commit()

def fetch_messages(limit: int = 200, offset: int = 0, sentiment: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve messages from SQLite with optional sentiment filter."""
    query = "SELECT raw_json FROM processed_messages"
    params: List[Any] = []
    conditions = []
    if sentiment:
        conditions.append("LOWER(sentiment_label) = LOWER(?)")
        params.append(sentiment)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with db_lock, sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    return [json.loads(r[0]) for r in rows]

def count_messages() -> int:
    with db_lock, sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT COUNT(*) FROM processed_messages").fetchone()
        return row[0] if row else 0

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
        save_processed_message(message)
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
    init_db()
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

    db_ok = DB_PATH.exists()
    message_count = count_messages()

    return {
        "status": "ok",
        "rabbitmq_host": RABBITMQ_HOST,
        "rabbitmq_connected": rabbitmq_connected,
        "db_path": str(DB_PATH),
        "db_initialized": db_ok,
        "processed_data_count": message_count,
    }

@app.get("/sentiment/timeline")
async def get_sentiment_timeline(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment label (positive|neutral|negative)"),
):
    """
    Returns processed sentiment data with pagination and optional sentiment filter.
    """
    try:
        data = fetch_messages(limit=limit, offset=offset, sentiment=sentiment)
        return {"data": data, "count": count_messages(), "limit": limit, "offset": offset}
    except Exception as e:
        logging.error(f"API Gateway: Failed to fetch timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch sentiment timeline")
