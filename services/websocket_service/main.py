from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    title="CryptoVibe WebSocket Service",
    description="Real-time data push for cryptocurrency sentiment and price data.",
    version="0.1.0",
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.broadcast_lock = threading.Lock() # Ensure thread-safe broadcasting

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logging.info(f"WebSocket disconnected: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        # Use a lock for broadcasting if connections list can be modified by multiple threads
        # Though FastAPI's WebSocket is async, the underlying list modification might not be.
        # This prevents 'list changed during iteration' errors if disconnect() happens mid-broadcast.
        with self.broadcast_lock:
            disconnected_clients = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except RuntimeError as e: # Handle WebSocket not connected error
                    logging.warning(f"Failed to send to disconnected client: {e}. Removing.")
                    disconnected_clients.append(connection)
                except Exception as e:
                    logging.error(f"Error broadcasting to client: {e}")
                    disconnected_clients.append(connection)
            for client in disconnected_clients:
                self.active_connections.remove(client)

        logging.info(f"Broadcasted message to {len(self.active_connections)} clients.")

manager = ConnectionManager()

def connect_to_rabbitmq(host):
    """Connects to RabbitMQ."""
    max_retries = 10
    retry_delay = 5  # seconds
    for i in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            channel = connection.channel()
            channel.queue_declare(queue=PROCESSED_POSTS_QUEUE, durable=True)
            logging.info(f"WebSocket Service: Connected to RabbitMQ at {host}, queue '{PROCESSED_POSTS_QUEUE}' declared.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.warning(f"WebSocket Service: RabbitMQ connection failed ({e}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
            time.sleep(retry_delay)
    logging.critical("WebSocket Service: Failed to connect to RabbitMQ after multiple retries.")
    exit(1)

def rabbitmq_consumer_callback(ch, method, properties, body):
    """Callback function to broadcast received messages to WebSocket clients."""
    message = body.decode('utf-8')
    logging.info(f"WebSocket Service: Consumed processed message for broadcast.")
    
    # We need to run async broadcast in an event loop. FastAPI's app.loop is available at startup.
    # For a blocking consumer thread, we can create a new event loop or use asyncio.run
    # if not already in an async context. Since this is in a separate thread,
    # we need to create a new event loop for it.
    import asyncio
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    new_loop.run_until_complete(manager.broadcast(message))
    new_loop.close()
    
    ch.basic_ack(method.delivery_tag)

def start_rabbitmq_consumer():
    """Starts the RabbitMQ consumer in a separate thread."""
    connection, channel = connect_to_rabbitmq(RABBITMQ_HOST)
    
    channel.basic_consume(queue=PROCESSED_POSTS_QUEUE, on_message_callback=rabbitmq_consumer_callback, auto_ack=False)
    logging.info(f"WebSocket Service: Starting RabbitMQ consumer for queue '{PROCESSED_POSTS_QUEUE}'...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info("WebSocket Service: RabbitMQ consumer interrupted.")
    except Exception as e:
        logging.critical(f"WebSocket Service: RabbitMQ consumer failed: {e}", exc_info=True)
    finally:
        connection.close()
        logging.info("WebSocket Service: RabbitMQ connection closed.")

@app.on_event("startup")
async def startup_event():
    """Event handler for application startup."""
    logging.info("WebSocket Service: Startup event triggered.")
    # Start RabbitMQ consumer in a background thread
    consumer_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    consumer_thread.start()
    logging.info("WebSocket Service: RabbitMQ consumer thread started.")


@app.get("/")
async def root():
    return {"message": "CryptoVibe WebSocket Service is running!"}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        # This service now primarily broadcasts. It doesn't expect clients to send messages
        # continuously. The loop is removed.
        # Keep the connection alive until client disconnects.
        while True:
            # You might optionally listen for specific control messages from the client
            # e.g., "subscribe to X", "unsubscribe from Y"
            await websocket.receive_text() # This will keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info(f"Client #{client_id} disconnected.")
    except Exception as e:
        logging.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket)