// dashboard/src/hooks/useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';

// Define a type for the processed message
export interface ProcessedMessage {
  id: string;
  text: string;
  date: string;
  source: string;
  author: string;
  score: number;
  type: string;
  sentiment: {
    score: number;
    label: string;
  };
  processed_at: string;
  price?: {
    ticker: string;
    price: number;
    as_of: string;
  } | null;
}

export const useWebSocket = (
  url: string,
  onMessage: (message: ProcessedMessage) => void,
  clientId: string = 'frontend_client'
) => {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectInterval = 5000; // 5 seconds

  const connect = useCallback(() => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return; // Already connected or connecting
    }

    ws.current = new WebSocket(`${url}/ws/${clientId}`);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setError(null);
      reconnectAttempts.current = 0;
    };

    ws.current.onmessage = (event) => {
      try {
        const message: ProcessedMessage = JSON.parse(event.data);
        onMessage(message);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.current.onclose = (event) => {
      console.log('WebSocket disconnected:', event);
      setIsConnected(false);
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
        setTimeout(connect, reconnectInterval);
      } else {
        console.error('Max reconnect attempts reached. Please refresh the page.');
        setError(new Event('Max reconnect attempts reached'));
      }
    };

    ws.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError(event);
      ws.current?.close(); // Attempt to close and trigger onclose for reconnect logic
    };
  }, [url, clientId, onMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
    };
  }, [connect]);

  return { isConnected, error };
};
