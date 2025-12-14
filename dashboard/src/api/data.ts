// dashboard/src/api/data.ts

import { ProcessedMessage } from '../hooks/useWebSocket';

const API_GATEWAY_URL = 'http://localhost:8000';

export const dataService = {
  fetchSentimentTimeline: async (
    limit: number = 500,
    offset: number = 0,
    sentiment?: string
  ): Promise<{ data: ProcessedMessage[]; count: number }> => {
    try {
      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
      });
      if (sentiment) {
        params.append('sentiment', sentiment);
      }
      const response = await fetch(`${API_GATEWAY_URL}/sentiment/timeline?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return { data: data.data ?? [], count: data.count ?? 0 };
    } catch (error) {
      console.error("Error fetching sentiment timeline:", error);
      return { data: [], count: 0 };
    }
  }
};
