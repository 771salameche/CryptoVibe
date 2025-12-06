// dashboard/src/api/data.ts

import { ProcessedMessage } from '../hooks/useWebSocket';

const API_GATEWAY_URL = 'http://localhost:8000';

export const dataService = {
  fetchSentimentTimeline: async (): Promise<ProcessedMessage[]> => {
    try {
      const response = await fetch(`${API_GATEWAY_URL}/sentiment/timeline`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data.data; // Assuming the API returns {"data": [...]}
    } catch (error) {
      console.error("Error fetching sentiment timeline:", error);
      return [];
    }
  }
};
