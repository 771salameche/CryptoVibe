export interface CryptoEvent {
  date: string; // ISO string or similar
  eventType: 'LISTING' | 'HACK' | 'REGULATION' | 'PARTNERSHIP' | 'UPGRADE' | string;
  crypto: string[]; // BTC, ETH, SOL, or multiple
  title: string;
  textSnippet: string;
  mentions: number;
  sentiment: number; // Avg sentiment score
  // Add more fields if needed from events_timeline.csv
}

export interface WordData {
  text: string;
  value: number; // Frequency or weight
  sentiment?: 'positive' | 'negative' | 'neutral'; // For coloring word clouds
}

export interface FilterState {
  eventTypes: string[];
  cryptos: string[];
  startDate: Date | null;
  endDate: Date | null;
  sentiment: 'all' | 'positive' | 'negative' | 'neutral';
  searchTerm: string;
}

export interface EventStatistics {
  totalEvents: number;
  mostActiveCrypto: { crypto: string; count: number } | null;
  mostCommonEventType: { type: string; count: number } | null;
}
