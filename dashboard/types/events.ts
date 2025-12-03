export type EventType = 'LISTING' | 'HACK' | 'REGULATION' | 'PARTNERSHIP' | 'UPGRADE';

export interface WordData {
  text: string;
  value: number;
  sentiment?: 'positive' | 'negative' | 'neutral';
}

export interface Event {
  date: string;
  event_type: EventType;
  crypto: string;
  title: string;
  mentions: number;
  sentiment: number;
  text: string;
}