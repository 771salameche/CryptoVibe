export interface CryptoData {
  symbol: string;
  name: string;
  avgSentiment: number;
  totalMentions: number;
  trend: number;
    sparklineData: number[];
    distribution: {positive: number, neutral: number, negative: number};
    sentimentVolatility: number;
    correlationWithPrice: number;
    eventFrequency: number;
  fullTimeSeries: { Date: string; Sentiment: number; Mentions: number; Volume?: number; }[];
}