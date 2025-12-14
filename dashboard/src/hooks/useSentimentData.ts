import { useEffect, useMemo, useState } from 'react';
import { dataService } from '../api/data';
import { ProcessedMessage } from './useWebSocket';

export interface SentimentMetrics {
  latest?: ProcessedMessage;
  priceChangePct?: number | null;
  sentimentPriceCorrelation?: number | null;
}

const mean = (values: number[]) => {
  if (!values.length) return 0;
  return values.reduce((acc, val) => acc + val, 0) / values.length;
};

const computePriceChange = (messages: ProcessedMessage[]) => {
  const pricePoints = messages.filter((msg) => msg.price && typeof msg.price.price === 'number');
  if (pricePoints.length < 2) return null;
  const first = pricePoints[0].price!.price;
  const last = pricePoints[pricePoints.length - 1].price!.price;
  if (!first) return null;
  return ((last - first) / first) * 100;
};

const computeCorrelation = (messages: ProcessedMessage[]) => {
  const pairs = messages
    .filter((msg) => msg.price && typeof msg.sentiment?.score === 'number')
    .map((msg) => ({
      sentiment: msg.sentiment.score,
      price: msg.price!.price,
    }));

  if (pairs.length < 2) return null;

  const sentiments = pairs.map((p) => p.sentiment);
  const prices = pairs.map((p) => p.price);
  const meanSent = mean(sentiments);
  const meanPrice = mean(prices);

  const numerator = pairs.reduce(
    (acc, p) => acc + (p.sentiment - meanSent) * (p.price - meanPrice),
    0
  );
  const denomSent = Math.sqrt(pairs.reduce((acc, p) => acc + (p.sentiment - meanSent) ** 2, 0));
  const denomPrice = Math.sqrt(pairs.reduce((acc, p) => acc + (p.price - meanPrice) ** 2, 0));
  const denom = denomSent * denomPrice;
  if (!denom) return null;
  return numerator / denom;
};

export const useSentimentData = () => {
  const [historicalData, setHistoricalData] = useState<ProcessedMessage[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const loadHistoricalData = async () => {
      setLoading(true);
      const response = await dataService.fetchSentimentTimeline(500, 0);
      if (!mounted) return;
      const data = response.data ?? [];
      const sorted = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      setHistoricalData(sorted);
      setTotalCount(response.count ?? sorted.length);
      setLoading(false);
    };

    loadHistoricalData();
    return () => {
      mounted = false;
    };
  }, []);

  const metrics: SentimentMetrics = useMemo(() => {
    const latest = historicalData[historicalData.length - 1];
    return {
      latest,
      priceChangePct: computePriceChange(historicalData),
      sentimentPriceCorrelation: computeCorrelation(historicalData),
    };
  }, [historicalData]);

  return {
    historicalData,
    totalCount,
    loading,
    metrics,
  };
};
