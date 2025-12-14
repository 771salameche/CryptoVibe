import React, { useEffect, useState } from 'react';
import KPICard from './KPICard';
import SentimentPriceOverlayChart from './charts/SentimentPriceOverlayChart';
import SentimentBreakdown from './SentimentBreakdown';
import PostsPanel from './PostsPanel';
import { Smile, TrendingUp, Link as LinkIcon, Database, Signal } from 'lucide-react';
import { ProcessedMessage } from '../src/hooks/useWebSocket';
import { dataService } from '../src/api/data';

interface DashboardProps {
  realtimeData: ProcessedMessage[];
  isWsConnected: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({ realtimeData, isWsConnected }) => {
  const [historicalData, setHistoricalData] = useState<ProcessedMessage[]>([]);
  const [loadingHistorical, setLoadingHistorical] = useState(true);
  const [totalCount, setTotalCount] = useState(0);

  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoadingHistorical(true);
      const response = await dataService.fetchSentimentTimeline(500, 0);
      const data = response.data ?? [];
      // Sort data by date before setting
      const sorted = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      setHistoricalData(sorted);
      setTotalCount(response.count ?? sorted.length);
      setLoadingHistorical(false);
    };

    loadHistoricalData();
  }, []);

  // Combine historical and real-time data
  // For simplicity, real-time data is just appended.
  // In a real app, you'd merge based on ID/date to avoid duplicates.
  const combinedData = [...historicalData, ...realtimeData];
  // Ensure combined data is sorted by date before passing to chart
  combinedData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  const pricePoints = combinedData.filter((msg) => msg.price && typeof msg.price.price === 'number');
  const priceChange = (() => {
    if (pricePoints.length < 2) return null;
    const first = pricePoints[0].price!.price;
    const last = pricePoints[pricePoints.length - 1].price!.price;
    if (first === 0) return null;
    return ((last - first) / first) * 100;
  })();

  const sentimentPriceCorrelation = (() => {
    const pairs = combinedData
      .filter((msg) => msg.price && typeof msg.sentiment?.score === 'number')
      .map((msg) => ({
        sentiment: msg.sentiment.score,
        price: msg.price!.price,
      }));
    if (pairs.length < 2) return null;
    const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const sentiments = pairs.map((p) => p.sentiment);
    const prices = pairs.map((p) => p.price);
    const meanSent = mean(sentiments);
    const meanPrice = mean(prices);
    const numerator = pairs.reduce((acc, p) => acc + (p.sentiment - meanSent) * (p.price - meanPrice), 0);
    const denomSent = Math.sqrt(pairs.reduce((acc, p) => acc + (p.sentiment - meanSent) ** 2, 0));
    const denomPrice = Math.sqrt(pairs.reduce((acc, p) => acc + (p.price - meanPrice) ** 2, 0));
    const denom = denomSent * denomPrice;
    if (denom === 0) return null;
    return numerator / denom;
  })();

  const latestMsg = combinedData[combinedData.length - 1];

  // Aggregate data for the chart (assuming chart expects a specific format)
  const chartData = combinedData.map(msg => ({
    date: msg.date,
    sentiment: msg.sentiment.score,
    price: msg.price?.price ?? 0,
    mentions: msg.score ?? 0,
  }));

  return (
    <div className="py-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <KPICard
          title="Current Sentiment"
          value={latestMsg ? `${latestMsg.sentiment.label} (${latestMsg.sentiment.score.toFixed(3)})` : "N/A"}
          trend={latestMsg && latestMsg.sentiment.score >= 0 ? "improving" : "declining"}
          icon={<Smile className="text-accent-secondary" />}
        />
        <KPICard
          title="24h Price Change"
          value={priceChange !== null ? `${priceChange.toFixed(2)}%` : "N/A"}
          trend={priceChange !== null ? (priceChange >= 0 ? "improving" : "declining") : "stable"}
          icon={<TrendingUp className="text-status-positive" />}
        />
        <KPICard
          title="Sentiment/Price Correlation"
          value={sentimentPriceCorrelation !== null ? sentimentPriceCorrelation.toFixed(2) : "N/A"}
          icon={<LinkIcon className="text-accent-primary" />}
        />
        <KPICard 
          title="WS Status" 
          value={isWsConnected ? "Connected" : "Disconnected"} 
          trend={isWsConnected ? "improving" : "declining"} 
          icon={<Signal className={isWsConnected ? "text-status-positive" : "text-status-negative"} />} 
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-bg-surface backdrop-blur-lg border border-border-default p-6 rounded-lg shadow-card">
          <h2 className="text-xl font-bold mb-4 text-fg-text">Sentiment vs. Price</h2>
          {loadingHistorical ? (
            <div className="flex items-center justify-center h-48 bg-bg-alt rounded-lg">
              <svg className="animate-spin h-8 w-8 text-accent-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="ml-3 text-fg-text-muted">Loading historical data...</span>
            </div>
          ) : (
            <SentimentPriceOverlayChart data={chartData} crypto="BTC" />
          )}
        </div>
        <div className="lg:col-span-1 space-y-8">
          <div className="bg-bg-surface backdrop-blur-lg border border-border-default p-6 rounded-lg shadow-card">
            <h2 className="text-xl font-bold mb-4 text-fg-text">Sentiment Breakdown</h2>
            <SentimentBreakdown />
          </div>
          <div className="bg-bg-surface backdrop-blur-lg border border-border-default p-6 rounded-lg shadow-card">
            <h2 className="text-xl font-bold mb-4 text-fg-text">Top Posts</h2>
            <PostsPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
