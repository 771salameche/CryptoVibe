import React, { useMemo } from 'react';
import KPICard from './KPICard';
import SentimentPriceOverlayChart from './charts/SentimentPriceOverlayChart';
import SentimentBreakdown from './SentimentBreakdown';
import PostsPanel from './PostsPanel';
import { Smile, TrendingUp, Link as LinkIcon, Signal } from 'lucide-react';
import { ProcessedMessage } from '../src/hooks/useWebSocket';
import { useSentimentData } from '../src/hooks/useSentimentData';

interface DashboardProps {
  realtimeData: ProcessedMessage[];
  isWsConnected: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({ realtimeData, isWsConnected }) => {
  const { historicalData, totalCount, loading, metrics } = useSentimentData();

  // Combine historical and real-time data (append and re-sort)
  const combinedData = useMemo(() => {
    const merged = [...historicalData, ...realtimeData];
    return merged.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }, [historicalData, realtimeData]);

  const latestMsg = combinedData[combinedData.length - 1];

  // Aggregate data for the chart (assuming chart expects a specific format)
  const chartData = combinedData.map(msg => ({
    date: msg.date,
    sentiment: msg.sentiment.score,
    price: msg.price?.price ?? 0,
    mentions: msg.score ?? 0,
  }));

  return (
    <div className="py-4 space-y-8">
      <div className="rounded-2xl border border-border-default/70 bg-gradient-to-r from-accent-primary/15 via-white/5 to-accent-secondary/10 text-fg-text shadow-card p-6 sm:p-7">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-fg-text-muted">Live Intelligence</p>
            <h1 className="text-2xl sm:text-3xl font-bold text-white mt-1">Real-time Crypto Sentiment & Price Pulse</h1>
            <p className="text-fg-text-muted max-w-2xl mt-2">
              Streaming FinBERT sentiment, price snapshots, and Reddit velocity into an interactive command center.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center rounded-full bg-white/10 px-3 py-1 text-xs font-semibold text-white border border-white/15">
              {isWsConnected ? 'WebSocket Live' : 'WebSocket Offline'}
            </span>
            <span className="inline-flex items-center rounded-full bg-fg-text text-slate-900 px-3 py-1 text-xs font-semibold">
              {totalCount} signals stored
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Current Sentiment"
          value={latestMsg ? `${latestMsg.sentiment.label} (${latestMsg.sentiment.score.toFixed(3)})` : "N/A"}
          trend={latestMsg && latestMsg.sentiment.score >= 0 ? "improving" : "declining"}
          icon={<Smile className="text-accent-secondary" />}
        />
        <KPICard
          title="24h Price Change"
          value={metrics.priceChangePct !== null && metrics.priceChangePct !== undefined ? `${metrics.priceChangePct.toFixed(2)}%` : "N/A"}
          trend={metrics.priceChangePct !== null && metrics.priceChangePct !== undefined ? (metrics.priceChangePct >= 0 ? "improving" : "declining") : "stable"}
          icon={<TrendingUp className="text-status-positive" />}
        />
        <KPICard
          title="Sentiment/Price Correlation"
          value={metrics.sentimentPriceCorrelation !== null && metrics.sentimentPriceCorrelation !== undefined ? metrics.sentimentPriceCorrelation.toFixed(2) : "N/A"}
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
          {loading ? (
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
