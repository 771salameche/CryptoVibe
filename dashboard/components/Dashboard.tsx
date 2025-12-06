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

  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoadingHistorical(true);
      const data = await dataService.fetchSentimentTimeline();
      // Sort data by date before setting
      data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      setHistoricalData(data);
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

  // Aggregate data for the chart (assuming chart expects a specific format)
  const chartData = combinedData.map(msg => ({
    date: msg.date,
    sentiment: msg.sentiment.score,
    // Add dummy price and mentions for now, as real historical price data isn't integrated yet
    price: 50000 + (Math.random() * 2000 - 1000), // Placeholder
    mentions: Math.floor(Math.random() * 1000), // Placeholder
  }));

  return (
    <div className="py-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <KPICard title="Current Sentiment" value="0.58" change="+0.05" trend="improving" icon={<Smile className="text-accent-secondary" />} />
        <KPICard title="24h Price Change" value="+2.5%" change="+1.5%" trend="improving" icon={<TrendingUp className="text-status-positive" />} />
        <KPICard title="Sentiment/Price Correlation" value="0.72" icon={<LinkIcon className="text-accent-primary" />} />
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
            <div>Loading historical data...</div>
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