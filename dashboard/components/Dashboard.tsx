import React from 'react';
import KPICard from './KPICard';
import SentimentPriceOverlayChart from './charts/SentimentPriceOverlayChart';
import SentimentBreakdown from './SentimentBreakdown';
import PostsPanel from './PostsPanel';
import { Smile, TrendingUp, Link as LinkIcon, Database } from 'lucide-react';

const MOCK_CHART_DATA = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(2023, 11, i + 1).toISOString(),
  sentiment: Math.random() * 2 - 1,
  price: 50000 + (Math.random() * 2000 - 1000) * i,
  mentions: Math.floor(Math.random() * 1000),
}));

const Dashboard: React.FC = () => {
  return (
    <div className="py-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <KPICard title="Current Sentiment" value="0.58" change="+0.05" trend="improving" icon={<Smile className="text-accent-secondary" />} />
        <KPICard title="24h Price Change" value="+2.5%" change="+1.5%" trend="improving" icon={<TrendingUp className="text-status-positive" />} />
        <KPICard title="Sentiment/Price Correlation" value="0.72" icon={<LinkIcon className="text-accent-primary" />} />
        <KPICard title="Data Coverage" value="95%" icon={<Database className="text-fg-text-muted" />} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-bg-surface backdrop-blur-lg border border-border-default p-6 rounded-lg shadow-card">
          <h2 className="text-xl font-bold mb-4 text-fg-text">Sentiment vs. Price</h2>
          <SentimentPriceOverlayChart data={MOCK_CHART_DATA} crypto="BTC" />
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
