import React from 'react';
import { CryptoData } from '../types/crypto';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface ComparisonCryptoCardProps {
  crypto: CryptoData;
  accentColor: string;
  onCardClick: (symbol: string) => void;
  isSelected: boolean;
}

const ComparisonCryptoCard: React.FC<ComparisonCryptoCardProps> = ({
  crypto,
  accentColor,
  onCardClick,
  isSelected,
}) => {
  const sentimentColor = crypto.avgSentiment >= 0 ? 'text-green-500' : 'text-red-500';
  const trendColor = crypto.trend >= 0 ? 'text-green-500' : 'text-red-500';
  const TrendIcon = crypto.trend >= 0 ? TrendingUp : TrendingDown;

  const cardClasses = `
    relative p-6 rounded-xl shadow-lg backdrop-filter backdrop-blur-lg border
    hover:shadow-xl transition-all duration-300 cursor-pointer
    ${isSelected ? 'border-2 border-opacity-80' : 'border-opacity-20'}
  `;

  return (
    <div
      className={cardClasses}
      style={{ borderColor: isSelected ? accentColor : 'rgba(255, 255, 255, 0.2)' }}
      onClick={() => onCardClick(crypto.symbol)}
    >
      {/* Header */}
      <div className="flex items-center space-x-3 mb-4">
        {/* Crypto Icon/Logo - Placeholder for now */}
        <div className="w-8 h-8 rounded-full flex items-center justify-center text-lg font-bold" style={{ backgroundColor: accentColor }}>
          {crypto.symbol[0]}
        </div>
        <h3 className="text-xl font-semibold text-white">{crypto.name} <span className="text-gray-400">{crypto.symbol}</span></h3>
      </div>

      {/* Main Metrics */}
      <div className="mb-4">
        <div className="flex items-baseline justify-between mb-2">
          <p className="text-gray-400 text-sm">Avg Sentiment</p>
          <p className={`text-3xl font-bold ${sentimentColor}`}>{crypto.avgSentiment.toFixed(2)}</p>
        </div>
        <div className="flex items-baseline justify-between">
          <p className="text-gray-400 text-sm">Total Mentions</p>
          <p className="text-2xl font-bold text-white">{crypto.totalMentions.toLocaleString()} posts</p>
        </div>
      </div>

      {/* Trend Indicator */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <TrendIcon size={20} className={trendColor} />
          <span className={`text-lg font-semibold ${trendColor}`}>{crypto.trend.toFixed(1)}%</span>
        </div>
        <p className="text-gray-400 text-sm">vs previous period</p>
      </div>

      {/* Mini Sparkline Chart */}
      <div className="h-[50px] w-full mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={crypto.sparklineData.map(value => ({ value }))}>
            <Line
              type="monotone"
              dataKey="value"
              stroke={accentColor}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment Breakdown */}
      <div className="text-sm text-gray-400">
        <p>Positive: <span className="text-green-400">{crypto.distribution.positive}%</span></p>
        <p>Neutral: <span className="text-yellow-400">{crypto.distribution.neutral}%</span></p>
        <p>Negative: <span className="text-red-400">{crypto.distribution.negative}%</span></p>
      </div>
    </div>
  );
};

export default ComparisonCryptoCard;
