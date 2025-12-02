import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import { CryptoData } from '../../types/crypto';

interface RadarComparisonChartProps {
  data: CryptoData[];
  selectedCrypto: string | null;
}

const RadarComparisonChart: React.FC<RadarComparisonChartProps> = ({ data, selectedCrypto }) => {
  const getAccentColor = (symbol: string) => {
    switch (symbol) {
      case 'BTC':
        return '#f7931a'; // Orange
      case 'ETH':
        return '#627EEA'; // Blue
      case 'SOL':
        return '#14F195'; // Green
      default:
        return '#cccccc';
    }
  };

  // Prepare data for the RadarChart
  // The RadarChart expects data in a specific format:
  // [{ subject: 'Metric', A: valueA, B: valueB, ...}, ...]
  const radarData = [
    { subject: 'Avg Sentiment', A: 0, fullMark: 1 }, // Assuming sentiment is -1 to 1, normalized for chart
    { subject: 'Volatility', A: 0, fullMark: 1 }, // Normalized
    { subject: 'Correlation', A: 0, fullMark: 1 }, // Normalized
    { subject: 'Social Volume', A: 0, fullMark: 1 }, // Normalized
    { subject: 'Event Freq', A: 0, fullMark: 1 }, // Normalized
  ];

  // Helper to normalize values (simple min-max normalization for radar chart)
  const normalize = (value: number, min: number, max: number) => {
    if (max === min) return 0.5; // Avoid division by zero
    return (value - min) / (max - min);
  };

  // Extract all values for normalization
  const allAvgSentiment = data.map(d => d.avgSentiment);
  const allSentimentVolatility = data.map(d => d.sentimentVolatility);
  const allCorrelationWithPrice = data.map(d => d.correlationWithPrice);
  const allTotalMentions = data.map(d => d.totalMentions);
  const allEventFrequency = data.map(d => d.eventFrequency);

  const minMax = {
    avgSentiment: { min: Math.min(...allAvgSentiment), max: Math.max(...allAvgSentiment) },
    sentimentVolatility: { min: Math.min(...allSentimentVolatility), max: Math.max(...allSentimentVolatility) },
    correlationWithPrice: { min: Math.min(...allCorrelationWithPrice), max: Math.max(...allCorrelationWithPrice) },
    totalMentions: { min: Math.min(...allTotalMentions), max: Math.max(...allTotalMentions) },
    eventFrequency: { min: Math.min(...allEventFrequency), max: Math.max(...allEventFrequency) },
  };

  // Populate radarData for each crypto
  data.forEach(crypto => {
    radarData[0][crypto.symbol] = normalize(crypto.avgSentiment, minMax.avgSentiment.min, minMax.avgSentiment.max);
    radarData[1][crypto.symbol] = normalize(crypto.sentimentVolatility, minMax.sentimentVolatility.min, minMax.sentimentVolatility.max);
    radarData[2][crypto.symbol] = normalize(crypto.correlationWithPrice, minMax.correlationWithPrice.min, minMax.correlationWithPrice.max);
    radarData[3][crypto.symbol] = normalize(crypto.totalMentions, minMax.totalMentions.min, minMax.totalMentions.max);
    radarData[4][crypto.symbol] = normalize(crypto.eventFrequency, minMax.eventFrequency.min, minMax.eventFrequency.max);
  });

  // Custom Tooltip for Radar Chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-3 bg-gray-800 bg-opacity-70 rounded-lg shadow-md border border-gray-700 text-white text-sm">
          <p className="font-bold mb-1">{payload[0].payload.subject}</p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              {entry.name}: {entry.value !== undefined ? (entry.value * 100).toFixed(1) : 'N/A'}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };


  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
        <PolarGrid stroke="#444" />
        <PolarAngleAxis dataKey="subject" stroke="#999" />
        <PolarRadiusAxis angle={30} domain={[0, 1]} stroke="#999" />
        {data.map(crypto => (
          <Radar
            key={crypto.symbol}
            name={crypto.name}
            dataKey={crypto.symbol}
            stroke={getAccentColor(crypto.symbol)}
            fill={getAccentColor(crypto.symbol)}
            fillOpacity={selectedCrypto === null || selectedCrypto === crypto.symbol ? 0.6 : 0.2}
            strokeOpacity={selectedCrypto === null || selectedCrypto === crypto.symbol ? 1 : 0.4}
          />
        ))}
        <Legend />
        <Tooltip content={<CustomTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  );
};

export default RadarComparisonChart;