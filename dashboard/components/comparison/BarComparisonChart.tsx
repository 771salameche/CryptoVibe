import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { CryptoData } from '../../types/crypto';

interface BarComparisonChartProps {
  data: CryptoData[];
  selectedCrypto: string | null;
}

const BarComparisonChart: React.FC<BarComparisonChartProps> = ({ data, selectedCrypto }) => {
  // Sort data by average sentiment for display
  const sortedData = [...data].sort((a, b) => b.avgSentiment - a.avgSentiment);

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

  // Custom Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-3 bg-gray-800 bg-opacity-70 rounded-lg shadow-md border border-gray-700 text-white text-sm">
          <p className="font-bold mb-1">{label}</p>
          <p style={{ color: payload[0].color }}>
            Avg Sentiment: {payload[0].value !== undefined ? payload[0].value.toFixed(2) : 'N/A'}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        layout="vertical"
        data={sortedData}
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
        <XAxis type="number" stroke="#999" />
        <YAxis dataKey="name" type="category" stroke="#999" />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Bar dataKey="avgSentiment" name="Average Sentiment">
          {sortedData.map((entry, index) => (
            <Bar
              key={`bar-${index}`}
              dataKey="avgSentiment"
              fill={getAccentColor(entry.symbol)}
              opacity={selectedCrypto === null || selectedCrypto === entry.symbol ? 1 : 0.4}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default BarComparisonChart;