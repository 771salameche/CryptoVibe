import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { CryptoData } from '../../types/crypto';

interface MultiLineChartProps {
  data: CryptoData[];
  selectedCrypto: string | null;
  metricType: 'sentiment' | 'volume' | 'mentions';
}

const MultiLineChart: React.FC<MultiLineChartProps> = ({ data, selectedCrypto, metricType }) => {
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

  // Prepare data for Recharts
  // Combine all crypto's fullTimeSeries into a single array, aligning by Date
  const chartDataMap = new Map<string, any>(); // Map<Date, { Date: string, BTC_Sentiment: number, ETH_Sentiment: number, ... }>

  data.forEach(crypto => {
    crypto.fullTimeSeries.forEach(entry => {
      const date = entry.Date;
      if (!chartDataMap.has(date)) {
        chartDataMap.set(date, { Date: date });
      }
      const current = chartDataMap.get(date);
      let value: number = 0;
      if (metricType === 'sentiment') {
        value = entry.Sentiment;
      } else if (metricType === 'mentions') {
        value = entry.Mentions;
      }
      // Add more conditions for 'volume' if that data becomes available
      current[`${crypto.symbol}_${metricType}`] = value;
      chartDataMap.set(date, current);
    });
  });

  const chartData = Array.from(chartDataMap.values()).sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());

  // Custom Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-3 bg-gray-800 bg-opacity-70 rounded-lg shadow-md border border-gray-700 text-white text-sm">
          <p className="font-bold mb-1">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              {entry.name}: {entry.value !== undefined ? entry.value.toFixed(2) : 'N/A'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={chartData}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
        <XAxis dataKey="Date" stroke="#999" />
        <YAxis stroke="#999" />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        {data.map((crypto) => (
          <Line
            key={crypto.symbol}
            type="monotone"
            dataKey={`${crypto.symbol}_${metricType}`}
            stroke={getAccentColor(crypto.symbol)}
            strokeWidth={selectedCrypto === crypto.symbol ? 3 : 1.5}
            activeDot={{ r: 8 }}
            name={`${crypto.name} ${metricType.charAt(0).toUpperCase() + metricType.slice(1)}`}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MultiLineChart;