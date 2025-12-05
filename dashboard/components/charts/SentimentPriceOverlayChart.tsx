/**
 * @file This file contains the SentimentPriceOverlayChart component, which displays a time-series chart
 * that overlays sentiment scores, cryptocurrency prices, and mention volumes.
 *
 * @how-to-use
 * To use this component, you need to pass a `data` prop, which is an array of data points.
 * Each data point should be an object with the following properties:
 * - `date`: A string representing the date of the data point (e.g., '2023-01-01T00:00:00.000Z').
 * - `sentiment`: A number representing the sentiment score (e.g., 0.5).
 * - `price`: A number representing the price of the cryptocurrency (e.g., 50000).
 * - `mentions`: A number representing the number of mentions (e.g., 100).
 *
 * You also need to pass a `crypto` prop, which is a string representing the cryptocurrency
 * (e.g., 'BTC').
 *
 * Example:
 * ```
 * <SentimentPriceOverlayChart data={chartData} crypto="BTC" />
 * ```
 */

import React from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';

interface DataPoint {
  date: string;
  sentiment: number;
  price: number;
  mentions: number;
}

interface SentimentPriceOverlayChartProps {
  data: DataPoint[];
  crypto: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const date = format(new Date(label), 'MMM d, yyyy');
    const sentiment = payload.find((p: any) => p.dataKey === 'sentiment')?.value;
    const price = payload.find((p: any) => p.dataKey === 'price')?.value;
    const mentions = payload.find((p: any) => p.dataKey === 'mentions')?.value;

    return (
      <div className="bg-card/80 backdrop-blur-lg border border-border p-3 rounded-lg shadow-lg text-sm">
        <p className="font-bold text-card-foreground mb-1">{date}</p>
        {sentiment !== undefined && <p className="text-primary">Sentiment: {sentiment.toFixed(3)}</p>}
        {price !== undefined && <p className="text-green-500">Price: ${price.toFixed(2)}</p>}
        {mentions !== undefined && <p className="text-card-foreground/70">Mentions: {mentions}</p>}
      </div>
    );
  }
  return null;
};

const SentimentPriceOverlayChart: React.FC<SentimentPriceOverlayChartProps> = ({ data, crypto }) => {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="date"
          tickFormatter={(tick) => format(new Date(tick), 'MMM dd')}
          minTickGap={30}
          stroke="var(--foreground)"
        />
        <YAxis
          yAxisId="left"
          label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft', fill: 'var(--primary)' }}
          stroke="var(--primary)"
          domain={[-1, 1]}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          label={{ value: 'Price (USD)', angle: 90, position: 'insideRight', fill: '#34D399' }}
          stroke="#34D399"
        />
        <YAxis
          yAxisId="mentions"
          orientation="right"
          label={{ value: 'Mentions', angle: 90, position: 'insideRight', fill: 'var(--foreground)', offset: 50 }}
          stroke="var(--foreground)"
          domain={[0, 'dataMax + 100']}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: 'var(--foreground)', paddingTop: '10px' }} />
        <Bar yAxisId="mentions" dataKey="mentions" barSize={20} fill="var(--border)" />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="sentiment"
          stroke="var(--primary)"
          strokeWidth={2}
          dot={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="price"
          stroke="#34D399"
          strokeWidth={2}
          dot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default SentimentPriceOverlayChart;