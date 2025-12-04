import React, { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  Area,
} from 'recharts';
import { format } from 'date-fns';

interface SentimentPriceOverlayChartProps {
  data: any[]; // Data should contain date, sentiment, price, and priceChange (for color)
  crypto: string;
}

const SentimentPriceOverlayChart: React.FC<SentimentPriceOverlayChartProps> = ({ data, crypto }) => {

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const date = format(new Date(label), 'MMM d, yyyy');
      const sentiment = payload.find((p: any) => p.dataKey === 'sentiment')?.value;
      const price = payload.find((p: any) => p.dataKey === 'price')?.value;
      const priceChange = payload.find((p: any) => p.dataKey === 'priceChange')?.value;

      return (
        <div className="bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-700 text-sm">
          <p className="font-bold text-white mb-1">{date}</p>
          {sentiment !== undefined && <p className="text-blue-400">Sentiment: {sentiment.toFixed(3)}</p>}
          {price !== undefined && <p className="text-green-400">Price: ${price.toFixed(2)}</p>}
          {priceChange !== undefined && <p className={priceChange >= 0 ? 'text-green-400' : 'text-red-400'}>Price Change: {priceChange.toFixed(2)}%</p>}
        </div>
      );
    }
    return null;
  };

  // Calculate correlation coefficient
  const correlationCoefficient = useMemo(() => {
    if (data.length < 2) return 0;
    const n = data.length;
    let sumSentiment = 0;
    let sumPrice = 0;
    let sumSentimentSq = 0;
    let sumPriceSq = 0;
    let sumProduct = 0;

    for (let i = 0; i < n; i++) {
      const s = data[i].sentiment;
      const p = data[i].price;
      sumSentiment += s;
      sumPrice += p;
      sumSentimentSq += s * s;
      sumPriceSq += p * p;
      sumProduct += s * p;
    }

    const numerator = n * sumProduct - sumSentiment * sumPrice;
    const denominator = Math.sqrt((n * sumSentimentSq - sumSentiment * sumSentiment) * (n * sumPriceSq - sumPrice * sumPrice));

    if (denominator === 0) return 0;
    return numerator / denominator;
  }, [data]);


  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
          <XAxis
            dataKey="date"
            tickFormatter={(tick) => format(new Date(tick), 'MMM dd')}
            minTickGap={30}
            stroke="#9CA3AF"
          />
          <YAxis
            yAxisId="left"
            label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft', fill: '#60A5FA' }}
            stroke="#60A5FA"
            domain={[-1, 1]}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            label={{ value: 'Price (USD)', angle: 90, position: 'insideRight', fill: '#34D399' }}
            stroke="#34D399"
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ color: '#F9FAFB', paddingTop: '10px' }} />

          {/* Highlight areas where both sentiment and price rise */}
          {data.map((entry, index) => {
            if (index === 0) return null;
            const prevEntry = data[index - 1];
            if (
              entry.sentiment > prevEntry.sentiment &&
              entry.price > prevEntry.price
            ) {
              return (
                <ReferenceArea
                  key={`rise-${index}`}
                  yAxisId="left"
                  x1={prevEntry.date}
                  x2={entry.date}
                  stroke="none"
                  fill="#86EFAC" // Green-200
                  fillOpacity={0.1}
                />
              );
            }
            return null;
          })}

          {/* Highlight areas where sentiment rises but price falls (divergence) */}
          {data.map((entry, index) => {
            if (index === 0) return null;
            const prevEntry = data[index - 1];
            if (
              entry.sentiment > prevEntry.sentiment &&
              entry.price < prevEntry.price
            ) {
              return (
                <ReferenceArea
                  key={`divergence-${index}`}
                  yAxisId="left"
                  x1={prevEntry.date}
                  x2={entry.date}
                  stroke="none"
                  fill="#FDE68A" // Yellow-200
                  fillOpacity={0.1}
                />
              );
            }
            return null;
          })}

          <Line
            yAxisId="left"
            type="monotone"
            dataKey="sentiment"
            stroke="#60A5FA" // Blue-400
            strokeWidth={2}
            dot={false}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="price"
            stroke={(entry) => (entry.priceChange >= 0 ? '#34D399' : '#EF4444')} // Green-500 or Red-500
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="absolute top-0 right-0 p-2 text-sm bg-gray-700 bg-opacity-70 rounded-bl-lg text-white">
        Correlation: r = {correlationCoefficient.toFixed(2)}
      </div>
    </div>
  );
};

export default SentimentPriceOverlayChart;