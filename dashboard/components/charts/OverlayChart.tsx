import React from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
} from 'recharts';

interface OverlayChartProps {
  data: any[];
}

const OverlayChart: React.FC<OverlayChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
        <XAxis dataKey="date" stroke="#9CA3AF" />
        <YAxis yAxisId="left" label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft', fill: '#60A5FA' }} stroke="#60A5FA" />
        <YAxis yAxisId="right" orientation="right" label={{ value: 'Price (USD)', angle: 90, position: 'insideRight', fill: '#34D399' }} stroke="#34D399" />
        <Tooltip
          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #4B5563' }}
          labelStyle={{ color: '#F9FAFB' }}
        />
        <Legend wrapperStyle={{ color: '#F9FAFB' }} />
        <Area yAxisId="left" type="monotone" dataKey="sentiment" fill="#60A5FA" stroke="#60A5FA" fillOpacity={0.3} />
        <Line yAxisId="right" type="monotone" dataKey="price" stroke="#34D399" />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default OverlayChart;