import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ErrorBar,
  LabelList,
} from 'recharts';

interface EventImpactChartProps {
  data: {
    eventType: string;
    priceChange: number;
    // For error bars, typically you'd have min and max values or a standard deviation
    // For now, we'll use a dummy value for error bar demonstration
    error?: number; 
  }[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-700 text-sm text-white">
        <p className="font-bold mb-1">{label}</p>
        <p className={data.priceChange >= 0 ? 'text-green-400' : 'text-red-400'}>
          Avg. Price Change: {data.priceChange.toFixed(2)}%
        </p>
        {data.error !== undefined && <p>Error Margin: ±{data.error.toFixed(2)}%</p>}
      </div>
    );
  }
  return null;
};

const EventImpactChart: React.FC<EventImpactChartProps> = ({ data }) => {
  // Sort data by absolute priceChange for better visualization
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => Math.abs(b.priceChange) - Math.abs(a.priceChange));
  }, [data]);

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={sortedData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
        <XAxis type="number" stroke="#9CA3AF" />
        <YAxis type="category" dataKey="eventType" stroke="#9CA3AF" width={120} tick={{ fill: '#9CA3AF' }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: '#F9FAFB', paddingTop: '10px' }} />
        <Bar dataKey="priceChange" name="Avg. Price Change (%)">
          {sortedData.map((entry, index) => (
            <React.Fragment key={`bar-${index}`}>
              <Bar
                dataKey="priceChange"
                fill={entry.priceChange >= 0 ? '#34D399' : '#EF4444'} // Green-500 or Red-500
              >
                <LabelList dataKey="priceChange" position="right" formatter={(value: number) => `${value.toFixed(1)}%`} fill="#F9FAFB" />
              </Bar>
              {/* Dummy error bar for demonstration */}
              {entry.error && (
                <ErrorBar dataKey="error" width={4} strokeWidth={2} stroke="white" direction="x" />
              )}
            </React.Fragment>
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default EventImpactChart;