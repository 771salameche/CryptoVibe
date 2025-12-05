import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Label,
  Line,
} from 'recharts';

interface ScatterPlotProps {
  data: { sentiment: number; priceChange: number; }[];
  crypto: string;
}

// Helper function for linear regression
const calculateLinearRegression = (data: { sentiment: number; priceChange: number; }[]) => {
  const n = data.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  data.forEach(point => {
    sumX += point.sentiment;
    sumY += point.priceChange;
    sumXY += point.sentiment * point.priceChange;
    sumXX += point.sentiment * point.sentiment;
  });

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
};

// Helper function to calculate R-squared
const calculateRSquared = (data: { sentiment: number; priceChange: number; }[], slope: number, intercept: number) => {
  const n = data.length;
  const meanY = data.reduce((sum, point) => sum + point.priceChange, 0) / n;
  let totalSumSquares = 0;
  let residualSumSquares = 0;

  data.forEach(point => {
    const predictedY = slope * point.sentiment + intercept;
    totalSumSquares += (point.priceChange - meanY) * (point.priceChange - meanY);
    residualSumSquares += (point.priceChange - predictedY) * (point.priceChange - predictedY);
  });

  return 1 - (residualSumSquares / totalSumSquares);
};

const ScatterPlot: React.FC<ScatterPlotProps> = ({ data, crypto }) => {
  const { slope, intercept } = useMemo(() => calculateLinearRegression(data), [data]);
  const rSquared = useMemo(() => calculateRSquared(data, slope, intercept), [data, slope, intercept]);

  const regressionLineData = useMemo(() => {
    if (data.length === 0) return [];
    const minSentiment = Math.min(...data.map(d => d.sentiment));
    const maxSentiment = Math.max(...data.map(d => d.sentiment));
    return [
      { sentiment: minSentiment, priceChange: slope * minSentiment + intercept },
      { sentiment: maxSentiment, priceChange: slope * maxSentiment + intercept },
    ];
  }, [data, slope, intercept]);

  return (
    <div>
      <h3 className="text-xl font-bold mb-4 text-white text-center">{crypto} - Sentiment vs. Price Change</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
          <XAxis type="number" dataKey="sentiment" name="Sentiment" unit="" stroke="#9CA3AF">
            <Label value="Daily Sentiment Score" offset={0} position="bottom" fill="#9CA3AF" />
          </XAxis>
          <YAxis type="number" dataKey="priceChange" name="Price Change" unit="%" stroke="#9CA3AF">
            <Label value="Daily Price Change %" offset={0} position="left" angle={-90} fill="#9CA3AF" />
          </YAxis>
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #4B5563' }} />
          <Legend wrapperStyle={{ color: '#F9FAFB', paddingTop: '10px' }} />

          {/* Quadrant Labels */}
          <ReferenceLine y={0} stroke="#9CA3AF" strokeDasharray="3 3" />
          <ReferenceLine x={0} stroke="#9CA3AF" strokeDasharray="3 3" />
          <Label value="High sentiment, Price up ✓" x="90%" y="10%" position="insideTopRight" fill="#34D399" />
          <Label value="Low sentiment, Price up" x="10%" y="10%" position="insideTopLeft" fill="#FDE68A" />
          <Label value="High sentiment, Price down" x="90%" y="90%" position="insideBottomRight" fill="#FDE68A" />
          <Label value="Low sentiment, Price down ✓" x="10%" y="90%" position="insideBottomLeft" fill="#EF4444" />


          <Scatter name="Days" data={data} fill="#8884d8" >
            {data.map((entry, index) => (
              <Scatter key={`cell-${index}`} fill={entry.priceChange >= 0 ? '#34D399' : '#EF4444'} />
            ))}
          </Scatter>
          {/* Regression Line */}
          <Line
            dataKey="priceChange"
            data={regressionLineData}
            dot={false}
            activeDot={false}
            stroke="#FBBF24" // Amber-400
            strokeWidth={2}
            isAnimationActive={false}
            legendType="none"
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="text-center text-white mt-2">
        Regression: y = {slope.toFixed(2)}x + {intercept.toFixed(2)} (R² = {rSquared.toFixed(2)})
      </div>
    </div>
  );
};

export default ScatterPlot;