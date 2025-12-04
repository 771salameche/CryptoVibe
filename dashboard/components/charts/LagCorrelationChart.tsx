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
  ReferenceLine,
  ReferenceDot,
  ReferenceArea,
  Label,
} from 'recharts';

interface LagCorrelationChartProps {
  data: any[]; // Data should contain lag, and correlation for BTC, ETH, SOL
}

const LagCorrelationChart: React.FC<LagCorrelationChartProps> = ({ data }) => {

  // Function to find the optimal lag for a given crypto
  const findOptimalLag = (cryptoKey: string) => {
    let maxCorr = -Infinity;
    let optimalLag = null;
    let pValue = 1; // Placeholder for p-value

    data.forEach(entry => {
      if (entry[cryptoKey] > maxCorr) {
        maxCorr = entry[cryptoKey];
        optimalLag = entry.lag;
        // In a real scenario, you'd retrieve the actual p-value here
        // For mock, let's assume higher correlation means lower p-value
        pValue = 1 - Math.abs(entry[cryptoKey]); 
      }
    });
    return { optimalLag, maxCorr, pValue };
  };

  const btcOptimal = findOptimalLag('BTC');
  const ethOptimal = findOptimalLag('ETH');
  const solOptimal = findOptimalLag('SOL');


  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
        <XAxis dataKey="lag" name="Lag (hours)" stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }}>
          <Label value="Time Lag (hours)" offset={-5} position="insideBottom" fill="#9CA3AF" />
        </XAxis>
        <YAxis label={{ value: 'Correlation Coefficient', angle: -90, position: 'insideLeft', fill: '#F9FAFB' }} stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} domain={[-1, 1]} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #4B5563' }}
          labelStyle={{ color: '#F9FAFB' }}
        />
        <Legend wrapperStyle={{ color: '#F9FAFB', paddingTop: '10px' }} />
        <ReferenceLine y={0} stroke="#9CA3AF" strokeDasharray="3 3" />

        {/* Shaded area for statistical significance (p<0.05) - Placeholder */}
        {/* This would ideally come from actual data about significance thresholds */}
        {/* For now, assume a correlation coefficient above 0.5 or below -0.5 is "significant" */}
        <ReferenceArea y1={0.5} y2={1} stroke="none" fill="#34D399" fillOpacity={0.05} />
        <ReferenceArea y1={-1} y2={-0.5} stroke="none" fill="#EF4444" fillOpacity={0.05} />
        <ReferenceLine y={0.5} stroke="#34D399" strokeDasharray="3 3" label={{ position: 'top', value: 'p < 0.05', fill: '#34D399', fontSize: 12 }} />
        <ReferenceLine y={-0.5} stroke="#EF4444" strokeDasharray="3 3" label={{ position: 'bottom', value: 'p < 0.05', fill: '#EF4444', fontSize: 12 }} />


        <Line type="monotone" dataKey="BTC" stroke="#FBBF24" name="BTC" activeDot={{ r: 6 }} />
        <Line type="monotone" dataKey="ETH" stroke="#60A5FA" name="ETH" activeDot={{ r: 6 }} />
        <Line type="monotone" dataKey="SOL" stroke="#34D399" name="SOL" activeDot={{ r: 6 }} />

        {/* Optimal Lag Annotations */}
        {btcOptimal.optimalLag !== null && (
          <ReferenceDot x={btcOptimal.optimalLag} y={btcOptimal.maxCorr} r={5} fill="#FBBF24" stroke="#FBBF24" strokeWidth={2}>
            <Label value={`BTC optimal: ${btcOptimal.optimalLag}h (r=${btcOptimal.maxCorr.toFixed(2)})`} position="top" offset={10} fill="#FBBF24" />
          </ReferenceDot>
        )}
        {ethOptimal.optimalLag !== null && (
          <ReferenceDot x={ethOptimal.optimalLag} y={ethOptimal.maxCorr} r={5} fill="#60A5FA" stroke="#60A5FA" strokeWidth={2}>
            <Label value={`ETH optimal: ${ethOptimal.optimalLag}h (r=${ethOptimal.maxCorr.toFixed(2)})`} position="top" offset={10} fill="#60A5FA" />
          </ReferenceDot>
        )}
        {solOptimal.optimalLag !== null && (
          <ReferenceDot x={solOptimal.optimalLag} y={solOptimal.maxCorr} r={5} fill="#34D399" stroke="#34D399" strokeWidth={2}>
            <Label value={`SOL optimal: ${solOptimal.optimalLag}h (r=${solOptimal.maxCorr.toFixed(2)})`} position="top" offset={10} fill="#34D399" />
          </ReferenceDot>
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LagCorrelationChart;