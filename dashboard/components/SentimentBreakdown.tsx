import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const MOCK_DATA = [
  { source: 'Twitter', sentiment: 0.65, color: '#1DA1F2' },
  { source: 'Reddit', sentiment: 0.55, color: '#FF4500' },
  { source: 'News', sentiment: 0.45, color: '#FFD700' },
];

const SentimentBreakdown: React.FC = () => {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={MOCK_DATA} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 20 }}>
        <XAxis type="number" domain={[-1, 1]} hide />
        <YAxis type="category" dataKey="source" hide />
        <Tooltip
          cursor={{ fill: 'var(--border)' }}
          contentStyle={{
            backgroundColor: 'var(--card)',
            backdropFilter: 'blur(10px)',
            border: '1px solid var(--border)',
            color: 'var(--card-foreground)',
          }}
        />
        <Bar dataKey="sentiment" radius={[4, 4, 4, 4]}>
          {MOCK_DATA.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default SentimentBreakdown;
