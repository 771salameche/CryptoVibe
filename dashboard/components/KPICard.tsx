import React, { ReactNode } from 'react';

interface KPICardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'improving' | 'declining' | 'stable';
  icon: ReactNode;
}

const KPICard: React.FC<KPICardProps> = ({ title, value, change, trend, icon }) => {
  const trendColor = trend === 'improving' ? 'text-status-positive' : trend === 'declining' ? 'text-status-negative' : 'text-status-neutral';

  return (
    <div className="group bg-bg-surface backdrop-blur-lg border border-border-default p-6 rounded-lg shadow-card transition-all duration-300 ease-in-out hover:scale-105 hover:shadow-xl hover:border-accent-primary">
      <div className="flex justify-between items-start">
        <p className="text-fg-text-muted text-sm">{title}</p>
        <div className="transition-transform duration-300 ease-in-out group-hover:scale-110">
          {icon}
        </div>
      </div>
      <p className="text-3xl font-bold text-fg-text">{value}</p>
      {change && (
        <p className={`text-sm ${trendColor}`}>
          {change}
        </p>
      )}
    </div>
  );
};

export default KPICard;
