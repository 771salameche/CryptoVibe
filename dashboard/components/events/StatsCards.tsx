import React from 'react';
import { EventStatistics } from '../../types/events';

interface StatsCardsProps {
  stats: EventStatistics;
}

const StatsCards: React.FC<StatsCardsProps> = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {/* Total Events Detected */}
      <div className="bg-gray-800 bg-opacity-30 p-6 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg text-center">
        <p className="text-sm text-gray-400 mb-1">Total Events Detected</p>
        <p className="text-3xl font-bold text-white">{stats.totalEvents} events</p>
      </div>

      {/* Most Active Crypto */}
      <div className="bg-gray-800 bg-opacity-30 p-6 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg text-center">
        <p className="text-sm text-gray-400 mb-1">Most Active Crypto</p>
        {stats.mostActiveCrypto ? (
          <p className="text-3xl font-bold text-white">{stats.mostActiveCrypto.crypto} ({stats.mostActiveCrypto.count} events)</p>
        ) : (
          <p className="text-xl text-gray-500">N/A</p>
        )}
      </div>

      {/* Most Common Event Type */}
      <div className="bg-gray-800 bg-opacity-30 p-6 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg text-center">
        <p className="text-sm text-gray-400 mb-1">Most Common Event Type</p>
        {stats.mostCommonEventType ? (
          <p className="text-3xl font-bold text-white">{stats.mostCommonEventType.type} ({stats.mostCommonEventType.count})</p>
        ) : (
          <p className="text-xl text-gray-500">N/A</p>
        )}
      </div>
    </div>
  );
};

export default StatsCards;
