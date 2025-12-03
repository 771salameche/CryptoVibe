import React from 'react';
import { EventCsvRow } from '../../utils/eventsDataProcessor'; // Import EventCsvRow

interface StatsCardsProps {
    events: EventCsvRow[];
}

const StatsCards: React.FC<StatsCardsProps> = ({ events }) => {
  const totalEvents = events.length;

  const mostActiveCrypto = events.reduce((acc, event) => {
    acc[event.crypto] = (acc[event.crypto] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const mostActiveCryptoName = Object.keys(mostActiveCrypto).reduce((a, b) => mostActiveCrypto[a] > mostActiveCrypto[b] ? a : b, '');

  const mostCommonEventType = events.reduce((acc, event) => {
    acc[event.event_type] = (acc[event.event_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const mostCommonEventTypeName = Object.keys(mostCommonEventType).reduce((a, b) => mostCommonEventType[a] > mostCommonEventType[b] ? a : b, '');

  if (events.length === 0) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-800 p-4 rounded-lg animate-pulse h-24"></div>
            <div className="bg-gray-800 p-4 rounded-lg animate-pulse h-24"></div>
            <div className="bg-gray-800 p-4 rounded-lg animate-pulse h-24"></div>
        </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div className="bg-gray-800 p-4 rounded-lg">
        <h3 className="text-sm font-medium text-gray-400">Total Events</h3>
        <p className="text-2xl font-bold">{totalEvents}</p>
      </div>
      <div className="bg-gray-800 p-4 rounded-lg">
        <h3 className="text-sm font-medium text-gray-400">Most Active Crypto</h3>
        <p className="text-2xl font-bold">{mostActiveCryptoName} ({mostActiveCrypto[mostActiveCryptoName] || 0})</p>
      </div>
      <div className="bg-gray-800 p-4 rounded-lg">
        <h3 className="text-sm font-medium text-gray-400">Most Common Event</h3>
        <p className="text-2xl font-bold">{mostCommonEventTypeName} ({mostCommonEventType[mostCommonEventTypeName] || 0})</p>
      </div>
    </div>
  );
};

export default StatsCards;
