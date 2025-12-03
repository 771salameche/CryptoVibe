import React, { useState } from 'react';
import { CryptoEvent } from '../../types/events';
import { ChevronDown, ChevronUp } from 'lucide-react'; // Icons

interface EventCardProps {
  event: CryptoEvent;
}

const EventCard: React.FC<EventCardProps> = ({ event }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getEventTypeColor = (eventType: string) => {
    switch (eventType) {
      case 'LISTING': return 'bg-green-600';
      case 'HACK': return 'bg-red-600';
      case 'REGULATION': return 'bg-yellow-600';
      case 'PARTNERSHIP': return 'bg-blue-600';
      case 'UPGRADE': return 'bg-purple-600';
      default: return 'bg-gray-600';
    }
  };

  const getCryptoColor = (cryptoSymbol: string) => {
    switch (cryptoSymbol) {
      case 'BTC': return '#f7931a'; // Orange
      case 'ETH': return '#627EEA'; // Blue
      case 'SOL': return '#14F195'; // Green
      default: return '#cccccc';
    }
  };

  const sentimentColor = event.sentiment >= 0 ? 'text-green-500' : 'text-red-500';

  const formatTimestamp = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
    });
  };

  return (
    <div
      className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg cursor-pointer transition-all duration-300 hover:shadow-xl"
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div className="flex justify-between items-start mb-2">
        {/* Event Type Badge */}
        <span className={`${getEventTypeColor(event.eventType)} text-white text-xs px-3 py-1 rounded-full font-semibold`}>
          {event.eventType}
        </span>
        
        {/* Crypto Logos */}
        <div className="flex -space-x-2">
          {event.crypto.map((c, index) => (
            <div
              key={index}
              className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 border-gray-800"
              style={{ backgroundColor: getCryptoColor(c) }}
              title={c}
            >
              {c[0]}
            </div>
          ))}
        </div>
      </div>

      <h4 className="text-lg font-semibold text-white mb-2">{event.title}</h4>

      <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
        <span>{event.mentions} mentions</span>
        <span className={`font-semibold ${sentimentColor}`}>{event.sentiment.toFixed(2)}</span>
      </div>

      {isExpanded && (
        <p className="text-gray-300 text-sm mb-2">
          {event.textSnippet}
        </p>
      )}

      <div className="flex justify-between items-center text-xs text-gray-500">
        <span>{formatTimestamp(event.date)}</span>
        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </div>
    </div>
  );
};

export default EventCard;
