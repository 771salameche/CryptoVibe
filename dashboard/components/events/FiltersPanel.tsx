import React, { useState, useEffect } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css'; // Import styles
import { FilterState, CryptoEvent } from '../../types/events';
import { Search, XCircle } from 'lucide-react'; // Icons

interface FiltersPanelProps {
  filterState: FilterState;
  onFilterChange: (newFilters: Partial<FilterState>) => void;
  onResetFilters: () => void;
  allEvents: CryptoEvent[]; // To derive available filter options
}

const FiltersPanel: React.FC<FiltersPanelProps> = ({
  filterState,
  onFilterChange,
  onResetFilters,
  allEvents,
}) => {
  const [availableEventTypes, setAvailableEventTypes] = useState<string[]>([]);
  const [availableCryptos, setAvailableCryptos] = useState<string[]>([]);

  useEffect(() => {
    // Extract unique event types and cryptos from allEvents
    const types = Array.from(new Set(allEvents.map(event => event.eventType)));
    setAvailableEventTypes(types);

    const cryptos = Array.from(new Set(allEvents.flatMap(event => event.crypto)));
    setAvailableCryptos(cryptos);
  }, [allEvents]);

  const handleEventTypeChange = (type: string) => {
    onFilterChange({
      eventTypes: filterState.eventTypes.includes(type)
        ? filterState.eventTypes.filter(t => t !== type)
        : [...filterState.eventTypes, type],
    });
  };

  const handleCryptoChange = (crypto: string) => {
    onFilterChange({
      cryptos: filterState.cryptos.includes(crypto)
        ? filterState.cryptos.filter(c => c !== crypto)
        : [...filterState.cryptos, crypto],
    });
  };

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

  return (
    <div className="bg-gray-800 bg-opacity-30 p-6 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg mb-8">
      <h3 className="text-xl font-semibold text-white mb-4">Filter Events</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Event Type Filter */}
        <div>
          <label className="block text-gray-300 text-sm font-bold mb-2">Event Type</label>
          <div className="flex flex-wrap gap-2">
            {availableEventTypes.map(type => (
              <button
                key={type}
                className={`${getEventTypeColor(type)} text-white text-xs px-3 py-1 rounded-full transition-all duration-200
                  ${filterState.eventTypes.includes(type) ? 'opacity-100 ring-2 ring-offset-2 ring-offset-gray-800' : 'opacity-70 hover:opacity-100'}`}
                onClick={() => handleEventTypeChange(type)}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Crypto Filter */}
        <div>
          <label className="block text-gray-300 text-sm font-bold mb-2">Cryptocurrency</label>
          <div className="flex flex-wrap gap-2">
            {availableCryptos.map(crypto => (
              <button
                key={crypto}
                className={`px-3 py-1 text-xs rounded-full transition-all duration-200
                  ${filterState.cryptos.includes(crypto) ? 'bg-blue-500 text-white ring-2 ring-offset-2 ring-offset-gray-800' : 'bg-gray-700 text-gray-300 hover:bg-blue-500'}`}
                onClick={() => handleCryptoChange(crypto)}
              >
                {crypto}
              </button>
            ))}
          </div>
        </div>

        {/* Date Range Picker */}
        <div>
          <label className="block text-gray-300 text-sm font-bold mb-2">Date Range</label>
          <div className="flex gap-2">
            <DatePicker
              selected={filterState.startDate}
              onChange={(date: Date) => onFilterChange({ startDate: date })}
              selectsStart
              startDate={filterState.startDate}
              endDate={filterState.endDate}
              placeholderText="Start Date"
              className="p-2 rounded bg-gray-700 text-white w-full text-sm"
              wrapperClassName="w-1/2"
            />
            <DatePicker
              selected={filterState.endDate}
              onChange={(date: Date) => onFilterChange({ endDate: date })}
              selectsEnd
              startDate={filterState.startDate}
              endDate={filterState.endDate}
              minDate={filterState.startDate}
              placeholderText="End Date"
              className="p-2 rounded bg-gray-700 text-white w-full text-sm"
              wrapperClassName="w-1/2"
            />
          </div>
        </div>

        {/* Sentiment Filter */}
        <div>
          <label className="block text-gray-300 text-sm font-bold mb-2">Sentiment</label>
          <div className="flex gap-2">
            {['all', 'positive', 'neutral', 'negative'].map(sentiment => (
              <button
                key={sentiment}
                className={`px-3 py-1 text-xs rounded-full transition-all duration-200 capitalize
                  ${filterState.sentiment === sentiment ? 'bg-indigo-500 text-white ring-2 ring-offset-2 ring-offset-gray-800' : 'bg-gray-700 text-gray-300 hover:bg-indigo-500'}`}
                onClick={() => onFilterChange({ sentiment: sentiment as 'all' | 'positive' | 'negative' | 'neutral' })}
              >
                {sentiment}
              </button>
            ))}
          </div>
        </div>

        {/* Search Input */}
        <div className="lg:col-span-1">
          <label className="block text-gray-300 text-sm font-bold mb-2">Search</label>
          <div className="relative">
            <input
              type="text"
              placeholder="Search events..."
              className="p-2 pl-10 rounded bg-gray-700 text-white w-full text-sm"
              value={filterState.searchTerm}
              onChange={(e) => onFilterChange({ searchTerm: e.target.value })}
            />
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
          </div>
        </div>

        {/* Reset Filters Button */}
        <div className="flex items-end justify-end">
          <button
            onClick={onResetFilters}
            className="flex items-center gap-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all duration-200 text-sm"
          >
            <XCircle size={16} /> Reset Filters
          </button>
        </div>
      </div>
    </div>
  );
};

export default FiltersPanel;
