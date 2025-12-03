import React from 'react';
import { EventType } from '../types/events';

const eventTypes: EventType[] = ['LISTING', 'HACK', 'REGULATION', 'PARTNERSHIP', 'UPGRADE'];
const cryptos = ['BTC', 'ETH', 'SOL'];

interface FiltersPanelProps {
    searchTerm: string;
    setSearchTerm: (searchTerm: string) => void;
    selectedEventTypes: string[];
    setSelectedEventTypes: (eventTypes: string[]) => void;
    selectedCryptos: string[];
    setSelectedCryptos: (cryptos: string[]) => void;
    setDateRange: (dateRange: [Date | null, Date | null]) => void;
}

const FiltersPanel: React.FC<FiltersPanelProps> = ({
    searchTerm,
    setSearchTerm,
    selectedEventTypes,
    setSelectedEventTypes,
    selectedCryptos,
    setSelectedCryptos,
    setDateRange
}) => {
    
    const handleEventTypeChange = (eventType: string) => {
        const newSelection = selectedEventTypes.includes(eventType)
            ? selectedEventTypes.filter(t => t !== eventType)
            : [...selectedEventTypes, eventType];
        setSelectedEventTypes(newSelection);
    };

    const handleCryptoChange = (crypto: string) => {
        const newSelection = selectedCryptos.includes(crypto)
            ? selectedCryptos.filter(c => c !== crypto)
            : [...selectedCryptos, crypto];
        setSelectedCryptos(newSelection);
    };

    const handleReset = () => {
        setSearchTerm('');
        setSelectedEventTypes([]);
        setSelectedCryptos([]);
        setDateRange([null, null]);
    }

  return (
    <div className="bg-gray-800 p-4 rounded-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            {/* Search Input */}
            <div className="col-span-1 lg:col-span-1">
                <label htmlFor="search" className="block text-sm font-medium text-gray-400 mb-1">Search Events</label>
                <input
                    type="text"
                    id="search"
                    placeholder="Search events..."
                    className="w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>

            {/* Event Type Filter */}
            <div className="col-span-1 lg:col-span-2">
                <label className="block text-sm font-medium text-gray-400 mb-1">Event Type</label>
                <div className="flex items-center space-x-2 mt-2 flex-wrap">
                    {eventTypes.map(type => (
                        <button 
                            key={type}
                            onClick={() => handleEventTypeChange(type)}
                            className={`px-3 py-1 text-sm font-medium rounded-full ${selectedEventTypes.includes(type) ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                        >
                            {type}
                        </button>
                    ))}
                </div>
            </div>

            {/* Crypto Filter */}
            <div className="col-span-1 lg:col-span-1">
                <label className="block text-sm font-medium text-gray-400 mb-1">Cryptocurrency</label>
                <div className="flex items-center space-x-4 mt-2">
                    {cryptos.map(crypto => (
                        <label key={crypto} className="flex items-center text-white">
                            <input 
                                type="checkbox" 
                                className="h-4 w-4 bg-gray-700 border-gray-600 rounded text-indigo-600 focus:ring-indigo-500"
                                checked={selectedCryptos.includes(crypto)}
                                onChange={() => handleCryptoChange(crypto)}
                            />
                            <span className="ml-2">{crypto}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Date Range Filter */}
            <div className="col-span-1 lg:col-span-1">
                <label htmlFor="start-date" className="block text-sm font-medium text-gray-400 mb-1">Start Date</label>
                <input 
                    type="date" 
                    id="start-date" 
                    className="w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    onChange={(e) => setDateRange([e.target.valueAsDate, null])} // Simple implementation for now
                />
            </div>
            <div className="col-span-1 lg:col-span-1">
                <label htmlFor="end-date" className="block text-sm font-medium text-gray-400 mb-1">End Date</label>
                <input 
                    type="date" 
                    id="end-date" 
                    className="w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    onChange={(e) => setDateRange([null, e.target.valueAsDate])} // Simple implementation for now
                />
            </div>

            {/* Reset Button */}
            <div className="col-span-1 lg:col-span-1 flex items-end">
                <button 
                    onClick={handleReset}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                    Reset
                </button>
            </div>
        </div>
    </div>
  );
};

export default FiltersPanel;
