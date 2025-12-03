import React, { useState, useEffect, useMemo } from 'react';
import { loadEventsData, calculateWordCloudsData } from '../utils/eventsDataProcessor';
import { CryptoEvent, WordData, FilterState, EventStatistics } from '../types/events';

// Import sub-components (to be created)
import FiltersPanel from './events/FiltersPanel';
import StatsCards from './events/StatsCards';
import Timeline from './events/Timeline';
import WordClouds from './events/WordClouds';

const initialFilterState: FilterState = {
  eventTypes: [],
  cryptos: [],
  startDate: null,
  endDate: null,
  sentiment: 'all',
  searchTerm: '',
};

const EventsSection: React.FC = () => {
  const [allEvents, setAllEvents] = useState<CryptoEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<CryptoEvent[]>([]);
  const [allWordData, setAllWordData] = useState<{ overall: WordData[]; positive: WordData[]; negative: WordData[] } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterState, setFilterState] = useState<FilterState>(initialFilterState);
  const [eventStats, setEventStats] = useState<EventStatistics | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const events = await loadEventsData();
      setAllEvents(events);

      // Calculate initial stats
      const stats = calculateEventStatistics(events);
      setEventStats(stats);

      // Calculate word clouds (might take time, consider memoization or pre-calculation)
      const wordClouds = await calculateWordCloudsData(events);
      setAllWordData(wordClouds);

      setLoading(false);
    };
    fetchData();
  }, []);

  // --- Filtering Logic ---
  useEffect(() => {
    let eventsToShow = allEvents;

    // Apply event type filter
    if (filterState.eventTypes.length > 0) {
      eventsToShow = eventsToShow.filter(event =>
        filterState.eventTypes.includes(event.eventType)
      );
    }

    // Apply crypto filter
    if (filterState.cryptos.length > 0) {
      eventsToShow = eventsToShow.filter(event =>
        event.crypto.some(c => filterState.cryptos.includes(c))
      );
    }

    // Apply date range filter
    if (filterState.startDate) {
      eventsToShow = eventsToShow.filter(event => new Date(event.date) >= filterState.startDate!);
    }
    if (filterState.endDate) {
      eventsToShow = eventsToShow.filter(event => new Date(event.date) <= filterState.endDate!);
    }

    // Apply sentiment filter
    if (filterState.sentiment !== 'all') {
      eventsToShow = eventsToShow.filter(event => {
        if (filterState.sentiment === 'positive') return event.sentiment > 0;
        if (filterState.sentiment === 'negative') return event.sentiment < 0;
        if (filterState.sentiment === 'neutral') return event.sentiment === 0; // Assuming 0 is neutral
        return true;
      });
    }

    // Apply search term filter
    if (filterState.searchTerm) {
      const searchTermLower = filterState.searchTerm.toLowerCase();
      eventsToShow = eventsToShow.filter(event =>
        event.title.toLowerCase().includes(searchTermLower) ||
        event.textSnippet.toLowerCase().includes(searchTermLower)
      );
    }

    setFilteredEvents(eventsToShow);

    // Recalculate stats for filtered events
    const stats = calculateEventStatistics(eventsToShow);
    setEventStats(stats);

    // Recalculate word clouds for filtered events (if needed, or only for overall)
    // For now, word clouds are based on all data. This can be optimized later.
  }, [filterState, allEvents]);

  const handleFilterChange = (newFilters: Partial<FilterState>) => {
    setFilterState(prev => ({ ...prev, ...newFilters }));
  };

  const resetFilters = () => {
    setFilterState(initialFilterState);
  };

  if (loading) {
    return <div className="text-white text-center p-8 text-xl">Loading events and word clouds...</div>;
  }

  return (
    <section className="p-8 bg-gray-900 text-white min-h-screen">
      <h2 className="text-4xl font-bold mb-8 text-center">Crypto Events & Trends</h2>

      {/* Filters Panel */}
      <FiltersPanel filterState={filterState} onFilterChange={handleFilterChange} onResetFilters={resetFilters} allEvents={allEvents} />

      {/* Event Statistics */}
      {eventStats && <StatsCards stats={eventStats} />}

      {/* Main Content: Timeline and Word Clouds */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
        {/* Timeline Section */}
        <div>
          <Timeline events={filteredEvents} />
        </div>

        {/* Word Clouds Section */}
        <div>
          {allWordData ? (
            <WordClouds wordCloudsData={allWordData} />
          ) : (
            <div className="text-center text-gray-500">Loading word clouds...</div>
          )}
        </div>
      </div>
    </section>
  );
};

// Helper function to calculate event statistics
const calculateEventStatistics = (events: CryptoEvent[]): EventStatistics => {
  const totalEvents = events.length;

  const cryptoCounts: Record<string, number> = {};
  events.forEach(event => {
    event.crypto.forEach(c => {
      cryptoCounts[c] = (cryptoCounts[c] || 0) + 1;
    });
  });
  let mostActiveCrypto: { crypto: string; count: number } | null = null;
  if (Object.keys(cryptoCounts).length > 0) {
    const sorted = Object.entries(cryptoCounts).sort(([, countA], [, countB]) => countB - countA);
    mostActiveCrypto = { crypto: sorted[0][0], count: sorted[0][1] };
  }

  const eventTypeCounts: Record<string, number> = {};
  events.forEach(event => {
    eventTypeCounts[event.eventType] = (eventTypeCounts[event.eventType] || 0) + 1;
  });
  let mostCommonEventType: { type: string; count: number } | null = null;
  if (Object.keys(eventTypeCounts).length > 0) {
    const sorted = Object.entries(eventTypeCounts).sort(([, countA], [, countB]) => countB - countA);
    mostCommonEventType = { type: sorted[0][0], count: sorted[0][1] };
  }

  return {
    totalEvents,
    mostActiveCrypto,
    mostCommonEventType,
  };
};

export default EventsSection;
