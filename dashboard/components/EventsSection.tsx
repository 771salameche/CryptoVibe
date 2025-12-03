import React, { useState, useEffect } from 'react';
import StatsCards from './events/StatsCards';
import FiltersPanel from './events/FiltersPanel';
import Timeline from './events/Timeline';
import WordClouds from './events/WordClouds';
import { loadEventsData } from '../utils/dataLoader';
import { EventCsvRow } from '../../utils/eventsDataProcessor'; // Import EventCsvRow

const EventsSection: React.FC = () => {
  const [allEvents, setAllEvents] = useState<EventCsvRow[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<EventCsvRow[]>([]);
  const [loading, setLoading] = useState(true);

  // Filter states
  // const [searchTerm, setSearchTerm] = useState(''); // Removed as EventCsvRow doesn't have title/text
  const [selectedEventTypes, setSelectedEventTypes] = useState<string[]>([]);
  const [selectedCryptos, setSelectedCryptos] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState<[Date | null, Date | null]>([null, null]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const data = await loadEventsData();
      setAllEvents(data);
      setFilteredEvents(data);
      setLoading(false);
    };
    fetchData();
  }, []);

  useEffect(() => {
    let events = allEvents;

    // SearchTerm filtering removed as EventCsvRow doesn't have title/text
    /*
    if (searchTerm) {
        events = events.filter(event =>
            event.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            event.text.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }
    */

    if (selectedEventTypes.length > 0) {
        events = events.filter(event => selectedEventTypes.includes(event.event_type));
    }

    if (selectedCryptos.length > 0) {
        events = events.filter(event => selectedCryptos.includes(event.crypto));
    }
    
    if (dateRange[0] || dateRange[1]) {
        const startDate = dateRange[0] ? new Date(dateRange[0]).setHours(0, 0, 0, 0) : null;
        const endDate = dateRange[1] ? new Date(dateRange[1]).setHours(23, 59, 59, 999) : null;

        events = events.filter(event => {
            const eventDate = new Date(event.date).getTime();
            let matchesStartDate = true;
            let matchesEndDate = true;

            if (startDate) {
                matchesStartDate = eventDate >= startDate;
            }
            if (endDate) {
                matchesEndDate = eventDate <= endDate;
            }
            return matchesStartDate && matchesEndDate;
        });
    }

    setFilteredEvents(events);
  }, [selectedEventTypes, selectedCryptos, dateRange, allEvents]); // searchTerm removed from dependency array


  return (
    <div className="bg-gray-900 text-white p-6 rounded-lg">
      <header className="mb-6">
        <h2 className="text-3xl font-bold mb-2">Events & Trends</h2>
        <p className="text-gray-400">Track major cryptocurrency events and market-moving keywords.</p>
      </header>

      <StatsCards events={allEvents} />
      <FiltersPanel 
        searchTerm={''} // Pass empty string or handle differently if needed
        setSearchTerm={() => {}} // No-op for setSearchTerm
        selectedEventTypes={selectedEventTypes}
        setSelectedEventTypes={setSelectedEventTypes}
        selectedCryptos={selectedCryptos}
        setSelectedCryptos={setSelectedCryptos}
        setDateRange={setDateRange}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
        <div className="col-span-1">
          <Timeline events={filteredEvents} loading={loading} />
        </div>
        <div className="col-span-1">
          <WordClouds events={filteredEvents} />
        </div>
      </div>
    </div>
  );
};

export default EventsSection;