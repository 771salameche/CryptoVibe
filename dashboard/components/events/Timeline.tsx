import React from 'react';
import { EventCsvRow } from '../../utils/eventsDataProcessor'; // Import EventCsvRow
import EventCard from './EventCard';

interface TimelineProps {
    events: EventCsvRow[];
    loading: boolean;
}

const Timeline: React.FC<TimelineProps> = ({ events, loading }) => {
  if (loading) {
    return (
        <div>
            {[...Array(5)].map((_, i) => (
                <div key={i} className="bg-gray-800 p-4 rounded-lg animate-pulse mb-4 h-28"></div>
            ))}
        </div>
    );
  }

  if (events.length === 0) {
    return <div className="text-center text-gray-500 mt-8">No events found for the selected filters.</div>;
  }

  return (
    <div className="relative border-l-2 border-gray-700">
      {events.map((event, index) => (
        <EventCard key={index} event={event} />
      ))}
    </div>
  );
};

export default Timeline;