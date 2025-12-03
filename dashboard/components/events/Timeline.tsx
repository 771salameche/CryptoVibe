import React, { useState } from 'react';
import { VerticalTimeline, VerticalTimelineElement } from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css'; // Import default styles
import { CryptoEvent } from '../../types/events';
import EventCard from './EventCard';

interface TimelineProps {
  events: CryptoEvent[];
}

const Timeline: React.FC<TimelineProps> = ({ events }) => {
  const [visibleEventsCount, setVisibleEventsCount] = useState(10); // Show 10 initially

  const loadMoreEvents = () => {
    setVisibleEventsCount(prevCount => prevCount + 10); // Load 10 more events
  };

  return (
    <div className="relative pt-4"> {/* Added padding top to account for potential fixed headers */}
      {events.length === 0 ? (
        <div className="text-center text-gray-500 text-lg p-4">No events found for the selected filters.</div>
      ) : (
        <VerticalTimeline lineColor={'#444'} animate={false}>
          {events.slice(0, visibleEventsCount).map((event, index) => (
            <VerticalTimelineElement
              key={index}
              className="vertical-timeline-element--work"
              contentStyle={{ background: 'transparent', boxShadow: 'none', padding: '0' }}
              contentArrowStyle={{ borderRight: '7px solid #444' }}
              date={new Date(event.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
              })}
              iconStyle={{ background: '#666', color: '#fff' }}
              icon={<div className="flex items-center justify-center h-full w-full text-xs">{event.crypto[0]}</div>}
            >
              <EventCard event={event} />
            </VerticalTimelineElement>
          ))}
        </VerticalTimeline>
      )}

      {events.length > visibleEventsCount && (
        <div className="text-center mt-4">
          <button
            onClick={loadMoreEvents}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200"
          >
            Load More Events
          </button>
        </div>
      )}
    </div>
  );
};

export default Timeline;
