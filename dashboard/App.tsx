
import React from 'react';
import SentimentTimeline from './components/SentimentTimeline';
// import CryptoComparison from './components/CryptoComparison';
import EventsSection from './components/EventsSection'; // New import

function App() {
  return (
    <div className="bg-gray-900 min-h-screen text-white text-2xl p-8">
      <SentimentTimeline isLoading={false} />
      {/* Hello World! The app is rendering. */}
      {/* <CryptoComparison /> */}
      <EventsSection />
    </div>
  );
}

export default App;
