import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import SentimentTimeline from './components/SentimentTimeline';
import EventsSection from './components/EventsSection';
import PriceCorrelation from './components/PriceCorrelation';

function App() {
  return (
    <Router>
      <div className="bg-gray-900 min-h-screen text-white">
        <Header />
        <div className="p-8">
          <Routes>
            <Route path="/" element={
              <>
                <SentimentTimeline isLoading={false} />
                <EventsSection />
              </>
            } />
            <Route path="/correlation" element={<PriceCorrelation />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
