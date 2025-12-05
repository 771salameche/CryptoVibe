import React, { useState, useEffect, lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import SentimentTimeline from './components/SentimentTimeline';
import LoadingIndicator from './components/LoadingIndicator';
import { Toaster } from 'react-hot-toast';

const EventsSection = lazy(() => import('./components/EventsSection'));
const PriceCorrelation = lazy(() => import('./components/PriceCorrelation'));

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
      // To test the error state, uncomment the following line
      // setIsError(true);
    }, 2000); // Simulate a 2-second loading time

    return () => clearTimeout(timer);
  }, []);

  return (
    <Router>
      {isLoading && <LoadingIndicator />}
      <div className={`bg-gray-900 min-h-screen text-white transition-opacity duration-500 ${isLoading ? 'opacity-0' : 'opacity-100'}`}>
        <Header />
        <div className="p-8">
          <Suspense fallback={<LoadingIndicator />}>
            <Routes>
              <Route path="/" element={
                <>
                  <SentimentTimeline isLoading={isLoading} isError={isError} />
                  <EventsSection />
                </>
              } />
              <Route path="/correlation" element={<PriceCorrelation />} />
            </Routes>
          </Suspense>
        </div>
      </div>
      <Toaster />
    </Router>
  );
}

export default App;
