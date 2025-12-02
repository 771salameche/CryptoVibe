
import React from 'react';
import SentimentTimeline from './components/SentimentTimeline';
import CryptoComparison from './components/CryptoComparison'; // New import

function App() {
  return (
    <div className="bg-gray-900 min-h-screen">
      <SentimentTimeline isLoading={false} />
    </div>
  );
}

export default App;
