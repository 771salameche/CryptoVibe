
import React from 'react';
import SentimentTimeline from './components/SentimentTimeline';

function App() {
  return (
    <div className="bg-gray-900 min-h-screen flex items-center justify-center">
      <SentimentTimeline isLoading={false} />
    </div>
  );
}

export default App;
