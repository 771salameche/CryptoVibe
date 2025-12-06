import React, { lazy, Suspense, useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import LoadingIndicator from './components/LoadingIndicator';
import { Toaster } from 'react-hot-toast';
import Dashboard from './components/Dashboard';
import Layout from './components/Layout';
import { useWebSocket, ProcessedMessage } from './src/hooks/useWebSocket'; // Assuming src/hooks is correct path

const PriceCorrelation = lazy(() => import('./components/PriceCorrelation'));

function App() {
  const [realtimeData, setRealtimeData] = useState<ProcessedMessage[]>([]);

  // Callback to handle new messages from the WebSocket
  const handleNewMessage = useCallback((message: ProcessedMessage) => {
    setRealtimeData((prevData) => {
      // Keep only a certain number of messages to prevent infinite growth
      const MAX_REALTIME_MESSAGES = 50; 
      const newData = [...prevData, message];
      return newData.slice(Math.max(newData.length - MAX_REALTIME_MESSAGES, 0));
    });
  }, []);

  // Establish WebSocket connection
  const { isConnected, error } = useWebSocket('ws://localhost:8001', handleNewMessage);

  // Optional: Log WebSocket status
  useEffect(() => {
    if (isConnected) {
      console.log('WebSocket is live!');
    }
    if (error) {
      console.error('WebSocket encountered an error:', error);
    }
  }, [isConnected, error]);

  return (
    <Router>
      <Header />
      <main>
        <Layout>
          <Suspense fallback={<LoadingIndicator />}>
            <Routes>
              {/* Pass realtimeData to the Dashboard component */}
              <Route path="/" element={<Dashboard realtimeData={realtimeData} isWsConnected={isConnected} />} />
              <Route path="/correlation" element={<PriceCorrelation />} />
            </Routes>
          </Suspense>
        </Layout>
      </main>
      <Toaster />
    </Router>
  );
}

export default App;