import React, { lazy, Suspense, useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import LoadingIndicator from './components/LoadingIndicator';
import toast, { Toaster } from 'react-hot-toast'; // Import toast
import Dashboard from './components/Dashboard';
import Layout from './components/Layout';
import Footer from './components/Footer';
import { useWebSocket, ProcessedMessage } from './src/hooks/useWebSocket';

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
      toast.success('WebSocket Connected!', { id: 'websocket-status' });
    } else if (!isConnected && !error) {
      // Only show disconnected if it was previously connected or trying to connect
      console.log('WebSocket Disconnected.');
      toast.error('WebSocket Disconnected.', { id: 'websocket-status' });
    }
    if (error) {
      console.error('WebSocket encountered an error:', error);
      toast.error('WebSocket Error: ' + error.message, { id: 'websocket-status' });
    }
  }, [isConnected, error]);

  return (
    <Router>
      <div className="relative min-h-screen overflow-hidden">
        <div className="app-grid-overlay" aria-hidden />
        <Header />
        <main className="relative z-10">
          <Layout>
            <Suspense fallback={<LoadingIndicator />}>
              <Routes>
                <Route path="/" element={<Dashboard realtimeData={realtimeData} isWsConnected={isConnected} />} />
                <Route path="/correlation" element={<PriceCorrelation />} />
              </Routes>
            </Suspense>
          </Layout>
          <Footer />
        </main>
      </div>
      <Toaster />
    </Router>
  );
}

export default App;
