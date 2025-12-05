import React, { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import LoadingIndicator from './components/LoadingIndicator';
import { Toaster } from 'react-hot-toast';
import Dashboard from './components/Dashboard';
import Layout from './components/Layout';

const PriceCorrelation = lazy(() => import('./components/PriceCorrelation'));

function App() {
  return (
    <Router>
      <Header />
      <main>
        <Layout>
          <Suspense fallback={<LoadingIndicator />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
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
