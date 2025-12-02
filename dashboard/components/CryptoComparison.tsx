import React, { useState, useEffect } from 'react';
import ComparisonCryptoCard from './ComparisonCryptoCard';
import { loadAndProcessCryptoData } from '../utils/cryptoDataProcessor';
import { CryptoData } from '../types/crypto';

// Import chart components (will be created later)
import MultiLineChart from './comparison/MultiLineChart';
import BarComparisonChart from './comparison/BarComparisonChart';
import RadarComparisonChart from './comparison/RadarComparisonChart';

type ChartType = 'sentiment' | 'volume' | 'mentions'; // Placeholder for chart types

const CryptoComparison: React.FC = () => {
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedCrypto, setSelectedCrypto] = useState<string | null>(null);
  const [selectedChartType, setSelectedChartType] = useState<ChartType>('sentiment');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const data = await loadAndProcessCryptoData();
      setCryptoData(data);
      setLoading(false);
    };
    fetchData();
  }, []);

  const handleCardClick = (symbol: string) => {
    setSelectedCrypto(symbol === selectedCrypto ? null : symbol); // Toggle selection
  };

  const getAccentColor = (symbol: string) => {
    switch (symbol) {
      case 'BTC':
        return '#f7931a'; // Orange
      case 'ETH':
        return '#627EEA'; // Blue
      case 'SOL':
        return '#14F195'; // Green
      default:
        return '#cccccc';
    }
  };

  if (loading) {
    return <div className="text-white text-center p-8">Loading crypto data...</div>;
  }

  return (
    <div className="p-8">
      <h2 className="text-3xl font-bold text-white mb-8 text-center">Crypto Comparison Dashboard</h2>

      {/* Crypto Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        {cryptoData.map((crypto) => (
          <ComparisonCryptoCard
            key={crypto.symbol}
            crypto={crypto}
            accentColor={getAccentColor(crypto.symbol)}
            onCardClick={handleCardClick}
            isSelected={selectedCrypto === crypto.symbol}
          />
        ))}
      </div>

      {/* Comparison Charts Section */}
      <div className="bg-gray-800 bg-opacity-30 p-6 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg">
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setSelectedChartType('sentiment')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              selectedChartType === 'sentiment' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-blue-500'
            }`}
          >
            Sentiment
          </button>
          <button
            onClick={() => setSelectedChartType('volume')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              selectedChartType === 'volume' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-blue-500'
            }`}
          >
            Volume
          </button>
          <button
            onClick={() => setSelectedChartType('mentions')}
            className={`px-4 py-2 rounded-lg transition-all duration-200 ${
              selectedChartType === 'mentions' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-blue-500'
            }`}
          >
            Mentions
          </button>
        </div>

        <div className="h-96 w-full"> {/* Adjust height as needed */}
          {selectedChartType === 'sentiment' && <MultiLineChart data={cryptoData} selectedCrypto={selectedCrypto} metricType={selectedChartType} />}
          {selectedChartType === 'volume' && <BarComparisonChart data={cryptoData} selectedCrypto={selectedCrypto} />}
          {selectedChartType === 'mentions' && <RadarComparisonChart data={cryptoData} selectedCrypto={selectedCrypto} />}
        </div>
      </div>
    </div>
  );
};

export default CryptoComparison;
