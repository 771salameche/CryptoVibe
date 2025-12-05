
import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';
import { format, subDays } from 'date-fns';
import toast from 'react-hot-toast'; // Import react-hot-toast

// To export as PNG, a library like html-to-image is needed.
// import { toPng } from 'html-to-image';

// --- TYPE DEFINITIONS ---
type SentimentDataPoint = {
  date: string;
  BTC?: number;
  ETH?: number;
  SOL?: number;
  BTC_posts?: number;
  ETH_posts?: number;
  SOL_posts?: number;
};

type SentimentTimelineProps = {
  data?: SentimentDataPoint[];
  isLoading: boolean;
  isError: boolean;
};

type Crypto = 'BTC' | 'ETH' | 'SOL';

const MOCK_DATA: SentimentDataPoint[] = Array.from({ length: 90 }, (_, i) => {
    const date = format(subDays(new Date(), 89 - i), 'yyyy-MM-dd');
    return {
        date,
        BTC: Math.random() * 2 - 1,
        ETH: Math.random() * 2 - 1,
        SOL: Math.random() * 2 - 1,
        BTC_posts: Math.floor(Math.random() * 1000),
        ETH_posts: Math.floor(Math.random() * 800),
        SOL_posts: Math.floor(Math.random() * 600),
    };
});

const CRYPTO_CONFIG: { [key in Crypto]: { color: string } } = {
  BTC: { color: '#F7931A' },
  ETH: { color: '#627EEA' },
  SOL: { color: '#14F195' },
};

// --- HELPER FUNCTIONS ---
const calculateMovingAverage = (data: SentimentDataPoint[], crypto: Crypto, period: number): (number | null)[] => {
    const key = crypto as keyof SentimentDataPoint;
    const movingAverages: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            movingAverages.push(null);
        } else {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += (data[i - j][key] as number) || 0;
            }
            movingAverages.push(sum / period);
        }
    }
    return movingAverages;
};

const downloadCSV = (data: SentimentDataPoint[], selectedCryptos: Crypto[]) => {
    const headers = ['date', ...selectedCryptos];
    const csvContent = [
        headers.join(','),
        ...data.map(item => headers.map(header => (item as any)[header]).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    if (link.href) {
        URL.revokeObjectURL(link.href);
    }
    link.href = URL.createObjectURL(blob);
    link.download = 'sentiment_data.csv';
    link.click();
    toast.success('CSV exported successfully!');
};

// --- SUB-COMPONENTS ---
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
        <p className="font-bold text-gray-800 dark:text-gray-100">{label}</p>
        {payload.map((p: any, index: number) => (
          !p.dataKey.includes('_ma') && (
            <div key={`${p.dataKey}-${index}`} style={{ color: p.color }} className="flex justify-between items-center space-x-4">
              <span>{p.name}:</span>
              <span className="font-bold">{p.value.toFixed(3)}</span>
            </div>
          )
        ))}
      </div>
    );
  }
  return null;
};


const MetricCard = ({ title, value, change, trend, info }: { title: string, value: string, change?: string, trend?: 'improving' | 'declining' | 'stable', info?: string }) => (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md flex-1 transition-transform duration-300 ease-in-out hover:-translate-y-1 hover:shadow-lg">
        <div className="flex items-center">
          <p className="text-gray-500 dark:text-gray-400 text-sm">{title}</p>
          {info && <div className="ml-2"><InfoIcon id={title} content={info} /></div>}
        </div>
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
        {change && (
            <p className={`text-sm ${trend === 'improving' ? 'text-green-500' : trend === 'declining' ? 'text-red-500' : 'text-gray-500'}`}>
                {change}
            </p>
        )}
    </div>
);

const SkeletonLoader = () => (
    <div className="bg-gray-50 dark:bg-gray-900 p-4 sm:p-6 rounded-2xl w-full">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="h-24 rounded-xl bg-gray-200 dark:bg-gray-700 shimmer"></div>
            <div className="h-24 rounded-xl bg-gray-200 dark:bg-gray-700 shimmer"></div>
            <div className="h-24 rounded-xl bg-gray-200 dark:bg-gray-700 shimmer"></div>
        </div>
        <div className="h-[500px] rounded-lg bg-gray-200 dark:bg-gray-700 shimmer"></div>
    </div>
);


import InfoIcon from './InfoIcon';
import ErrorBoundary from './ErrorBoundary';
import ErrorState from './ErrorState';
import EmptyState from './EmptyState';

const SentimentTimeline: React.FC<SentimentTimelineProps> = ({ data = MOCK_DATA, isLoading, isError }) => {
  const [selectedCryptos, setSelectedCryptos] = useState<Crypto[]>(['BTC', 'ETH', 'SOL']);
  const [dateRange, setDateRange] = useState<number>(30); // 7, 30, 90 days
  const [showMA, setShowMA] = useState<boolean>(false);
  const [isAutoRefreshEnabled, setIsAutoRefreshEnabled] = useState<boolean>(false);
  const chartRef = useRef<HTMLDivElement>(null);

  // Simulated data fetching function
  const fetchData = () => {
    toast.promise(
      new Promise((resolve, reject) => {
        setTimeout(() => {
          if (Math.random() > 0.1) { // 90% success rate
            console.log('Fetching new sentiment data...');
            resolve('Data refreshed successfully!');
          } else {
            reject('Failed to refresh data.');
          }
        }, 1000);
      }),
      {
        loading: 'Refreshing data...',
        success: (message) => message,
        error: (error) => error,
      }
    );
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoRefreshEnabled) {
      interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
      toast.success('Auto-refresh enabled (every 30s)');
    } else {
      toast.error('Auto-refresh disabled');
    }
    return () => clearInterval(interval);
  }, [isAutoRefreshEnabled]);

  const handleRetry = () => {
    console.log('Retrying...');
    fetchData(); // Attempt to refetch data on retry
  };

  const handleManualRefresh = () => {
    fetchData(); // Trigger manual data fetch
  };

  const filteredData = useMemo(() => {
    const endDate = new Date();
    const startDate = subDays(endDate, dateRange - 1);
    return data
        .filter(d => {
            const date = new Date(d.date);
            return date >= startDate && date <= endDate;
        })
        .map(d => {
            const point: SentimentDataPoint = { date: d.date };
            selectedCryptos.forEach(crypto => {
                point[crypto] = d[crypto];
                (point as any)[`${crypto}_posts`] = d[`${crypto}_posts` as keyof SentimentDataPoint];
            });
            return point;
        });
  }, [data, selectedCryptos, dateRange]);

  const processedData = useMemo(() => {
    if (!showMA) return filteredData;
    return filteredData.map((d, i) => {
        const point = { ...d };
        selectedCryptos.forEach(crypto => {
            const ma = calculateMovingAverage(filteredData, crypto, 7);
            (point as any)[`${crypto}_ma`] = ma[i];
        });
        return point;
    });
  }, [filteredData, showMA, selectedCryptos]);

  const metrics = useMemo(() => {
    if (filteredData.length === 0) {
        return { current: 'N/A', average: 'N/A', trend: 'stable' as 'stable', trendDescription: 'No data' };
    }
    const latest = filteredData[filteredData.length - 1];
    const current = selectedCryptos.reduce((acc, c) => acc + (latest[c] || 0), 0) / selectedCryptos.length;

    const average = filteredData.reduce((acc, day) => {
        return acc + selectedCryptos.reduce((dayAcc, c) => dayAcc + (day[c] || 0), 0);
    }, 0) / (filteredData.length * selectedCryptos.length);

    const firstWeekAvg = filteredData.slice(0, 7).reduce((acc, day) => {
        return acc + selectedCryptos.reduce((dayAcc, c) => dayAcc + (day[c] || 0), 0);
    }, 0) / (7 * selectedCryptos.length);

    const lastWeekAvg = filteredData.slice(-7).reduce((acc, day) => {
        return acc + selectedCryptos.reduce((dayAcc, c) => dayAcc + (day[c] || 0), 0);
    }, 0) / (7 * selectedCryptos.length);

    let trend: 'improving' | 'declining' | 'stable' = 'stable';
    if (lastWeekAvg > firstWeekAvg) trend = 'improving';
    if (lastWeekAvg < firstWeekAvg) trend = 'declining';

    return {
        current: current.toFixed(3),
        average: average.toFixed(3),
        trend,
        trendDescription: `${trend.charAt(0).toUpperCase() + trend.slice(1)}`
    };
  }, [filteredData, selectedCryptos]);

  const handleExportPNG = () => {
    if (chartRef.current) {
      toast.promise(
        import('html-to-image').then(({ toPng }) => {
          return toPng(chartRef.current as HTMLElement);
        }).then((dataUrl) => {
          const link = document.createElement('a');
          link.download = 'sentiment-chart.png';
          link.href = dataUrl;
          link.click();
          return 'PNG exported successfully!';
        }),
        {
          loading: 'Exporting PNG...',
          success: (message) => message,
          error: 'Failed to export PNG.',
        }
      );
    } else {
        toast.error('Chart reference not found for PNG export.');
    }
  };

  if (isLoading) {
    return <SkeletonLoader />;
  }

  if (isError) {
    return <ErrorState message="Failed to load sentiment data." onRetry={handleRetry} />;
  }

  if (!data || data.length === 0) {
    return <EmptyState message="No data available for the selected period." />;
  }

  return (
    <ErrorBoundary fallback={<ErrorState message="An unexpected error occurred." onRetry={handleRetry} />}>
    <div className="bg-gray-50 dark:bg-gray-900 p-4 sm:p-6 rounded-2xl w-full">

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-6">
          <MetricCard title="Current Avg. Sentiment" value={metrics.current} info="The average sentiment score of the selected cryptocurrencies for the most recent day. Ranges from -1 (very negative) to +1 (very positive)." />
          <MetricCard title="Average Sentiment" value={metrics.average} info="The average sentiment score of the selected cryptocurrencies over the selected period. Ranges from -1 (very negative) to +1 (very positive)." />
          <MetricCard title="Trend" value={metrics.trendDescription} trend={metrics.trend} info="The direction of the sentiment change over the selected period." />
      </div>

      {/* Filters and Export */}
      <div className="flex flex-col sm:flex-row flex-wrap items-center justify-between mb-4 p-4 bg-white dark:bg-gray-800 rounded-lg">
        <div className="flex items-center space-x-4 mb-2 sm:mb-0">
          <span className="font-semibold text-gray-700 dark:text-gray-300">Cryptos:</span>
          {Object.keys(CRYPTO_CONFIG).map((crypto) => (
            <label key={crypto} className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={selectedCryptos.includes(crypto as Crypto)} onChange={() => { setSelectedCryptos(prev => prev.includes(crypto as Crypto) ? prev.filter(c => c !== crypto) : [...prev, crypto as Crypto]); }} className={`form-checkbox h-5 w-5 rounded focus:ring-2 focus:ring-offset-2`} style={{accentColor: CRYPTO_CONFIG[crypto as Crypto].color}} aria-label={`Select ${crypto}`} />
              <span className="text-gray-800 dark:text-gray-200">{crypto}</span>
            </label>
          ))}
        </div>

        <div className="flex items-center space-x-2 mb-2 sm:mb-0">
            {[7, 30, 90].map(days => (
                <button key={days} onClick={() => setDateRange(days)} className={`px-4 py-2 rounded-md text-sm font-medium transition-transform duration-150 ease-in-out active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2 ${dateRange === days ? 'bg-blue-600 text-white focus:ring-blue-500' : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:ring-gray-500'}`} aria-label={`Show data for ${days} days`}>
                    {days} Days
                </button>
            ))}
        </div>
        
        <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer">
                <input type="checkbox" checked={showMA} onChange={() => setShowMA(!showMA)} className="form-checkbox h-5 w-5 rounded text-blue-600 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500" aria-label="Show 7-Day Moving Average"/>
                <span className="text-gray-700 dark:text-gray-300">Show 7-Day MA</span>
            </label>
            <button onClick={handleExportPNG} className="px-3 py-2 text-sm bg-gray-200 dark:bg-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500" aria-label="Export chart as PNG">Export PNG</button>
            <button onClick={() => downloadCSV(filteredData, selectedCryptos)} className="px-3 py-2 text-sm bg-gray-200 dark:bg-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500" aria-label="Export data as CSV">Export CSV</button>
            <button onClick={handleManualRefresh} className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500" aria-label="Refresh data">Refresh Data</button>
            <label className="flex items-center space-x-2 cursor-pointer">
                <input type="checkbox" checked={isAutoRefreshEnabled} onChange={() => setIsAutoRefreshEnabled(!isAutoRefreshEnabled)} className="form-checkbox h-5 w-5 rounded text-blue-600 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500" aria-label="Toggle auto refresh"/>
                <span className="text-gray-700 dark:text-gray-300">Auto Refresh</span>
            </label>
        </div>
      </div>

      {/* Chart */}
      <div className="h-[300px] sm:h-[500px] w-full bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm" ref={chartRef} aria-label="Sentiment timeline chart showing cryptocurrency sentiment over time">
        <ResponsiveContainer width="99%" height="99%">
            <AreaChart data={processedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }} animationDuration={500}>
                 <defs>
                    {selectedCryptos.map(crypto => (
                        <linearGradient key={crypto} id={`gradient-${crypto}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={CRYPTO_CONFIG[crypto].color} stopOpacity={0.4}/>
                            <stop offset="95%" stopColor={CRYPTO_CONFIG[crypto].color} stopOpacity={0}/>
                        </linearGradient>
                    ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} tickFormatter={(date) => format(new Date(date), 'MMM d')} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend onClick={(e) => { setSelectedCryptos(prev => prev.includes(e.dataKey as Crypto) ? prev.filter(c => c !== e.dataKey) : [...prev, e.dataKey as Crypto] ); }} />
                <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="4 4" />

                {selectedCryptos.map(crypto => (
                    <React.Fragment key={crypto}>
                        <Area type="monotone" dataKey={crypto} strokeWidth={0} fill={`url(#gradient-${crypto})`} animationDuration={1000} />
                        <Line type="monotone" dataKey={crypto} stroke={CRYPTO_CONFIG[crypto].color} strokeWidth={2} dot={false} activeDot={{ r: 6 }} animationDuration={1500} />
                    </React.Fragment>
                ))}
                 {showMA && selectedCryptos.map(crypto => (
                    <Line key={`${crypto}_ma`} type="monotone" dataKey={`${crypto}_ma`} stroke={CRYPTO_CONFIG[crypto].color} strokeWidth={2} strokeDasharray="5 5" dot={false} name={`${crypto} MA`} animationDuration={1500} />
                ))}
            </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
    </ErrorBoundary>
  );
};

export default SentimentTimeline;
