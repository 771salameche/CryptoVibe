import React, { useState, useEffect, useMemo } from 'react';
import SentimentPriceOverlayChart from './charts/SentimentPriceOverlayChart';
import ScatterPlot from './charts/ScatterPlot';
import CorrelationHeatmap from './charts/CorrelationHeatmap';
import LagCorrelationChart from './charts/LagCorrelationChart';
import EventImpactChart from './charts/EventImpactChart';
import CorrelationTable from './charts/CorrelationTable';
import { loadCorrelationData } from '../utils/dataLoader';
import { subDays, format } from 'date-fns';

const PriceCorrelation: React.FC = () => {
    const [activeChart, setActiveChart] = useState('overlay');
    const [activeCrypto, setActiveCrypto] = useState('BTC');
    const [highlightedCrypto, setHighlightedCrypto] = useState<string | null>(null);
    const [dateRange, setDateRange] = useState(30); // Default to last 30 days
    const [showSmoothing, setShowSmoothing] = useState(false); // Toggle for moving average smoothing

    const [allChartData, setAllChartData] = useState<any | null>(null); // Store raw loaded data
    const [chartData, setChartData] = useState<any>({
        overlay: [],
        scatter: [],
        heatmap: { data: [], colHeaders: [], rowHeaders: [] },
        lag: [],
        eventImpact: [],
        table: [],
    });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            const rawData = await loadCorrelationData();
            setAllChartData(rawData); // Store raw data
            setLoading(false);
        };
        fetchData();
    }, []);

    useEffect(() => {
        if (!allChartData) return;

        const { sentimentPrice, correlationResults } = allChartData;

        // --- Filter data by date range ---
        const endDate = new Date();
        const startDate = subDays(endDate, dateRange - 1);

        const filteredSentimentPrice = sentimentPrice.data.filter((row: any) => {
            const rowDate = new Date(row[sentimentPrice.header.indexOf('date')]);
            return rowDate >= startDate && rowDate <= endDate;
        });

        // --- Process Overlay Data ---
        let processedOverlayData = filteredSentimentPrice.map((row: any) => ({
            date: row[sentimentPrice.header.indexOf('date')],
            sentiment: parseFloat(row[sentimentPrice.header.indexOf(`${activeCrypto.toLowerCase()}_sentiment`)]),
            price: parseFloat(row[sentimentPrice.header.indexOf(`${activeCrypto.toLowerCase()}_price`)]),
            priceChange: parseFloat(row[sentimentPrice.header.indexOf(`${activeCrypto.toLowerCase()}_price_change`)]),
        }));

        if (showSmoothing) {
            // Apply 7-day moving average (simplified for example)
            processedOverlayData = processedOverlayData.map((entry: any, i: number, arr: any[]) => {
                const slice = arr.slice(Math.max(0, i - 6), i + 1);
                const avgSentiment = slice.reduce((sum, d) => sum + d.sentiment, 0) / slice.length;
                const avgPrice = slice.reduce((sum, d) => sum + d.price, 0) / slice.length;
                return { ...entry, sentiment: avgSentiment, price: avgPrice };
            });
        }
        
        // --- Process Scatter Data (filtered by activeCrypto) ---
        const scatterData = filteredSentimentPrice.map((row: any) => ({
            sentiment: parseFloat(row[sentimentPrice.header.indexOf(`${activeCrypto.toLowerCase()}_sentiment`)]),
            priceChange: parseFloat(row[sentimentPrice.header.indexOf(`${activeCrypto.toLowerCase()}_price_change`)]),
        }));


        // --- Heatmap Data ---
        // Dynamically extract headers for heatmap from correlationResults if available
        const heatmapColHeaders = correlationResults.header.filter((h: string) => h.includes('price') || h.includes('volume')).map((h: string) => {
            return h.replace('_price', ' Price').replace('_volume', ' Volume');
        });
        const heatmapRowHeaders = ['BTC Sentiment', 'ETH Sentiment', 'SOL Sentiment'];

        const processedHeatmapData = heatmapRowHeaders.map((sentimentType: string) => {
            const row: any = { type: sentimentType };
            const cryptoPrefix = sentimentType.split(' ')[0].toLowerCase(); // 'btc', 'eth', 'sol'
            heatmapColHeaders.forEach((colHeader: string) => {
                const colKey = colHeader.toLowerCase().replace(' price', '_price').replace(' volume', '_volume');
                const correlationValue = correlationResults.data[0][correlationResults.header.indexOf(`${cryptoPrefix}_${colKey}_correlation`)]
                row[colHeader] = parseFloat(correlationValue) || 0;
            });
            return row;
        });


        // --- Lag Data ---
        let processedLagData = correlationResults.data.map((row: any) => ({
            lag: parseInt(row[correlationResults.header.indexOf('lag')]),
            BTC: parseFloat(row[correlationResults.header.indexOf('BTC_correlation')]),
            ETH: parseFloat(row[correlationResults.header.indexOf('ETH_correlation')]),
            SOL: parseFloat(row[correlationResults.header.indexOf('SOL_correlation')]),
            BTC_p: parseFloat(row[correlationResults.header.indexOf('BTC_p_value')]), // Assuming p-values are available
            ETH_p: parseFloat(row[correlationResults.header.indexOf('ETH_p_value')]),
            SOL_p: parseFloat(row[correlationResults.header.indexOf('SOL_p_value')]),
        }));
        
        // --- Event Impact Data (static for now) ---
        const eventImpactData = [
            { eventType: 'LISTING', priceChange: 5.2, error: 0.5 },
            { eventType: 'PARTNERSHIP', priceChange: 3.1, error: 0.3 },
            { eventType: 'UPGRADE', priceChange: 2.5, error: 0.2 },
            { eventType: 'REGULATION', priceChange: -2.8, error: 0.4 },
            { eventType: 'HACK', priceChange: -8.7, error: 1.0 },
        ];

        // --- Table Data ---
        const tableData = [
            { crypto: 'BTC', sameDay: 0.45, nextDay: 0.52, optimalLag: '24h', pValue: 0.002, isSignificant: true },
            { crypto: 'ETH', sameDay: 0.38, nextDay: 0.41, optimalLag: '18h', pValue: 0.008, isSignificant: true },
            { crypto: 'SOL', sameDay: 0.22, nextDay: 0.28, optimalLag: '12h', pValue: 0.089, isSignificant: false },
        ];

        setChartData({
            overlay: processedOverlayData,
            scatter: scatterData,
            heatmap: { data: processedHeatmapData, colHeaders: heatmapColHeaders, rowHeaders: heatmapRowHeaders },
            lag: processedLagData,
            eventImpact: eventImpactData,
            table: tableData,
        });

    }, [allChartData, activeCrypto, dateRange, showSmoothing]);


    const handleRowClick = (crypto: string) => {
        setHighlightedCrypto(crypto);
        setActiveCrypto(crypto); // Also update activeCrypto when a row is clicked
    };

    const generateInsights = useMemo(() => {
        const insights = [];
        const currentTableData = chartData.table.find((row: any) => row.crypto === activeCrypto);
        
        if (currentTableData) {
            insights.push(`💡 ${activeCrypto} sentiment predicts price with ${currentTableData.optimalLag} delay (r=${currentTableData.nextDay.toFixed(2)})`);
            if (currentTableData.isSignificant) {
                insights.push(`✅ This correlation is statistically significant (p=${currentTableData.pValue.toFixed(3)})`);
            } else {
                insights.push(`⚠️ ${activeCrypto} shows weak correlation (r=${currentTableData.nextDay.toFixed(2)}) - other factors might dominate (p=${currentTableData.pValue.toFixed(3)})`);
            }
        }

        const highestImpactEvent = chartData.eventImpact.reduce((max: any, event: any) => Math.abs(event.priceChange) > Math.abs(max.priceChange) ? event : max, { priceChange: 0 });
        if (highestImpactEvent.priceChange > 0) {
            insights.push(`📈 ${highestImpactEvent.eventType} events are associated with a +${highestImpactEvent.priceChange.toFixed(2)}% average price increase.`);
        } else if (highestImpactEvent.priceChange < 0) {
            insights.push(`📉 ${highestImpactEvent.eventType} events are associated with a ${highestImpactEvent.priceChange.toFixed(2)}% average price drop.`);
        }

        return insights;
    }, [activeCrypto, chartData.table, chartData.eventImpact]);


    const renderChart = () => {
        if(loading) {
            return <div className="flex items-center justify-center h-full text-gray-500">Loading...</div>;
        }

        switch (activeChart) {
            case 'overlay':
                return <SentimentPriceOverlayChart data={chartData.overlay} crypto={activeCrypto} />;
            case 'scatter':
                return <ScatterPlot data={chartData.scatter} crypto={activeCrypto} />;
            case 'heatmap':
                return <CorrelationHeatmap 
                    data={chartData.heatmap.data} 
                    colHeaders={chartData.heatmap.colHeaders} 
                    rowHeaders={chartData.heatmap.rowHeaders} 
                />;
            case 'lag':
                return <LagCorrelationChart data={chartData.lag} />;
            case 'events':
                return <EventImpactChart data={chartData.eventImpact} />;
            default:
                return null;
        }
    };

    return (
        <div className="p-6 bg-gray-900 min-h-screen text-white">
            <header className="mb-8">
                <h1 className="text-4xl font-bold">Sentiment-Price Correlation Analysis</h1>
                <p className="text-gray-400">Exploring the relationship between market sentiment and cryptocurrency prices.</p>
            </header>

            {/* Filters and Controls */}
            <div className="mb-8 flex flex-wrap justify-between items-center bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg">
                {/* Crypto Selector */}
                <div className="flex items-center space-x-2 mb-4 md:mb-0">
                    <span className="font-semibold text-gray-300">Crypto:</span>
                    {['BTC', 'ETH', 'SOL'].map(crypto => ( // Removed 'All' as it complicates data filtering for single-crypto charts
                        <button 
                            key={crypto}
                            className={`px-4 py-2 rounded-md text-sm font-medium ${activeCrypto === crypto ? 'bg-indigo-600' : 'bg-gray-700'}`}
                            onClick={() => setActiveCrypto(crypto)}
                        >
                            {crypto}
                        </button>
                    ))}
                </div>

                {/* Date Range Selector */}
                <div className="flex items-center space-x-2 mb-4 md:mb-0">
                    <span className="font-semibold text-gray-300">Period:</span>
                    {[7, 30, 90].map(days => (
                        <button 
                            key={days}
                            className={`px-4 py-2 rounded-md text-sm font-medium ${dateRange === days ? 'bg-indigo-600' : 'bg-gray-700'}`}
                            onClick={() => setDateRange(days)}
                        >
                            Last {days} days
                        </button>
                    ))}
                </div>

                {/* Chart Type Selector */}
                <div className="flex items-center space-x-2 mb-4 md:mb-0">
                    <span className="font-semibold text-gray-300">Chart Type:</span>
                    {['Overlay', 'Scatter', 'Heatmap', 'Lag', 'Events'].map(chart => (
                        <button 
                            key={chart}
                            className={`px-4 py-2 rounded-md text-sm font-medium ${activeChart === chart.toLowerCase() ? 'bg-indigo-600' : 'bg-gray-700'}`}
                            onClick={() => setActiveChart(chart.toLowerCase())}
                        >
                            {chart}
                        </button>
                    ))}
                </div>

                {/* Smoothing Toggle */}
                {activeChart === 'overlay' && ( // Only show smoothing toggle for overlay chart
                    <div className="flex items-center space-x-2">
                        <input
                            type="checkbox"
                            id="smoothing-toggle"
                            checked={showSmoothing}
                            onChange={() => setShowSmoothing(!showSmoothing)}
                            className="form-checkbox h-4 w-4 text-indigo-600"
                        />
                        <label htmlFor="smoothing-toggle" className="text-gray-300 text-sm">7-day MA Smoothing</label>
                    </div>
                )}

                {/* Export Buttons */}
                <div className="flex items-center space-x-2">
                    <button
                        className="px-4 py-2 rounded-md text-sm font-medium bg-gray-700 text-white"
                        onClick={() => alert('Exporting chart as PNG... (Functionality not yet implemented)')}
                    >
                        Export Chart PNG
                    </button>
                    <button
                        className="px-4 py-2 rounded-md text-sm font-medium bg-gray-700 text-white"
                        onClick={() => alert('Exporting data as CSV... (Functionality not yet implemented)')}
                    >
                        Export Data CSV
                    </button>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Chart Display */}
                <div className="lg:col-span-2">
                    <div className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg">
                        {renderChart()}
                    </div>
                </div>

                {/* Key Insights Panel */}
                <div className="lg:col-span-1">
                    <div className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg h-96">
                        <h2 className="text-2xl font-bold mb-4">Key Insights</h2>
                        <ul className="space-y-4">
                            {generateInsights.map((insight: string, index: number) => (
                                <li key={index} className="text-gray-300">{insight}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="mt-8">
                <h2 className="text-2xl font-bold mb-4 text-white">Sentiment vs Price Movement</h2>
                <CorrelationTable data={chartData.table} onRowClick={handleRowClick} highlightedCrypto={highlightedCrypto} />
            </div>
        </div>
    );
}

export default PriceCorrelation;