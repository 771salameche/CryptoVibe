import { CryptoData } from '../types/crypto';
import Papa from 'papaparse';

interface SentimentTimeSeriesRow {
  Date: string;
  Crypto: string;
  Sentiment: number;
  Mentions: number;
  // Add other fields if present in sentiment_timeseries.csv
}

interface EnrichedDatasetRow {
  Crypto: string;
  SentimentScore: number;
  Date: string;
  CorrelationWithPrice: number; // Assuming this column exists for radar chart
  EventFrequency: number; // Assuming this column exists for radar chart
  SentimentCategory: 'Positive' | 'Negative' | 'Neutral';
  // Add other fields if present in enriched_dataset.csv
}

export const loadAndProcessCryptoData = async (): Promise<CryptoData[]> => {
  const sentimentTimeSeriesUrl = '/data/sentiment_timeseries.csv';
  const enrichedDatasetUrl = '/data/enriched_dataset.csv';

  // Fetch and parse sentiment_timeseries.csv
  const sentimentTimeSeriesResponse = await fetch(sentimentTimeSeriesUrl);
  const sentimentTimeSeriesText = await sentimentTimeSeriesResponse.text();
  const sentimentTimeSeriesParsed = Papa.parse<SentimentTimeSeriesRow>(sentimentTimeSeriesText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  const sentimentTimeSeriesData = sentimentTimeSeriesParsed.data.filter(row => row.Crypto && row.Date); // Filter out rows with missing essential data

  // Fetch and parse enriched_dataset.csv
  const enrichedDatasetResponse = await fetch(enrichedDatasetUrl);
  const enrichedDatasetText = await enrichedDatasetResponse.text();
  const enrichedDatasetParsed = Papa.parse<EnrichedDatasetRow>(enrichedDatasetText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  const enrichedDatasetData = enrichedDatasetParsed.data.filter(row => row.Crypto && row.Date); // Filter out rows with missing essential data


  // Group data by crypto
  const groupedSentimentTimeSeries = sentimentTimeSeriesData.reduce((acc, row) => {
    if (!acc[row.Crypto]) {
      acc[row.Crypto] = [];
    }
    acc[row.Crypto].push(row);
    return acc;
  }, {} as Record<string, SentimentTimeSeriesRow[]>);

  const groupedEnrichedDataset = enrichedDatasetData.reduce((acc, row) => {
    if (!acc[row.Crypto]) {
      acc[row.Crypto] = [];
    }
    acc[row.Crypto].push(row);
    return acc;
  }, {} as Record<string, EnrichedDatasetRow[]>);

  const cryptos = ['BTC', 'ETH', 'SOL'];
  const processedData: CryptoData[] = [];

  for (const symbol of cryptos) {
    const sentimentData = groupedSentimentTimeSeries[symbol] || [];
    const enrichedData = groupedEnrichedDataset[symbol] || [];

    // Sort sentiment data by date to ensure correct sparkline and trend calculation
    sentimentData.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime());

    // --- Calculations ---

    // Average Sentiment
    const avgSentiment = sentimentData.length > 0
      ? sentimentData.reduce((sum, row) => sum + (row.Sentiment || 0), 0) / sentimentData.length
      : 0;

    // Total Mentions
    const totalMentions = sentimentData.length > 0
      ? sentimentData.reduce((sum, row) => sum + (row.Mentions || 0), 0)
      : 0;
    
    // Sparkline Data (Last 7 days sentiment)
    const sparklineData = sentimentData
      .slice(Math.max(sentimentData.length - 7, 0)) // Get last 7 entries
      .map(row => row.Sentiment || 0);

    // Trend (Percentage change vs previous period - simplified: current vs last 7 day average vs previous 7 day average)
    // For simplicity, calculate trend based on the difference between the average sentiment of the last 7 days
    // and the average sentiment of the 7 days before that.
    let trend = 0;
    if (sentimentData.length >= 14) {
      const last7DaysSentiment = sentimentData.slice(-7).map(row => row.Sentiment || 0);
      const prev7DaysSentiment = sentimentData.slice(-14, -7).map(row => row.Sentiment || 0);

      const avgLast7Days = last7DaysSentiment.reduce((sum, val) => sum + val, 0) / last7DaysSentiment.length;
      const avgPrev7Days = prev7DaysSentiment.reduce((sum, val) => sum + val, 0) / prev7DaysSentiment.length;

      if (avgPrev7Days !== 0) {
        trend = ((avgLast7Days - avgPrev7Days) / avgPrev7Days) * 100;
      }
    } else if (sentimentData.length >= 2) { // Fallback for less than 14 days data
        const lastSentiment = sentimentData[sentimentData.length - 1].Sentiment || 0;
        const firstSentiment = sentimentData[0].Sentiment || 0;
        if (firstSentiment !== 0) {
            trend = ((lastSentiment - firstSentiment) / firstSentiment) * 100;
        }
    }


    // Sentiment Distribution
    const sentimentCategories = enrichedData.map(row => row.SentimentCategory).filter(Boolean); // Filter out undefined/null
    const positiveCount = sentimentCategories.filter(cat => cat === 'Positive').length;
    const neutralCount = sentimentCategories.filter(cat => cat === 'Neutral').length;
    const negativeCount = sentimentCategories.filter(cat => cat === 'Negative').length;
    const totalSentimentEntries = sentimentCategories.length;

    const distribution = totalSentimentEntries > 0
      ? {
          positive: (positiveCount / totalSentimentEntries) * 100,
          neutral: (neutralCount / totalSentimentEntries) * 100,
          negative: (negativeCount / totalSentimentEntries) * 100,
        }
      : { positive: 0, neutral: 0, negative: 0 };
      
    // Sentiment Volatility (Standard Deviation)
    const sentimentScores = sentimentData.map(row => row.Sentiment || 0);
    const meanSentiment = avgSentiment; // Already calculated
    const squaredDifferences = sentimentScores.map(score => Math.pow(score - meanSentiment, 2));
    const variance = squaredDifferences.length > 0
      ? squaredDifferences.reduce((sum, val) => sum + val, 0) / squaredDifferences.length
      : 0;
    const sentimentVolatility = Math.sqrt(variance);

    // Correlation with Price (taking average from enriched data)
    const correlationWithPriceScores = enrichedData.map(row => row.CorrelationWithPrice || 0);
    const correlationWithPrice = correlationWithPriceScores.length > 0
      ? correlationWithPriceScores.reduce((sum, val) => sum + val, 0) / correlationWithPriceScores.length
      : 0;

    // Event Frequency (taking average from enriched data)
    const eventFrequencyScores = enrichedData.map(row => row.EventFrequency || 0);
    const eventFrequency = eventFrequencyScores.length > 0
      ? eventFrequencyScores.reduce((sum, val) => sum + val, 0) / eventFrequencyScores.length
      : 0;


    processedData.push({
      symbol: symbol,
      name: symbol === 'BTC' ? 'Bitcoin' : symbol === 'ETH' ? 'Ethereum' : 'Solana', // Hardcoded names for now
      avgSentiment: parseFloat(avgSentiment.toFixed(2)),
      totalMentions: Math.round(totalMentions),
      trend: parseFloat(trend.toFixed(1)),
      sparklineData: sparklineData.map(s => parseFloat(s.toFixed(2))),
      distribution: {
        positive: parseFloat(distribution.positive.toFixed(1)),
        neutral: parseFloat(distribution.neutral.toFixed(1)),
        negative: parseFloat(distribution.negative.toFixed(1)),
      },
      sentimentVolatility: parseFloat(sentimentVolatility.toFixed(2)),
      correlationWithPrice: parseFloat(correlationWithPrice.toFixed(2)),
      eventFrequency: parseFloat(eventFrequency.toFixed(2)),
      fullTimeSeries: sentimentData.map(d => ({
        Date: d.Date,
        Sentiment: parseFloat(d.Sentiment.toFixed(2)),
        Mentions: d.Mentions,
        // Assuming Volume is not in sentiment_timeseries.csv for now
      })),
    });
  }

  return processedData;
};