import { WordData } from '../types/events';

export interface EventCsvRow {
  date: string;
  event_type: string;
  crypto: string;
  mention_count: number;
  avg_sentiment: number;
  total_importance: number;
}

const stopWords = new Set([
  // Generic stop words (a very minimal set, as our "words" are specific categories/cryptos)
  'the', 'is', 'and', 'to', 'of', 'a', 'in', 'it', 'with', 'for', 'on', 'as', 'at', 'by', 'be',
  'this', 'that', 'was', 'we', 'i', 'he', 'she', 'you', 'they', 'but', 'not', 'or', 'from', 'an',
  'has', 'have', 'had', 'will', 'would', 'can', 'could', 'do', 'does', 'did',
  // Common event types and crypto names to exclude if they appear too frequently and don't add value
  'market crash', 'network outage', 'regulatory action', 'hack / exploit', 'exchange listing', 'partnership / collaboration',
  'bitcoin', 'ethereum', 'solana', 'xrp', 'cardano', 'bnb', 'dogecoin', 'litecoin', 'polkadot',
  'chainlink', 'tron', 'monero', 'zcash', 'near protocol', 'stellar', 'ethereum classic', 'avalanche',
  'cronos', 'hedera', 'hyperliquid', 'toncoin', 'sui', 'bittensor', 'kaspa', 'ondo', 'world liberty financial',
  'world liberty financial usd', 'usdc', 'tether', 'aster',
  // Further filtering for case-insensitivity and common variations
  'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'xrp', 'cardano', 'ada', 'bnb', 'dogecoin', 'doge', 'litecoin', 'ltc',
  'polkadot', 'dot', 'chainlink', 'link', 'tron', 'trx', 'monero', 'xmr', 'zcash', 'zec', 'near protocol', 'near',
  'stellar', 'xlm', 'ethereum classic', 'etc', 'avalanche', 'avax', 'cronos', 'cro', 'hedera', 'hbar',
  'hyperliquid', 'toncoin', 'ton', 'sui', 'bittensor', 'tao', 'kaspa', 'kas', 'ondo', 'usdc', 'tether', 'usdt', 'aster', 'atr',
  'market', 'crash', 'network', 'outage', 'regulatory', 'action', 'hack', 'exploit', 'exchange', 'listing',
  'partnership', 'collaboration', 'trading', 'price', 'coin', 'crypto', 'cryptocurrency', 'token',
  'events', 'news', 'update', 'major', 'new', 'report', 'announcement', 'launch', 'upgrade', 'system',
]);


export function processEventsForWordCloud(events: EventCsvRow[]): { overall: WordData[]; positive: WordData[]; negative: WordData[] } {
  const overallWordCounts: Record<string, { value: number; sentimentSum: number; count: number }> = {};
  const positiveWordCounts: Record<string, { value: number; sentimentSum: number; count: number }> = {};
  const negativeWordCounts: Record<string, { value: number; sentimentSum: number; count: number }> = {};

  const addWord = (word: string, importance: number, sentiment: number, targetCounts: Record<string, { value: number; sentimentSum: number; count: number }>) => {
    const lowerCaseWord = word.toLowerCase();
    if (!stopWords.has(lowerCaseWord) && lowerCaseWord.length > 1) { // Filter out very short words
      if (!targetCounts[word]) {
        targetCounts[word] = { value: 0, sentimentSum: 0, count: 0 };
      }
      targetCounts[word].value += importance;
      targetCounts[word].sentimentSum += sentiment;
      targetCounts[word].count += 1;
    }
  };

  events.forEach(event => {
    const wordsToConsider = [event.event_type, event.crypto].filter(Boolean) as string[]; // Ensure no empty strings
    const importance = event.total_importance || 1; // Default importance to 1 if not present
    const sentiment = event.avg_sentiment || 0; // Default sentiment to 0 if not present

    wordsToConsider.forEach(word => {
      addWord(word, importance, sentiment, overallWordCounts);

      if (sentiment > 0.1) { // Threshold for positive sentiment
        addWord(word, importance, sentiment, positiveWordCounts);
      } else if (sentiment < -0.1) { // Threshold for negative sentiment
        addWord(word, importance, sentiment, negativeWordCounts);
      }
    });
  });

  const convertToWordData = (counts: Record<string, { value: number; sentimentSum: number; count: number }>, topN: number = 50): WordData[] => {
    return Object.entries(counts)
      .sort(([, dataA], [, dataB]) => dataB.value - dataA.value)
      .slice(0, topN)
      .map(([text, data]) => {
        const avgSentiment = data.count > 0 ? data.sentimentSum / data.count : 0;
        let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
        if (avgSentiment > 0.1) {
          sentiment = 'positive';
        } else if (avgSentiment < -0.1) {
          sentiment = 'negative';
        }
        return { text, value: Math.max(1, Math.round(data.value)), sentiment }; // Ensure value is at least 1 and an integer
      });
  };

  return {
    overall: convertToWordData(overallWordCounts, 100), // More words for overall
    positive: convertToWordData(positiveWordCounts),
    negative: convertToWordData(negativeWordCounts),
  };
}