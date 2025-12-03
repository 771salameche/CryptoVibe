import Papa from 'papaparse';
import { CryptoEvent, WordData } from '../types/events';

interface EventTimelineRow {
  date: string;
  event_type: string;
  crypto: string; // Can be a comma-separated string
  text: string;
  mentions: number;
  sentiment: number;
}

interface EnrichedDatasetRow {
  Crypto: string;
  SentimentCategory: 'Positive' | 'Negative' | 'Neutral';
  CleanedText: string; // Assuming a column with cleaned text for word cloud
  // Add other fields from enriched_dataset.csv relevant for word cloud, if any
}

export const loadEventsData = async (): Promise<CryptoEvent[]> => {
  const eventsTimelineUrl = '/data/events_timeline.csv';

  const eventsTimelineResponse = await fetch(eventsTimelineUrl);
  const eventsTimelineText = await eventsTimelineResponse.text();
  const eventsTimelineParsed = Papa.parse<EventTimelineRow>(eventsTimelineText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  const eventsData = eventsTimelineParsed.data.map(row => ({
    date: row.date,
    eventType: row.event_type,
    crypto: row.crypto ? row.crypto.split(',').map(c => c.trim().toUpperCase()) : [], // Split and clean crypto names
    title: row.text.substring(0, 100) + '...', // Simple title from text
    textSnippet: row.text,
    mentions: row.mentions || 0,
    sentiment: row.sentiment || 0,
  })).filter(event => event.date); // Filter out events without a date

  // Sort events by date, most recent first
  eventsData.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  return eventsData;
};

export const calculateWordCloudsData = async (
  events: CryptoEvent[],
  enrichedDatasetUrl: string = '/data/enriched_dataset.csv'
): Promise<{ overall: WordData[]; positive: WordData[]; negative: WordData[] }> => {
  const wordCounts: Record<string, { count: number; positive: number; negative: number; neutral: number }> = {};
  const stopWords = new Set(['the', 'is', 'and', 'or', 'a', 'an', 'to', 'of', 'for', 'on', 'with', 'in', 'it', 'that', 'this', 'was', 'are', 'be', 'as', 'by', 'at', 'from', 'but', 'not', 'we', 'i', 'you', 'he', 'she', 'they', 'it', 'do', 'say', 'go', 'get', 'make', 'know', 'think', 'take', 'see', 'come', 'want', 'look', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'crypto', 'bitcoin', 'ethereum', 'solana', 'blockchain', 'market', 'price', 'coin', 'token']); // Add more common words and crypto terms

  // Fetch and parse enriched_dataset.csv for word cloud data
  const enrichedDatasetResponse = await fetch(enrichedDatasetUrl);
  const enrichedDatasetText = await enrichedDatasetResponse.text();
  const enrichedDatasetParsed = Papa.parse<EnrichedDatasetRow>(enrichedDatasetText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  const enrichedData = enrichedDatasetParsed.data.filter(row => row.CleanedText);

  enrichedData.forEach(row => {
    const words = (row.CleanedText || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '') // Remove special characters
      .split(/\s+/) // Split by whitespace
      .filter(word => word.length > 2 && !stopWords.has(word)); // Filter short words and stop words

    words.forEach(word => {
      if (!wordCounts[word]) {
        wordCounts[word] = { count: 0, positive: 0, negative: 0, neutral: 0 };
      }
      wordCounts[word].count++;
      if (row.SentimentCategory === 'Positive') {
        wordCounts[word].positive++;
      } else if (row.SentimentCategory === 'Negative') {
        wordCounts[word].negative++;
      } else {
        wordCounts[word].neutral++;
      }
    });
  });

  const allWords: WordData[] = [];
  const positiveWords: WordData[] = [];
  const negativeWords: WordData[] = [];

  Object.entries(wordCounts).forEach(([word, data]) => {
    // Determine overall sentiment for coloring
    let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
    if (data.positive > data.negative && data.positive > data.neutral) {
      sentiment = 'positive';
    } else if (data.negative > data.positive && data.negative > data.neutral) {
      sentiment = 'negative';
    }

    allWords.push({ text: word, value: data.count, sentiment });

    if (data.positive > data.negative) { // Simple heuristic for positive cloud
      positiveWords.push({ text: word, value: data.positive });
    }
    if (data.negative > data.positive) { // Simple heuristic for negative cloud
      negativeWords.push({ text: word, value: data.negative });
    }
  });

  // Sort and take top N words for sentiment-specific clouds
  positiveWords.sort((a, b) => b.value - a.value);
  negativeWords.sort((a, b) => b.value - a.value);

  return {
    overall: allWords,
    positive: positiveWords.slice(0, 50),
    negative: negativeWords.slice(0, 50),
  };
};
