import { EventCsvRow } from './eventsDataProcessor'; // Import EventCsvRow

function parseCSV(csvText: string): string[][] {
  const lines = csvText.trim().split('\n');
  return lines.map(line => {
    // This is a simple parser, it doesn't handle commas inside quoted strings
    return line.split(',');
  });
}

export async function loadEventsData(): Promise<EventCsvRow[]> {
  try {
    const response = await fetch('/data/events_timeline.csv');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const csvText = await response.text();
    const rows = parseCSV(csvText);
    const header = rows[0];
    const data = rows.slice(1);

    const events: EventCsvRow[] = data.map(row => {
      const dateIndex = header.indexOf('date');
      const eventTypeIndex = header.indexOf('event_type');
      const cryptoIndex = header.indexOf('crypto');
      const mentionCountIndex = header.indexOf('mention_count');
      const avgSentimentIndex = header.indexOf('avg_sentiment');
      const totalImportanceIndex = header.indexOf('total_importance'); // New index

      return {
        date: dateIndex !== -1 ? row[dateIndex] : '',
        event_type: eventTypeIndex !== -1 ? row[eventTypeIndex] : '',
        crypto: cryptoIndex !== -1 ? row[cryptoIndex] : '',
        mention_count: mentionCountIndex !== -1 ? parseInt(row[mentionCountIndex], 10) : 0,
        avg_sentiment: avgSentimentIndex !== -1 ? parseFloat(row[avgSentimentIndex]) : 0,
        total_importance: totalImportanceIndex !== -1 ? parseFloat(row[totalImportanceIndex]) : 0, // Parse total_importance
      };
    });

    return events;
  } catch (error) {
    console.error("Failed to load events data:", error);
    return [];
  }
}

export async function loadEnrichedData(): Promise<any[]> {
    try {
      const response = await fetch('/public/data/enriched_dataset.csv');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const csvText = await response.text();
      const rows = parseCSV(csvText);
      const header = rows[0];
      const data = rows.slice(1);
  
      const enrichedData = data.map(row => {
        const rowData: any = {};
        header.forEach((h, i) => {
            rowData[h] = row[i];
        });
        return rowData;
      });
  
      return enrichedData;
    } catch (error) {
      console.error("Failed to load enriched data:", error);
      return [];
    }
  }