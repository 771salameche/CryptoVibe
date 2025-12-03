import React from 'react';
import ReactWordcloud from 'wordcloud-react-18';
import { WordData } from '../../types/events';

interface WordCloudProps {
  words: WordData[];
  title: string;
  sentimentType?: 'overall' | 'positive' | 'negative';
}

const WordCloud: React.FC<WordCloudProps> = ({ words, title, sentimentType = 'overall' }) => {
  const getWordColor = (word: WordData) => {
    if (sentimentType === 'overall' && word.sentiment) {
      if (word.sentiment === 'positive') return 'rgb(34, 197, 94)'; // Tailwind green-500
      if (word.sentiment === 'negative') return 'rgb(239, 68, 68)'; // Tailwind red-500
      return 'rgb(107, 114, 128)'; // Tailwind gray-500
    }
    if (sentimentType === 'positive') return 'rgb(34, 197, 94)'; // Green gradient
    if (sentimentType === 'negative') return 'rgb(239, 68, 68)'; // Red gradient
    return 'rgb(107, 114, 128)'; // Default gray
  };

  const options = {
    rotations: 0,
    rotationAngles: [0, 0],
    fontFamily: 'Impact',
    enableTooltip: true,
    deterministic: true,
    colors: (word: WordData) => getWordColor(word),
  };

  // Format words for react-wordcloud (requires text and value)
  const formattedWords = words.map(word => ({
    text: word.text,
    value: word.value,
    sentiment: word.sentiment, // Pass sentiment for custom coloring
  }));

  return (
    <div className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg h-96 flex flex-col">
      <h4 className="text-xl font-semibold text-white mb-2 text-center">{title}</h4>
      {formattedWords.length > 0 ? (
        <div className="flex-grow">
          <ReactWordcloud words={formattedWords} options={options} />
        </div>
      ) : (
        <div className="flex-grow flex items-center justify-center text-gray-500 text-lg">
          Not enough data for word cloud.
        </div>
      )}
    </div>
  );
};

export default WordCloud;
