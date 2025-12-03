import React from 'react';
import { WordData } from '../../types/events';
import { scaleLog } from '@visx/scale';
import { Wordcloud } from '@visx/wordcloud';

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

  const fontScale = scaleLog({
    domain: [Math.min(...words.map((w) => w.value)), Math.max(...words.map((w) => w.value))],
    range: [10, 100],
  });

  const fontSizeSetter = (datum: WordData) => fontScale(datum.value);

  return (
    <div className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg h-96 flex flex-col">
      <h4 className="text-xl font-semibold text-white mb-2 text-center">{title}</h4>
      {words.length > 0 ? (
        <div className="flex-grow flex items-center justify-center">
          <Wordcloud
            words={words}
            width={400} // Adjust width as needed
            height={300} // Adjust height as needed
            fontSize={fontSizeSetter}
            font="Impact"
            padding={5}
            spiral="archimedean"
            rotate={() => 0} // No rotation for simplicity
            fill={getWordColor}
          >
            {(cloudWords) =>
              cloudWords.map((w, i) => (
                <text
                  key={w.text}
                  fill={w.fill}
                  textAnchor="middle"
                  transform={`translate(${w.x}, ${w.y}) rotate(${w.rotate})`}
                  fontSize={w.size}
                  fontFamily={w.font}
                >
                  {w.text}
                </text>
              ))
            }
          </Wordcloud>
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