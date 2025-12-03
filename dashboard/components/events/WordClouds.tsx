import React from 'react';
import { WordData } from '../../types/events';
import WordCloud from './WordCloud';

interface WordCloudsProps {
  wordCloudsData: {
    overall: WordData[];
    positive: WordData[];
    negative: WordData[];
  };
}

const WordClouds: React.FC<WordCloudsProps> = ({ wordCloudsData }) => {
  return (
    <div className="space-y-6">
      {/* Overall Word Cloud */}
      <WordCloud words={wordCloudsData.overall} title="Overall Keyword Trends" sentimentType="overall" />

      {/* Positive and Negative Word Clouds */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <WordCloud words={wordCloudsData.positive} title="Positive Sentiment Keywords" sentimentType="positive" />
        <WordCloud words={wordCloudsData.negative} title="Negative Sentiment Keywords" sentimentType="negative" />
      </div>
    </div>
  );
};

export default WordClouds;
