import React, { useState, useEffect, useMemo } from 'react';
import WordCloud from './WordCloud'; // Import the local WordCloud component
import { EventCsvRow } from '../../utils/eventsDataProcessor';
import { processEventsForWordCloud } from '../../utils/eventsDataProcessor';
import { WordData } from '../../types/events';

interface WordCloudsProps {
    events: EventCsvRow[];
}

const WordClouds: React.FC<WordCloudsProps> = ({ events }) => {
    const [activeTab, setActiveTab] = useState<'overall' | 'positive' | 'negative'>('overall');
    const [wordData, setWordData] = useState<{ overall: WordData[]; positive: WordData[]; negative: WordData[] }>({
        overall: [],
        positive: [],
        negative: [],
    });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        const processed = processEventsForWordCloud(events);
        setWordData(processed);
        setLoading(false);
    }, [events]);

    const getCloudWords = useMemo(() => {
        switch (activeTab) {
            case 'positive':
                return wordData.positive;
            case 'negative':
                return wordData.negative;
            case 'overall':
            default:
                return wordData.overall;
        }
    }, [activeTab, wordData]);

    const getTitle = useMemo(() => {
        switch (activeTab) {
            case 'positive':
                return 'Positive Event & Crypto Word Cloud';
            case 'negative':
                return 'Negative Event & Crypto Word Cloud';
            case 'overall':
            default:
                return 'Overall Event & Crypto Word Cloud';
        }
    }, [activeTab]);

    if (loading) {
        return (
            <div className="bg-gray-800 p-4 rounded-lg animate-pulse h-96 flex items-center justify-center text-white">
                Loading Word Clouds...
            </div>
        );
    }

    return (
        <div className="w-full">
            <div className="flex justify-center mb-4">
                <button
                    className={`px-4 py-2 rounded-l-lg ${activeTab === 'overall' ? 'bg-indigo-600' : 'bg-gray-700'} text-white`}
                    onClick={() => setActiveTab('overall')}
                >
                    Overall
                </button>
                <button
                    className={`px-4 py-2 ${activeTab === 'positive' ? 'bg-indigo-600' : 'bg-gray-700'} text-white`}
                    onClick={() => setActiveTab('positive')}
                >
                    Positive
                </button>
                <button
                    className={`px-4 py-2 rounded-r-lg ${activeTab === 'negative' ? 'bg-indigo-600' : 'bg-gray-700'} text-white`}
                    onClick={() => setActiveTab('negative')}
                >
                    Negative
                </button>
            </div>
            {getCloudWords.length > 0 ? (
                <WordCloud words={getCloudWords} title={getTitle} sentimentType={activeTab} />
            ) : (
                <div className="bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg h-96 flex flex-col items-center justify-center text-gray-500 text-lg">
                    Not enough data for word cloud.
                </div>
            )}
        </div>
    );
};

export default WordClouds;