import React from 'react';
import { EventCsvRow } from '../../utils/eventsDataProcessor'; // Import EventCsvRow
import { formatDistanceToNow } from 'date-fns';

interface EventCardProps {
  event: EventCsvRow;
}

const eventTypeColors: Record<string, string> = {
    'Regulatory Action': 'bg-yellow-500',
    'Hack / Exploit': 'bg-red-500',
    'Market Crash': 'bg-red-700',
    'Network Outage': 'bg-orange-500',
    'Exchange Listing': 'bg-green-500',
    'Partnership / Collaboration': 'bg-blue-500',
    // Add more as needed
    'LISTING': 'bg-green-500', // Kept for compatibility if old EventType is somehow used
    'HACK': 'bg-red-500',
    'REGULATION': 'bg-yellow-500',
    'PARTNERSHIP': 'bg-blue-500',
    'UPGRADE': 'bg-purple-500',
};

const sentimentColor = (sentiment: number) => {
    if (sentiment > 0.1) return 'text-green-400';
    if (sentiment < -0.1) return 'text-red-400';
    return 'text-gray-400';
}

const CryptoLogo: React.FC<{ crypto: string }> = ({ crypto }) => {
    const logos: Record<string, string> = {
        BITCOIN: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/btc.svg',
        ETHEREUM: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/eth.svg',
        SOLANA: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/sol.svg',
        XRP: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/xrp.svg',
        CARDANO: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/ada.svg',
        BNB: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/bnb.svg',
        DOGECOIN: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/doge.svg',
        POLKADOT: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/dot.svg',
        CHAINLINK: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/link.svg',
        LITECOIN: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/ltc.svg',
        TRON: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/trx.svg',
        MONERO: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/xmr.svg',
        ZCASH: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/zec.svg',
        'NEAR PROTOCOL': 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/near.svg',
        STELLAR: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/xlm.svg',
        'ETHEREUM CLASSIC': 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/etc.svg',
        AVALANCHE: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/avax.svg',
        CRONOS: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/cro.svg',
        HEDERA: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/hbar.svg',
        TONCOIN: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/ton.svg',
        SUI: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/sui.svg',
        KASPA: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/kas.svg',
        USDC: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/usdc.svg',
        TETHER: 'https://cdn.jsdelivr.net/gh/atomiclabs/cryptocurrency-icons@1a63530be6e374711a8554f31b17e4cb92c258d5/svg/color/usdt.svg',
        // Add other crypto logos as needed
    };
    const logoUrl = logos[crypto.toUpperCase()];
    if (!logoUrl) return null;
    return <img src={logoUrl} alt={`${crypto} logo`} className="w-5 h-5 mr-2" />;
};


const EventCard: React.FC<EventCardProps> = ({ event }) => {
  const eventTitle = `${event.event_type} - ${event.crypto}`;
  const eventDescription = `Mentions: ${event.mention_count}, Importance: ${event.total_importance.toFixed(2)}`;

  return (
    <div className="mb-8 ml-10 relative">
        <span className={`absolute -left-10 top-1 flex items-center justify-center w-8 h-8 ${eventTypeColors[event.event_type]} rounded-full ring-8 ring-gray-900`}>
            {/* Can be an icon later */}
        </span>

        <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-white/20 hover:border-white/40 transition-all duration-200 ease-in-out transform hover:-translate-y-1">
            <div className="flex justify-between items-start">
                <div className="flex items-center">
                    <CryptoLogo crypto={event.crypto} />
                    <h3 className="text-lg font-bold text-white">{eventTitle}</h3>
                </div>
                <span className={`px-2 py-1 text-xs font-bold rounded-full text-white ${eventTypeColors[event.event_type]}`}>{event.event_type}</span>
            </div>

            <p className="text-sm text-gray-300 mt-2">{eventDescription}</p>
            
            <div className="flex justify-between items-center mt-4 text-sm text-gray-400">
                <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586L5.586 15.414A1.994 1.994 0 015 14V6a2 2 0 012-2h10a2 2 0 012 2v2z" />
                    </svg>
                    <span>{event.mention_count} mentions</span>
                </div>
                <span className={sentimentColor(event.avg_sentiment)}>Sentiment: {event.avg_sentiment.toFixed(2)}</span>
                <span>{formatDistanceToNow(new Date(event.date), { addSuffix: true })}</span>
            </div>
        </div>
    </div>
  );
};

export default EventCard;