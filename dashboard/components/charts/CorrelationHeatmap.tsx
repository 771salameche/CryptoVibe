import React, { useState } from 'react';

interface CorrelationHeatmapProps {
  data: {
    // Example: { type: 'BTC Sentiment', 'BTC Price': 0.8, 'ETH Price': 0.6, 'SOL Price': 0.4, 'BTC Volume': 0.5, ... }
    type: string;
    [key: string]: number | string;
  }[];
  colHeaders: string[]; // e.g., ['BTC Price', 'ETH Price', 'SOL Price', 'BTC Volume']
  rowHeaders: string[]; // e.g., ['BTC Sentiment', 'ETH Sentiment', 'SOL Sentiment']
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ data, colHeaders, rowHeaders }) => {
  const [hoveredCell, setHoveredCell] = useState<{ row: string; col: string; value: number; pValue?: number } | null>(null);

  const getColor = (value: number) => {
    const intensity = Math.abs(value);
    const red = value < 0 ? 255 : Math.round(255 * (1 - intensity));
    const green = value > 0 ? 255 : Math.round(255 * (1 - intensity));
    const blue = Math.round(255 * (1 - intensity));
    return `rgb(${red}, ${green}, ${blue})`;
  };

  const correlationData = data.reduce((acc, row) => {
    acc[row.type] = row;
    return acc;
  }, {} as { [key: string]: any });

  return (
    <div className="p-4 relative">
      <h3 className="text-xl font-bold mb-4 text-white text-center">Cross-Crypto Correlation Matrix</h3>
      <div className="flex flex-col items-center">
        {/* Column Headers */}
        <div className="grid gap-1 mb-1" style={{ gridTemplateColumns: `80px repeat(${colHeaders.length}, 1fr)` }}>
          <div className="h-8"></div> {/* Corner spacer */}
          {colHeaders.map((header) => (
            <div key={header} className="text-center font-semibold text-gray-300 text-sm rotate-45 transform origin-bottom-left w-max py-2 px-1">
              {header.replace(' Sentiment', '').replace(' Price', '').replace(' Volume', '')}
            </div>
          ))}
        </div>

        {/* Heatmap Grid */}
        <div className="grid gap-1" style={{ gridTemplateColumns: `80px repeat(${colHeaders.length}, 1fr)` }}>
          {rowHeaders.map((rowHeader) => (
            <React.Fragment key={rowHeader}>
              {/* Row Header */}
              <div className="h-10 flex items-center justify-end pr-2 text-right font-semibold text-gray-300 text-sm">
                {rowHeader.replace(' Sentiment', '')}
              </div>
              {/* Cells */}
              {colHeaders.map((colHeader) => {
                const value = correlationData[rowHeader] ? (correlationData[rowHeader][colHeader] || 0) : 0;
                // For p-value and significance, you'd need that data in correlationData
                // For now, mock pValue and significance
                const pValue = value > 0.7 || value < -0.7 ? 0.01 : (value > 0.3 || value < -0.3 ? 0.05 : 0.2);
                const isSignificant = pValue < 0.05;

                return (
                  <div
                    key={`${rowHeader}-${colHeader}`}
                    className="relative w-10 h-10 flex items-center justify-center rounded transition-all duration-100"
                    style={{ backgroundColor: getColor(value) }}
                    onMouseEnter={() => setHoveredCell({ row: rowHeader, col: colHeader, value, pValue })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    <span className="text-xs font-bold text-gray-900 drop-shadow-sm">
                      {value.toFixed(2)}
                    </span>
                    {hoveredCell && hoveredCell.row === rowHeader && hoveredCell.col === colHeader && (
                      <div className="absolute z-10 -top-2 left-1/2 -translate-x-1/2 -translate-y-full bg-gray-700 text-white text-xs px-2 py-1 rounded-md shadow-lg whitespace-nowrap">
                        <p>Corr: {hoveredCell.value.toFixed(2)}</p>
                        {hoveredCell.pValue && <p>P-value: {hoveredCell.pValue.toFixed(3)}</p>}
                        <p>{isSignificant ? 'Significant (p<0.05)' : 'Not Significant'}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CorrelationHeatmap;