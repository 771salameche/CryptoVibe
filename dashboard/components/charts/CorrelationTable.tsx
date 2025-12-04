import React, { useState } from 'react';

interface CorrelationTableProps {
  data: {
    crypto: string;
    sameDay: number;
    nextDay: number;
    optimalLag: string;
    pValue: number;
    isSignificant: boolean;
  }[];
  onRowClick: (crypto: string) => void;
  highlightedCrypto?: string | null;
}

const CorrelationTable: React.FC<CorrelationTableProps> = ({ data, onRowClick, highlightedCrypto }) => {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const sortedData = [...data].sort((a, b) => {
    if (!sortColumn) return 0;

    const aValue = (a as any)[sortColumn];
    const bValue = (b as any)[sortColumn];

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortDirection === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
    }
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
    }
    return 0;
  });

  const getSortIndicator = (column: string) => {
    if (sortColumn === column) {
      return sortDirection === 'asc' ? ' ▲' : ' ▼';
    }
    return '';
  };

  return (
    <div className="overflow-x-auto bg-gray-800 bg-opacity-30 p-4 rounded-xl backdrop-filter backdrop-blur-lg border border-gray-700 shadow-lg">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="p-2 cursor-pointer" onClick={() => handleSort('crypto')}>
              Crypto {getSortIndicator('crypto')}
            </th>
            <th className="p-2 cursor-pointer" onClick={() => handleSort('sameDay')}>
              Same-Day Corr {getSortIndicator('sameDay')}
            </th>
            <th className="p-2 cursor-pointer" onClick={() => handleSort('nextDay')}>
              Next-Day Corr {getSortIndicator('nextDay')}
            </th>
            <th className="p-2 cursor-pointer" onClick={() => handleSort('optimalLag')}>
              Optimal Lag {getSortIndicator('optimalLag')}
            </th>
            <th className="p-2 cursor-pointer" onClick={() => handleSort('pValue')}>
              P-value {getSortIndicator('pValue')}
            </th>
            <th className="p-2 cursor-pointer" onClick={() => handleSort('isSignificant')}>
              Significant? {getSortIndicator('isSignificant')}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map(row => (
            <tr
              key={row.crypto}
              className={`hover:bg-gray-700 cursor-pointer ${highlightedCrypto === row.crypto ? 'bg-indigo-700' : ''}`}
              onClick={() => onRowClick(row.crypto)}
            >
              <td className="p-2">{row.crypto}</td>
              <td className="p-2">{row.sameDay.toFixed(2)}</td>
              <td className="p-2">{row.nextDay.toFixed(2)}</td>
              <td className="p-2">{row.optimalLag}</td>
              <td className="p-2">{row.pValue.toFixed(3)}</td>
              <td className={`p-2 ${row.isSignificant ? 'text-green-500' : 'text-red-500'}`}>
                {row.isSignificant ? '✓ Yes' : '✗ No'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default CorrelationTable;