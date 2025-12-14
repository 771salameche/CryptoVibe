import React, { ReactNode } from 'react';
import CollapsibleBottomPanel from './CollapsibleBottomPanel';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="rounded-3xl border border-border-default/60 bg-bg-surface shadow-card backdrop-blur-2xl p-6 sm:p-8 mt-6 ring-1 ring-white/5">
        {children}
      </div>
      <CollapsibleBottomPanel title="Activity Log">
        <div className="p-4 text-fg-text-muted">
          <p>Live pipeline checkpoints (mocked for now):</p>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li>[10:30] BTC price updated from yfinance.</li>
            <li>[10:25] New FinBERT sentiment batch processed.</li>
            <li>[10:15] WebSocket broadcast delivered to dashboard.</li>
          </ul>
        </div>
      </CollapsibleBottomPanel>
    </div>
  );
};

export default Layout;
