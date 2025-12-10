import React, { ReactNode } from 'react';
import CollapsibleBottomPanel from './CollapsibleBottomPanel';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {children}
      <CollapsibleBottomPanel title="Activity Log">
        <div className="p-4 text-gray-300">
          <p>This is your activity log. Future updates will appear here.</p>
          <ul className="list-disc pl-5 mt-2">
            <li>[10:30 AM] Bitcoin price updated.</li>
            <li>[10:25 AM] New sentiment data processed.</li>
            <li>[10:15 AM] Ethereum event detected.</li>
          </ul>
        </div>
      </CollapsibleBottomPanel>
    </div>
  );
};

export default Layout;
