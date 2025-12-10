import React, { useState, ReactNode, useEffect } from 'react';

interface CollapsibleBottomPanelProps {
  children: ReactNode;
  title: string;
}

const CollapsibleBottomPanel: React.FC<CollapsibleBottomPanelProps> = ({ children, title }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(
    () => {
      // Initialize from localStorage, default to false if not found
      const savedState = localStorage.getItem('collapsibleBottomPanelExpanded');
      return savedState ? JSON.parse(savedState) : false;
    }
  );

  useEffect(() => {
    // Save state to localStorage whenever it changes
    localStorage.setItem('collapsibleBottomPanelExpanded', JSON.stringify(isExpanded));
  }, [isExpanded]);

  const togglePanel = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-800 text-white shadow-lg z-50 transition-all duration-300 ease-in-out"
         style={{ height: isExpanded ? '50vh' : '40px' }}>
      <div className="flex justify-between items-center p-2 cursor-pointer" onClick={togglePanel}>
        <h3 className="text-lg font-semibold">{title}</h3>
        <button className="p-1 focus:outline-none">
          {isExpanded ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          )}
        </button>
      </div>
      <div className={`overflow-y-auto px-4 pb-4 ${isExpanded ? 'block' : 'hidden'}`} style={{ maxHeight: 'calc(50vh - 40px)' }}>
        {children}
      </div>
    </div>
  );
};

export default CollapsibleBottomPanel;
