import React from 'react';
import { Tooltip } from 'react-tooltip';
import 'react-tooltip/dist/react-tooltip.css';

interface InfoIconProps {
  id: string;
  content: string;
}

const InfoIcon: React.FC<InfoIconProps> = ({ id, content }) => {
  return (
    <>
      <a data-tooltip-id={id} data-tooltip-content={content}>
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 hover:text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </a>
      <Tooltip id={id} />
    </>
  );
};

export default InfoIcon;
