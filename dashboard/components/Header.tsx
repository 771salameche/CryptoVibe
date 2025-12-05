import React from 'react';
import { Link } from 'react-router-dom';
import ThemeSwitcher from './ThemeSwitcher';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 text-white p-4 flex justify-between items-center">
      <Link to="/" className="text-2xl font-bold">CryptoVibe</Link>
      <nav className="flex items-center space-x-4">
        <Link to="/" className="px-4">Sentiment Timeline</Link>
        <Link to="/correlation" className="px-4">Price Correlation</Link>
        <ThemeSwitcher />
      </nav>
    </header>
  );
};

export default Header;