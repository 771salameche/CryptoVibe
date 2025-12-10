import React from 'react';
import { NavLink } from 'react-router-dom';
import ThemeSwitcher from './ThemeSwitcher';
import Layout from './Layout';
import { LayoutGrid, Link as LinkIcon, Bitcoin } from 'lucide-react'; // Removed HelpCircle icon

interface HeaderProps {
  // Removed startTour prop
}

const Header: React.FC<HeaderProps> = () => {
  return (
    <header className="sticky top-0 z-50 py-4 bg-bg-surface/80 backdrop-blur-lg border-b border-border-default">
      <Layout>
        <div className="flex justify-between items-center">
          <NavLink to="/" className="flex items-center space-x-2 text-2xl font-bold text-accent-primary">
            <Bitcoin />
            <span>CryptoVibe</span>
          </NavLink>
          <nav className="flex items-center space-x-6">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? 'text-accent-primary' : 'text-fg-text-muted hover:text-fg-text'
                }`
              }
            >
              <LayoutGrid size={18} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink
              to="/correlation"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? 'text-accent-primary' : 'text-fg-text-muted hover:text-fg-text'
                }`
              }
            >
              <LinkIcon size={18} />
              <span>Correlation</span>
            </NavLink>
            {/* Removed Help button */}
            <ThemeSwitcher />
          </nav>
        </div>
      </Layout>
    </header>
  );
};

export default Header;