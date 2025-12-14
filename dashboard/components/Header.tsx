import React from 'react';
import { NavLink } from 'react-router-dom';
import ThemeSwitcher from './ThemeSwitcher';
import Layout from './Layout';
import { LayoutGrid, Link as LinkIcon, Bitcoin } from 'lucide-react';

interface HeaderProps {
  // Removed startTour prop
}

const Header: React.FC<HeaderProps> = () => {
  return (
    <header className="sticky top-0 z-40 py-4 bg-bg-surface/70 backdrop-blur-2xl border-b border-border-default/60 shadow-card">
      <Layout>
        <div className="flex justify-between items-center gap-4">
          <NavLink
            to="/"
            aria-label="Go to CryptoVibe dashboard home"
            className="flex items-center space-x-2 text-2xl font-semibold text-fg-text"
          >
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-primary to-accent-secondary text-slate-900 shadow-lg">
              <Bitcoin size={22} aria-hidden="true" />
            </span>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-accent-primary via-white to-accent-secondary">
              CryptoVibe
            </span>
          </NavLink>
          <nav className="flex items-center space-x-3" aria-label="Primary">
            <NavLink
              to="/"
              aria-label="Dashboard"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-3.5 py-2 rounded-full text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-white/10 text-white shadow-card border border-white/10'
                    : 'text-fg-text-muted hover:text-fg-text hover:bg-white/5'
                }`
              }
            >
              <LayoutGrid size={18} aria-hidden="true" />
              <span>Dashboard</span>
            </NavLink>
            <NavLink
              to="/correlation"
              aria-label="Correlation"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-3.5 py-2 rounded-full text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-white/10 text-white shadow-card border border-white/10'
                    : 'text-fg-text-muted hover:text-fg-text hover:bg-white/5'
                }`
              }
            >
              <LinkIcon size={18} aria-hidden="true" />
              <span>Correlation</span>
            </NavLink>
            <div className="pl-4 border-l border-border-default/60">
              <ThemeSwitcher aria-label="Toggle theme" />
            </div>
          </nav>
        </div>
      </Layout>
    </header>
  );
};

export default Header;
