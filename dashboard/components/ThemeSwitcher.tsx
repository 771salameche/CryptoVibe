import React, { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';

const ThemeSwitcher: React.FC = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof localStorage !== 'undefined' && localStorage.theme) {
      return localStorage.theme as 'light' | 'dark';
    }
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-full bg-bg-surface-muted text-fg-text-muted hover:text-fg-text transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent-primary"
      aria-label="Toggle theme"
    >
      <Sun size={20} className={`transition-transform duration-500 ${theme === 'dark' ? 'rotate-90 scale-0' : 'rotate-0 scale-100'}`} />
      <Moon size={20} className={`absolute transition-transform duration-500 ${theme === 'light' ? 'rotate-90 scale-0' : 'rotate-0 scale-100'}`} />
    </button>
  );
};

export default ThemeSwitcher;
