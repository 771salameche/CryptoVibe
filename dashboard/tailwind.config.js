
/** @type {import('tailwindcss').Config} */
import { theme } from './styles/theme';

export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-app': 'var(--bg-app)',
        'bg-surface': 'var(--bg-surface)',
        'bg-surface-muted': 'var(--bg-surface-muted)',
        'fg-text': 'var(--fg-text)',
        'fg-text-muted': 'var(--fg-text-muted)',
        'accent-primary': 'var(--accent-primary)',
        'accent-secondary': 'var(--accent-secondary)',
        'status-positive': 'var(--status-positive)',
        'status-negative': 'var(--status-negative)',
        'status-neutral': 'var(--status-neutral)',
        'border-default': 'var(--border-default)',
      },
      spacing: theme.spacing,
      borderRadius: theme.radii,
      boxShadow: {
        card: 'var(--shadow-card)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
      },
      animation: {
        shimmer: 'shimmer 2s infinite linear',
      },
    },
  },
  plugins: [],
}
