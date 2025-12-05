// dashboard/styles/theme.ts

const tokens = {
  spacing: {
    '0': '0',
    '1': '0.25rem',
    '2': '0.5rem',
    '3': '0.75rem',
    '4': '1rem',
    '5': '1.25rem',
    '6': '1.5rem',
    '8': '2rem',
    '10': '2.5rem',
    '12': '3rem',
    '16': '4rem',
    '24': '6rem',
    '32': '8rem',
  },
  radii: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '1rem',
    full: '9999px',
  },
};

export const theme = {
  ...tokens,
  light: {
    bg: {
      app: 'linear-gradient(to bottom, #f0f4f8, #d9e2ec)',
      surface: 'rgba(255, 255, 255, 0.5)',
      surfaceMuted: 'rgba(255, 255, 255, 0.2)',
    },
    fg: {
      text: '#1a202c',
      textMuted: '#718096',
    },
    accent: {
      primary: '#3b82f6',
      secondary: '#8b5cf6',
    },
    status: {
      positive: '#10b981',
      negative: '#ef4444',
      neutral: '#6b7280',
    },
    border: {
      default: 'rgba(255, 255, 255, 0.2)',
    },
    shadow: {
      card: '0 4px 6px rgba(0, 0, 0, 0.05)',
    },
  },
  dark: {
    bg: {
      app: 'linear-gradient(to bottom, #1e293b, #312e81)',
      surface: 'rgba(30, 41, 59, 0.5)',
      surfaceMuted: 'rgba(30, 41, 59, 0.2)',
    },
    fg: {
      text: '#f7fafc',
      textMuted: '#a0aec0',
    },
    accent: {
      primary: '#3b82f6',
      secondary: '#8b5cf6',
    },
    status: {
      positive: '#34d399',
      negative: '#f87171',
      neutral: '#9ca3af',
    },
    border: {
      default: 'rgba(255, 255, 255, 0.1)',
    },
    shadow: {
      card: '0 4px 6px rgba(0, 0, 0, 0.2)',
    },
  },
};
