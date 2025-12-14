import React from 'react';
import { BookOpen, Code2, Github } from 'lucide-react';

const links = [
  { label: 'GitHub', href: 'https://github.com', icon: <Github size={18} aria-hidden="true" /> },
  { label: 'Docs', href: '#', icon: <BookOpen size={18} aria-hidden="true" /> },
  { label: 'API', href: '#', icon: <Code2 size={18} aria-hidden="true" /> },
];

const Footer: React.FC = () => {
  return (
    <footer className="relative mt-10 mb-6 rounded-3xl border border-border-default/60 bg-bg-surface/90 shadow-card backdrop-blur-xl px-6 sm:px-8 py-6 text-sm text-fg-text">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <p className="text-fg-text-muted">Powered by CryptoVibe | Data updated every 5 minutes</p>
          <div className="flex items-center gap-4 text-fg-text">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.href}
                aria-label={link.label}
                className="inline-flex items-center gap-2 px-2 py-1 rounded-full hover:bg-white/5 transition-colors"
              >
                {link.icon}
                <span>{link.label}</span>
              </a>
            ))}
          </div>
        </div>
        <hr className="border-border-default/40" />
        <div className="flex justify-center text-fg-text-muted">
          © 2025 CryptoVibe. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
