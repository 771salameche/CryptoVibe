import React from 'react';

const MOCK_POSTS = [
  { id: 1, source: 'Twitter', content: 'BTC to the moon! 🚀', sentiment: 0.95 },
  { id: 2, source: 'Reddit', content: 'Crypto is the future of finance.', sentiment: 0.85 },
  { id: 3, source: 'News', content: 'Bitcoin price surges after positive news.', sentiment: 0.75 },
  { id: 4, source: 'Twitter', content: 'I am not sure about the future of ETH', sentiment: -0.25 },
  { id: 5, source: 'Reddit', content: 'This is a scam.', sentiment: -0.95 },
];

const PostsPanel: React.FC = () => {
  return (
    <div>
      <div className="mb-4">
        <h3 className="text-lg font-bold mb-2">Top Positive</h3>
        <ul>
          {MOCK_POSTS.filter(p => p.sentiment > 0).map(post => (
            <li key={post.id} className="mb-2 p-3 bg-card/80 backdrop-blur-lg border border-border rounded-lg">
              <p className="text-sm text-card-foreground/70">{post.source}</p>
              <p className="text-card-foreground">{post.content}</p>
              <p className={`text-sm ${post.sentiment > 0 ? 'text-status-positive' : 'text-status-negative'}`}>{post.sentiment.toFixed(2)}</p>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-bold mb-2">Top Negative</h3>
        <ul>
          {MOCK_POSTS.filter(p => p.sentiment < 0).map(post => (
            <li key={post.id} className="mb-2 p-3 bg-card/80 backdrop-blur-lg border border-border rounded-lg">
              <p className="text-sm text-card-foreground/70">{post.source}</p>
              <p className="text-card-foreground">{post.content}</p>
              <p className={`text-sm ${post.sentiment > 0 ? 'text-status-positive' : 'text-status-negative'}`}>{post.sentiment.toFixed(2)}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default PostsPanel;
