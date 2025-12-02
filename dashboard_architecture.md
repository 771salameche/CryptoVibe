
# CryptoSense Dashboard Architecture

## 1. Overall Layout Structure

The dashboard will use a modern, responsive layout that adapts to different screen sizes.

- **Header:**
    - **Logo:** "CryptoSense" text or a simple logo on the left.
    - **Navigation Menu:** Links to "Dashboard", "About", and "Settings".
    - **Theme Toggle:** A button to switch between light and dark modes.
- **Main Content Area:**
    - A vertically scrollable area containing all the dashboard sections.
- **Sidebar:**
    - A collapsible sidebar on the left (desktop) or a slide-in panel (mobile) for filters and controls.
- **Footer:**
    - **Credits:** "Made with ❤️ by CryptoSense"
    - **Links:** Links to GitHub, Twitter, etc.
- **Responsive Breakpoints:**
    - **Mobile:** `< 640px`
    - **Tablet:** `640px - 1024px`
    - **Desktop:** `> 1024px`

## 2. Page Sections (Top to Bottom)

The main content area will be organized into the following sections:

1.  **Hero Section:**
    - Three prominent metric cards displaying key real-time data:
        - **Current Sentiment:** A gauge or a card showing the overall market sentiment (e.g., "Bullish", "Bearish", "Neutral").
        - **Total Posts Analyzed:** A number showing the total volume of social media posts analyzed in the selected time frame.
        - **Active Cryptos:** The number of cryptocurrencies currently being tracked.
2.  **Main Chart:**
    - A large, interactive time-series line chart showing sentiment scores for selected cryptocurrencies over time.
3.  **Crypto Comparison:**
    - Three side-by-side cards, one for each major cryptocurrency (BTC, ETH, SOL), showing key stats:
        - Current price and 24h change.
        - Current sentiment score.
        - Volume of mentions.
4.  **Events Timeline:**
    - A vertically scrollable list of significant market events (e.g., "Binance Listing", "SEC Announcement"). Each event will have a date, a short description, and an icon.
5.  **Sentiment Distribution:**
    - A donut or pie chart showing the percentage breakdown of positive, negative, and neutral sentiment across all posts.
6.  **Additional Analytics:**
    - Optional expandable sections for more detailed analysis, such as:
        - Top Influencers
        - Hot Topics/Keywords

## 3. Color Scheme & Theme

A consistent and accessible color scheme will be used for both light and dark themes.

### Light Theme

- **Primary:** `#007BFF` (Blue)
- **Background:** `#F8F9FA` (Light Gray)
- **Cards/Widgets:** `#FFFFFF` (White)
- **Text:** `#212529` (Dark Gray)
- **Accent:** `#6C757D` (Gray)

### Dark Theme

- **Primary:** `#007BFF` (Blue)
- **Background:** `#121212` (Almost Black)
- **Cards/Widgets:** `#1E1E1E` (Dark Gray)
- **Text:** `#E0E0E0` (Light Gray)
- **Accent:** `#8C8C8C` (Gray)

### Semantic Colors

- **Positive:** `#28A745` (Green)
- **Negative:** `#DC3545` (Red)
- **Neutral:** `#6C757D` (Gray)

### Color Variables (CSS)

```css
:root {
  --primary-color: #007BFF;
  --background-color: #F8F9FA;
  --card-background: #FFFFFF;
  --text-color: #212529;
  --accent-color: #6C757D;
  --positive-color: #28A745;
  --negative-color: #DC3545;
  --neutral-color: #6C757D;
}

[data-theme='dark'] {
  --background-color: #121212;
  --card-background: #1E1E1E;
  --text-color: #E0E0E0;
  --accent-color: #8C8C8C;
}
```

### Accessibility

All color combinations will be checked to meet WCAG AA contrast ratio standards.

## 4. Component Specifications

### `MetricCard`

- **Purpose:** To display a single key metric.
- **Props:**
    - `title: string` (e.g., "Current Sentiment")
    - `value: string | number` (e.g., "Bullish", 12.345)
    - `change?: number` (e.g., 5.2 for a 5.2% change)
    - `icon?: React.ReactNode`
- **Data Format:** Simple primitives.
- **Interactions:** Hover effect to show a tooltip with more details.
- **Loading State:** A skeleton loader will be displayed while data is being fetched.
- **Error State:** A message like "Data unavailable" will be shown if fetching fails.

### `SentimentChart`

- **Purpose:** To visualize sentiment over time.
- **Props:**
    - `data: Array<{ date: string; sentiment: number; crypto: string }>`
- **Data Format:** An array of objects with date, sentiment score, and crypto name.
- **Interactions:**
    - Hover to show a tooltip with details for a specific date.
    - Click on a crypto in the legend to toggle its visibility.
    - Zoom and pan functionality.
- **Loading State:** A spinner or skeleton chart will be shown.
- **Error State:** A "Chart data could not be loaded" message.

### `CryptoComparisonCard`

- **Purpose:** To show a snapshot of a single cryptocurrency.
- **Props:**
    - `crypto: { name: string; symbol: string; price: number; change24h: number; sentiment: number; mentions: number }`
- **Data Format:** An object with crypto details.
- **Interactions:** Click to navigate to a more detailed page for that crypto.
- **Loading State:** Skeleton loader.
- **Error State:** "Data unavailable".

### `EventTimeline`

- **Purpose:** To display a list of market events.
- **Props:**
    - `events: Array<{ date: string; title: string; description: string; type: 'positive' | 'negative' | 'neutral' }>`
- **Data Format:** An array of event objects.
- **Interactions:** Click on an event to show a modal with more details.
- **Loading State:** A series of skeleton list items.
- **Error State:** "Events could not be loaded".

### `SentimentDonut`

- **Purpose:** To show the distribution of sentiment.
- **Props:**
    - `data: Array<{ name: 'positive' | 'negative' | 'neutral'; value: number }>`
- **Data Format:** An array of objects with sentiment name and percentage.
- **Interactions:** Hover over a segment to show its percentage.
- **Loading State:** A skeleton donut chart.
- **Error State:** "Data unavailable".

### `FilterSidebar`

- **Purpose:** To provide filtering options for the dashboard.
- **Props:**
    - `onFilterChange: (filters: { cryptos: string[]; dateRange: [Date, Date] }) => void`
- **Interactions:** Checkboxes to select cryptocurrencies, a date range picker to select the time frame.
- **Loading State:** Not applicable.
- **Error State:** Not applicable.

### `DateRangePicker`

- **Purpose:** To select a date range.
- **Props:**
    - `onChange: (dateRange: [Date, Date]) => void`
- **Interactions:** Calendar view to select start and end dates, or presets like "7d", "30d", "90d".
- **Loading State:** Not applicable.
- **Error State:** Not applicable.

## 5. Data Flow Architecture

- **Data Sources:**
    - Initial data will be loaded from JSON files (`sentiment_data.json`, `events_data.json`).
    - The architecture should be flexible to switch to a REST API or WebSocket connection in the future.
- **Data Loading Strategy:**
    - **On Mount:** Initial data for the default time frame will be loaded when the dashboard mounts.
    - **Lazy Load:** Data for sections below the fold (e.g., "Additional Analytics") can be loaded as the user scrolls them into view.
    - **Refresh:** A manual refresh button will be available. The dashboard can also be configured to refresh automatically at a set interval (e.g., every 5 minutes).
- **State Management:**
    - **React `useState`:** For simple, local component state.
    - **React `useContext` or a lightweight library like Zustand:** For global state like filters, theme, and fetched data that needs to be shared across multiple components.
- **Caching:**
    - Fetched data will be cached in the global state to avoid re-fetching when not necessary.
    - `React.memo` and `useMemo` will be used to memoize components and expensive calculations.

## 6. Interaction Patterns

- **Crypto Selector:** Checkboxes in the sidebar will allow users to select which cryptocurrencies to display on the main sentiment chart.
- **Date Range Filter:** A date range picker with presets will filter all data on the dashboard to the selected time frame.
- **Chart Interactions:**
    - **Hover:** Tooltips will provide detailed information on data points.
    - **Zoom/Pan:** Users can zoom in on specific time ranges in the main chart.
- **Export Functionality:** A "Download CSV" button will allow users to export the current view's data.
- **Responsive Behaviors:** On mobile, the sidebar will be a slide-in panel, and cards will stack vertically.

## 7. Technical Stack

- **Frontend:** React + TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Icons:** Lucide React
- **Animations:** Framer Motion (optional, for subtle animations)

## 8. File Structure

```
dashboard/
├── components/
│   ├── MetricCard.tsx
│   ├── SentimentChart.tsx
│   ├── CryptoCard.tsx
│   ├── EventTimeline.tsx
│   ├── SentimentDonut.tsx
│   ├── FilterSidebar.tsx
│   └── DateRangePicker.tsx
├── data/
│   ├── sentiment_data.json
│   └── events_data.json
├── utils/
│   ├── dataLoader.ts
│   └── formatters.ts
├── styles/
│   └── globals.css
└── App.tsx
```

## 9. Wireframes/Mockups

### Desktop Layout

```
+----------------------------------------------------------------------------------+
| CryptoSense | Dashboard | About | Settings | [Theme Toggle]                     |
+----------------------------------------------------------------------------------+
| [Filters] |                                                                    |
|           |  +-----------------+  +-----------------+  +-----------------+      |
|           |  | MetricCard 1    |  | MetricCard 2    |  | MetricCard 3    |      |
|           |  +-----------------+  +-----------------+  +-----------------+      |
|           |                                                                    |
|           |  +-------------------------------------------------------------+   |
|           |  |                                                             |   |
|           |  |                      SentimentChart                         |   |
|           |  |                                                             |   |
|           |  +-------------------------------------------------------------+   |
|           |                                                                    |
|           |  +-----------------+  +-----------------+  +-----------------+      |
|           |  | CryptoCard BTC  |  | CryptoCard ETH  |  | CryptoCard SOL  |      |
|           |  +-----------------+  +-----------------+  +-----------------+      |
|           |                                                                    |
+----------------------------------------------------------------------------------+
```

### Mobile Layout

```
+------------------------------------+
| [Menu] CryptoSense   [Theme Toggle]|
+------------------------------------+
| +--------------------------------+ |
| |          MetricCard 1          | |
| +--------------------------------+ |
| +--------------------------------+ |
| |          MetricCard 2          | |
| +--------------------------------+ |
| +--------------------------------+ |
| |          MetricCard 3          | |
| +--------------------------------+ |
| +--------------------------------+ |
| |                                | |
| |         SentimentChart         | |
| |                                | |
| +--------------------------------+ |
| ... (other sections stack) ...     |
+------------------------------------+
```

## 10. Performance Considerations

- **Lazy Load:** Components below the fold (like the events timeline and additional analytics) will be lazy-loaded using `React.lazy`.
- **Virtualize:** The events timeline will use a virtualization library like `react-window` or `react-virtual` to efficiently render long lists.
- **Debounce:** Filter inputs in the sidebar will be debounced to prevent excessive re-renders while the user is typing or selecting.
- **Memoize:** `React.memo` and `useMemo` will be used to prevent unnecessary re-renders of components and re-computation of expensive calculations.
- **Target Load Time:** Aim for a First Contentful Paint (FCP) of less than 3 seconds on a standard 3G connection.

## 11. Accessibility

- **Keyboard Navigation:** All interactive elements will be reachable and operable via the keyboard.
- **ARIA Labels:** ARIA attributes will be used to provide context for screen readers, especially for charts and interactive controls.
- **Screen Reader Compatibility:** The application will be tested with screen readers (e.g., NVDA, VoiceOver) to ensure a good user experience.
- **Focus Indicators:** Clear and visible focus indicators will be provided for all focusable elements.
- **Alt Text:** Alt text or equivalent alternatives will be provided for all visualizations.

## 12. Responsive Design

- **Mobile (<640px):** A single-column layout. Cards will stack vertically. The sidebar will be a slide-in panel.
- **Tablet (640px - 1024px):** A two-column grid for cards and other content.
- **Desktop (>1024px):** A multi-column layout with a persistent sidebar on the left.
- **Breakpoints:**
    - `sm`: 640px
    - `md`: 768px
    - `lg`: 1024px
    - `xl`: 1280px
