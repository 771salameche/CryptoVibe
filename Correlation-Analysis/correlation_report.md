# Crypto Sentiment-Price Correlation Analysis

This report details the statistical correlation between daily social media sentiment and cryptocurrency price changes.

| lag      | type     |          r |   p_value | crypto   | is_significant   | strength   | interpretation                 |
|:---------|:---------|-----------:|----------:|:---------|:-----------------|:-----------|:-------------------------------|
| Same-Day | Pearson  |  0.0485271 | 0.757307  | BTC      | No               | Weak       | Not statistically significant. |
| Same-Day | Spearman |  0.051193  | 0.744412  | BTC      | No               | Weak       | Not statistically significant. |
| Next-Day | Pearson  | -0.111276  | 0.477459  | BTC      | No               | Moderate   | Not statistically significant. |
| Next-Day | Spearman | -0.146481  | 0.3486    | BTC      | No               | Moderate   | Not statistically significant. |
| Same-Day | Pearson  |  0.20874   | 0.184631  | ETH      | No               | Moderate   | Not statistically significant. |
| Same-Day | Spearman |  0.237349  | 0.130154  | ETH      | No               | Moderate   | Not statistically significant. |
| Next-Day | Pearson  | -0.292345  | 0.0602841 | ETH      | No               | Moderate   | Not statistically significant. |
| Next-Day | Spearman | -0.266845  | 0.0875818 | ETH      | No               | Moderate   | Not statistically significant. |
| Same-Day | Pearson  |  0.0260282 | 0.868411  | SOL      | No               | Weak       | Not statistically significant. |
| Same-Day | Spearman | -0.0581102 | 0.711279  | SOL      | No               | Weak       | Not statistically significant. |
| Next-Day | Pearson  | -0.177716  | 0.254234  | SOL      | No               | Moderate   | Not statistically significant. |
| Next-Day | Spearman | -0.209075  | 0.178456  | SOL      | No               | Moderate   | Not statistically significant. |
## Key Insights

- **Strongest Relationship:** No statistically significant (p < 0.05) correlations were found between sentiment and price changes.
- **Same-Day vs. Next-Day:** The strongest same-day correlation was 0.237 (p=0.130) for ETH, while the strongest next-day (predictive) correlation was -0.292 (p=0.060) for ETH.
- **Predictive Power:** No significant link was found between today's sentiment and tomorrow's price changes for any of the analyzed cryptocurrencies.