# Phase 0 approved copy deck

This is the single source of truth for all user-facing Phase 0 wording. It is
binding for the summary linter in issue #71 and the UI copy audit in issue #75.
Use the strings below verbatim unless a documented placeholder is being
substituted. Describe coverage; do not claim that coverage caused a market move,
predict a move, or give investment advice.

## Approved UI copy

| Surface | Approved copy | Usage |
| --- | --- | --- |
| Product/page title | `Ticker Narratives` | Browser and page title. |
| Ticker tab | `TSLA — Tesla` | Use the same pattern for all five tabs below. |
| Ticker tab | `NVDA — NVIDIA` |  |
| Ticker tab | `AMD — Advanced Micro Devices` |  |
| Ticker tab | `AAPL — Apple` |  |
| Ticker tab | `META — Meta Platforms` |  |
| Themes section heading | `Themes dominating current coverage` | Primary heading above theme cards. |
| Current-activity framing | `Key narratives around today’s move` | Use only as a framing label; it does not explain the move. |
| Coverage descriptor | `Coverage today is dominated by: {theme}` | `{theme}` is a cited, neutral theme label. |
| Coverage descriptor | `The most-covered storyline is: {theme}` | Use where a single theme is highlighted. |
| Theme-card label | `Theme` | Label for a theme card. |
| Theme-card label | `Other coverage` | Bucket for stories outside listed themes. |
| Theme-card label | `Stories` | Story-list label. |
| Source/citation label | `Cited sources` | Heading for linked publisher stories. |
| Source/citation action | `Open source` | Link action; opens the publisher article. |
| Freshness | `Data as of HH:MM` | Replace `HH:MM` with the latest successful run time in the displayed timezone. |
| Stale state | `Data may be delayed. Last successful update: HH:MM.` | Show when the last successful run is more than 60 minutes old during market hours. |
| Empty state | `No current coverage for {ticker}.` | `{ticker}` is the active ticker symbol. |
| Empty-state support | `Check back after the next update.` | Pair with the empty-state message. |
| Degraded-summary state | `Summary unavailable — source stories are still available` | Required fallback when a summary fails guardrails. |
| Loading state | `Loading current coverage…` | While the active ticker is loading. |
| Error state | `Coverage is temporarily unavailable. Please try again shortly.` | For recoverable display failures only. |
| Standing disclosure | `AI-generated from cited sources. Informational only — not investment advice.` | Render persistently on the page. |

## Copy rules

- Use “Themes dominating current coverage” and “Key narratives around today’s
  move” as descriptions of coverage, never as causal explanations.
- Every summary sentence needs a cited source; source labels link to the
  publisher rather than restating a conclusion as fact.
- Do not expose product-authored model confidence, probability, forecast,
  recommendation, or trading-action language. A factual, cited description of
  an analyst rating or price-target change is allowed; do not turn it into a
  product endorsement.
- The stale, empty, degraded, loading, and error states above are the only
  approved Phase 0 state messages.

## Implementation notes

`config/banned_phrases.txt` is for generated product copy and summaries only.
It must not be applied to raw publisher headlines, descriptions, or source
quotes. Match rules case-insensitively. Future linter work may add context-aware
checks, but must preserve the literal and regex rule semantics documented in
that file. The executable content-scope contract is:

- Apply rules to `generated_summary`, `generated_label`, and `product_ui_copy`.
- Do not apply rules to `publisher_headline`, `publisher_description`, or
  `publisher_quotation`.
