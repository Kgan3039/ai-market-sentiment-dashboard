import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const TICKER_LABELS = {
  TSLA: "TSLA — Tesla",
  NVDA: "NVDA — NVIDIA",
  AMD: "AMD — Advanced Micro Devices",
  AAPL: "AAPL — Apple",
  META: "META — Meta Platforms",
};

function formatTime(value) {
  if (!value) return "--:--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "--:--";
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

async function requestJson(path, signal) {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
  return response.json();
}

function CitationLinks({ sentence, citations }) {
  const citationById = new Map(citations.map((citation) => [citation.id, citation]));

  return (
    <sup className="citation-links" aria-label="Cited sources">
      {sentence.citation_ids.map((citationId, index) => {
        const citation = citationById.get(citationId);
        if (!citation) return null;
        return (
          <a
            key={citationId}
            href={citation.url}
            target="_blank"
            rel="noreferrer"
            title={`Open source: ${citation.headline}`}
          >
            {index + 1}
          </a>
        );
      })}
    </sup>
  );
}

function StoryList({ stories }) {
  if (!stories.length) return null;

  return (
    <details className="story-list">
      <summary>Stories ({stories.length})</summary>
      <ul>
        {stories.map((story) => (
          <li key={story.id}>
            <a href={story.url} target="_blank" rel="noreferrer">
              {story.headline}
            </a>
            <span>{story.outlet}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function ThemeCard({ theme }) {
  return (
    <article className="theme-card">
      <div className="theme-card-heading">
        <span className="theme-rank">{String(theme.rank).padStart(2, "0")}</span>
        <div>
          <p className="eyebrow">Theme</p>
          <h3>{theme.label}</h3>
        </div>
        <span className="coverage-count">
          {theme.story_count} {theme.story_count === 1 ? "story" : "stories"} · {theme.outlet_count}{" "}
          {theme.outlet_count === 1 ? "outlet" : "outlets"}
        </span>
      </div>

      {theme.degraded ? (
        <p className="degraded-copy">Summary unavailable — source stories are still available</p>
      ) : (
        <div className="summary-copy">
          {theme.sentences.map((sentence, index) => (
            <p key={`${theme.id}-${index}`}>
              {sentence.text}
              <CitationLinks sentence={sentence} citations={theme.citations} />
            </p>
          ))}
        </div>
      )}

      <div className="source-block">
        <p className="source-heading">Cited sources</p>
        <div className="citation-list">
          {theme.citations.map((citation) => (
            <a
              key={citation.id}
              className="citation-chip"
              href={citation.url}
              target="_blank"
              rel="noreferrer"
            >
              <span>{citation.outlet}</span>
              Open source
            </a>
          ))}
        </div>
      </div>

      <StoryList stories={theme.stories} />
    </article>
  );
}

export default function App() {
  const [tickers, setTickers] = useState([]);
  const [activeTicker, setActiveTicker] = useState("TSLA");
  const [themesPayload, setThemesPayload] = useState(null);
  const [metaStatus, setMetaStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    document.title = "Ticker Narratives";

    async function loadPageMeta() {
      try {
        const [tickerPayload, statusPayload] = await Promise.all([
          requestJson("/api/v1/tickers", controller.signal),
          requestJson("/api/v1/meta/status", controller.signal),
        ]);
        setTickers(tickerPayload.tickers);
        setMetaStatus(statusPayload);
        setActiveTicker((currentTicker) =>
          tickerPayload.tickers.some((ticker) => ticker.ticker === currentTicker)
            ? currentTicker
            : tickerPayload.tickers[0]?.ticker || "TSLA"
        );
      } catch (requestError) {
        if (requestError.name !== "AbortError") setError(true);
      }
    }

    loadPageMeta();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(false);

    async function loadThemes() {
      try {
        const payload = await requestJson(`/api/v1/tickers/${activeTicker}/themes`, controller.signal);
        setThemesPayload(payload);
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          setThemesPayload(null);
          setError(true);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    loadThemes();
    return () => controller.abort();
  }, [activeTicker]);

  const dataAsOf = themesPayload?.data_as_of || metaStatus?.data_as_of;
  const isStale = metaStatus?.is_stale || tickers.find((ticker) => ticker.ticker === activeTicker)?.is_stale;
  const activeLabel = TICKER_LABELS[activeTicker] || activeTicker;

  return (
    <main className="narratives-app">
      <div className="topographic-wash" aria-hidden="true" />
      <header className="masthead">
        <div>
          <h1>Ticker Narratives</h1>
        </div>
        <p className="data-stamp">Data as of {formatTime(dataAsOf)}</p>
      </header>

      {isStale ? (
        <p className="stale-banner">Data may be delayed. Last successful update: {formatTime(dataAsOf)}.</p>
      ) : null}

      <nav className="ticker-tabs" aria-label="Ticker narratives">
        {(tickers.length ? tickers : Object.keys(TICKER_LABELS).map((ticker) => ({ ticker }))).map((ticker) => (
          <button
            key={ticker.ticker}
            className={ticker.ticker === activeTicker ? "ticker-tab active" : "ticker-tab"}
            type="button"
            onClick={() => setActiveTicker(ticker.ticker)}
          >
            {TICKER_LABELS[ticker.ticker] || ticker.ticker}
            {typeof ticker.theme_count === "number" ? <span>{ticker.theme_count}</span> : null}
          </button>
        ))}
      </nav>

      <section className="narratives-heading" aria-labelledby="coverage-heading">
        <p>Key narratives around today’s move</p>
        <h2 id="coverage-heading">Themes dominating current coverage</h2>
        <span>{activeLabel}</span>
      </section>

      {loading ? <section className="state-card">Loading current coverage…</section> : null}
      {error ? <section className="state-card error">Coverage is temporarily unavailable. Please try again shortly.</section> : null}

      {!loading && !error && themesPayload?.themes.length === 0 ? (
        <section className="state-card">
          <p>No current coverage for {activeTicker}.</p>
          <p>Check back after the next update.</p>
        </section>
      ) : null}

      {!loading && !error && themesPayload?.themes.length ? (
        <section className="theme-grid">
          {themesPayload.themes.map((theme) => (
            <ThemeCard key={theme.id} theme={theme} />
          ))}
        </section>
      ) : null}

      {!loading && !error && themesPayload?.other_coverage?.story_count ? (
        <section className="other-coverage">
          <div>
            <p className="eyebrow">Other coverage</p>
            <h2>Other coverage</h2>
          </div>
          <StoryList stories={themesPayload.other_coverage.stories} />
        </section>
      ) : null}

      <footer>AI-generated from cited sources. Informational only — not investment advice.</footer>
    </main>
  );
}
