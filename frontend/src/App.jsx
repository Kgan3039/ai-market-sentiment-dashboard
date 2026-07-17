import { useEffect, useMemo, useState } from "react";
import Header from "./components/Header";
import CompanyOverview from "./components/CompanyOverview";
import FinancialMetrics from "./components/FinancialMetrics";
import NewsFeed from "./components/NewsFeed";
import SentimentPanel from "./components/SentimentPanel";
import Alerts from "./components/Alerts";
import PredictionModel from "./components/PredictionModel";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DEMO_TICKERS = ["NVDA", "TSLA"];

function formatTimestamp(value) {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleString();
}

function getAvailabilityLabel(item) {
  if (!item) return "Not queried";
  return item.status || (item.available ? "ready" : "unavailable");
}

function getAvailabilityType(item) {
  if (!item) return "info";
  if (item.status === "ready" || item.status === "live" || item.status === "cached") return "success";
  if (item.status === "fallback" || item.status === "partial") return "warning";
  if (item.status === "unavailable") return "warning";
  return item.available ? "success" : "warning";
}

function getAvailabilityMessage(item) {
  return item?.message || "Backend availability has not been reported yet.";
}

function getAvailabilityTitle(key, label, item) {
  if (key === "social_posts" && item?.source?.toLowerCase().includes("publisher")) {
    return "Publisher items";
  }
  return label;
}

function buildStatusAlerts(summary, error, loading) {
  if (error) {
    const isUnknownTicker = error.status === 404;
    const isInvalidTicker = error.status === 400;
    return [
      {
        id: "api-error",
        type: isUnknownTicker || isInvalidTicker ? "warning" : "danger",
        title: isUnknownTicker
          ? "Unknown symbol"
          : isInvalidTicker
          ? "Invalid ticker"
          : "Data request",
        message: error.message || "Unable to load dashboard data.",
        status: "Not available",
      },
    ];
  }

  if (!summary) {
    return [
      {
        id: "empty",
        type: loading ? "info" : "warning",
        title: loading ? "Loading data" : "No ticker loaded",
        message: loading
          ? "Fetching the latest market summary."
          : "Load a ticker to view the dashboard.",
        status: loading ? "Pending" : "Not available",
      },
    ];
  }

  const availability = summary.availability || summary.status || {};
  const sections = [
    ["market_data", "Market data"],
    ["sentiment", "Sentiment"],
    ["prediction", "Experimental signal"],
    ["headlines", "Headlines"],
    ["social_posts", "Social posts"],
    ["fundamentals", "Fundamentals"],
  ];

  return sections.map(([key, label]) => {
    const item = availability[key];

    return {
      id: key,
      type: getAvailabilityType(item),
      title: getAvailabilityTitle(key, label, item),
      status: getAvailabilityLabel(item),
      count: item?.count,
      source: item?.source,
      message: getAvailabilityMessage(item),
    };
  });
}

export default function App() {
  const [queryTicker, setQueryTicker] = useState("NVDA");
  const [activeTicker, setActiveTicker] = useState("NVDA");
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    async function fetchSummary() {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `${API_BASE}/dashboard/summary/${activeTicker}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          let detail = `Request failed with status ${response.status}`;
          try {
            const errorPayload = await response.json();
            detail = errorPayload?.detail || detail;
          } catch {
            // Keep the generic status message when the API does not return JSON.
          }
          const requestError = new Error(detail);
          requestError.status = response.status;
          throw requestError;
        }

        const payload = await response.json();
        setSummary(payload);
      } catch (err) {
        if (err.name === "AbortError") return;
        console.error(err);
        setSummary(null);
        setError({
          message: err.message || "Unable to load dashboard data.",
          status: err.status,
        });
      } finally {
        setLoading(false);
      }
    }

    fetchSummary();
    return () => controller.abort();
  }, [activeTicker]);

  function handleSubmit(event) {
    event.preventDefault();
    const normalized = queryTicker.trim().toUpperCase();
    if (!normalized) return;
    setQueryTicker(normalized);
    setActiveTicker(normalized);
  }

  function loadTicker(ticker) {
    setQueryTicker(ticker);
    setActiveTicker(ticker);
  }

  const alerts = useMemo(
    () => buildStatusAlerts(summary, error, loading),
    [summary, error, loading]
  );
  const availability = summary?.availability || summary?.status || {};
  const errorMessage = error?.message || null;
  const isUnsupportedTicker = error?.status === 404 || error?.status === 400;
  const showDashboardContent = !errorMessage || loading;

  return (
    <div className="app">
      <div className="app-bg" />
      <div className="dashboard">
        <section className="card search-bar">
          <div className="search-controls">
            <form className="search-form" onSubmit={handleSubmit}>
              <label className="search-label" htmlFor="ticker-input">
                Ticker
              </label>
              <input
                id="ticker-input"
                className="ticker-input"
                type="text"
                value={queryTicker}
                placeholder="NVDA"
                onChange={(event) =>
                  setQueryTicker(event.target.value.toUpperCase())
                }
              />
              <button className="search-submit" type="submit" disabled={loading}>
                Load
              </button>
            </form>
            <div className="demo-presets" aria-label="Demo tickers">
              <span className="demo-presets-label">Demo-ready</span>
              {DEMO_TICKERS.map((ticker) => (
                <button
                  key={ticker}
                  className={`preset-button ${activeTicker === ticker ? "active" : ""}`}
                  type="button"
                  onClick={() => loadTicker(ticker)}
                  disabled={loading && activeTicker === ticker}
                >
                  {ticker}
                </button>
              ))}
            </div>
          </div>
          <div className="status-strip">
            <span className="status-text">
              {errorMessage
                ? isUnsupportedTicker
                  ? "Local demo data is ready for NVDA and TSLA."
                  : errorMessage
                : summary?.updated_at
                ? `Last updated ${formatTimestamp(summary.updated_at)}`
                : "Enter a ticker to load market data"}
            </span>
          </div>
        </section>

        {errorMessage && !loading ? (
          <section className="card dashboard-error-state">
            <h2 className="section-title">
              {error?.status === 404
                ? "Symbol not found"
                : error?.status === 400
                ? "Check ticker format"
                : "Unable to load ticker"}
            </h2>
            <p className="overview-text">{errorMessage}</p>
            {isUnsupportedTicker ? (
              <div className="error-actions" aria-label="Load demo ticker">
                {DEMO_TICKERS.map((ticker) => (
                  <button
                    key={ticker}
                    className="preset-button active"
                    type="button"
                    onClick={() => loadTicker(ticker)}
                  >
                    Load {ticker}
                  </button>
                ))}
              </div>
            ) : null}
          </section>
        ) : null}

        {showDashboardContent ? (
          <>
            <Header
              ticker={summary?.ticker || activeTicker}
              marketData={summary?.market_data}
              exchange={summary ? "Market summary" : "Waiting for data"}
              updatedAt={formatTimestamp(summary?.updated_at)}
            />

            <CompanyOverview
              ticker={summary?.ticker || activeTicker}
              fundamentals={summary?.fundamentals}
              updatedAt={formatTimestamp(summary?.updated_at)}
            />

            <div className="grid-2col">
              <FinancialMetrics
                fundamentals={summary?.fundamentals}
                availability={availability?.fundamentals}
                loading={loading && !summary}
              />
              <Alerts alerts={alerts} />
            </div>

            <SentimentPanel
              sentiment={summary?.sentiment}
              marketHistory={summary?.market_history || []}
              ticker={summary?.ticker || activeTicker}
              loading={loading && !summary}
            />
            <NewsFeed
              news={summary?.headlines || []}
              socialPosts={summary?.social_posts || []}
              socialAvailability={availability?.social_posts}
              loading={loading && !summary}
            />
            <PredictionModel
              prediction={summary?.prediction}
              updatedAt={formatTimestamp(summary?.updated_at)}
            />
          </>
        ) : null}
      </div>
    </div>
  );
}
