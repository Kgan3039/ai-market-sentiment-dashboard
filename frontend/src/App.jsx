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

function formatTimestamp(value) {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleString();
}

function buildStatusAlerts(summary, error, loading) {
  if (error) {
    return [
      {
        id: "api-error",
        type: "danger",
        title: "API Error",
        message: error,
        status: "failed",
      },
    ];
  }

  if (!summary) {
    return [
      {
        id: "empty",
        type: loading ? "info" : "warning",
        title: loading ? "Loading Summary" : "No Data Loaded",
        message: loading
          ? "Fetching the latest dashboard summary from the backend."
          : "Load a ticker to render the backend summary response.",
        status: loading ? "loading" : "empty",
      },
    ];
  }

  const availability = summary.availability || summary.status || {};
  const sections = [
    ["sentiment", "Sentiment"],
    ["market_data", "Market Data"],
    ["prediction", "Prediction"],
    ["headlines", "Headlines"],
    ["fundamentals", "Fundamentals"],
  ];

  return sections.map(([key, label]) => {
    const item = availability[key];
    const available = item?.available ?? false;
    const status = item?.status || (available ? "ready" : "unavailable");
    const type =
      status === "fallback" || status === "partial"
        ? "warning"
        : available
          ? "success"
          : "warning";

    return {
      id: key,
      type,
      title: label,
      status,
      source: item?.source,
      count: item?.count,
      message:
        item?.message ||
        `${label} data is ${available ? "available" : "not available"}.`,
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
          throw new Error(`Request failed with status ${response.status}`);
        }

        const payload = await response.json();
        setSummary(payload);
      } catch (err) {
        if (err.name === "AbortError") return;
        console.error(err);
        setSummary(null);
        setError(err.message || "Unable to load dashboard data.");
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
    setActiveTicker(normalized);
  }

  const alerts = useMemo(
    () => buildStatusAlerts(summary, error, loading),
    [summary, error, loading]
  );

  return (
    <div className="app">
      <div className="app-bg" />
      <div className="dashboard">
        <section className="card search-bar">
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
            <button className="search-submit" type="submit">
              Load
            </button>
          </form>
          <div className="status-strip">
            <span
              className={`status-pill ${
                error ? "error" : loading ? "loading" : "idle"
              }`}
            >
              {error ? "Error" : loading ? "Loading" : "Ready"}
            </span>
            <span className="status-text">
              Source: `/dashboard/summary/{activeTicker}`
            </span>
          </div>
        </section>

        <Header
          ticker={summary?.ticker || activeTicker}
          price={summary?.market_data?.price}
          exchange={summary ? "API SUMMARY" : "WAITING"}
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
            availability={summary?.availability?.fundamentals}
            loading={loading && !summary}
          />
          <Alerts alerts={alerts} />
        </div>

        <SentimentPanel sentiment={summary?.sentiment} loading={loading && !summary} />
        <NewsFeed
          news={summary?.headlines || []}
          availability={summary?.availability?.headlines}
          loading={loading && !summary}
        />
        <PredictionModel
          prediction={summary?.prediction}
          updatedAt={formatTimestamp(summary?.updated_at)}
        />
      </div>
    </div>
  );
}
