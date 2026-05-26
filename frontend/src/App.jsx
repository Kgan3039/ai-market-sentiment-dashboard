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

function getAvailabilityLabel(item) {
  if (!item) return "Pending";
  if (item.status === "fallback" || item.status === "partial") return "Pending";
  return item.available ? "Available" : "Not available";
}

function getAvailabilityType(item) {
  if (!item) return "info";
  if (item.status === "fallback" || item.status === "partial") return "warning";
  return item.available ? "success" : "warning";
}

function getAvailabilityMessage(label, item) {
  if (item?.available) return `${label} available.`;
  if (label === "Sentiment") {
    return "Aggregate sentiment unavailable until enough validated text data is available.";
  }
  if (label === "Prediction") {
    return "Prediction unavailable until enough validated input data is available.";
  }
  if (label === "Social posts") return "No social posts available for this ticker yet.";
  if (label === "Headlines") return "No headlines available for this ticker yet.";
  if (label === "Fundamentals") return "Fundamentals are not available in the current MVP.";
  return `${label} not available.`;
}

function buildStatusAlerts(summary, error, loading) {
  if (error) {
    return [
      {
        id: "api-error",
        type: "danger",
        title: "Data request",
        message: error,
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
    ["prediction", "Prediction"],
    ["headlines", "Headlines"],
    ["social_posts", "Social posts"],
  ];

  return sections.map(([key, label]) => {
    const item = availability[key];

    return {
      id: key,
      type: getAvailabilityType(item),
      title: label,
      status: getAvailabilityLabel(item),
      count: item?.count,
      message: getAvailabilityMessage(label, item),
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
            <span className="status-text">
              {summary?.updated_at
                ? `Last updated ${formatTimestamp(summary.updated_at)}`
                : "Enter a ticker to load market data"}
            </span>
          </div>
        </section>

        <Header
          ticker={summary?.ticker || activeTicker}
          price={summary?.market_data?.price}
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
            loading={loading && !summary}
          />
          <Alerts alerts={alerts} />
        </div>

        <SentimentPanel sentiment={summary?.sentiment} loading={loading && !summary} />
        <NewsFeed
          news={summary?.headlines || []}
          socialPosts={summary?.social_posts || []}
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
