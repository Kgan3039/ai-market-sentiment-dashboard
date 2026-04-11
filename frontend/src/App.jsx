import { useState, useEffect } from "react";
import Header from "./components/Header";
import CompanyOverview from "./components/CompanyOverview";
import FinancialMetrics from "./components/FinancialMetrics";
import NewsFeed from "./components/NewsFeed";
import SentimentPanel from "./components/SentimentPanel";
import Alerts from "./components/Alerts";
import PredictionModel from "./components/PredictionModel";
import "./App.css";

// Backend API base URL
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Fallback dummy data if API fails
const DUMMY_DATA = {
  ticker: "AAPL",
  companyName: "Apple Inc.",
  exchange: "NASDAQ",
  price: 213.49,
  priceChange: +1.24,
  priceChangePct: +0.58,
  overview:
    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. It also sells various related services including the App Store, Apple Music, iCloud, and Apple TV+.",

  financials: {
    annual: {
      revenue: "$383.9B",
      netIncome: "$97.0B",
      operatingCashFlow: "$113.0B",
      eps: "$6.13",
    },
    quarterly: {
      revenue: "$94.9B",
      netIncome: "$23.6B",
      operatingCashFlow: "$29.9B",
      eps: "$1.52",
    },
  },

  ratios: {
    pe: 32.4,
    eps: 6.13,
    roe: "160.6%",
    debtToEquity: 1.87,
    revenueGrowthYoY: "+4.9%",
    grossMargin: "45.6%",
  },

  news: [
    {
      id: 1,
      source: "Reuters",
      headline: "Apple Eyes AI Partnership Expansion Amid Competitive Pressure",
      time: "2h ago",
      sentiment: "positive",
      url: "#",
    },
    {
      id: 2,
      source: "Bloomberg",
      headline: "iPhone 17 Supply Chain Concerns Emerge from Asian Suppliers",
      time: "4h ago",
      sentiment: "negative",
      url: "#",
    },
    {
      id: 3,
      source: "CNBC",
      headline: "Apple Services Revenue Hits Record High in Q1 2026",
      time: "6h ago",
      sentiment: "positive",
      url: "#",
    },
    {
      id: 4,
      source: "WSJ",
      headline: "EU Regulators Scrutinize Apple App Store Fee Structure",
      time: "9h ago",
      sentiment: "negative",
      url: "#",
    },
    {
      id: 5,
      source: "TechCrunch",
      headline: "Apple Vision Pro 2 Rumored for Late 2026 Release",
      time: "12h ago",
      sentiment: "neutral",
      url: "#",
    },
  ],

  socialPosts: [
    {
      id: 1,
      platform: "X (Twitter)",
      handle: "@TechInvestorPro",
      content:
        "AAPL services revenue is the real story here. Software margins are insane. Long term this is the most important metric to watch. 🚀",
      likes: "14.2K",
      sentiment: "positive",
    },
    {
      id: 2,
      platform: "Reddit",
      handle: "r/investing",
      content:
        "Anyone else concerned about AAPL's China exposure? Revenue from China down 11% YoY. That's a real risk that isn't priced in.",
      likes: "8.7K",
      sentiment: "negative",
    },
  ],

  sentiment: {
    score: 72, // 0-100
    label: "Bullish",
    change: +3,
    positiveDrivers: [
      "Strong services revenue growth exceeding analyst expectations",
      "AI integration announcements generating significant market excitement",
    ],
    negativeDrivers: [
      "China market revenue declining for third consecutive quarter",
      "Regulatory pressure in EU threatening App Store business model",
    ],
    sourceDistribution: {
      analystReports: 28,
      news: 35,
      socialMedia: 27,
      companyFilings: 10,
    },
    // Time series: [date, sentimentScore, stockPrice]
    timeSeries: [
      { date: "Oct", score: 58, price: 172 },
      { date: "Nov", score: 63, price: 181 },
      { date: "Dec", score: 61, price: 179 },
      { date: "Jan", score: 67, price: 195 },
      { date: "Feb", score: 65, price: 190 },
      { date: "Mar", score: 70, price: 208 },
      { date: "Apr", score: 72, price: 213 },
    ],
  },

  alerts: [
    {
      id: 1,
      type: "warning",
      title: "Earnings Report",
      message: "Q2 2026 earnings call scheduled in 8 days. High volatility expected.",
    },
    {
      id: 2,
      type: "danger",
      title: "Regulatory Risk",
      message: "EU Digital Markets Act compliance deadline may impact App Store revenue.",
    },
    {
      id: 3,
      type: "success",
      title: "Analyst Upgrade",
      message: "Goldman Sachs raised price target from $220 to $245. Maintains 'Buy' rating.",
    },
  ],

  prediction: {
    shortTerm: {
      horizon: "1–2 Weeks",
      direction: "Bullish",
      confidence: 68,
      rationale: "Pre-earnings momentum and positive sentiment trend suggest near-term upside. Watch for any supply chain updates.",
    },
    mediumTerm: {
      horizon: "1–3 Months",
      direction: "Neutral",
      confidence: 54,
      rationale: "Strong services growth offset by hardware slowdown and China headwinds. Fundamentals solid but priced in at current P/E.",
    },
    catalysts: [
      { date: "Apr 9", event: "Q2 2026 Earnings Call", impact: "high" },
      { date: "Apr 22", event: "WWDC Developer Conference", impact: "medium" },
      { date: "May 15", event: "EU DMA Compliance Ruling", impact: "high" },
      { date: "Jun 1", event: "iPhone 17 Supply Chain Update", impact: "medium" },
    ],
  },
};

// Transform API response to dashboard format
function transformDashboardData(apiResponse, ticker) {
  if (!apiResponse) return null;

  const responseTicker = apiResponse.ticker?.toUpperCase() ?? ticker.toUpperCase();
  const marketPrice = apiResponse.market_data?.price ?? 0;
  const sentimentScore = apiResponse.sentiment?.sentiment_score || 0;
  const normalizedSentimentScore = Math.max(
    0,
    Math.min(100, Math.round(((sentimentScore + 1) / 2) * 100))
  );
  const predictionDirection = apiResponse.prediction?.label || "neutral";

  return {
    ticker: responseTicker,
    companyName: `${responseTicker} Stock Dashboard`,
    exchange: "NASDAQ",
    price: marketPrice,
    priceChange: marketPrice,
    priceChangePct: 0.58,
    overview: `Real-time sentiment and prediction data for ${responseTicker} stock`,
    financials: {
      annual: {
        revenue: "N/A",
        netIncome: "N/A",
        operatingCashFlow: "N/A",
        eps: "N/A",
      },
      quarterly: {
        revenue: "N/A",
        netIncome: "N/A",
        operatingCashFlow: "N/A",
        eps: "N/A",
      },
    },
    ratios: {
      pe: "N/A",
      eps: "N/A",
      roe: "N/A",
      debtToEquity: "N/A",
      revenueGrowthYoY: "N/A",
      grossMargin: "N/A",
    },
    news: [],
    socialPosts: [],
    sentiment: {
      score: normalizedSentimentScore,
      label:
        apiResponse.sentiment?.sentiment_label?.charAt(0).toUpperCase() +
          (apiResponse.sentiment?.sentiment_label?.slice(1) || "neutral"),
      change: 3,
      positiveDrivers: ["Market data integrated from real-time pipeline"],
      negativeDrivers: [],
      sourceDistribution: {
        analystReports: 25,
        news: 35,
        socialMedia: 30,
        companyFilings: 10,
      },
      timeSeries: [
        {
          date: "Now",
          score: normalizedSentimentScore,
          price: apiResponse.market_data?.price || 0,
        },
      ],
    },
    alerts: [
      {
        id: 1,
        type: "info",
        title: "Real-Time Data",
        message: `Showing live sentiment and market data for ${responseTicker}`,
      },
    ],
    prediction: {
      shortTerm: {
        horizon: "1–2 Weeks",
        direction: predictionDirection.toUpperCase(),
        confidence: Math.round((apiResponse.prediction?.confidence || 0.5) * 100),
        rationale: `ML prediction: ${predictionDirection} with ${Math.round((apiResponse.prediction?.confidence || 0.5) * 100)}% confidence`,
      },
      mediumTerm: {
        horizon: "1–3 Months",
        direction: "NEUTRAL",
        confidence: 50,
        rationale: "Medium-term outlook based on aggregated signals",
      },
      catalysts: [
        { date: apiResponse.date ?? "Today", event: `${responseTicker} Market Update`, impact: "high" },
      ],
    },
  };
}

export default function App() {
  const [activeTicker, setActiveTicker] = useState("NVDA");
  const [data, setData] = useState(DUMMY_DATA);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch dashboard data from backend
  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE}/dashboard/summary/${activeTicker}`);
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const apiData = await response.json();
        const transformedData = transformDashboardData(apiData, activeTicker);
        setData(transformedData || DUMMY_DATA);
      } catch (err) {
        console.error("Error fetching dashboard data:", err);
        setError(err.message);
        // Fall back to dummy data on error
        setData({ ...DUMMY_DATA, ticker: activeTicker.toUpperCase() });
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [activeTicker]);

  return (
    <div className="app">
      <div className="app-bg" />
      <div className="dashboard">
        {/* Search/Ticker Input */}
        <div style={{ padding: "20px", textAlign: "center", background: "rgba(0,0,0,0.3)" }}>
          <input
            type="text"
            placeholder="Enter ticker (e.g., NVDA, TSLA, AAPL)"
            value={activeTicker}
            onChange={(e) => setActiveTicker(e.target.value.toUpperCase())}
            style={{ padding: "8px 12px", fontSize: "16px", borderRadius: "4px", border: "1px solid #666" }}
          />
          {loading && <span style={{ marginLeft: "10px", color: "#00ff00" }}>Loading...</span>}
          {error && <span style={{ marginLeft: "10px", color: "#ff0000" }}>Error: {error}</span>}
        </div>

        {/* 
          We pass 'data' as a prop to each section component.
          Think of props like function arguments — each component
          receives only the slice of data it needs.
        */}
        <Header ticker={data.ticker} companyName={data.companyName} price={data.price} priceChange={data.priceChange} priceChangePct={data.priceChangePct} exchange={data.exchange} />

        <CompanyOverview overview={data.overview} />

        <div className="grid-2col">
          <FinancialMetrics financials={data.financials} ratios={data.ratios} />
          <Alerts alerts={data.alerts} />
        </div>

        <SentimentPanel sentiment={data.sentiment} />

        <NewsFeed news={data.news} socialPosts={data.socialPosts} />

        <PredictionModel prediction={data.prediction} />
      </div>
    </div>
  );
}
