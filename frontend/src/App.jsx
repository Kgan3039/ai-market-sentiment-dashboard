import { useState } from "react";
import Header from "./components/Header";
import CompanyOverview from "./components/CompanyOverview";
import FinancialMetrics from "./components/FinancialMetrics";
import NewsFeed from "./components/NewsFeed";
import SentimentPanel from "./components/SentimentPanel";
import Alerts from "./components/Alerts";
import PredictionModel from "./components/PredictionModel";
import "./App.css";

// DUMMY DATA — replace with real API calls once backend is ready
// This lives here in App.jsx so all child components can access it via props
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

export default function App() {
  // useState: ticker could be swapped out when you add a search bar via setActiveTicker
  const [activeTicker, setActiveTicker] = useState("AAPL");
  const data = DUMMY_DATA; // later: fetch from backend based on activeTicker

  return (
    <div className="app">
      <div className="app-bg" />
      <div className="dashboard">
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
