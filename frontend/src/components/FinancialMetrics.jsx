// useState lets us toggle between Annual and Quarterly view
import { useState } from "react";

export default function FinancialMetrics({ financials, ratios }) {
  // 'period' is either "annual" or "quarterly"
  // setperiod is the function we call to change it
  const [period, setPeriod] = useState("annual");

  const figures = financials[period];

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Financials & Ratios</h2>
        {/* Toggle buttons — clicking them calls setPeriod to swap the data shown */}
        <div className="toggle-group">
          <button
            className={`toggle-btn ${period === "annual" ? "active" : ""}`}
            onClick={() => setPeriod("annual")}
          >
            Annual
          </button>
          <button
            className={`toggle-btn ${period === "quarterly" ? "active" : ""}`}
            onClick={() => setPeriod("quarterly")}
          >
            Quarterly
          </button>
        </div>
      </div>

      <div className="metrics-grid">
        <MetricCard label="Revenue" value={figures.revenue} />
        <MetricCard label="Net Income" value={figures.netIncome} />
        <MetricCard label="Op. Cash Flow" value={figures.operatingCashFlow} />
        <MetricCard label="EPS" value={figures.eps} />
      </div>

      <div className="divider" />

      <h3 className="subsection-title">Key Ratios</h3>
      <div className="metrics-grid">
        <MetricCard label="P/E Ratio" value={ratios.pe} />
        <MetricCard label="EPS (TTM)" value={`$${ratios.eps}`} />
        <MetricCard label="ROE" value={ratios.roe} />
        <MetricCard label="Debt/Equity" value={ratios.debtToEquity} />
        <MetricCard label="Revenue Growth YoY" value={ratios.revenueGrowthYoY} highlight />
        <MetricCard label="Gross Margin" value={ratios.grossMargin} highlight />
      </div>
    </section>
  );
}

// A small reusable sub-component — notice how React lets you nest components
// 'highlight' is a boolean prop — if true we apply an extra CSS class
function MetricCard({ label, value, highlight }) {
  return (
    <div className={`metric-card ${highlight ? "metric-highlight" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
