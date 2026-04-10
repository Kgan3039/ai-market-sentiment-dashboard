import { useState } from "react";

export default function SentimentPanel({ sentiment }) {
  // Toggle to show/hide the stock price line on the chart
  const [showPrice, setShowPrice] = useState(false);

  const { score, label, change, positiveDrivers, negativeDrivers, sourceDistribution, timeSeries } = sentiment;

  // Normalize price data to 0-100 scale so it overlays cleanly with sentiment score
  const prices = timeSeries.map((d) => d.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const normalizePrice = (p) => ((p - minPrice) / (maxPrice - minPrice)) * 80 + 10;

  // SVG chart dimensions
  const W = 600;
  const H = 180;
  const PAD = 30;
  const chartW = W - PAD * 2;
  const chartH = H - PAD * 2;

  // Convert data points to SVG coordinates
  // SVG y-axis is inverted (0 is top), so we flip: higher score = lower y value
  const toX = (i) => PAD + (i / (timeSeries.length - 1)) * chartW;
  const toY = (val) => PAD + chartH - (val / 100) * chartH;

  // Build SVG polyline points string
  const sentimentPoints = timeSeries.map((d, i) => `${toX(i)},${toY(d.score)}`).join(" ");
  const pricePoints = timeSeries.map((d, i) => `${toX(i)},${toY(normalizePrice(d.price))}`).join(" ");

  return (
    <section className="card sentiment-card">
      <h2 className="section-title">Sentiment Analysis</h2>

      <div className="sentiment-top">
        {/* Score Gauge */}
        <div className="score-block">
          <ScoreGauge score={score} label={label} />
          <div className={`score-change ${change >= 0 ? "positive" : "negative"}`}>
            {change >= 0 ? "▲" : "▼"} {Math.abs(change)} pts vs yesterday
          </div>
        </div>

        {/* Drivers */}
        <div className="drivers-block">
          <div className="driver-group">
            <h4 className="driver-title positive-title">▲ Top Positives</h4>
            {positiveDrivers.map((d, i) => (
              <div key={i} className="driver-item positive-driver">{d}</div>
            ))}
          </div>
          <div className="driver-group">
            <h4 className="driver-title negative-title">▼ Top Negatives</h4>
            {negativeDrivers.map((d, i) => (
              <div key={i} className="driver-item negative-driver">{d}</div>
            ))}
          </div>
        </div>

        {/* Source Distribution */}
        <div className="source-block">
          <h4 className="subsection-title">Source Distribution</h4>
          {Object.entries(sourceDistribution).map(([key, pct]) => (
            <SourceBar key={key} label={formatSourceLabel(key)} pct={pct} />
          ))}
        </div>
      </div>

      {/* Time Series Chart */}
      <div className="chart-section">
        <div className="chart-header">
          <h4 className="subsection-title">Sentiment Over Time</h4>
          <button
            className={`toggle-btn ${showPrice ? "active" : ""}`}
            onClick={() => setShowPrice(!showPrice)}
          >
            {showPrice ? "Hide" : "Show"} Stock Price
          </button>
        </div>

        {/* 
          SVG chart — built manually so you understand the fundamentals.
          Later you can swap this for a library like Recharts.
          viewBox="0 0 W H" sets the coordinate system.
          preserveAspectRatio makes it scale responsively.
        */}
        <svg viewBox={`0 0 ${W} ${H}`} className="sentiment-chart" preserveAspectRatio="xMidYMid meet">
          {/* Horizontal grid lines */}
          {[25, 50, 75].map((y) => (
            <g key={y}>
              <line x1={PAD} y1={toY(y)} x2={W - PAD} y2={toY(y)} stroke="var(--border)" strokeDasharray="4 4" strokeWidth="0.8" />
              <text x={PAD - 5} y={toY(y) + 4} textAnchor="end" fontSize="9" fill="var(--text-muted)">{y}</text>
            </g>
          ))}

          {/* X-axis labels */}
          {timeSeries.map((d, i) => (
            <text key={d.date} x={toX(i)} y={H - 6} textAnchor="middle" fontSize="9" fill="var(--text-muted)">{d.date}</text>
          ))}

          {/* Stock price line (toggleable) */}
          {showPrice && (
            <polyline points={pricePoints} fill="none" stroke="var(--accent-secondary)" strokeWidth="1.5" strokeDasharray="5 3" opacity="0.7" />
          )}

          {/* Sentiment score line */}
          <polyline points={sentimentPoints} fill="none" stroke="var(--accent)" strokeWidth="2.5" />

          {/* Dots at each data point */}
          {timeSeries.map((d, i) => (
            <circle key={d.date} cx={toX(i)} cy={toY(d.score)} r="3.5" fill="var(--accent)" />
          ))}
        </svg>

        {showPrice && (
          <div className="chart-legend">
            <span className="legend-item"><span className="legend-dot accent" />Sentiment Score</span>
            <span className="legend-item"><span className="legend-dot secondary" />Stock Price (normalized)</span>
          </div>
        )}
      </div>
    </section>
  );
}

// Circular gauge built with SVG
function ScoreGauge({ score, label }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  // strokeDashoffset controls how much of the circle arc is "filled"
  const offset = circumference - (score / 100) * circumference;

  const color = score >= 65 ? "var(--positive)" : score >= 40 ? "var(--accent)" : "var(--negative)";

  return (
    <div className="gauge-wrapper">
      <svg width="130" height="130" viewBox="0 0 130 130">
        <circle cx="65" cy="65" r={radius} fill="none" stroke="var(--border)" strokeWidth="10" />
        <circle
          cx="65" cy="65" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 65 65)"
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
        <text x="65" y="60" textAnchor="middle" fontSize="26" fontWeight="700" fill="var(--text-primary)">{score}</text>
        <text x="65" y="78" textAnchor="middle" fontSize="11" fill="var(--text-muted)">{label}</text>
      </svg>
    </div>
  );
}

function SourceBar({ label, pct }) {
  return (
    <div className="source-bar-row">
      <span className="source-bar-label">{label}</span>
      <div className="source-bar-track">
        <div className="source-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="source-bar-pct">{pct}%</span>
    </div>
  );
}

function formatSourceLabel(key) {
  const map = {
    analystReports: "Analyst Reports",
    news: "News",
    socialMedia: "Social Media",
    companyFilings: "Company Filings",
  };
  return map[key] || key;
}
