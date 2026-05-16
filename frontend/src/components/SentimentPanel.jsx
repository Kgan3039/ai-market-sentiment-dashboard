function formatLabel(value = "neutral") {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export default function SentimentPanel({ sentiment, loading = false }) {
  const hasSentiment = Boolean(sentiment);
  const sentimentScore = sentiment?.sentiment_score ?? 0;
  const normalizedScore = Math.max(
    0,
    Math.min(100, Math.round(((sentimentScore + 1) / 2) * 100))
  );

  const sourceDistribution = {
    positive: Math.round((sentiment?.positive_prob ?? 0) * 100),
    negative: Math.round((sentiment?.negative_prob ?? 0) * 100),
    neutral: Math.round((sentiment?.neutral_prob ?? 0) * 100),
  };

  const positiveDrivers = [
    `Positive probability: ${sourceDistribution.positive}%`,
    `Confidence: ${Math.round((sentiment?.sentiment_confidence ?? 0) * 100)}%`,
  ];

  const negativeDrivers = [
    `Negative probability: ${sourceDistribution.negative}%`,
    `Neutral probability: ${sourceDistribution.neutral}%`,
  ];

  const W = 600;
  const H = 180;
  const PAD = 30;
  const chartW = W - PAD * 2;
  const chartH = H - PAD * 2;

  const toX = () => PAD + chartW / 2;
  const toY = (val) => PAD + chartH - (val / 100) * chartH;
  const label = formatLabel(sentiment?.sentiment_label || "neutral");

  return (
    <section className="card sentiment-card">
      <div className="section-header">
        <h2 className="section-title">Sentiment Analysis</h2>
        <span
          className={`section-status ${
            hasSentiment ? "status-ready" : "status-unavailable"
          }`}
        >
          {loading ? "loading" : hasSentiment ? "ready" : "unavailable"}
        </span>
      </div>

      {!hasSentiment && !loading ? (
        <div className="empty-state">
          Sentiment data is not available in the current summary response.
        </div>
      ) : null}

      <div className="sentiment-top">
        <div className="score-block">
          <ScoreGauge score={normalizedScore} label={loading ? "Loading" : label} />
          <div className="score-change muted">
            {loading ? "Awaiting backend sentiment" : `Raw score: ${sentimentScore.toFixed(2)}`}
          </div>
        </div>

        <div className="drivers-block">
          <div className="driver-group">
            <h4 className="driver-title positive-title">Positive Signals</h4>
            {positiveDrivers.map((item) => (
              <div key={item} className="driver-item positive-driver">
                {item}
              </div>
            ))}
          </div>
          <div className="driver-group">
            <h4 className="driver-title negative-title">Risk Signals</h4>
            {negativeDrivers.map((item) => (
              <div key={item} className="driver-item negative-driver">
                {item}
              </div>
            ))}
          </div>
        </div>

        <div className="source-block">
          <h4 className="subsection-title">Probability Mix</h4>
          {Object.entries(sourceDistribution).map(([key, pct]) => (
            <SourceBar key={key} label={formatLabel(key)} pct={pct} />
          ))}
        </div>
      </div>

      <div className="chart-section">
        <div className="chart-header">
          <h4 className="subsection-title">Current Sentiment Snapshot</h4>
          <span className="chart-note">single point</span>
        </div>

        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="sentiment-chart"
          preserveAspectRatio="xMidYMid meet"
        >
          {[25, 50, 75].map((y) => (
            <g key={y}>
              <line
                x1={PAD}
                y1={toY(y)}
                x2={W - PAD}
                y2={toY(y)}
                stroke="var(--border)"
                strokeDasharray="4 4"
                strokeWidth="0.8"
              />
              <text
                x={PAD - 5}
                y={toY(y) + 4}
                textAnchor="end"
                fontSize="9"
                fill="var(--text-muted)"
              >
                {y}
              </text>
            </g>
          ))}

          <line
            x1={PAD}
            y1={toY(normalizedScore)}
            x2={W - PAD}
            y2={toY(normalizedScore)}
            stroke="var(--accent)"
            strokeWidth="1.5"
            strokeOpacity="0.5"
          />
          <circle cx={toX()} cy={toY(normalizedScore)} r="6" fill="var(--accent)" />
          <circle
            cx={toX()}
            cy={toY(normalizedScore)}
            r="15"
            fill="none"
            stroke="var(--accent)"
            strokeOpacity="0.3"
          />
          <text
            x={toX()}
            y={H - 6}
            textAnchor="middle"
            fontSize="9"
            fill="var(--text-muted)"
          >
            Latest summary
          </text>
        </svg>

        <p className="snapshot-copy">
          The backend currently returns one aggregate sentiment point, so this
          view highlights the latest probability mix instead of implying a
          historical trend.
        </p>
      </div>
    </section>
  );
}

function ScoreGauge({ score, label }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color =
    score >= 65 ? "var(--positive)" : score >= 40 ? "var(--accent)" : "var(--negative)";

  return (
    <div className="gauge-wrapper">
      <svg width="130" height="130" viewBox="0 0 130 130">
        <circle
          cx="65"
          cy="65"
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth="10"
        />
        <circle
          cx="65"
          cy="65"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 65 65)"
        />
        <text
          x="65"
          y="60"
          textAnchor="middle"
          fontSize="26"
          fontWeight="700"
          fill="var(--text-primary)"
        >
          {score}
        </text>
        <text
          x="65"
          y="78"
          textAnchor="middle"
          fontSize="11"
          fill="var(--text-muted)"
        >
          {label}
        </text>
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
