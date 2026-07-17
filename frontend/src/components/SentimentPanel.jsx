function formatLabel(value = "neutral") {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "--";
  return `$${value.toFixed(2)}`;
}

function formatShortDate(value) {
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function SentimentPanel({
  sentiment,
  marketHistory = [],
  ticker = "",
  loading = false,
}) {
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

  const historyPoints = marketHistory
    .map((point) => ({
      ...point,
      close: Number(point.close),
    }))
    .filter((point) => Number.isFinite(point.close));
  const latestHistoryPoint = historyPoints[historyPoints.length - 1];
  const firstHistoryPoint = historyPoints[0];
  const historyChange =
    latestHistoryPoint && firstHistoryPoint && firstHistoryPoint.close
      ? ((latestHistoryPoint.close - firstHistoryPoint.close) / firstHistoryPoint.close) * 100
      : 0;
  const historyTone = historyChange > 0 ? "positive" : historyChange < 0 ? "negative" : "neutral";
  const label = formatLabel(sentiment?.sentiment_label || "neutral");

  return (
    <section className="card sentiment-card">
      <div className="section-header">
        <h2 className="section-title">Sentiment Analysis</h2>
      </div>

      {!hasSentiment && !loading ? (
        <div className="empty-state">
          Aggregate sentiment unavailable until enough validated text data is available.
        </div>
      ) : null}

      {hasSentiment || loading ? (
        <>
          <div className="sentiment-top">
            <div className="score-block">
              <ScoreGauge score={normalizedScore} label={loading ? "Loading" : label} />
              <div className="score-change muted">
                {loading
                  ? "Loading sentiment"
                  : `Sentiment score: ${sentimentScore.toFixed(2)}`}
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
              <h4 className="subsection-title">{ticker} Price History</h4>
              <span className={`chart-note ${historyTone}`}>
                {historyPoints.length > 1
                  ? `${historyChange >= 0 ? "+" : ""}${historyChange.toFixed(1)}% 1M`
                  : "history pending"}
              </span>
            </div>

            <PriceHistoryChart points={historyPoints} />

            <p className="snapshot-copy">
              Recent close-price history from the market data provider, shown alongside the latest sentiment signal.
            </p>
          </div>
        </>
      ) : null}
    </section>
  );
}

function PriceHistoryChart({ points }) {
  const W = 640;
  const H = 150;
  const PAD_X = 44;
  const PAD_TOP = 18;
  const PAD_BOTTOM = 28;
  const chartW = W - PAD_X * 2;
  const chartH = H - PAD_TOP - PAD_BOTTOM;

  if (points.length < 2) {
    return (
      <div className="history-empty">
        Price history unavailable for this ticker right now.
      </div>
    );
  }

  const closes = points.map((point) => point.close);
  const minClose = Math.min(...closes);
  const maxClose = Math.max(...closes);
  const range = maxClose - minClose || Math.max(maxClose * 0.01, 1);
  const latest = points[points.length - 1];
  const first = points[0];

  const toX = (index) => PAD_X + (index / (points.length - 1)) * chartW;
  const toY = (close) => PAD_TOP + chartH - ((close - minClose) / range) * chartH;

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${toX(index).toFixed(2)} ${toY(point.close).toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L ${toX(points.length - 1).toFixed(2)} ${H - PAD_BOTTOM} L ${PAD_X} ${H - PAD_BOTTOM} Z`;

  return (
    <div className="history-chart-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="history-chart"
        preserveAspectRatio="none"
        aria-label="Recent price history"
      >
        {[maxClose, (maxClose + minClose) / 2, minClose].map((value) => (
          <g key={value}>
            <line
              x1={PAD_X}
              y1={toY(value)}
              x2={W - PAD_X}
              y2={toY(value)}
              stroke="var(--border)"
              strokeDasharray="4 5"
              strokeWidth="1"
            />
            <text x={PAD_X - 8} y={toY(value) + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">
              {formatMoney(value)}
            </text>
          </g>
        ))}
        <path d={areaPath} fill="var(--accent)" opacity="0.08" />
        <path d={linePath} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={toX(points.length - 1)} cy={toY(latest.close)} r="4.5" fill="var(--accent)" />
        <text x={PAD_X} y={H - 7} fontSize="10" fill="var(--text-muted)">
          {formatShortDate(first.date)}
        </text>
        <text x={W - PAD_X} y={H - 7} textAnchor="end" fontSize="10" fill="var(--text-muted)">
          {formatShortDate(latest.date)}
        </text>
      </svg>
      <div className="history-stats">
        <span>Latest {formatMoney(latest.close)}</span>
        <span>High {formatMoney(maxClose)}</span>
        <span>Low {formatMoney(minClose)}</span>
      </div>
    </div>
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
