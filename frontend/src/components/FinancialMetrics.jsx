function formatMoney(value, currency) {
  if (value === null || value === undefined) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency || "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatNumber(value, suffix = "") {
  if (value === null || value === undefined) return "N/A";
  return `${new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2,
  }).format(value)}${suffix}`;
}

export default function FinancialMetrics({ fundamentals, availability, loading = false }) {
  const metrics = [
    ["Revenue", formatMoney(fundamentals?.revenue, fundamentals?.currency)],
    ["Net Income", formatMoney(fundamentals?.net_income, fundamentals?.currency)],
    [
      "Op. Cash Flow",
      formatMoney(fundamentals?.operating_cash_flow, fundamentals?.currency),
    ],
    ["EPS", formatNumber(fundamentals?.eps)],
    ["P/E Ratio", formatNumber(fundamentals?.trailing_pe)],
    ["Debt/Equity", formatNumber(fundamentals?.debt_to_equity)],
  ];
  const context = [
    fundamentals?.company_name,
    fundamentals?.sector,
    fundamentals?.industry,
  ].filter(Boolean);

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Financials & Ratios</h2>
        <span
          className={`section-status ${
            fundamentals ? "status-ready" : "status-unavailable"
          }`}
        >
          {loading ? "loading" : fundamentals ? "ready" : "unavailable"}
        </span>
      </div>

      <p className="support-note">
        {loading
          ? "Fetching company fundamentals from the backend summary."
          : fundamentals
          ? `Source: ${fundamentals.source}`
          : availability?.message ||
            "Fundamentals are not available from the backend response yet."}
      </p>

      {context.length > 0 ? (
        <div className="fundamentals-context">
          {context.map((item) => (
            <span key={item}>{item}</span>
          ))}
          <span>Market cap {formatMoney(fundamentals?.market_cap, fundamentals?.currency)}</span>
        </div>
      ) : null}

      <div className="metrics-grid">
        {metrics.map(([label, value]) => (
          <div
            key={label}
            className={`metric-card ${value === "N/A" ? "metric-unavailable" : ""}`}
          >
            <span className="metric-label">{label}</span>
            <span className="metric-value">{value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
