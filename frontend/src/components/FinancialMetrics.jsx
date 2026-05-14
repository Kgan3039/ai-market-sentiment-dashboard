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

export default function FinancialMetrics({ fundamentals }) {
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

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Financials & Ratios</h2>
      </div>

      <p className="support-note">
        {fundamentals
          ? `Source: ${fundamentals.source}`
          : "Fundamentals are not available from the backend response yet."}
      </p>

      <div className="metrics-grid">
        {metrics.map(([label, value]) => (
          <div key={label} className="metric-card">
            <span className="metric-label">{label}</span>
            <span className="metric-value">{value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
