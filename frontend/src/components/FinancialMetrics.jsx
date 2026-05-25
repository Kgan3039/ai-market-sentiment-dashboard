function formatMoney(value, currency) {
  if (value === null || value === undefined) return null;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency || "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatNumber(value, suffix = "") {
  if (value === null || value === undefined) return null;
  return `${new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2,
  }).format(value)}${suffix}`;
}

export default function FinancialMetrics({ fundamentals, loading = false }) {
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
  const availableMetrics = metrics.filter(([, value]) => value !== null);
  const context = [
    fundamentals?.company_name,
    fundamentals?.sector,
    fundamentals?.industry,
  ].filter(Boolean);

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Financials & Ratios</h2>
      </div>

      <p className="support-note">
        {loading
          ? "Loading company fundamentals."
          : fundamentals
          ? "Company fundamentals available."
          : "Fundamentals are not available in the current MVP."}
      </p>

      {context.length > 0 ? (
        <div className="fundamentals-context">
          {context.map((item) => (
            <span key={item}>{item}</span>
          ))}
          {fundamentals?.market_cap ? (
            <span>Market cap {formatMoney(fundamentals.market_cap, fundamentals?.currency)}</span>
          ) : null}
        </div>
      ) : null}

      {availableMetrics.length > 0 ? (
        <div className="metrics-grid">
          {availableMetrics.map(([label, value]) => (
            <div key={label} className="metric-card">
              <span className="metric-label">{label}</span>
              <span className="metric-value">{value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          Fundamentals are not available in the current MVP.
        </div>
      )}
    </section>
  );
}
