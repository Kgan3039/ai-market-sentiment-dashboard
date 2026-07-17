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

function formatStatus(status) {
  if (!status) return null;
  const labels = {
    live: "Live",
    cached: "Cached",
    fallback: "Fallback",
    ready: "Available",
    unavailable: "Unavailable",
  };
  return labels[status] || status.replace(/_/g, " ");
}

function getFundamentalsNote(fundamentals, availability, loading) {
  if (loading) return "Loading company fundamentals.";
  if (availability?.message) return availability.message;
  if (fundamentals) return "Company fundamentals are available.";
  return "Fundamentals are unavailable from the configured providers.";
}

export default function FinancialMetrics({
  fundamentals,
  availability,
  loading = false,
}) {
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
  const statusLabel = formatStatus(
    availability?.status || (fundamentals ? "ready" : "unavailable")
  );
  const sourceLabel = availability?.source || fundamentals?.source;
  const note = getFundamentalsNote(fundamentals, availability, loading);

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Financials & Ratios</h2>
      </div>

      <p className="support-note">{note}</p>

      {statusLabel || sourceLabel ? (
        <div className="fundamentals-status">
          {statusLabel ? <span>{statusLabel}</span> : null}
          {sourceLabel ? <span>{sourceLabel}</span> : null}
        </div>
      ) : null}

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
          {note}
        </div>
      )}
    </section>
  );
}
