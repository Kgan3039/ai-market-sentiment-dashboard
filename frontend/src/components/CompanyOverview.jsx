export default function CompanyOverview({ ticker, fundamentals, updatedAt }) {
  const companyName = fundamentals?.company_name || `${ticker} Dashboard`;
  const businessLine = [fundamentals?.sector, fundamentals?.industry]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="card overview-card">
      <h2 className="section-title">Dashboard Overview</h2>
      <p className="overview-text">
        {companyName} market summary for {ticker}, including market data,
        sentiment, model output, headlines, and fundamentals when available.
      </p>
      {businessLine ? <p className="overview-meta">{businessLine}</p> : null}
      <p className="overview-meta">
        Last updated: {updatedAt || "Not available"}
      </p>
    </section>
  );
}
