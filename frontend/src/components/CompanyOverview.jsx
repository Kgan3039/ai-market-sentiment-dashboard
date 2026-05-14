export default function CompanyOverview({ ticker, fundamentals, updatedAt }) {
  const companyName = fundamentals?.company_name || `${ticker} Dashboard`;
  const businessLine = [fundamentals?.sector, fundamentals?.industry]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="card overview-card">
      <h2 className="section-title">Dashboard Overview</h2>
      <p className="overview-text">
        {companyName} is rendered from the live dashboard summary contract for{" "}
        {ticker}. The response now includes sentiment, market data, prediction
        output, headline availability, and fundamentals when the provider returns
        them.
      </p>
      {businessLine ? <p className="overview-meta">{businessLine}</p> : null}
      <p className="overview-meta">
        Last backend update: {updatedAt || "N/A"}
      </p>
    </section>
  );
}
