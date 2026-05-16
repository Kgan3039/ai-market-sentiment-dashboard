export default function Header({ ticker, price, exchange, updatedAt }) {
  const hasPrice = typeof price === "number" && Number.isFinite(price) && price > 0;

  return (
    <header className="header">
      <div className="header-left">
        <div className="ticker-badge">{ticker}</div>
        <div className="header-name-block">
          <h1 className="company-name">{ticker} Dashboard</h1>
          <span className="exchange-label">{exchange}</span>
        </div>
      </div>
      <div className="header-right">
        <div className="price-block">
          <span className="current-price">
            {hasPrice ? `$${price.toFixed(2)}` : "N/A"}
          </span>
          <span className="price-change muted">
            {updatedAt ? `Updated ${updatedAt}` : "Awaiting backend summary"}
          </span>
        </div>
        <div className="live-dot-wrapper">
          <span className="live-dot" />
          <span className="live-label">{hasPrice ? "SYNCED" : "SPARSE"}</span>
        </div>
      </div>
    </header>
  );
}
