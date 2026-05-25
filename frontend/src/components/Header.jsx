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
            {hasPrice ? `$${price.toFixed(2)}` : "Not available"}
          </span>
          <span className="price-change muted">
            {updatedAt ? `Last updated ${updatedAt}` : "Data not loaded"}
          </span>
        </div>
      </div>
    </header>
  );
}
