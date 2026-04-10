// Header.jsx
// This component receives 'props' — data passed from the parent (App.jsx)
// We destructure them directly in the function signature for cleaner code

export default function Header({ ticker, companyName, price, priceChange, priceChangePct, exchange }) {
  const isPositive = priceChange >= 0;

  return (
    <header className="header">
      <div className="header-left">
        <div className="ticker-badge">{ticker}</div>
        <div className="header-name-block">
          <h1 className="company-name">{companyName}</h1>
          <span className="exchange-label">{exchange}</span>
        </div>
      </div>
      <div className="header-right">
        <div className="price-block">
          <span className="current-price">${price.toFixed(2)}</span>
          <span className={`price-change ${isPositive ? "positive" : "negative"}`}>
            {isPositive ? "▲" : "▼"} {Math.abs(priceChange).toFixed(2)} ({isPositive ? "+" : ""}{priceChangePct.toFixed(2)}%)
          </span>
        </div>
        <div className="live-dot-wrapper">
          <span className="live-dot" />
          <span className="live-label">LIVE</span>
        </div>
      </div>
    </header>
  );
}
