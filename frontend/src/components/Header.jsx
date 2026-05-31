function formatSignedNumber(value, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}`;
}

function formatPriceChange(delta, percentChange) {
  const formattedDelta = formatSignedNumber(delta);
  const formattedPercent = formatSignedNumber(percentChange);

  if (!formattedDelta && !formattedPercent) return "Change not reported";
  if (!formattedDelta) return `${formattedPercent}%`;
  if (!formattedPercent) return `$${formattedDelta}`;
  return `$${formattedDelta} (${formattedPercent}%)`;
}

function formatVolumeDelta(value) {
  const formattedValue = formatSignedNumber(value * 100, 1);
  return formattedValue ? `Volume delta ${formattedValue}%` : null;
}

export default function Header({ ticker, marketData, exchange, updatedAt }) {
  const price = marketData?.price;
  const hasPrice = typeof price === "number" && Number.isFinite(price) && price > 0;
  const priceDelta = marketData?.price_delta_24h;
  const percentChange = marketData?.percent_change_24h;
  const volumeDelta = marketData?.volume_delta;
  const hasDelta = typeof priceDelta === "number" && Number.isFinite(priceDelta);
  const deltaClass = hasDelta ? (priceDelta >= 0 ? "positive" : "negative") : "muted";
  const source = marketData?.source || "Source not reported";
  const status = marketData?.status || (hasPrice ? "live" : "unavailable");
  const volumeDeltaLabel =
    typeof volumeDelta === "number" && Number.isFinite(volumeDelta)
      ? formatVolumeDelta(volumeDelta)
      : null;

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
          <span className={`price-change ${deltaClass}`}>
            {hasPrice ? formatPriceChange(priceDelta, percentChange) : "Data not loaded"}
          </span>
          <div className="market-provenance" aria-label="Market data provenance">
            <span className="market-status">{status}</span>
            <span>{source}</span>
            {volumeDeltaLabel ? <span>{volumeDeltaLabel}</span> : null}
          </div>
          <span className="market-updated">
            {updatedAt ? `Last updated ${updatedAt}` : "Update time not reported"}
          </span>
        </div>
      </div>
    </header>
  );
}
