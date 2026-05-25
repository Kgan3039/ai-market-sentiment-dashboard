function getDirectionColor(direction) {
  if (direction === "up") return "var(--positive)";
  if (direction === "down") return "var(--negative)";
  return "var(--accent)";
}

export default function PredictionModel({ prediction, updatedAt }) {
  const hasPrediction = Boolean(prediction);
  const direction = prediction?.predicted_movement || "neutral";
  const confidence = Math.round((prediction?.confidence || 0) * 100);
  const probability = Math.round((prediction?.probability || 0) * 100);
  const color = getDirectionColor(direction);

  return (
    <section className="card prediction-card">
      <div className="section-header">
        <h2 className="section-title">Prediction Model</h2>
      </div>

      {!hasPrediction ? (
        <div className="empty-state">
          Prediction output is not available in the current summary response.
        </div>
      ) : null}

      {hasPrediction ? (
        <div className="prediction-grid">
          <div className="horizon-card">
            <div className="horizon-label">Predicted direction</div>
            <div className="horizon-direction" style={{ color }}>
              {direction.toUpperCase()}
            </div>
            <div className="confidence-bar-label">
              Model confidence: {confidence}%
            </div>
            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{ width: `${confidence}%`, background: color }}
              />
            </div>
            <p className="horizon-rationale">
              For informational use only. This model output is not financial advice.
            </p>
          </div>

          <div className="horizon-card">
            <div className="horizon-label">Model probability</div>
            <div className="horizon-direction">{probability}%</div>
            <p className="horizon-rationale">
              Last updated: {updatedAt || "Not available"}
            </p>
          </div>

          <div className="catalyst-block">
            <h3 className="subsection-title">Model note</h3>
            <div className="catalyst-item">
              <span className="catalyst-date">Info</span>
              <span className="catalyst-event">
                Predictions are experimental analytics based on available sentiment and market data.
              </span>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
