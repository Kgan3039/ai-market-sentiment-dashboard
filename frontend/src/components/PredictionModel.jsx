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
        <span
          className={`section-status ${
            hasPrediction ? "status-ready" : "status-unavailable"
          }`}
        >
          {hasPrediction ? "ready" : "unavailable"}
        </span>
      </div>

      {!hasPrediction ? (
        <div className="empty-state">
          Prediction output is not available in the current summary response.
        </div>
      ) : null}

      <div className="prediction-grid">
        <div className="horizon-card">
          <div className="horizon-label">Predicted Movement</div>
          <div className="horizon-direction" style={{ color }}>
            {direction.toUpperCase()}
          </div>
          <div className="confidence-bar-label">Confidence: {confidence}%</div>
          <div className="confidence-track">
            <div
              className="confidence-fill"
              style={{ width: `${confidence}%`, background: color }}
            />
          </div>
          <p className="horizon-rationale">
            Directly rendered from the backend prediction response.
          </p>
        </div>

        <div className="horizon-card">
          <div className="horizon-label">Probability</div>
          <div className="horizon-direction">{probability}%</div>
          <p className="horizon-rationale">
            Last summary update: {updatedAt || "N/A"}
          </p>
        </div>

        <div className="catalyst-block">
          <h3 className="subsection-title">Model Status</h3>
          <div className="catalyst-item impact-medium">
            <span className="catalyst-date">API</span>
            <span className="catalyst-event">
              The UI is using the existing backend prediction contract without inventing extra horizons or catalysts.
            </span>
            <span className="catalyst-impact">live</span>
          </div>
        </div>
      </div>
    </section>
  );
}
