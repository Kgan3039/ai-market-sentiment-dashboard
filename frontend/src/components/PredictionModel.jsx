function getDirectionColor(direction) {
  if (direction === "up") return "var(--positive)";
  if (direction === "down") return "var(--negative)";
  return "var(--accent)";
}

function formatModelSource(modelInfo) {
  const source = modelInfo?.artifact_source;
  if (source === "disk") return "Persisted artifact";
  if (source === "trained_and_persisted") return "Fresh artifact";
  if (source === "trained_in_memory") return "In-memory training";
  if (source === "none" || modelInfo?.name === "FallbackRule") return "Fallback rule";
  return "Model source unavailable";
}

function formatModelDetail(modelInfo) {
  if (!modelInfo) return "Model provenance is not available for this response.";
  if (modelInfo.name === "FallbackRule") {
    return "Fallback rule used because the persisted model path was unavailable.";
  }
  const version = modelInfo.version ? `version ${modelInfo.version}` : "unversioned artifact";
  const trainingData = modelInfo.training_data
    ? modelInfo.training_data.replaceAll("_", " ")
    : "training data not reported";
  return `${modelInfo.name || "Experimental signal"} serving from ${version}; trained on ${trainingData}.`;
}

function formatMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : null;
}

export default function PredictionModel({ prediction, updatedAt }) {
  const hasPrediction = Boolean(prediction);
  const direction = prediction?.predicted_movement || "neutral";
  const confidence = Math.round((prediction?.confidence || 0) * 100);
  const probability = Math.round((prediction?.probability || 0) * 100);
  const color = getDirectionColor(direction);
  const modelInfo = prediction?.model_info;
  const sourceLabel = formatModelSource(modelInfo);
  const modelDetail = formatModelDetail(modelInfo);
  const servingStatus = modelInfo?.status === "fallback" ? "Fallback" : "Serving";
  const trainedAt = modelInfo?.trained_at
    ? new Date(modelInfo.trained_at).toLocaleDateString()
    : null;

  return (
    <section className="card prediction-card">
      <div className="section-header">
        <h2 className="section-title">Experimental Signal</h2>
      </div>

      {!hasPrediction ? (
        <div className="empty-state">
          Signal unavailable until enough validated input data is available.
        </div>
      ) : null}

      {hasPrediction ? (
        <div className="prediction-grid">
          <div className="horizon-card">
            <div className="horizon-label">Signal direction</div>
            <div className="horizon-direction" style={{ color }}>
              {direction.toUpperCase()}
            </div>
            <div className="confidence-bar-label">
              	Signal confidence: {confidence}%
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
            <div className="horizon-label">Signal probability</div>
            <div className="horizon-direction">{probability}%</div>
            <p className="horizon-rationale">
              Last updated: {updatedAt || "Not available"}
            </p>
          </div>

          <div className="catalyst-block">
            <h3 className="subsection-title">Model provenance</h3>
            <div className={`model-source ${modelInfo?.status === "fallback" ? "fallback" : "ready"}`}>
              <span>{servingStatus}</span>
              <strong>{sourceLabel}</strong>
            </div>
            <p className="horizon-rationale">{modelDetail}</p>
            <div className="model-meta">
              {modelInfo?.version ? <span>{modelInfo.version}</span> : null}
              {trainedAt ? <span>Trained {trainedAt}</span> : null}
            </div>
            <p className="model-honesty">
              Experimental analytics only; outputs depend on available sentiment and market data.
            </p>
          </div>
        </div>
      ) : null}
    </section>
  );
}
