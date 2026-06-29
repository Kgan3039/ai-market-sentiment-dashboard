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
  if (!modelInfo) return "Signal provenance is not available for this response.";
  if (modelInfo.name === "FallbackRule") {
    return "No validated artifact is available for this response.";
  }
  const version = modelInfo.version ? `version ${modelInfo.version}` : "unversioned artifact";
  const trainingData = modelInfo.training_data
    ? modelInfo.training_data.replaceAll("_", " ")
    : "training data not reported";
  return `${modelInfo.name || "Experimental signal"} using ${version}; trained on ${trainingData}.`;
}

export default function PredictionModel({ prediction, updatedAt }) {
  const hasPrediction = Boolean(prediction);
  const hasRealModel = prediction?.model_info?.real_training_data === true;
  const direction = prediction?.predicted_movement || "neutral";
  const color = getDirectionColor(direction);
  const modelInfo = prediction?.model_info;
  const sourceLabel = formatModelSource(modelInfo);
  const modelDetail = formatModelDetail(modelInfo);
  const servingStatus = modelInfo?.status === "ready" ? "Available" : "Unavailable";
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
      No validated experimental signal is available. Synthetic or demo-trained
      outputs are withheld until a model is trained and evaluated on real
      historical outcomes.
    </div>
  ) : null}

  {hasPrediction && !hasRealModel ? (
    <div className="empty-state">
      This artifact is synthetic-only, so its output is withheld until it is
      evaluated against real historical outcomes.
    </div>
  ) : null}

  {hasPrediction && hasRealModel ? (
    <div className="prediction-grid">
      <div className="horizon-card">
        <div className="horizon-label">Signal direction</div>
        <div className="horizon-direction" style={{ color }}>
          {direction.toUpperCase()}
        </div>
        <p className="horizon-rationale">
          Experimental analytics only. This signal is not financial advice.
        </p>
      </div>

      <div className="horizon-card">
        <div className="horizon-label">Signal status</div>
        <div className="horizon-direction">VALIDATED</div>
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
