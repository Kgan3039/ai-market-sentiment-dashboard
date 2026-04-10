export default function PredictionModel({ prediction }) {
  const { shortTerm, mediumTerm, catalysts } = prediction;

  const directionColor = (dir) =>
    dir === "Bullish" ? "var(--positive)" : dir === "Bearish" ? "var(--negative)" : "var(--accent)";

  return (
    <section className="card prediction-card">
      <h2 className="section-title">Prediction Model</h2>

      <div className="prediction-grid">
        <PredictionHorizon data={shortTerm} directionColor={directionColor(shortTerm.direction)} />
        <PredictionHorizon data={mediumTerm} directionColor={directionColor(mediumTerm.direction)} />

        {/* Upcoming Catalysts */}
        <div className="catalyst-block">
          <h3 className="subsection-title">Upcoming Catalysts</h3>
          {catalysts.map((c, i) => (
            <div key={i} className={`catalyst-item impact-${c.impact}`}>
              <span className="catalyst-date">{c.date}</span>
              <span className="catalyst-event">{c.event}</span>
              <span className="catalyst-impact">{c.impact}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function PredictionHorizon({ data, directionColor }) {
  return (
    <div className="horizon-card">
      <div className="horizon-label">{data.horizon}</div>
      <div className="horizon-direction" style={{ color: directionColor }}>
        {data.direction}
      </div>
      <div className="confidence-bar-wrapper">
        <div className="confidence-bar-label">Confidence: {data.confidence}%</div>
        <div className="confidence-track">
          <div className="confidence-fill" style={{ width: `${data.confidence}%`, background: directionColor }} />
        </div>
      </div>
      <p className="horizon-rationale">{data.rationale}</p>
    </div>
  );
}
