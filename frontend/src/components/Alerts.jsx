export default function Alerts({ alerts }) {
  return (
    <section className="card">
      <h2 className="section-title">Important Alerts</h2>
      <div className="alerts-list">
        {alerts.map((alert) => (
          <AlertItem key={alert.id} alert={alert} />
        ))}
      </div>
    </section>
  );
}

function AlertItem({ alert }) {
  const icons = { warning: "⚡", danger: "🔴", success: "✅" };

  return (
    <div className={`alert-item alert-${alert.type}`}>
      <div className="alert-icon">{icons[alert.type]}</div>
      <div className="alert-body">
        <div className="alert-title">{alert.title}</div>
        <div className="alert-message">{alert.message}</div>
      </div>
    </div>
  );
}
