export default function Alerts({ alerts = [] }) {
  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">System Status</h2>
        <span className="section-kicker">{alerts.length} checks</span>
      </div>
      <div className="alerts-list">
        {alerts.length > 0 ? (
          alerts.map((alert) => <AlertItem key={alert.id} alert={alert} />)
        ) : (
          <div className="empty-state">No alerts to show.</div>
        )}
      </div>
    </section>
  );
}

function AlertItem({ alert }) {
  const icons = { info: "i", warning: "!", danger: "x", success: "ok" };
  const status = alert.status || alert.type || "info";

  return (
    <div className={`alert-item alert-${alert.type || "info"}`}>
      <div className="alert-icon">{icons[alert.type] || "i"}</div>
      <div className="alert-body">
        <div className="alert-title-row">
          <div className="alert-title">{alert.title}</div>
          <span className={`alert-status status-${status}`}>{status}</span>
        </div>
        <div className="alert-message">{alert.message}</div>
        {alert.source || typeof alert.count === "number" ? (
          <div className="alert-meta">
            {alert.source ? <span>{alert.source}</span> : null}
            {typeof alert.count === "number" ? <span>{alert.count} items</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
