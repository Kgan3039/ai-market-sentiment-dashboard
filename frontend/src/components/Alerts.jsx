export default function Alerts({ alerts = [] }) {
  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Data Availability</h2>
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
  return (
    <div className={`alert-item alert-${alert.type || "info"}`}>
      <div className="alert-body">
        <div className="alert-title-row">
          <div className="alert-title">{alert.title}</div>
          {alert.status ? <span className="alert-status">{alert.status}</span> : null}
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
