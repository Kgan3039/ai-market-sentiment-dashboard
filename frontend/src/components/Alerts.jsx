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

function formatAlertStatus(status) {
  if (!status) return null;
  return status.replace(/_/g, " ");
}

function AlertItem({ alert }) {
  const statusLabel = formatAlertStatus(alert.status);
  const itemLabel = alert.count === 1 ? "item" : "items";

  return (
    <div className={`alert-item alert-${alert.type || "info"}`}>
      <div className="alert-body">
        <div className="alert-title-row">
          <div className="alert-title">{alert.title}</div>
          {statusLabel ? <span className="alert-status">{statusLabel}</span> : null}
        </div>
        <div className="alert-message">{alert.message}</div>
        {alert.source || typeof alert.count === "number" ? (
          <div className="alert-meta">
            {alert.source ? <span>{alert.source}</span> : null}
            {typeof alert.count === "number" ? <span>{alert.count} {itemLabel}</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
