export default function CompanyOverview({ overview }) {
  return (
    <section className="card overview-card">
      <h2 className="section-title">Company Overview</h2>
      <p className="overview-text">{overview}</p>
    </section>
  );
}
