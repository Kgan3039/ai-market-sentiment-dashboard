# Phase 0 Read API Contract

Owner: Mihir. This contract is fixture-backed until Isaac's Phase 0 SQLite
pipeline is merged, then the same response shapes must be served from the
repository. The frontend must not depend on legacy sentiment, prediction,
market, or dashboard routes.

## Endpoints

`GET /api/v1/tickers` returns the fixed Phase 0 ticker universe in this order:
TSLA, NVDA, AMD, AAPL, META. Each item contains `ticker`, `company_name`,
`data_as_of`, `theme_count`, and `is_stale`.

`GET /api/v1/tickers/{ticker}/themes?date=YYYY-MM-DD` returns `ticker`,
`date`, `data_as_of`, `themes`, and `other_coverage`. Omitting `date` returns
the latest available trading day. A valid ticker with no coverage for the
requested date returns `200` with empty `themes` and `other_coverage`; a ticker
outside the five-symbol universe returns `404`.

Every theme contains `id`, `label`, `rank`, `sentences`, `citations`,
`stories`, `outlet_count`, `story_count`, and `degraded`.

- `sentences` matches `ai.summarization.ThemeSummary.sentences`: each entry is
  `{text, citation_ids}`.
- Every `citation_id` resolves to a member of the theme's `citations` array.
- Every citation and story has `id`, `headline`, `outlet`, `url`, and
  `published_at`.
- A degraded theme has `degraded: true`, an empty `sentences` array, and a
  non-empty story list whenever coverage exists.

`GET /api/v1/meta/status` returns `data_as_of`, `is_stale`, and one latest-run
record per pipeline stage. A record contains `stage`, `status`, `started_at`,
`completed_at`, `duration_ms`, and `error_count`.

## Fixture Source

The committed source is
`backend/app/phase0/fixtures/phase0_narratives.json`. It intentionally covers
normal summaries, a degraded summary, Other coverage, and an empty coverage
day so UI work and API tests can proceed independently of live ingestion.

## SQLite Handoff

The API depends only on `NarrativeReadRepository` in
`backend/app/phase0/repository.py`. When Isaac's `phase0.repository.Phase0Repository`
lands on `main`, add a `SQLiteNarrativeRepository` implementation behind that
interface.

- Read `themes` by `(ticker, trading_day)` ordered by `salience_rank`.
- Decode the stored `summary` into the `sentences` shape produced by
  `ai.summarization.ThemeSummary`.
- Resolve `citations` and canonical story members through `stories.member_ids`
  and `raw_items`; never emit a generated sentence with an unresolved ID.
- Return unclustered/noise stories through `other_coverage` once Matthew's
  clustering stage records that assignment.
- Map the newest `run_log` entry for each stage to `/meta/status` and compute
  the stale flag using the approved market-hours rule.

Do not change these response names during the handoff. Add live-data adapter
tests using a temporary SQLite database before switching the default source
from fixtures.
