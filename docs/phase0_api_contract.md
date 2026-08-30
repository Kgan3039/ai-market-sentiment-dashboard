# Phase 0 Read API Contract

Owner: Mihir. This contract is currently fixture-backed. The same response
shapes must be served when the API switches to the merged pipeline's persisted
output. The frontend must not depend on legacy sentiment, prediction, market,
or dashboard routes.

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

`status.data_as_of` is required, authoritative pipeline metadata. Missing or
unparseable values are invalid pipeline output and are never inferred from
ticker coverage timestamps.

## Fixture Source

The committed source is
`backend/app/phase0/fixtures/phase0_narratives.json`. It intentionally covers
normal summaries, a degraded summary, Other coverage, and an empty coverage
day so UI work and API tests can proceed independently of live ingestion.

## Live Persistence Handoff

The API depends only on `NarrativeReadRepository` in
`backend/app/phase0/repository.py`. The live-data adapter must preserve that
interface while reading the pipeline's persisted output.

- Read only API-eligible, completed narrative/theme output by
  `(ticker, trading_day)`, ordered by `salience_rank`.
- Decode the stored `summary` into the `sentences` shape produced by
  `ai.summarization.ThemeSummary`.
- Resolve every cited sentence to its persisted evidence and returned member
  stories; never emit a generated sentence with an unresolved citation ID.
- Return unclustered/noise stories through `other_coverage` once Matthew's
  clustering stage records that assignment.
- Map run and persistence health into `/meta/status` and the API degradation
  state, and compute freshness using the approved market-hours rule.
- Exclude incomplete or failed outputs instead of presenting intermediate
  records as a completed theme set.

### Live-data readiness gate

Do not switch the default API source from fixtures merely because the SQLite
repository is present. The switch requires all of the following:

- The pipeline completes the required downstream processing and persists
  API-eligible narrative/theme output. Raw items and intermediate story
  records, including degraded intermediate story output, do not satisfy this
  contract.
- The live-data adapter resolves every cited sentence to persisted evidence,
  maps run and persistence health into the API degradation state, and excludes
  incomplete or failed outputs.
- Temporary-SQLite contract tests cover `/tickers`, `/themes`, and
  `/meta/status` with real persisted rows before the source selection changes.

Do not change these response names during the handoff. Add live-data adapter
tests using a temporary SQLite database before switching the default source
from fixtures.
