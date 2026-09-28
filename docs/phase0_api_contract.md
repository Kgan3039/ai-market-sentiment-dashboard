# Phase 0 Read API Contract

Owner: Mihir. The same response shapes are served from two explicitly selected
sources: the committed fixture (the default) and the pipeline's persisted
SQLite output (see "SQLite Source" below). The frontend must not depend on
legacy sentiment, prediction, market, or dashboard routes.

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
  `published_at`. `published_at` is always present and is `null` when the
  publisher gave no publication time; a timestamp is never invented.
- `label` is non-empty and has no upper length bound. A degraded theme's label
  is the persisted M5 label, which is a real canonical headline, and no
  pipeline stage bounds headline length (Yahoo and RSS titles are only
  stripped, `display_text` only normalizes whitespace and typography, and
  `themes.label` has no length CHECK). Labels are never truncated. Generated
  labels are separately bounded by A2 to at most eight words.
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

## SQLite Source

`PHASE0_NARRATIVE_SOURCE=sqlite` selects
`backend/app/phase0/sqlite_repository.py`. The default is `fixture`; the
source is never switched because a database file exists, and SQLite mode never
falls back to the fixture. `PHASE0_DATABASE_PATH` and
`PHASE0_PIPELINE_VERSION` are read exactly as `pipeline.py` reads them, with
the same defaults.

**GET only reads.** The SQLite source holds a `phase0.repository.Phase0Reader`
(every query opens SQLite `mode=ro` with `query_only` and a write-denying
authorizer) and never a `Phase0Repository`. It never calls `ensure_summary`,
A2, or a provider, and needs no `GEMINI_API_KEY` and no network.

- **Tickers:** the fixed universe, in universe order, whatever is persisted.
  `theme_count` is the number of themes the latest day would show.
- **Omitted date:** the newest day on which the ticker has a live story. A day
  is never skipped because an older day has a summary.
- **Theme id:** `themes.theme_key`, which is stable across reruns
  (`themes.id` is re-minted whenever stories are reconciled).
- **Summaries:** a theme's generated label and sentences are served only when
  `phase0.summary_lifecycle.current_summary_artifact` returns a current
  artifact under `phase0.summary_runner.production_generation_policy()`, the
  scheduler's own policy. That theme's `stories` and `citations` are the
  artifact's frozen evidence, and sentence `citation_ids` are `story:<id>` in
  their persisted order. Any other theme is `degraded: true` with no
  sentences and its persisted M5 label; a stale, superseded, invalidated or
  corrupt artifact is never shown. If the policy cannot be resolved, every
  theme is degraded.
- **Unhealthy days:** a day A2's population gate refuses (no theme set,
  M2-only, mixed, or inconsistent) has `themes: []`, and every live story is
  in `other_coverage` in story order.
- **Other coverage:** persisted Other Coverage by position, then stories M5
  excluded for having no encodable text, by story id. No live story is lost,
  and each appears exactly once.
- **Status:** `last_runs` is the newest `run_log` row per stage for the
  pipeline version, with `error_count` only; error text is never exposed.
  `data_as_of` is the newest successful run's completion, else the newest
  degraded run's. A ticker day's `data_as_of` is that partition's newest
  successful or degraded run, else the global value.
- **503:** a missing, unreadable, corrupt or wrongly-versioned database, no
  completed run, or persisted rows the contract cannot represent without
  inventing data (a story without a URL, a timestamp without a UTC offset, a
  theme without a `theme_key`) answer `503` with the fixed detail "Coverage is
  temporarily unavailable." The cause is logged on the server only.

The default stays `fixture`. A deployment selects `sqlite` explicitly, once
the host checks in `docs/phase0_deployment_handoff.md` pass against real
persisted output.

### Deployment requirements

- The API process must see the same `GEMINI_MODEL` and
  `GEMINI_MAX_OUTPUT_TOKENS` as the scheduled pipeline, and the same checkout
  (copy rules). Otherwise it resolves a different policy fingerprint: current
  summaries appear degraded, or an older policy's still-valid artifact can be
  served. The API does not need, and should not be given, `GEMINI_API_KEY`.
- `PYTHONPATH` must include the project root as well as `backend/`, because
  the SQLite source imports `phase0`, `ai`, and `tools`.
- **The API is a read-only consumer of the writer-owned live WAL database.**
  Read-only here means the database *content*: every query opens SQLite
  `mode=ro` with `query_only` and a write-denying authorizer, so no row and no
  byte of the main database file changes. It does not mean SQLite touches no
  file. A WAL read needs the `-wal` and `-shm` coordination files, and SQLite
  deletes both when the writer's last connection closes -- so after each
  scheduled run the database is checkpointed and only the main file exists.
  What the API's next read then does depends on the directory:
  - if the API may create files there, SQLite creates an empty `-wal` and a
    `-shm` index, with the database file's permissions, and leaves them;
  - if it may not, the read fails ("attempt to write a readonly database")
    and the API answers the fixed `503`.
- **Supported deployment:** the writer side provisions and keeps the live WAL
  state -- for example a writer-owned process that holds one connection to
  the database open, so `-wal` and `-shm` survive the pipeline's own
  open/write/close cycles. The API user then needs only read access to the
  database, `-wal` and `-shm` files and no write access to the directory; it
  creates no file and sees each new commit. Without that, the API fails
  closed with `503` rather than being given ownership of database files.
- Never open the live database with `immutable=1`: SQLite would stop
  noticing the writer's commits. `nolock` or exclusive locking on the reader
  would likewise give up correct coordination with the writer.
