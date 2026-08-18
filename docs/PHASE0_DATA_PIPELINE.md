# Phase 0 Data Pipeline

`pipeline.py` orchestrates the Phase 0 ingestion components. It calls
Yahoo (#61) and RSS (#62); it persists nothing itself.

That division is the design, not an implementation detail. I1 (#57) made
every durable write happen inside a run that names exactly one partition —
one stage, one ticker, one trading day, one pipeline version — and both
fetchers settle their own partitions against that contract. An
orchestrator that also wrote run rows would be building a second, weaker
audit beside the authoritative one, so this one writes none: no
`log_stage`, no `set_source_state`, no `insert_raw_items`, no connection,
no `run_log` row of its own.

## Two different things called a "run"

| | Pipeline invocation | Repository run |
|---|---|---|
| What it is | one execution of `pipeline.py` | one stage against one valid partition |
| Identity | `invocation_id`, a correlation id | `run_id`, a partition identity |
| Scope | many tickers, days, and feeds | exactly one ticker/day/version/stage |
| Owner | this file | `stage_run` and the component APIs |
| Durability | in memory and in the structured log | a `run_log` row, in the same transaction as the data |
| How many | one per process | several per invocation |

The invocation id is handed to a component as a **base**, and the
component derives its own partition identities from it — `partition_run_id`
in `phase0/yahoo.py` and `phase0/rss.py` appends the partition before
anything is recorded:

```
phase0-6ce1e25c…:yahoo:NVDA:2026-08-18          one ticker, one day
phase0-6ce1e25c…:rss:marketwatch:snapshot:2026-08-18   one feed's evidence
phase0-6ce1e25c…:rss:marketwatch:aapl:2026-08-18       one ticker's relevance
```

Nothing ever opens a run under the bare base. That is what keeps
`run_log`'s `UNIQUE(run_id, stage)` meaning "one partition" rather than
"one process" — and it is what the previous version of this file got
wrong, by minting one uuid and one trading day and handing both to every
fetcher.

There is no pipeline-level audit table in the current schema, and no
migration adds one here. Faking one into `run_log` would mean writing a
row whose `run_id` names no partition, which is the single thing I1's
identity rule exists to prevent. Until a product requirement justifies
real invocation-level schema, the structured log is the invocation record
and the per-partition `run_log` rows remain the durable truth.

## Commands

```bash
.venv/bin/python pipeline.py                      # live ingestion
.venv/bin/python pipeline.py --status             # latest durable stage rows
.venv/bin/python pipeline.py --database-info       # schema version, migrations, counts
.venv/bin/python pipeline.py --replay             # rebuild RSS relevance, no network
.venv/bin/python pipeline.py --database /var/lib/ticker-narratives/phase0.sqlite3
PHASE0_DATABASE_PATH=/var/lib/ticker-narratives/phase0.sqlite3 \
  .venv/bin/python pipeline.py
```

### Exit codes

| Code | Status | Meaning |
|---|---|---|
| 0 | `success` | every component completed with nothing unsettled |
| 0 | `skipped` | another invocation held the lock; nothing was attempted |
| 1 | `degraded` | real evidence was persisted, and something is incomplete |
| 2 | `failed` | every mandatory component settled nothing |

Three outcomes need three codes. A degraded run stored a usable day with
one source down; reporting it as success hides an outage, and reporting it
as failure throws away the day. Alert on 2 and trend 1.

### There is no `--date`

Both fetchers dropped their `trading_day` argument deliberately: a run's
day is a partition identity the evidence decides, derived from
`published_at` falling back to `fetched_at`. The repository refuses a
batch whose day disagrees with its run, so a day announced by the
scheduler could only ever be ignored or fatal. A fetch that starts at
23:55 and returns yesterday's article stores it under yesterday.

The pipeline computes an `invocation_day` in `America/New_York` for log
and CLI organisation only. It labels the invocation; it never labels
evidence, and it cannot override a component's partition derivation.

## Failure isolation

Components run in sequence and fail independently. There is no
cross-source transaction, so evidence is durable the moment its own
partition commits:

* Yahoo succeeds, RSS fails → Yahoo's five ticker-days stay committed;
  the invocation is `degraded`.
* One ticker's provider fails → the other four partitions are untouched;
  the failed one is recorded `degraded` in its own `run_log` row.
* One feed is unreachable → the other feeds still run.
* A component raises an unexpected exception → it is recorded `failed`,
  and the next component still runs.
* A component's **construction** fails — a missing or malformed
  `feeds.yaml`/`aliases.yaml`, a blank `pipeline_version` — → same
  treatment. Components are built inside their own stage, not while the
  stage list is assembled, so a YAML typo costs one component rather than
  the invocation, and the CLI answers with an exit code rather than a
  traceback.

## Replay

`--replay` calls I3's `reclassify_persisted`. It reads persisted evidence
and nothing else — the fetcher is built with an HTTP callable that raises,
so a replay that reached for a feed would fail loudly rather than quietly
refetch.

Each `(ticker, day)` partition's derived state is **replaced** inside that
partition's own terminal run. Nothing is deleted, no stage key is reset,
raw evidence is never touched, and a partition that fails keeps the
derived state it already had. Running it twice produces the same result as
running it once.

**What replay does *not* do today.** `--replay` reports this itself, and
the claim is deliberately narrow:

| | Status |
|---|---|
| RSS relevance | replayable |
| Yahoo refetch | not replayed — replay never fetches |
| Dedup, clustering, summarization (M1–M5) | not implemented, not registered |
| Scoped replay (one ticker/day/version) | unavailable — `reclassify_persisted` takes no scope |

Replay currently covers **all** persisted RSS evidence, because that is
the only scope the public API offers. Downstream stages register in
`DOWNSTREAM_STAGES` as they land; the tuple is empty on purpose, so the
CLI's answer about what it can rebuild stays true without being updated.

## Scheduling

`deploy/phase0-pipeline.cron` is the production template: every 30 minutes
from 09:00 through 16:30 America/New_York on weekdays, hourly otherwise.

**Timezone.** Every expression is written in `America/New_York` and means
nothing in another zone. `CRON_TZ` is what makes that true, and it is a
Vixie-cron/cronie extension — an implementation that ignores it will run
these lines in host local time, and the file will look installed and
correct while the market window lands somewhere else. Verify on the
deployment host. DST needs no handling: the zone carries EST and EDT with
it, which a UTC schedule would not.

**Overlap.** A fetch can outlast its interval, and cron starts the next
copy regardless.

`pipeline.py` acquires the lock itself and is the only thing that does.
There is no outer `flock` in the crontab, deliberately: two nested locks
are two *different* locks, and a cron run holding the shell's while a
manual run held `pipeline.py`'s own default would leave both believing
they were alone.

**Every production entrypoint must pass the same `--lock-file`.** Cron
does; so must anything else pointed at the same deployment:

```bash
.venv/bin/python pipeline.py --lock-file /var/lock/phase0-pipeline.lock
```

Omitting it falls back to `<database>.lock`. That default is right for
local development — two checkouts on one laptop should not block each
other — and wrong for a deployment, where cron, systemd, and an operator
may each spell the database path differently while targeting one pipeline.

The loser is refused immediately rather than queued behind a run that may
itself be wedged: it logs `invocation_skipped`, does no component work at
all, and exits 0. The lock is released however the invocation ends —
success, degraded, or failed.

**This is a template, not a proven unattended deployment.** It has not
been run against a real host, there is no alerting, no log rotation, no
health check, and no restart policy here. Replace `/opt/ticker-narratives`,
the Python path, the database path, and the log destination before use.

## Logging

One JSON object per line on `phase0.pipeline`: `invocation_started`,
`component_completed` per component, `invocation_completed`. Every payload
carries the `invocation_id`, and every payload goes through I1's
`redact_secrets` — including errors this file built from exception
messages, which are exactly where a bearer token tends to end up.

These logs are a process-level summary. They are not an audit: the
`run_log` rows the components wrote are.
