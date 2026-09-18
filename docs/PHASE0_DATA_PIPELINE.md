# Phase 0 Data Pipeline

`pipeline.py` orchestrates the Phase 0 components. It calls Yahoo (#61)
and RSS (#62), then the intelligence component that turns what they
persisted into stories and themes; it persists nothing itself.

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

## Stories and themes

After both ingestion components have settled, the `intelligence`
component reconciles persisted stories and themes for the days that need
it. It calls `PartitionCoordinator` and nothing else: the coordinator owns
the ordering that makes the output correct — capture the previous theme
identities, reconcile stories, reconcile themes with the captured
identities — and isolates each ticker-day from the others: a failure in
any of the three steps settles that partition and the next one runs. A
capture failure is recorded as a `failed` `themes` run for the partition
and its stories are left untouched. Each partition's work lands as its
own `stories` and `themes` run-log rows, written by the reconcilers under
their own stage names. **No summary is generated by the scheduled path.**
The narrative API is unaffected and may still serve fixtures; a live run
producing themes does not establish release provenance for anything.

**Which partitions.** Every scheduling fact is kept per ticker/day
partition — that is the grain the ledger records it at — and the day is
only the unit the coordinator executes. Three sources, unioned, and the
days they name are what runs:

* *Touched* — the partitions this invocation's evidence-stage runs
  (`fetch_yahoo`, `ingest_rss`, `classify_rss`, `reclassify_rss`)
  **durably changed**, read back from `run_log` under the invocation's
  id prefix: a ticker-scoped run of one of those stages whose counts
  record `raw_items_inserted > 0` or `relevance_changed > 0`. Those
  counters are written by the mutations themselves — an insert that
  created the row, a classification after which the item's association
  with that ticker, or its eligibility, is different from before. A
  late-arriving article inserts under its published day, so that
  partition is touched and rebuilt. A provider serving an article already
  stored, a feed re-listing an old entry, or a classifier re-deciding an
  association that already stood opens runs that record no change and
  touches nothing; `relevance_assigned` counts decisions, not changes,
  and is not consulted. A duplicate observation therefore cannot reopen
  an expired failure.
* *Retried* — partitions whose newest `stories` or `themes` outcome is
  *unresolved* — a `failed` run, or a `degraded` run carrying a
  `stage_degraded` marker — and whose failure episode is still inside its
  window. Without this, a transient failure — a model cache missing on
  Friday's last run — would stay failed until fresh evidence happened to
  arrive for that partition. The marker is what separates that case from
  an identical replay, which also settles `degraded` (unchanged rows count
  as partial work) and is deliberately *not* retried: nothing went wrong
  and nothing would change.

* *Recovered* — partitions, not touched, on days where an evidence-writing
  run completed within the horizon, whose intelligence work under the
  running `pipeline_version` either never began or was interrupted
  before themes. Per ticker, read from the run ledger:

  | newest `stories` run | `themes` run | decided by |
  |---|---|---|
  | none | none | **recovery** — never attempted |
  | `success`, or `degraded` with no marker | none | **recovery** — interrupted between stories and themes |
  | `failed`, or `degraded` with a `stage_degraded` marker | none | **the retry window** — this is an episode |
  | any | any | the outcome itself, and the retry window if it is unresolved |

  "No `themes` row" alone is *not* the criterion. The coordinator does not
  open themes over a story failure, so a partition whose stories failed
  also has no `themes` row — and it already has an episode with an anchor
  and a deadline. Recovering it under the evidence-triggered identity
  would give it a new anchor on every invocation and the deadline would
  never arrive; so the newest `stories` outcome is read, and an
  unresolved one keeps the partition out of recovery whether its window
  is open or expired. A settled `stories` outcome the coordinator would
  have carried on from, with no `themes` row, is the interrupted case and
  is recovered. The `themes` row is the witness that a partition was
  carried through — the coordinator writes one for every partition it
  finishes, healthy, empty, M2-only, or failed at capture — and a healthy
  completed day has one for every partition and is never selected again.
  The check is per ticker, so one processed ticker never vouches for
  another on the same day.

**Identity is per partition — touched, then episode, then everything
else.** The identity a partition runs under decides whether a failure
there may anchor a new retry window, so it follows the strongest fact
about *that partition*, never about its neighbours on the day. Evidence
this invocation durably changed is the strongest: it is a new reason to
process, and a failure on it legitimately opens a fresh window, even
over an old episode. An existing episode comes next: a partition whose
newest `stories` or `themes` outcome is unresolved, and which this
invocation did not touch, runs under the retry identity — whether it was
selected as a retry, is rerun because its day was, or is an expired
episode that came along — so no attempt on it can renew a deadline it
already owns. Every other partition — never attempted, interrupted
before themes, or a settled neighbour rerun with its day — runs under
the evidence-triggered identity, so if its attempt fails, that failure
is an anchor and the partition is retried on its own account. One day
can therefore hold a touched partition, a retried one and a recovered
one, each under its own run id, and no partition's identity is
inherited from another's.

**The retry window is anchored, not rolling.** An episode is every
unresolved attempt since the last one that resolved the partition, and
its window opens at the newest attempt that was *not itself a retry* —
the failure that started it, or a later failure that followed new
evidence. A partition with an episode this invocation did not touch
runs under `<invocation>:intelligence-retry:<ticker>:<day>`; a retry is
recognised **structurally**, by splitting the run id three times from the
right and comparing the pipeline-owned component exactly, never by
searching the string, so an invocation id a caller chose is free to
contain that text and still be an ordinary evidence-triggered run.
Retries never move the anchor, so a permanent failure retried every half
hour stops being retried `RETRY_HORIZON` (three days) after it began,
whatever the retries recorded. Recovery never claims a partition with an
unresolved `stories` outcome, and a repeat observation of stored
evidence never touches one, so neither moves it either: the window is
anchored to the first unresolved evidence-triggered failure, and only
a genuine durable change to the partition's evidence can open a new
one. A success closes the episode. New evidence for the partition is
not blocked by an expired episode: it arrives through the touched path,
and if that attempt fails too, a fresh window opens on it.
Selection is scoped to the running `pipeline_version`; one version's
`stories` and `themes` outcomes neither hide, schedule, nor recover
another's.

The same three-day horizon bounds both recovery and retry; a day the
operator would want caught up for one reason is a day they would want
caught up for the other. Evidence persisted before intelligence could run
and never reached within that window is not swept afterwards.

Rerunning a day brings that day's other partitions with it. Healthy
ones do not rewrite their stories or themes, but each still writes its
own `stories` and `themes` run rows, under its own identity — an
identical replay is recorded as `degraded` with no errors, which is the
ledger's word for "unchanged".

**One clock.** The repository owns it: `Phase0Repository(clock=…)`,
defaulting to UTC wall time, stamps every run's `started_at` and
`completed_at`, and `run_live` reads the same clock for the invocation's
day label and for every cutoff the intelligence component computes. There
is deliberately no `now=` parameter on `run_live`, `run_replay`, or the
intelligence builder — a caller-supplied instant would govern the
comparison but not the timestamps it compares against. A clock that
returns a naive datetime is refused. Evidence timestamps
(`raw_items.fetched_at`, `published_at`) are written by the fetchers and
are not consulted by scheduling; they remain source semantics.

**Status.** The component is `success` when every partition's stories and
themes landed, `degraded` when any partition failed or degraded on top of
persisted stories, and `failed` when no story generation was written at
all. It is not mandatory — a day with nothing to reconcile is an honest
`success` and must not stop an invocation whose fetches all failed from
being `failed` — but that cannot make its own failure look green:
`success` requires every component to succeed, so a failed or degraded
intelligence run is a `degraded` invocation at best. Semantic dedup being
unavailable retains M2 stories marked `m2.exact` and ships no themes for
that partition, exactly as before.

### Guarded summary generation (A2) exists, and is not wired in

`ai/guarded_summary.py` and `phase0/summaries.py` implement A2: one
persisted theme, read back as a `Phase0Reader.theme_population` snapshot,
is projected onto a frozen generation input (`story:<stories.id>`
citation ids, canonical title, the M3/M5 standfirst, outlet, timestamp),
sent to Gemini at most twice, and validated against that same frozen
input. The result is a typed value — `accepted` with a `ThemeSummary`, or
`unavailable` with a reason and the attempts made — never a fabricated
fallback.

What it does and does not claim, exactly:

| | |
|---|---|
| Registered in `pipeline.py` / the coordinator | **no** — nothing schedules it |
| Writes `themes.summary`, `themes.status`, `themes.citations`, `run_log` | **no** — it holds no repository handle |
| Persists, caches, or invalidates summaries | **no** — A3 owns persistence, cache, accounting and lifecycle |
| Served by the narrative API | **no** — the API is still fixture-backed |
| "accepted" means | structurally grounded (every sentence cites ids that exist in the frozen input) and clean under `config/banned_phrases.txt` |
| "accepted" does **not** mean | semantically faithful — sentence support remains the G2 human-review gate (A4b) |

The population health gate refuses stale, M2-only, mixed-stage, or
inconsistent theme sets with a stable code before any prompt is built.

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
| Stories and themes (M2–M5) | produced by the **live** path; `--replay` does not drive them |
| Summarization | implemented (A2, guarded), not registered; nothing persisted |
| Scoped replay (one ticker/day/version) | unavailable — `reclassify_persisted` takes no scope |

Replay currently covers **all** persisted RSS evidence, because that is
the only scope the public API offers. Downstream components register in
`DOWNSTREAM_STAGES` as builders bound at run time; the CLI reports what is
registered and, separately, what replay actually drives.

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
