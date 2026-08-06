# Phase 0 persistence (issue #57)

Single SQLite datastore (WAL) and the one repository module every Phase 0
stage and the read API go through. Nothing outside this package opens a
SQLite connection.

**Decision (spec section 2, Isaac's Day 1 call):** SQLite, not the existing
persistence layer. The existing layer stores prediction/sentiment
artifacts, which Phase 0 does not produce, and has no raw-item store, so
AC-8 replayability would have had to be built on top of it anyway.

## Layout

| Module | What it owns |
|---|---|
| `repository.py` | `Phase0Repository` — every public read and write |
| `schema.py` | migration loading, the ledger, atomic application |
| `migrations/` | additive `NNN_name.sql` files; never edited after release |
| `models.py` | transport records for the reconciliation boundaries |
| `tickers.py` | the approved universe and its normalization |
| `redaction.py` | credential removal for errors and metadata |
| `scalars.py` | the two scalar policies: redact diagnostics, reject identity |
| `embeddings.py` | validation for M1's durable embedding cache |
| `errors.py` | typed errors (`Phase0ValidationError` is a `ValueError`) |

## Guarantees

**No public writable connection, and `ATTACH` is denied.**
`read_connection()` is the only connection the public surface hands out.
It is opened `mode=ro` *and* carries a SQLite statement authorizer that
allows reads and refuses everything else. Both matter: `mode=ro` protects
the file it opened and says nothing about any other, so a caller could
otherwise turn `query_only` off, `ATTACH` the same database under a second
name, and write through the alias — committing data with no run log at
all. `ATTACH` and `DETACH` are denied outright, as are all DML, all DDL,
writes to `temp`, and any pragma that could re-enable writing, so `main`,
`temp`, and any alias are equally out of reach. Reads, joins, and
schema-inspection pragmas (`table_info`, `integrity_check`, `user_version`
in its query form, …) work normally. Raw writable access lives on
`repository.admin.connect_writable()`, for manual repair and migrations,
and is not restricted.

**One transaction per public write.** A batch lands whole or not at all.
Helpers that take a connection never commit; only the private writable
connection does.

**Migrations are additive and are never rewritten.** `schema_migrations`
records each applied file by name and SHA-256. Editing an already-applied
migration is refused — add a new file instead. Each migration runs inside
one `BEGIN IMMEDIATE` that also advances `user_version`, so a failure
leaves the schema, the ledger, and `user_version` untouched.

**The database enforces the contracts, not just the API.** Direct SQL
cannot store an unsupported ticker, cite a raw item that is not in one of
the theme's member stories, cite one raw item from two themes in the same
ticker-day, group a story from another ticker-day *or another pipeline
version*, or orphan an embedding. Each of those holds on `UPDATE` as well
as on `INSERT`, and a parent row — story, theme, or theme set — cannot be
relocated out from under the children that reference it.

**The approved universe is a constant, not a table.** The five symbols are
written as a literal into every ticker trigger, so nothing that ordinary
SQL can write changes what is accepted. `supported_tickers` remains as a
readable projection of that constant and is sealed against insert, update,
and delete — inserting `GOOG` there no longer widens anything, because
nothing consults it.

**Stage logging cannot be switched off.** `stage_run()` writes a `run_log`
row in a `finally` block, including when the stage body raises. No public
method takes a `persist_run_log`-style flag.

**Pipeline mutations carry their run.** `ingest_raw_items`,
`reconcile_stories`, `reconcile_themes`, `persist_embeddings`, and
`record_source_state` all require the `StageRunContext` that `stage_run()`
yields, and write their `run_log` row in the same transaction as the data.
A missing, completed, foreign, or lease-expired handle raises before
anything is written. The unlogged row helpers live on `repository.admin`,
where they are labelled administrative rather than presented as pipeline
entrypoints.

**The run handle cannot be forged.** `StageRunContext` refuses direct
construction, and authorization is by object identity against a registry
the owning repository keeps — so a copy, a pickle, an `object.__new__`
shell, or a look-alike with every field matching authorizes nothing. It is
read-only, and its counts are derived by the operations that run under it;
there is no caller-facing way to seed or edit them.

**Lease ownership is checked inside the write transaction, and released
there too.** `stage_run` proves ownership before the context exists: a
missing, foreign, completed, reclaimed, or expired key means no context,
no registration, and no run log. Then a logged mutation takes the write
lock, re-validates the capability and the key on that same connection,
mutates, writes the run log, and — when it is the stage's `terminal=True`
operation — transitions the key to its final status and releases the
lease, all before committing. There is no committed state in which the
data says success and the key is still reclaimable as `running`. Updating
zero rows while finishing a key is an error, never a silent success.

**A run may only write its own partition.** Every logged operation derives
the payload's partition and rejects the whole batch before writing if it
disagrees with the run: an NVDA run cannot ingest an AMD item, a v1 theme
set cannot cite a v2 story, and an embedding cannot name a source from
another ticker-day.

**Raw evidence is preserved; operational metadata is redacted.** These are
different serializers, not a flag: `serialize_raw_evidence` stores a
publisher payload exactly as supplied, and `serialize_operational_metadata`
redacts everything Phase 0 says *about* a fetch. A credential-looking
string inside publisher content is evidence and stays; a transport
credential belongs to I2/I3's fetch boundary and must never reach the
payload in the first place. Scalar columns follow the same split, in
`scalars.py`: diagnostics are redacted, identity and configuration
(provider namespace and item id, story keys, model name and revision,
algorithm version, config fingerprint) are *rejected* with a typed error,
because a silently rewritten identifier repoints the row it names.
Scheme-introduced credentials are removed at any length — `Bearer a` goes,
`bearer instrument` stays — because a short token is still a token.

The accurate statement of the security contract is therefore: **no
operational credential supplied through a diagnostic or configuration
surface reaches persistence; raw provider evidence is preserved
unchanged.**

**Fingerprints are not identifiers.** M2's `cluster_fingerprint`, M3's
`story_fingerprint`, and M5's `fingerprint` / `theme_key` are stored beside
the durable row ids as change-detection handles, never instead of them.

## The boundaries downstream stages use

```python
repository.migrate()

# Everything a stage writes happens inside its run (#68).
with repository.stage_run(
    run_id=run_id, stage="m3.semantic", trading_day="2026-07-23",
    pipeline_version="v1", ticker="NVDA", stage_key=key,
) as run:
    # Ingestion (#61, #62)
    repository.ingest_raw_items(items, run=run, source_state=state)

    # M2/M3 — one ticker/trading-day, atomically
    report = repository.reconcile_stories(
        run=run, ticker="NVDA", trading_day="2026-07-23",
        pipeline_version="v1", stories=[StoryRecord(...)],
    )

    # M5 — one ticker/trading-day theme set, atomically
    repository.reconcile_themes(
        run=run, ticker="NVDA", trading_day="2026-07-23",
        pipeline_version="v1", theme_set=ThemeSetRecord(...),
        themes=[ThemeRecord(...)],
        other_coverage=[OtherCoverageRecord(...)],
        excluded=[ExcludedStoryRecord(...)],
    )

# M1 — implements nlp.embeddings.EmbeddingRepository, so its single-vector
# cache reads and writes need no run; the batch (persist_embeddings) does.
service.embed_targets(targets, repository)

# Reading: the only public connection, and SQLite refuses writes on it.
with repository.read_connection() as connection:
    connection.execute("SELECT * FROM themes WHERE trading_day = ?", (day,))

# Fixtures, backfills, and repair — deliberately unlogged, deliberately
# named as such.
repository.admin.insert_raw_items(items)
with repository.admin.connect_writable() as connection:
    ...
```

`reconcile_stories` and `reconcile_themes` report
`inserted / updated / unchanged / deleted / invalidated`. A structural
story change (a story added, removed, or re-membered) drops the day's
theme set in the same transaction, because a theme whose membership no
longer exists is not a theme that can be replayed.

## Recovery

`claim_stage_key` takes a lease. A crashed owner's key becomes claimable
again once the lease expires, and exactly one concurrent caller wins the
reclaim. `recover_expired_leases()` makes that sweep explicit and
`stage_key_state()` exposes attempts, recoveries, and the last error.

## What #82, #83, and #68 must change on rebase

These calls **will fail after rebasing onto this branch**, by design. The
old names are gone from `Phase0Repository`; they are not deprecated
aliases, because an alias would let an unlogged write keep happening
silently.

| Old call | Replace with |
|---|---|
| `repository.insert_raw_item(item)` | `repository.ingest_raw_items([item], run=run)` |
| `repository.insert_raw_items(items, source_state=state)` | `repository.ingest_raw_items(items, run=run, source_state=state)` |
| `repository.set_source_state(source, ...)` | `repository.record_source_state(source, run=run, ...)` |
| `repository.reconcile_stories(ticker=..., ...)` | `repository.reconcile_stories(run=run, ticker=..., ...)` |
| `repository.reconcile_themes(ticker=..., ...)` | `repository.reconcile_themes(run=run, ticker=..., ...)` |
| `repository.insert_story(...)` / `insert_theme(...)` | `reconcile_stories` / `reconcile_themes` (or `repository.admin.*` in a fixture) |
| `repository.update_raw_item_ticker(...)` | `repository.admin.update_raw_item_ticker(...)` — fixtures and repair only |
| `repository.clear_derived_for_day(day)` | `repository.admin.clear_derived_for_day(day)` |
| `repository.log_stage(...)` | `with repository.stage_run(...)`, which logs by itself |
| `run.record_success(n)` / `run.update_counts({...})` | nothing — counts are derived from what the operation did |
| `with repository.connect() as c:` (reads) | `with repository.read_connection() as c:` |
| `with repository.connect() as c:` (writes) | a logged entrypoint, or `repository.admin.connect_writable()` for repair |
| `repository.admin.insert_story(...)` without a version | add `pipeline_version=...`; it is required |

The shape every stage now takes:

```python
key = {"stage": "ingest", "ticker": ticker,
       "trading_day": day, "pipeline_version": version}
if not repository.claim_stage_key(**key, run_id=run_id):
    return  # someone else owns this ticker-day

with repository.stage_run(
    run_id=run_id, stage="ingest", trading_day=day,
    pipeline_version=version, ticker=ticker, stage_key=key,
) as run:
    repository.ingest_raw_items(items, run=run)
    repository.record_source_state(source, run=run, successful=True)
```

The stage key's `stage`, `ticker`, `trading_day`, `pipeline_version`, and
owning `run_id` must all match the run's; a mismatch is refused when the
run opens, not discovered later.

**Exactly one operation per stage carries `terminal=True`.** That call
commits the data, the final run log, and the stage key's completion in a
single transaction. Operations before it are logged as `degraded` and
leave the key `running`; a stage that never declares a terminal operation
ends `degraded` with the key left retryable, because it never said the
work was finished.

**Payload partition.** Every logged operation validates what you hand it
against the run, and rejects the whole batch on any mismatch:

* `ingest_raw_items` — a raw item may carry `ticker=None` ("matches no
  ticker"), which is evidence the run may keep. Any ticker it *does*
  assert, in `ticker`, `tickers`, or `candidate_tickers`, must equal the
  run's. Its trading day, derived from `published_at` falling back to
  `fetched_at`, must equal the run's day. **#82 and #83 must slice their
  fetch batches per ticker-day before calling this.**
* `reconcile_stories` — every member raw item must already sit in the
  run's ticker-day.
* `reconcile_themes` — every story named in membership, Other Coverage, or
  exclusions must match the run's ticker, day, *and* pipeline version.
* `persist_embeddings` — every source must belong to the run's partition.
* `record_source_state` — an explicit `checked_at` must fall on the run's
  trading day; omitting it means "now" and asserts no day.

Two more things to expect on rebase:

* **Migrations renumber.** #82's `004_supported_ticker_universe.sql` is
  superseded by this branch's, and #83's `005_rss_evidence_and_provenance.sql`
  should become `011_...`. The ledger keys on filename, so a renumbered
  file still applies.
* **Identity scalars are validated.** If a fixture puts a credential-shaped
  string into `provider_namespace`, `provider_item_id`, a story key, or a
  model name, it now raises `Phase0ValidationError` instead of being
  silently redacted. Fix the value; do not route around the check.
* **Stories must state a `pipeline_version`.** Migration 011 removed NULL
  as a category — a version-less story could join a v1 theme and a v2
  theme at once. Existing NULL rows upgrade to the version inferred from
  their relationships, or to `legacy-v0` when they have none; a row
  attached to two different versions fails the migration rather than being
  assigned one arbitrarily.
