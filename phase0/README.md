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

**One transaction per public write.** A batch lands whole or not at all.
Helpers that take a connection never commit; only `connect()` does.

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

**Lease ownership is checked inside the write transaction.** A logged
mutation takes the write lock first, then validates the run capability,
the stage key's owner, its expiry, and its stage/ticker/day/version
partition *on that same connection*, then mutates, then writes the run log,
then commits. A concurrent reclaimer blocks at the first step and cannot
take the key between validation and commit, so the data and the stage
key's ownership cannot end up disagreeing.

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

# Fixtures, backfills, and repair — deliberately unlogged, deliberately
# named as such.
repository.admin.insert_raw_items(items)
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

The stage key's `stage`, `ticker`, `trading_day`, and `pipeline_version`
must match the run's; a mismatch is refused when the run opens, not
discovered later. Leaving the block writes the `run_log` row and releases
the key in one transaction, whether the stage succeeded or raised.

Two more things to expect on rebase:

* **Migrations renumber.** #82's `004_supported_ticker_universe.sql` is
  superseded by this branch's, and #83's `005_rss_evidence_and_provenance.sql`
  should become `011_...`. The ledger keys on filename, so a renumbered
  file still applies.
* **Identity scalars are validated.** If a fixture puts a credential-shaped
  string into `provider_namespace`, `provider_item_id`, a story key, or a
  model name, it now raises `Phase0ValidationError` instead of being
  silently redacted. Fix the value; do not route around the check.
