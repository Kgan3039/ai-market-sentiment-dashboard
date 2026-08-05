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
ticker-day, group a story from another ticker-day, or orphan an embedding.

**Stage logging cannot be switched off.** `stage_run()` writes a `run_log`
row in a `finally` block, including when the stage body raises. No public
method takes a `persist_run_log`-style flag.

**Fingerprints are not identifiers.** M2's `cluster_fingerprint`, M3's
`story_fingerprint`, and M5's `fingerprint` / `theme_key` are stored beside
the durable row ids as change-detection handles, never instead of them.

## The boundaries downstream stages use

```python
repository.migrate()

# Ingestion (#61, #62)
repository.insert_raw_items(items, source_state=state)

# M1 — implements nlp.embeddings.EmbeddingRepository
service.embed_targets(targets, repository)

# M2/M3 — one ticker/trading-day, atomically (#68)
report = repository.reconcile_stories(
    ticker="NVDA", trading_day="2026-07-23", pipeline_version="v1",
    stories=[StoryRecord(...)],
)

# M5 — one ticker/trading-day theme set, atomically
repository.reconcile_themes(
    ticker="NVDA", trading_day="2026-07-23", pipeline_version="v1",
    theme_set=ThemeSetRecord(...), themes=[ThemeRecord(...)],
    other_coverage=[OtherCoverageRecord(...)],
    excluded=[ExcludedStoryRecord(...)],
)

# Runner (#68)
with repository.stage_run(run_id=..., stage=..., trading_day=...,
                          pipeline_version=..., stage_key=key) as run:
    run.record_success(len(items))
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
