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

**The public surface never hands out a connection.** Reads go through
`repository.read`, a `Phase0Reader` with explicit methods — `story(id)`,
`raw_item(id)`, `run_log_rows(...)`, `stage_key_rows(...)`,
`table_columns(table)`, `integrity_check()`, `count(table)`, and the rest
listed below. Each opens a private read-only connection, runs *one query
this module wrote*, converts the rows to plain dictionaries, and closes
the connection before returning. There is no `execute`, no `cursor`, no
`executescript`, no `commit`/`rollback`, no `set_authorizer`, and no
attribute holding a connection: the reader's entire state is one `Path`.

This replaces the earlier `read_connection()`, and the replacement is the
point rather than a tidy-up. That method returned a real
`sqlite3.Connection`, hardened with `mode=ro`, `query_only`, and a
statement authorizer — and a caller holding a connection can undo all
three:

```python
connection.set_authorizer(None)          # the authorizer was theirs to remove
connection.execute("PRAGMA query_only = OFF")
connection.execute("ATTACH DATABASE '<same file>' AS alias")
connection.execute("INSERT INTO alias.raw_items ...")
connection.commit()                      # committed, with no run log at all
```

No amount of further hardening answers that, because every protection
lives on the object the caller was given. So the object is not given.

Raw writable access exists in exactly one place,
`repository.admin.connect_writable()`, spelled that way so the call site
says what it is; it is documented as manual-repair and migration only and
is deliberately unrestricted. Migrations and every internal transaction
use the private `_connect()`.

**One transaction per public write, and success means committed.** A
batch lands whole or not at all, and a terminal operation is successful
only once `commit()` has returned. Nothing is marked succeeded in memory
on the strength of statements that have not reached the disk.

Helpers that take a connection never commit; the transaction boundaries
are `_connect`, `_logged_mutation`, and `_write_settlement`, and a test
enumerates them so a fourth cannot appear unnoticed.

**Migrations are additive and are never rewritten.** `schema_migrations`
records each applied file by name and SHA-256. Editing an already-applied
migration is refused — add a new file instead. Each migration runs inside
one `BEGIN IMMEDIATE` that also advances `user_version`, so a failure
leaves the schema, the ledger, and `user_version` untouched. The one
documented exception is a *registered historical lineage*; see below.

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

**A stage key reaches `success` exactly one way.** There is no public
`complete_stage_key`: a caller could claim a key and declare it finished
with no data and no `run_log` row behind it. Only a terminal logged
mutation transitions a key to `success`, in the transaction that commits
the data and the log. `repository.admin.complete_stage_key()` remains for
an operator releasing a key they have repaired by hand, and writes no data
and no run log — which is exactly why it cannot be mistaken for a stage
finishing.

**A run settles once, after a commit, and the outcome is immutable.** The
lifecycle is `active`, `terminal_succeeded`, `terminal_failed`,
`settlement_failed`, and `closed_without_terminal`; only repository
internals move it, and **every move happens after the transaction that
earns it has committed**. The terminal sequence is exactly:

1. open a private writable connection;
2. `BEGIN IMMEDIATE`;
3. authorize the context, owner, lease, and partition;
4. validate the payload;
5. mutate;
6. write the derived final run log;
7. transition the stage key to `success` and release the lease;
8. `commit()`;
9. *only now*, mark the context `terminal_succeeded`.

Marking it at step 7 — which a `with`-managed transaction forces, since
the context manager commits after the body — meant an injected commit
failure rolled back the data, the run log, and the key release while the
object still said the stage had succeeded.

If any step raises, including `commit()`, the data rolls back, the
failure settlement runs in its own transaction, the context becomes
`terminal_failed` only once *that* commits, and the original exception
propagates untouched. If the settlement cannot commit either, the state
is `settlement_failed` — explicitly unknown, never successful. The stage
key is left exactly as it was, so the lease expires and ordinary recovery
reclaims it, which is also what a crash at that moment would have looked
like.

Settlement is conditional as well as atomic. A commit that raises has not
necessarily failed to land — an I/O error can be reported after the
transaction is durable — so settlement reads what is actually on disk
first and refuses to overwrite a committed success with a failure it
inferred from an exception.

A second terminal operation is refused *before* any transaction opens, so
it cannot rewrite the committed success it follows. Teardown does nothing
after any settled state, which is what stops a cleanup `StageKeyError`
from replacing the exception the caller actually needs. A run that never
declares a terminal operation gets exactly one degraded/retryable outcome
at teardown.

**Anything that can reject an operation runs inside it.** All five logged
entrypoints — including argument normalization — validate on the run's own
transaction, after `_logged_mutation` has taken responsibility. A caller
who catches the rejection inside the `stage_run` block and exits normally
still ends the run `failed`; validation that ran *before* the mutation let
that caller record a success for work that was rejected.

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

    # M5 — one ticker/trading-day theme set, atomically.  The stage's last
    # operation carries terminal=True: it commits the data, the final run
    # log, and the stage key's completion together.
    repository.reconcile_themes(
        run=run, ticker="NVDA", trading_day="2026-07-23",
        pipeline_version="v1", theme_set=ThemeSetRecord(...),
        themes=[ThemeRecord(...)],
        other_coverage=[OtherCoverageRecord(...)],
        excluded=[ExcludedStoryRecord(...)],
        terminal=True,
    )

# M1 — implements nlp.embeddings.EmbeddingRepository, so its single-vector
# cache reads and writes need no run; the batch (persist_embeddings) does.
service.embed_targets(targets, repository)

# Reading: named queries returning plain dictionaries.  No connection is
# handed out, so there is nothing to take the protections off.
repository.theme_set(ticker="NVDA", trading_day=day, pipeline_version="v1")
repository.stories_for_day(day, ticker="NVDA")
repository.read.story(story_id)
repository.read.run_log_rows(run_id=run_id)

# Fixtures, backfills, and repair — deliberately unlogged, deliberately
# named as such.  The one raw connection in the codebase is here.
repository.admin.insert_raw_items(items)
with repository.admin.connect_writable() as connection:
    ...
```

### The read surface

Higher-level reads stay on the repository: `stories_for_day`,
`theme_set`, `raw_items_for_day`, `raw_item_tickers`, `source_state`,
`run_log_entries`, `latest_stage_status`, `pipeline_status`,
`stage_key_state`, `stage_keys_for_day`, `supported_tickers`,
`schema_version`, `applied_migrations`, `count`, `get_embedding`.

Row- and schema-level reads live on `repository.read`:

| Method | Returns |
|---|---|
| `raw_item(id)` / `story(id)` / `theme(id)` | one row as a `dict`, or `None` |
| `raw_item_candidates(id=None)` | candidate ticker rows |
| `raw_item_associations(id=None)` | accepted `raw_item_tickers` rows |
| `source_state_rows()` | every `source_state` row |
| `run_log_rows(run_id=…, stage=…, trading_day=…)` | whole `run_log` rows, in order |
| `stage_key_rows(stage=…, ticker=…, trading_day=…, pipeline_version=…)` | whole stage-key rows |
| `table_names()` / `schema_objects()` | the schema, for rebasing branches |
| `table_columns(t)` / `foreign_keys(t)` / `indexes(t)` | per-table introspection |
| `integrity_check()` / `schema_version()` / `count(t)` | scalars |

A read this list does not cover is a method to add here, not a reason to
reach for raw SQL — which is what keeps the read surface reviewable. Table
names are checked against the live schema before they are ever
interpolated, and no method accepts SQL.

`reconcile_stories` and `reconcile_themes` report
`inserted / updated / unchanged / deleted / invalidated`. A structural
story change (a story added, removed, or re-membered) drops the day's
theme set in the same transaction, because a theme whose membership no
longer exists is not a theme that can be replayed.

## Historical lineages

Checksum immutability answers "was this file edited after it ran". It has
nothing to say about a **fork**: two branches wrote a different
`004_supported_ticker_universe.sql`, both were applied to real databases,
and the approved implementation supersedes the other. A database on the
superseded branch is not corrupt and its checksum is not wrong — it is
evidence of a different, known transition.

So the exception is a closed registry (`phase0/lineages.py`), not a
policy. One entry today:

| | |
|---|---|
| lineage | `remote-v4-supported-ticker-universe` |
| migration | `004_supported_ticker_universe.sql` |
| historical checksum | `fd4d208833984199a0a4307b82a8693767349c89e039ec1ec4d93eada78b9eab` |
| schema fingerprint | `8c4b16cb453668d3383263eafedadf4ff1c011857e39bdf289a3ee7044587b31` |
| convergence | `migrations/compat/004_remote_v4_convergence.sql` |

**What made it incompatible.** That `004` enforced the ticker universe
with literal `IN`-lists and twelve `enforce_*` triggers, and never created
`supported_tickers`. The approved `004` creates that table and drives
every trigger from it; approved `008` and `009` read it. So the approved
chain could not run on such a database at all — and, worse, the ledger
backfill used to write the *approved* checksum onto it, silently claiming
a migration had run that never had.

**What qualifies.** All of it, or none: `user_version` is 4; the
`001`–`003` ledger rows match the approved checksums exactly; the `004`
row is absent (that lineage predates the ledger) or holds exactly the
historical checksum; `supported_tickers` does not exist; the approved
`trg_*` v4 triggers do not exist; and a SHA-256 over the stored SQL of all
twelve `enforce_*` triggers equals the pinned fingerprint. A database that
merely claims version 4 does not qualify, and neither does the right
checksum with the wrong schema, or the right schema with the wrong
checksum.

**What happens then.** A convergence migration is spliced into the
schedule at version 4 — before `005`, because that is where the
divergence is. It drops the remote lineage's triggers and performs the
approved v4 transition, producing exactly an approved v4 schema; a test
asserts that equality against a real approved v4 database rather than
trusting the file. Migrations `005`–`011` then run normally, and the
result is schema-identical to a fresh database.

**What the record says.** The ledger goes on reporting the *historical*
checksum for `004`, because that is what ran; nothing is rewritten,
deleted, or reset. The convergence gets its own ledger row under its own
name and pinned checksum. Provenance lands in `schema_lineage`, readable
via `repository.schema_lineages()`, in the same transaction as the
convergence — so a rollback takes both and no database ever claims to have
converged when it has not. That table is created for *every* database, so
a converged one stays schema-identical to a fresh one; it just has a row.

Everything else still fails exactly as before: an unregistered variant, an
edited approved migration, a forged provenance row paired with a third
checksum, and a tampered convergence file are all refused.

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
| `repository.complete_stage_key(...)` (a stage finishing) | `terminal=True` on the stage's last logged mutation |
| `repository.complete_stage_key(...)` (operator repair) | `repository.admin.complete_stage_key(...)` |
| `run.record_success(n)` / `run.update_counts({...})` | nothing — counts are derived from what the operation did |
| `with repository.connect() as c:` (reads) | a repository read method, or `repository.read.*` |
| `with repository.read_connection() as c:` (reads) | the same — no connection is handed out any more |
| `with repository.connect() as c:` (writes) | a logged entrypoint, or `repository.admin.connect_writable()` for repair |
| `repository.admin.insert_story(...)` without a version | add `pipeline_version=...`; it is required |
| a story or embedding built from a tickerless raw item | associate the item with the ticker first (`raw_item_tickers`) |

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
    # The last operation says so, and that is what completes the key.
    repository.record_source_state(
        source, run=run, successful=True, terminal=True
    )
```

There is no `repository.complete_stage_key(...)` line to add after the
block, and no version of this loop where the stage marks itself finished
separately from the transaction that wrote its data.

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
  run's ticker-day **and be explicitly associated with its ticker**.
* `reconcile_themes` — every story named in membership, Other Coverage, or
  exclusions must match the run's ticker, day, *and* pipeline version.
* `persist_embeddings` — every source must belong to the run's partition.
* `record_source_state` — an explicit `checked_at` must fall on the run's
  trading day; omitting it means "now" and asserts no day.

**`candidate_tickers` has two accepted forms and one parser.** A bare
symbol — `"NVDA"`, `" nvda "` — which records the reason
`relevance_match`; or a mapping, `{"ticker": "NVDA", "reason": "..."}`.
Anything else raises, and one bad candidate rejects the whole item.
Duplicates keep the first mention's reason and the result is sorted by
symbol, so replays compare equal. Validation and persistence read the same
normalized output: two parsers is how `candidate_tickers=["AMD"]` came to
be stored under an NVDA run, because the validator understood only the
mapping form.

**Ticker-scoped derived processing needs an explicit association.**
Tickerless raw evidence is storable — a fetcher legitimately keeps what it
could not attribute — but it may not drift into being some ticker's story
member or embedding source just because that run held the transaction. A
raw item may enter `TICKER`'s derived output only when `raw_items.ticker`
is `TICKER`, or an accepted association exists in `raw_item_tickers`, the
authoritative relationship table. A `raw_item_candidates` row is a
suggestion nothing has accepted and does not count. An item associated
with several tickers is fine and serves each of them: the rule is
membership, not exclusivity, so each ticker's output is built
independently and an AMD-only item stays out of NVDA's. A ticker-agnostic
embedding pass, if one is ever needed, needs its own stage contract rather
than borrowing a ticker-scoped run.

Two more things to expect on rebase:

* **Migrations renumber.** #82's `004_supported_ticker_universe.sql` is
  superseded by this branch's. `011_` is now occupied by
  `011_required_story_pipeline_version.sql`, so #83's
  `005_rss_evidence_and_provenance.sql` — and anything else arriving from
  those branches — must take the next free numbers from `012_` upward. The
  ledger keys on filename, so a renumbered file still applies.
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
