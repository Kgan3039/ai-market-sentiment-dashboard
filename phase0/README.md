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
| `yahoo.py` | issue #61's Yahoo headline fetch, settled through `stage_run` |
| `urls.py` | URL canonicalization applied before a raw item is inserted |

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

That guarantee now covers the bootstrap too. `schema_migrations` and
`schema_lineage` used to be created and *committed* before the first
migration ran, and a pre-ledger database's truthful backfill committed on
its own as well — so a brand-new database whose first migration failed
kept two tables describing an attempt that never happened, and a v2
database kept a ledger it had not earned. Both now land inside the first
migration's transaction, which restores "a failed attempt changes
nothing" without weakening what follows: every later migration still
commits on its own, so a failure at step *n* still leaves a database
honestly at step *n-1*. The one thing a rollback cannot undo is the
file's *existence* — opening a connection creates it, before any
migration logic runs — so what a failed first attempt leaves behind is an
empty database, with no schema objects and `user_version` still 0.

**`user_version` is a claim, and a claim needs evidence.** The pre-ledger
backfill records every migration numbered at or below `user_version`, on
the reasoning that a database predating the ledger has already lived that
history. Nothing bounded the number. A database stamped 999 therefore had
its whole ledger *synthesized*, left nothing pending, and reported a
successful migration over a file whose Phase 0 schema had never been
created — after which every later run agreed it was finished. Reading a
high version as "all of this ran" is backwards: the further ahead a
database is, the less of it this code can vouch for.

Three read-only questions now bound that inference, and each refusal is
typed and total — schema, rows, both bootstrap tables, and `user_version`
exactly as they were:

- **Newer than this code.** `user_version` above the newest bundled
  migration means a later release wrote it. Refused first of all, before
  lineage recognition, because the answer must not depend on how far
  recognition gets.
- **Unledgered at any version but the one recognizer this code has.** The
  rule above stops one short on its own: at *equality* the same backfill
  claims everything and leaves nothing pending, so an empty file stamped
  with the current version was accepted as silently as one stamped 999.
  Every database this code creates keeps a ledger, written in the same
  transaction as its first migration, so an unledgered database is either
  the pre-ledger v2 schema or a registered lineage — and anything claiming
  any *other* version is asserting a history with nothing behind it.

  In both directions, which took two passes to get right. `> 2` was
  refused from the start; `0 < user_version < 2` was not, and that gap had
  teeth. A database stamped `1` had migration 001 backfilled from the
  number alone, then 002 **committed on its own**, and only then did 003
  look at the schema and refuse it — the attempt failed and the database
  had still been changed, left at version 2 carrying four tables it never
  asked for. There is no recognizer for a pre-ledger v1: the one
  pre-ledger schema this code knows how to check is v2, and 003 is what
  checks it. A version this code cannot recognize is now refused rather
  than partly upgraded. Zero is the exception and barely one — it means
  nothing has run, so there is no history to be evidence *of*.

  At the watermark itself the backfill is not a guess, and 003 is the
  *first* pending migration there, so a database that merely claims to be
  v2 is refused inside the same transaction that bootstrapped the ledger
  and rolls back whole.
- **A ledger out of step with the version.** The two are written together,
  so once a database keeps its own history they can only disagree if
  something outside this module moved one. A version *behind* its ledger
  used to be accepted outright, leaving the database reporting a version
  it had long since passed; a version *ahead* re-ran applied migrations
  and died on whatever they collided with, which is an untyped SQLite
  error rather than a refusal. This one runs after the checksum rule, not
  before it: a tampered ledger is usually also out of step, and "this
  migration was edited" says more than "these two numbers differ".

The convergence path has always asked the third question of the
historical lineages. This is the ordinary path catching up to it.

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

Migration 009 restores that projection by *rebuilding* the table rather
than repositioning it. `position` is `UNIQUE` and SQLite enforces it per
row, so moving TSLA to position 1 while NVDA still holds position 1
aborts the migration — and before 009 the table is not yet sealed, so any
ordering is a legitimate prior state. 119 of the 120 orderings collided.
No write order avoids that in general (any permutation with a cycle has a
row that must pass through an occupied position), so 009 empties the
table first and leaves nothing to collide with. Nothing references
`supported_tickers`, so emptying it costs nothing, and it converges from
any prior state: reordered, partial, sparse, duplicated, or carrying
unsupported rows.

**Stage logging cannot be switched off.** `stage_run()` writes a `run_log`
row in a `finally` block, including when the stage body raises. No public
method takes a `persist_run_log`-style flag.

**A run identity names exactly one partition, permanently.** `run_log` is
an upsert keyed on `(run_id, stage)`, and its conflict branch rewrote
`ticker` while leaving `trading_day` and `pipeline_version` alone. Both
halves were wrong in their own direction: reusing an identity under a
second ticker silently relabelled the first run's row, and reusing it
under a second day or pipeline version logged the new run under the old
one's partition. Either way one row described work no single run did, and
one run identity could accumulate two stage keys.

The partition is now settled by whoever writes first and is immutable
after that — `RUN_IDENTITY_COLUMNS` is `ticker`, `trading_day`,
`pipeline_version` — and reuse under a different one fails closed with a
`Phase0RunContextError` before anything is written. The check lives in
`_write_run_log`, the single place the row is written, so it holds for
`stage_run` and the admin log path alike; because settlement runs inside
the run's own transaction, a rejection rolls back the data too. Retrying
or replaying the same identity in the *same* partition is untouched —
that is the documented lifecycle, and it is the only thing this identity
is for.

**Pipeline mutations carry their run.** `ingest_raw_items`,
`reconcile_stories`, `reconcile_themes`, `persist_embeddings`, and
`record_source_state` all require the `StageRunContext` that `stage_run()`
yields, and write their `run_log` row in the same transaction as the data.
A missing, completed, foreign, or lease-expired handle raises before
anything is written. The unlogged row helpers live on `repository.admin`,
where they are labelled administrative rather than presented as pipeline
entrypoints.

**An outcome is stated once.** `record_source_state` takes both
`successful` and `status`, and the two used to be resolved separately:
the stored row followed `status` when it was given, while the run's
counters followed `successful` regardless. So a feed's record and the run
that wrote it could say opposite things — and not only when a caller
contradicted itself, because `status="unknown"` is perfectly valid, does
not count as a successful fetch, and still logged a success under the
default `successful=True`.

`status` is the richer statement — `partial`, `empty`, and `unknown` have
no boolean spelling — so when given it decides, and `successful` becomes a
claim about the same thing that must agree; stating both and disagreeing
is refused before anything is written. Whichever way it resolves, the
stored status, the `last_success_at` stamp, `consecutive_failures`, and
the run's own counters all derive from that one answer, using the same
succeeded set (`success`, `partial`, `empty`) the schema itself uses.
Omitting both means success — and that default lives in
`validate_source_state`, not at any call site. It briefly did live at one:
`record_source_state` substituted the default itself while the resolver
still collapsed `None` into `False`, so a payload that stated nothing
resolved to *success* through that one entrypoint and *failed* through
every other. `None` is "not stated", which is not the claim an explicit
`False` makes, and the resolver now keeps them apart.

There are four public ways a source state reaches the database —
`record_source_state`, `admin.set_source_state`, and the `source_state=`
argument of `ingest_raw_items` and `admin.insert_raw_items`, the last two
handing a raw payload straight through. All four resolve through
`validate_source_state`, so they agree by construction rather than by
four matching copies of the same rule; the truth table is asserted
against every one of them and against the resolver itself, which is
public precisely so a caller can ask what the repository will do before
committing.

The same shape appeared once more, in `_write_run_log`: an unstated status
*means* degraded when there are errors, so a caller stating `success`
alongside errors was contradicting the module's own rule. That pairing is
now refused too; `degraded` is the word this vocabulary has for "worked,
with problems".

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

**A stage key names a ticker, so a run holding one is never ticker-less.**
Omitting `ticker` on a `stage_run` that supplies a `stage_key` *adopts*
the key's ticker; it never means "any". The other reading is what the
code used to do — skip the ticker comparison when the run named none —
and it was a hole, because a ticker-less run's partition checks pass for
every ticker. An NVDA lease could open a ticker-less run and that run
could do AMD work, the key having authorized it by saying nothing.
Supplying a ticker that disagrees with the key is still refused, and
every other axis is compared as before.

**A lease is renewable or reclaimable, never both.** `heartbeat_stage_key`
extends only a lease that is still live: its condition is the exact
complement of the one `claim_stage_key` reclaims under (`lease_expires_at
IS NULL OR lease_expires_at <= now`). Expiry is the moment ownership
ends, not a suggestion the previous owner may decline — an owner able to
push a lapsed lease forward would keep working on a partition another
worker was already entitled to take, and both would hold it at once.

**A key that can be claimed is a key that can be settled.**
`claim_stage_key` normalized only `ticker` and `trading_day` and passed
`stage`, `pipeline_version`, and `run_id` through untouched, while
`stage_run` requires all five to be non-blank and stripped. So a claim
with `run_id=""` was written as `running` and then could be settled by
nobody: the identity holding the lease was one `stage_run` refuses, and
the partition stayed locked until the lease expired. A padded `stage` did
the same thing more quietly, storing the key under a name no normalized
lookup would match.

`_stage_key_identity` is now the one definition of those five fields,
built from the same helpers `stage_run` uses, and `claim_stage_key`,
`heartbeat_stage_key`, and `admin.complete_stage_key` all go through it —
so the lifecycle has one identity contract rather than three. Validation
happens before any write, so a rejected claim leaves no row and does not
touch a key already there. Ticker normalization, lease semantics, and
reclaim semantics are unchanged.

**Ticker membership is membership, not exclusivity.** One article can be
about two companies. `raw_item_tickers` is the authoritative table for
which tickers claim a raw item, and `raw_items.ticker` is the primary
attribution stored *beside* it — not a veto over the rest. So an
AMD-primary article carrying an accepted NVDA association is NVDA's
evidence as well as AMD's, and each ticker's derived output is built
independently from it. `_assert_raw_item_association` is the single rule;
paths that carried a second, stricter test against `raw_items.ticker`
were refusing evidence the association table had already accepted, and
disagreeing with `raw_items_for_day`, which had always read it correctly.

None of that loosens anything: an item with no association for this
ticker is refused, unattributed evidence is refused, a `raw_item_candidates`
row is a suggestion and still does not count, and day and pipeline-version
checks are untouched. A story's or a theme's `ticker` **is** exclusive —
those rows belong to exactly one partition — and is still compared as
such.

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

**A replay is "unchanged" only when a settlement would write exactly what
is already stored.** Reconciliation reports each story and theme as
inserted, updated, or unchanged, and `unchanged` skips the update path
entirely — so an equality test that is narrower than the write is not a
performance detail, it is a way to persist stale derived state forever.
Nothing downstream notices, because nothing downstream looks.

The contract is therefore stated once, as the columns each path owns:

* **Stories** — every column in `STORY_RECONCILED_COLUMNS` (which
  includes the `embedding` blob), plus `canonical_item_id`, plus whether
  the row is live (`invalidated_at IS NULL`, which the update path
  clears), plus the *full payload* of all three child relations:
  `story_members` down to each member's position, outlet, url,
  canonical url, match reason, and quarantine flag; every
  `story_provider_conflicts` row including its `item_ids` and `fields`;
  every `story_semantic_merges` row including its similarity and reason.
  Child *identities* are not enough — a conflict whose fields changed is
  a changed story.
* **Themes** — every column in `THEME_RECONCILED_COLUMNS`, which includes
  the `centroid` blob and all three salience components. These are
  derived outputs of the theme stage: a rerun that recomputes them has
  produced a different theme even when its label and membership are
  word-for-word identical. Plus both membership relations.

Everything else on those tables is named as not-owned and checked against
the table itself by
`test_every_story_column_is_owned_or_deliberately_exempt` and its theme
twin: the partition identity (which is what *pairs* an incoming row with
a stored one, not something to compare), the row id, and `updated_at`.

Two properties hold this in place rather than good intentions. First, the
same column mapping builds the INSERT, the UPDATE, and the comparison, so
a column cannot be written by a settlement and be invisible to the next
one. Second, equality is canonical on both sides — members by raw item,
conflicts by provider identity, merges by story-key pair, memberships and
the denormalized `themes.citations` list sorted — so equivalent inputs
compare equal however the stage happened to order them.

**One story, one accounting bucket — and "theme" counts against itself.**
Each canonical story in a partition is placed exactly once: in one theme,
in Other Coverage, or in exclusions. The rules enforcing that are
pairwise, and five of the six pairs were written — coverage against
themes, themes against coverage, exclusions against themes, exclusions
against coverage, coverage against exclusions — each on `INSERT` and on
`UPDATE`. The pair nobody wrote is the one where both sides are the same
table: a story in **two themes**.

`_prepare_coverage` flattens every theme's members into a *set* to check
coverage and exclusions against, so the one step that looks at every
member is the step that discards how many themes claimed it. Nothing
downstream caught it either. A theme's citations must belong to its member
stories and no two themes in a partition may cite the same raw item,
which blocks the obvious attempt — two themes sharing a one-item story
would have to share its citation. Give that story a second member and the
collision goes away: each theme cites a different item, both claim the
story, every remaining rule is satisfied, and the day's theme cards
double-count a story while its Other Coverage arithmetic stops adding up.

`reconcile_themes` now checks the incoming themes against each other
before any write, so a batch that overlaps anywhere writes nothing
anywhere, and the error names each shared story with the themes claiming
it. A repeat *inside* one record is a different thing and keeps its
existing treatment: `story_ids` and `citation_item_ids` both run through
`dict.fromkeys`, so duplicates are canonicalized rather than refused —
one owner named twice is not two owners.

Migration **014** puts the same rule in the database, on `INSERT` and on
`UPDATE`. The `UPDATE` guard excludes the row it is judging by its own
`(theme_id, story_id)`, because `BEFORE UPDATE` still sees the old row and
a membership merely moving between themes would otherwise collide with
itself. The invariant is stated per partition, matching the M5 triggers —
and that is the whole of it, because 011 made `stories.pipeline_version`
required and the partition triggers demand a member share its theme's
ticker, day, and version exactly. A story therefore belongs to one
partition, and the only themes that can claim it are in that partition.

The other two buckets never had this gap, for a structural reason worth
recording: `theme_sets` is unique per partition and both coverage tables
are keyed on `(theme_set_id, story_id)`, so their primary keys already say
"once". `theme_stories` is keyed on `(theme_id, story_id)`, and a
partition holds *many* themes — so the same shape of key permits exactly
the state 014 forbids.

A v13 database already carrying such a pair **refuses to upgrade** and
stays at 13, whole. Choosing which theme owns the story is a content
decision, and 011 set the policy when it could not infer a legacy
`pipeline_version`: abort, roll back, and leave it to an operator. Once
resolved, the upgrade goes through.

**A reallocation is one move, not two halves.** Because a story and a
citation each belong to one theme, settling theme by theme asks the
database to accept a *partial* rearrangement: the theme gaining a story
inserts while the theme losing it still holds it. That is not an invalid
outcome, it is a valid outcome half-applied — and a trigger sees only the
row in front of it, so it rightly refuses. One-way moves worked when the
donor happened to come first in the caller's list and failed when it came
second, so the same reallocation succeeded or failed on input order
alone; a swap had no ordering that worked at all.

The fix is in the write order, not in the rules. `reconcile_themes` makes
three passes: classify every incoming theme against what is stored,
**then** release the stories and citations of every theme that is
changing, **then** write the replacements. Citations are released before
memberships, for the same reason `_delete_themes` releases them in that
order — a citation is guarded against losing the member story underneath
it. All three passes sit in the transaction that already covered the
whole reconciliation, so the window where relations are released and not
yet rewritten cannot outlive a failure.

A theme whose stored relations already are the answer is left alone
rather than rebuilt, and nothing can be waiting on it: a final state
where two themes want the same story is refused before any of this, by
the check above. `_update_reconciled_theme` no longer clears anything
itself — reading as self-contained is exactly what made the interleaving
easy to write.

**A report accounts for everything the operation writes, not only what it
can name by id.** `ReconciliationReport`'s tuples hold row ids, so they
can only describe rows keyed by a fingerprint: stories, or themes.
`reconcile_themes` also owns three outputs with no theme id to report them
under — the `theme_sets` metadata row, the day's Other Coverage, and the
day's exclusions — and those used to be rewritten unconditionally and
reported not at all. A run could replace a whole day's coverage and the
report would say, truthfully about themes and falsely about the day, that
nothing had changed.

They are now settled the same way everything else is: compared first,
written only if different, and named in `changed_outputs` (from
`AUXILIARY_OUTPUTS`: `theme_set`, `other_coverage`, `excluded`) when they
are. `changed` counts them, so it means what it says in both directions —
true whenever any owned table would come out different, and false *only*
when the reconciliation wrote nothing at all. `theme_sets.updated_at` is
part of that: it is bookkeeping about the write, so it moves only when one
of the owned columns does, and two databases fed identical input stay
identical.

Both auxiliary lists are compared as *sets of rows*, keyed by story,
because neither table has an inherent row order. `position` is compared as
a value — it is a persisted column that ranking reads back — but the
sequence a caller happened to list two entries in is not a difference.
`THEME_SET_RECONCILED_COLUMNS` is checked against `theme_sets` itself by
`test_every_theme_set_column_is_owned_or_deliberately_exempt`, the same
way the story and theme lists are.

**Zero is a position, not a silence.** `StoryMemberRecord.position` and
`OtherCoverageRecord.position` both defaulted to `0`, and both call sites
read them as `value if value else index`. Zero is the *first* position, so
a field defaulting to it had no way to say "I did not state this", and an
explicit `0` on anything but the first element was replaced by that
element's index in the list. `[1, 5, 0]` is where it shows: the member the
caller put first came back second.

Both fields are now unset by default and both call sites ask `is None`.
Unset means "number these in the order given" — per element, not as a mode
the whole list is in — and anything stated is stored exactly. Making the
default `None` rather than `0` is what made the two distinguishable at
all; a caller who omits the field lands exactly where it used to, so the
only behaviour that changes is the one that was wrong.

The defect was self-concealing, which is worth recording: `position` is
part of story equality on *both* sides, and the write and the comparison
mangled an explicit zero identically, so a replay agreed with itself.
Only against a stored zero do the two disagree — and then the identical
input reports *changed*, forever.

That shape — `x if x else fallback` on an optional field whose falsy value
is real data — is now refused across the package by
`test_no_optional_field_reads_a_valid_falsy_value_as_absence`, so the next
optional field cannot quietly acquire it. Those two sites were the only
ones; every other optional value already goes through `_optional_text`,
`_optional_float`, or an explicit `is None`.

`reconcile_stories` has no equivalent: every table it owns is a story or a
story's child, and all of them already reach the per-story signature.
`test_reconcile_stories_reports_every_write_it_makes` pins that rather
than assuming it.

**An embedding source identity means what the schema says it means — and
only a run may use the partition-scoped half of it.** Migration 007's
ownership triggers define the identities the table accepts: a raw item by
`id`; a story by `id` *or* `cluster_fingerprint`; a theme by `id` *or*
`fingerprint` *or* `theme_key`. But `embeddings` is globally unique on
`(source_kind, source_id)`, while those fingerprints and keys are unique
only *within* one ticker/trading-day/pipeline-version. So the two halves
of that set are not equally safe, and the two entrypoints differ:

* **`persist_embeddings`** — run-scoped — resolves the whole set. It can
  afford to: it resolves the identity to a row, checks that row is the
  run's own ticker, day, and pipeline version, and refuses an identity
  that resolves to *more than one* row, since there is then no partition
  the vector could honestly belong to. The comparisons mirror the
  triggers' `CAST(id AS TEXT) = …` rather than leaning on SQLite's
  integer affinity, so `'01'` does not become story 1 here and get
  refused by the trigger a moment later.
* **`get_embedding` / `upsert_embedding` / `delete_embedding`** — M1's
  single-vector cache protocol — take **durable row ids only**. They
  carry no run, so they have no partition to judge a partition-scoped
  handle against; and "unambiguous right now" would not be enough anyway,
  because an identity that names one story today names two the moment
  another ticker-day clusters to the same fingerprint, with no write to
  this table in between to notice. A fingerprint reaching them is a
  caller error and is named as one.

This is the fingerprint rule above, applied where it had not been. Left
unapplied it was not theoretical: two tickers clustering to the same
fingerprint on the same day is ordinary, and the second `upsert_embedding`
silently replaced the first — after which *both* partitions read back one
ticker's vector for the other's text.

The M1 protocol is unaffected. `EmbeddingRepository` types `source_id` as
text and says nothing about which text, so this narrows which values are
cache keys, not the shape of any call.

**A vector dies with its last owner, not with its first.** Migration
007's cleanup triggers used to delete by every identity form
unconditionally, so dropping *any* story deleted embeddings keyed on its
`cluster_fingerprint` — including one another partition's still-live
story owned. The ordering that made this reachable is the ordinary one:
partition A caches a vector under a handle while it is the only owner,
partition B later produces its own row bearing the same handle, and B's
deletion takes A's vector. Nothing B did was invalid, and A was never
told.

The durable id needs no guard — it is globally unique, so nothing else
can be addressed by it. Every handle-shaped identity is now deleted only
once no live row still carries that handle, which the `AFTER DELETE`
triggers can ask directly: the row being deleted is already gone, so
`NOT EXISTS (SELECT 1 FROM stories WHERE cluster_fingerprint = …)` means
exactly "is anyone left". The orphan half of the contract still holds —
deleting the *last* owner removes the row — and the repository-side
`upsert_embedding` / `delete_embedding` are untouched, so a partition
still cannot reach another's cache entry through the API.

This was previously recorded here as an accepted residual, on the
reasoning that closing it meant rewriting a released migration. That
reasoning was wrong about the premise at the time: migration 007 had not
yet left this branch, so it was corrected in place before release, like
010 and 011.

**A rename is the same event as a deletion.** That guard fires on
`DELETE`, and an owner can reach the identical end state without dying:
it stays alive and rewrites the handle itself. `reconcile_themes` does
exactly this through the public API — it matches a theme by `fingerprint`
and writes `theme_key` as an owned column — so a re-run that assigns a
new key leaves the old key belonging to nobody while its vector stays
cached. Nothing reads that row afterwards, because every read resolves a
handle through a live parent, and nothing ever removed it. A handle later
reused by an unrelated partition would then find a vector encoding text
it never saw.

Migration **012** asks the same question on `UPDATE` of
`stories.cluster_fingerprint`, `themes.fingerprint`, and
`themes.theme_key`: who is left. A vector is collected only when no live
row would still be allowed to own the old handle — which is the
*ownership* predicate from `trg_embedding_owner_insert`, not one column.
That is what keeps a durable-id vector out of it: if `source_id` matches
some live row's id, that row is an owner, so a fingerprint moving out
from under a string that happens to be a row id changes nothing. Nothing
moves a vector to the new handle either; an embedding names the text that
produced it, so a renamed owner is a re-encode the repository performs
explicitly, never a rename the schema performs silently.

012 is additive rather than a correction to 007, because 007 is no longer
correctable: the branch carrying migrations 005–012 is now published as
`origin/agent/phase0-i1-persistence`, so every migration on it records a
checksum that a database somewhere may already hold. See the release-status
note below.

**And the delete path had to be brought up to the same predicate.** 012
asked the ownership question; 007's cleanup, which 012 did not touch,
still asked two narrower ones. The handle branch compared a single
column, so an alias colliding with another live row's *durable id* — or,
for themes, with its other alias — looked orphaned. The durable-id branch
compared nothing at all, so deleting story 1 took the vector cached under
`'1'` even while a live story's `cluster_fingerprint` was the string
`'1'`. Both are states the insert trigger would have called ownership a
moment earlier, and neither is exotic: handles are unique only within a
partition, so two of them repeating is ordinary, and a digest that
happens to spell a small integer is a coincidence nothing forbids.

Migration **013** replaces both cleanup triggers so that every verb asks
one question — *would any live row still be allowed to own this
`source_id`, through any accepted form?* — with the `NOT EXISTS`
correlated to `embeddings.source_id` rather than to one `OLD` column,
which is what lets the durable-id and handle branches share it. `raw_items`
is deliberately left alone: its ownership predicate has exactly one form,
the durable id, which is the one its cleanup already uses, and
`AUTOINCREMENT` means a deleted id is never issued again.

Replacing a trigger is additive. 007 and 012 keep their bytes and their
checksums; 013 drops what 007 created and creates the replacement, which
is a schema change like any other and settles in its own transaction. An
existing v12 database gains the fix on upgrade, because a trigger is
schema and the old one is sitting inside it.

What none of this changes is *who may name an owner*. The single-vector
cache still takes a durable id and means the row with that id, so another
row's alias spelling the same digits does not make it ambiguous there; a
handle is still refused there outright; and the run-scoped batch, which
resolves both forms, still refuses an identity that names two rows.

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
`inserted / updated / unchanged / deleted / invalidated`, plus
`changed_outputs` for what `reconcile_themes` owns outside the theme rows
themselves. A structural story change (a story added, removed, or
re-membered) drops the day's theme set in the same transaction, because a
theme whose membership no longer exists is not a theme that can be
replayed.

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
| historical version | `4` |
| application fingerprint | `ea38ea10cf7f0657f45f68fec86ffc18bd86a3f087d61924e67b235d8db7df5d` |
| `schema_migrations` fingerprint | `80eb31deee50ce560b411c6020c755dd3c3764dc4fc7b99a470f57d7f7788df4` |
| `schema_lineage` fingerprint | `1b5c479ea1ac90e14017ade4fe46f0df498b9d1097c743fd95b5fad7f9c6118c` |
| signature objects | the twelve `enforce_*` ticker triggers |
| convergence | `migrations/compat/004_remote_v4_convergence.sql` at version `4`, checksum `eb0609c3…` |

**What made it incompatible.** That `004` enforced the ticker universe
with literal `IN`-lists and twelve `enforce_*` triggers, and never created
`supported_tickers`. The approved `004` creates that table and drives
every trigger from it; approved `008` and `009` read it. So the approved
chain could not run on such a database at all — and, worse, the ledger
backfill used to write the *approved* checksum onto it, silently claiming
a migration had run that never had.

**What qualifies.** All of it, or none. The lineage's identity is five
things that must agree:

1. `user_version` is 4;
2. the **application fingerprint** matches — a SHA-256 over every object in
   `sqlite_master` with its stored SQL, plus every table's full
   `table_info` (name, declared type, nullability, default, primary-key
   position), `foreign_key_list`, and every index with its columns. A
   missing table, an extra table, an added or dropped index, a changed
   column, or a reworded trigger all fail it;
3. the **migration-metadata fingerprint** matches. `schema_migrations` and
   `schema_lineage` are excluded from the application fingerprint because
   they describe migration state rather than schema — but excluded is not
   unchecked. Each is digested the same way and pinned, so a ledger with
   an extra column, a renamed one, a widened type, a dropped `NOT NULL`, a
   different primary key, an added default, or an unexpected index is a
   ledger this code did not create, and nothing it reports can be taken at
   face value;
4. the **ledger contents** match, as whole rows. Either the ledger is
   absent — the genuine database predates it — or it holds exactly this
   lineage's history: `001`–`003` at the approved `(version, checksum)`
   and `004` at the historical one, no more and no less. A checksum is
   half a row: `004` recorded at version 99 with the right checksum
   describes a migration that does not exist, and is refused;
5. the historical **checksum** is exactly `fd4d2088…`.

**Claiming versus qualifying.** A database carrying any of the twelve
`enforce_*` triggers is *claiming* this lineage — nothing on the approved
lineage has ever created one. Claiming is not qualifying: such a database
must then match all five, or be **refused** before a single statement
runs. It may not fall through to the ordinary path, because that path
assumes a pre-ledger database ran the approved migrations up to its
`user_version`, and a fork is exactly what breaks that assumption.

**Two accepted states, and no third.**

| | |
|---|---|
| `pre-convergence` | `user_version` 4, the exact historical application schema, the exact metadata-table shapes, the exact historical ledger (or none at all), and no provenance yet |
| `post-convergence` | the historical row at `(4, fd4d2088…)`, the convergence row at `(4, eb0609c3…)`, **every other ledger row** whole and expected with none missing and none extra, `user_version` equal to the version the ledger reaches, the convergence's effects live in the schema, the application schema **exactly the one the approved migrations build at that version**, and provenance whose every field is the registry's |

The converged ledger is exact in both directions. Every row present must
be one a settlement writes — the approved files at their own
`(version, checksum)`, except `004`, which keeps reporting the historical
checksum because that is what ran, plus the convergence at its pinned
version — and every row that should be present at this `user_version` must
be. A row naming a migration that does not exist, `011` recorded at
version 99, a deleted `010`, or a `user_version` out of step with the
rows: none of these is a ledger this path wrote.

**The schema evidence is built, not pinned.** The decisive post-convergence
condition is that the live application schema equals the schema the
approved migrations produce at this database's `user_version` — computed by
running those files into an empty in-memory database and fingerprinting the
result, never stored as a constant. A pin would have to be re-derived for
every new migration, and a forgotten one fails *open*, going on comparing
against a schema the code no longer builds. This makes "a converged
database is identical to a fresh one" a rule the path enforces rather than
only something the tests assert. Before it existed, asking of the schema
only that it was *not* the fork's let three hand-written rows excuse the
historical checksum over an unknown fork, after which the remaining seven
approved migrations ran against it.

Anything that is neither fails closed. A historical schema carrying a
convergence row, a converged schema with a partial ledger, provenance over
an invalid ledger, a valid ledger under a tampered metadata table, partial
convergence metadata — none of these is a state, so none of them is
accepted.

**Provenance corroborates; the schema and the ledger decide.** The order is
the authority order: the live schema and the ledger are validated first and
on their own, and only then is `schema_lineage` consulted — to confirm what
they already say, or to contradict it. A provenance row can never supply
something they do not, and never upgrades an invalid ledger into a
recognized history.

One internal-consistency rule ties the settlement together: the historical
row, the convergence row, and the provenance row are written in the same
transaction and carry one timestamp. A ledger assembled from parts
afterwards does not, which is what refuses a fresh database dressed up with
a swapped `004` checksum, a hand-written convergence row, and a copied
lineage row.

**The compatibility settlement is one transaction.** The ordinary path
commits one migration at a time and still does; that is right for a
database taking steps every database takes, where a failure at step *n*
leaves it honestly at step *n-1*. None of that holds for a fork. A
half-converted database is at no version at all: bootstrap tables it did
not ask for, a ledger describing a history it has not lived, and a schema
partway between two branches.

So the conversion runs as a single settlement:

1. recognize the lineage **read-only** — nothing is created or backfilled
   by asking;
2. read and checksum the convergence file, before any transaction opens;
3. `BEGIN IMMEDIATE`;
4. re-validate the lineage inside the write lock;
5. create the ledger and provenance tables;
6. backfill the ledger **truthfully** — the historical checksum for the
   historical file, never the approved one;
7. run the convergence, then every remaining approved migration;
8. set `user_version`;
9. write provenance;
10. validate the result — every migration applied, the historical checksum
    intact, the convergence recorded, `supported_tickers` present, the
    `enforce_*` triggers gone, the version correct — and then the closing
    condition: the settlement may only commit a database that this code
    would itself recognize as `post-convergence`, exact ledger, exact
    schema, exact provenance and all;
11. commit.

Any failure rolls the whole thing back: the original tables and data, the
absence of `schema_migrations` and `schema_lineage`, and `user_version`,
all exactly as they were. Step 10 exists because marking a migration
applied is not the same as its schema existing — a settlement that only
claims to have converged is refused rather than committed — and its closing
condition is what ties this path to the next `migrate()`: whatever is
committed here must be exactly what recognition accepts there, so the
settlement can never produce a state the very next run has to refuse.
Provenance is written at step 9, before that check, for the same reason.

**What the record says.** The ledger goes on reporting the *historical*
checksum for `004`, because that is what ran; nothing is rewritten,
deleted, or reset. The convergence gets its own ledger row under its own
name and pinned checksum. `schema_lineage` is created for *every*
database, so a converged one stays schema-identical to a fresh one and
just has a row; read it with `repository.schema_lineages()`.

Everything else still fails exactly as before: an unregistered variant, an
edited approved migration, a forged provenance row, a tampered convergence
file, a partial ledger, and a database that merely resembles the lineage
are all refused — and refused before anything is written.

**What none of this can promise.** A converged database is *required* to be
schema-identical to a fresh one — that is what converging means — so no
schema evidence can separate the two, and anyone able to write to the
database file can also write the ledger and provenance rows a settlement
would have written. What the rules above guarantee is that doing so buys
nothing: the historical checksum is excused only for a database that is
already, in every respect, a valid database at head — exact schema, exact
ledger, every approved migration applied. A forgery cannot carry an
unknown schema, a stale one, a partial ledger, or a migration that never
ran, which is the whole of what the excusal was ever able to be abused
for. Anyone with write access to the file could always corrupt it
directly; the compatibility path grants no capability beyond that.

## The SQLite version this schema is written against

**The floor is SQLite 3.38**, declared as
`phase0.schema.MINIMUM_SQLITE_VERSION`. It comes from what the migrations
actually use — `ON CONFLICT ... DO UPDATE` (3.24) and the
`json_valid`/`json_type` CHECK constraints, JSON1 having become a default
build option in 3.38 — and nothing here needs anything newer. That is
worth keeping deliberately: several current distributions ship Python
against a SQLite in the 3.4x range.

The trap the floor exists to catch is `RAISE()`. Its message had to be a
string *literal* until 3.47 (October 2024), so
`RAISE(ABORT, 'a ' || 'b')` is a syntax error on anything older. The
consequence is far worse than one broken statement: SQLite parses the
whole schema the first time a connection touches it, so a single trigger
it cannot parse makes the entire database **unopenable**. Measured on
3.43.2, `SELECT count(*) FROM schema_migrations` returns
`malformed database schema (trg_story_partition_locked)` — and so does
`DROP TRIGGER`, which means such a database cannot even be repaired from
the runtime that cannot open it.

Migrations 010 and 011 originally carried three concatenated messages and
were corrected in place, before release, to single literals with the same
text. That is a pre-release correction, not a rewrite of released history:
010 and 011 have never existed outside this local branch — the published
I1 head carries migrations 001–004 only — so no database anywhere records
their old checksums. Once this branch lands they become immutable like
every other migration, and the only remaining route would be an additive
one. (An additive migration would only half-help anyway: it can re-create
a *persisted* trigger, but it cannot help fresh creation, because 010 must
parse before 012 exists.)

**Which migrations are actually released.** The rule is the same one
every time and it is worth stating once: a migration is released when a
database somewhere could have recorded its checksum, which means it
exists in an authoritative published lineage — not merely in an approved
local branch. Migrations **001–004** were released first: they are on
`origin/agent/phase0-i1-persistence`, and 004's checksum is pinned in
`lineages.py` as the remote-v4 lineage's migration. Nothing here has ever
edited them.

**005–011 have since joined them.** They were unreleased while this work
was in progress — each existed on exactly one local commit and on no
remote — which is why 007, 009, 010, and 011 could be corrected in place.
That window is closed: `origin/agent/phase0-i1-persistence` now carries
this branch, so every migration on it is published and immutable on the
same rule that has always protected 001–004. **Every further correction
to 001–011 is additive**, which is why the handle-move cleanup above is
migration 012 rather than an edit to 007. 012 itself is unreleased only
until the next push, so it is written as a file that never needs
correcting rather than one that can be corrected.

Applying the rule consistently matters more than which answer it gives.
The same branch that made 001–004 released is the one that has now
released 005–011; treating a push as release when it suited an earlier
conclusion and not when it constrains a later one would make the rule
decorative. What "additive" costs is visible in 010's case above: an
additive migration can re-create a *persisted* trigger but cannot help
fresh creation, because the earlier file must parse before the later one
exists. 012 has no such problem — a trigger added later governs every
update after it, on fresh and existing databases alike.

Three tests hold the line, and they are complementary:

- `test_no_migration_uses_a_non_literal_raise_message` — statically, in
  every migration including the compatibility one.
- `test_every_migration_parses_on_an_older_sqlite` — applies the whole
  sequence through an older `sqlite3` CLI than the one this process is
  linked against, then reads the resulting schema back, because a
  compatibility claim checked only by the library making the claim is not
  a check. It skips when the host has no older CLI to offer.
- `test_the_declared_sqlite_floor_is_the_one_the_schema_needs` — pins the
  floor and refuses the keywords that would raise it.

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
* `record_source_state` **and `ingest_raw_items(source_state=…)`** — a
  stated `checked_at` must fall on the run's trading day. Only
  `record_source_state` enforced this; the `source_state=` argument
  reaches the same write and did not, so one run could stamp another
  day's fetch as its own while its run log recorded the work as today's.
  The check now lives in `_set_source_state`, which both go through, so
  neither can drift from the other again. Omitting it still means "now"
  and asserts no day — though only `record_source_state` can omit it, as
  the mapping form requires `checked_at`. The comparison is on the
  *normalized* instant, so a stated offset that moves the day moves the
  answer with it. `admin.insert_raw_items` writes without a run and so
  has no day to check against; that is unchanged.

**`candidate_tickers` has two accepted forms and one parser.** A bare
symbol — `"NVDA"`, `" nvda "` — which records the reason
`relevance_match`; or a mapping, `{"ticker": "NVDA", "reason": "..."}`.
Anything else raises, and one bad candidate rejects the whole item.
Duplicates keep the first mention's reason and the result is sorted by
symbol, so replays compare equal. Validation and persistence read the same
normalized output: two parsers is how `candidate_tickers=["AMD"]` came to
be stored under an NVDA run, because the validator understood only the
mapping form.

**A duplicate names stored evidence, and that evidence has its own day.**
`(source, canonical_url)` is unique, so re-offering an item does not
create a row — it resolves to one that already exists, and the ingestion
path then writes *that* row's ticker associations and candidate reasons.
The partition check read only the incoming payload's timestamps, so a run
for day D could resolve to a row belonging to D-1 and mutate it while the
run log recorded the work as D's. It now derives the stored row's
effective day too — in SQL, with the same `COALESCE(published_at,
fetched_at)` the reader uses, so "which day is this evidence on" has one
definition rather than two that can drift — and refuses the whole batch
before anything is written when it is not the run's day. The stored
timestamp is never rewritten to make the partition match: the write is
refused, the evidence is left alone.

Inside the run's own day the duplicate path is unchanged, associations
and candidate updates included, so an idempotent replay still costs
nothing and still returns the existing id.

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

## How #82 (Yahoo, issue #61) was ported onto this contract

`yahoo.py` is the first consumer rebased onto the final surface, so it is
also the worked example of the table above. Everything it used is gone:
`insert_raw_items`, `set_source_state`, `log_stage`, and the public
`connect()` its tests read through.

**One response fans out into several partitions.** A single request for a
single symbol routinely returns articles published across several days,
and a run may speak for exactly one ticker-day. So the fetcher groups the
normalized batch by the *same* day derivation the repository uses —
`published_at` falling back to `fetched_at`, spelled once in
`yahoo.effective_day` — and opens one run per group. If those two
derivations ever drift, batches this module considers same-day get
rejected on arrival, which is the failure mode to look for.

**The run identity carries the partition.** `run_log` is
`UNIQUE(run_id, stage)` and recording one identity against a second
partition is refused, not merged — so a base id shared across the days of
one response would fail to persist the second day at all.
`yahoo.partition_run_id` spells the identity `<base>:<ticker>:<day>`.

**The source-state checkpoint rides on the fetch-day partition.** Source
state is keyed by feed and answers "when did we last check this feed",
which is the fetch day, not the day an article was published — and a
stated `checked_at` must fall on its run's day anyway, so no other
partition would accept it. It is written last, as that run's `terminal`
mutation, which is what makes the feed's checkpoint and the run that
wrote it commit together. Published-day partitions settle first, so the
checkpoint never claims the feed was checked before the evidence it
describes is durable.

**Handled provider outcomes use the repository's own vocabulary.**
`success`, `partial`, `empty`, and `failed` are `SOURCE_STATE_STATUSES`,
so the checkpoint and the run describe one event in one language. A
failed checkpoint resolves to a `degraded` run — `record_source_state`
counts a failed state as partial — and never to `success`. That is the
contradiction worth guarding: a run claiming clean success over a
checkpoint that says the fetch failed.

**A failed persist leaves the checkpoint alone.** The old code wrote a
`failed` source state from outside any transaction. Here the failure is
already durable — it is the partition's `failed` run-log row — and
stamping the feed on top of it would claim a check that never completed.
A complete *provider* failure is different: there is no evidence to
ingest, so the checkpoint is the run's only mutation and therefore its
terminal one. That is the path that used to call `log_stage`.

**`fetch()` no longer takes a `trading_day`.** A run's day is a partition
identity the evidence decides, not a label a caller supplies; an override
could only be ignored or fatal. The aggregate counters `fetch()` returns
are a summary for its caller — the durable audit is one `run_log` row per
ticker-day, written inside the same transaction as the data.

## What `request_timeout_seconds` bounds

yfinance cannot cancel a request in flight: `Ticker.news` takes no timeout
and the library hard-codes `timeout=30` into its own session calls. Timing
out therefore cannot mean "the request stopped" — only "we stopped
waiting". The first cut treated those as the same thing, and the cost was
unbounded: each attempt started its own daemon thread, so a hung provider
left one live request *per attempt per scheduled fetch*, growing without
limit for as long as the hang lasted.

`YahooProviderGate` bounds the work instead of pretending to stop it.

- **A ceiling.** `max_concurrent` requests may be outstanding — by default
  one per approved ticker, which is the most the sequential fetch loop can
  ever want. A request that cannot claim a slot is refused with
  `YahooProviderBusyError` rather than queued, so a hang degrades the next
  fetch immediately instead of parking work behind it.
- **Single flight.** Slots are keyed by ticker. A caller asking for a
  ticker whose request is still outstanding *joins* it. This is what keeps
  a retry — and the next scheduled fetch — from multiplying live requests
  against a provider that is already struggling. Retry counts and backoff
  are unchanged; what a retry now repeats is the wait, not the request.
- **The request frees its own slot.** Only the worker releases, in a
  `finally`, so a request abandoned at the timeout keeps counting against
  the ceiling for exactly as long as it is really running. The one
  exception is a worker that never starts: if `Thread.start` raises, no
  worker exists to reach that `finally`, so `call` undoes its own
  registration and frees the slot before the failure propagates. Which of
  the two retires a request is settled by identity under the gate's lock,
  so the slot is freed exactly once either way.

The gate is deliberately not a `ThreadPoolExecutor`: its workers are
non-daemon and joined at interpreter shutdown, so one hung provider call
would leave a scheduled job unable to exit. These workers are daemons,
capped by the same semaphore that caps the slots.

A fetcher built without an explicit gate uses the module-level
`SHARED_PROVIDER_GATE`, so a scheduler that builds a fresh fetcher per run
cannot escape the ceiling by bringing its own slots. Pass a private gate to
opt out. A refusal settles like any other exhausted request: a `failed`
checkpoint and a `degraded` run, with `counts["provider_busy"]` separating
it from a timeout.
