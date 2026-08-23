---
Status: Approved
Approved by: Kartik Gangwar
Approved date: August 21, 2026
Applies to: I5 — Downstream Integration on Real Evidence
Base commit: 2771e5becfd7fa3e48dace00cf5088c77b8f0c3d
---

# I5 decision record

## Why this file exists

I5 connects M1–M5 to the persistence and orchestration contracts that I1–I4
finalized. The audit that preceded it found that the modules are compatible
and do not need rebuilding, and that the work is a projection layer, a
partition enumerator, a per-partition runner, and inspection tooling.

Along the way eight questions came up that the code cannot answer for
itself, and four more that only appeared while specifying the answers. All
twelve are settled here **before** implementation, because three of them
(A, B, C) change what gets built rather than how it is written, and because
a decision that lives only in a review thread gets re-litigated by whoever
touches the code next.

Each entry records the decision, the reason, and — where one exists — the
condition under which it should be revisited. Nothing here is a permanent
commitment; several are explicitly Phase 0 expedients with a stated trigger.

**Scope note.** This document is a record, not a plan. It says what was
decided, not what order it gets built in.

---

## A — Embeddings are computed in memory for I5

**Decision.** Story and theme vectors are computed in memory and used
directly by M3 and M5. I5 introduces **no** two-pass story persistence
lifecycle whose only purpose is to obtain durable ids to cache vectors
against. The `embeddings` table may remain empty for these stages, and that
is an honest empty rather than a missing feature.

**Why.** M1's cache protocol (`nlp.embeddings.EmbeddingRepository`) is
keyed on a durable row id, and `Phase0Repository.upsert_embedding` enforces
that through `phase0.embeddings.require_durable_source_id`. The run-scoped
batch, `persist_embeddings`, additionally refuses a source that does not yet
exist. Stories have no `stories.id` until `reconcile_stories` runs, which is
*after* M3 — so at the moment M3 needs a vector there is nothing to key it
to.

Persisting M2 clusters first to obtain ids would work once and then fight
the schema: migration 013's `trg_embedding_story_cleanup` collects a story's
vector when that story row dies, and the M3 reconciliation supersedes and
deletes exactly those rows. The cache would be rebuilt and discarded on
every run.

Against that, re-encoding costs seconds: a local MiniLM over a few hundred
stories across five tickers. Paying an architectural price to avoid it is
the wrong trade at this stage.

**Revisit when.** Measured runtime makes re-encoding materially expensive at
the production cadence (every 30 minutes during market hours). The cheaper
mitigations come first — bounding partition enumeration to a recent window,
and skipping partitions whose evidence has not changed — and only if those
are insufficient does the caching question reopen.

---

## B — Publisher and outlet identity is canonicalized at the projection boundary

**Decision.**

- Canonicalization happens **only** in the projection layer. `nlp/dedup/text.py`
  is not modified in I5, so no M2/M3/M4 policy fingerprint moves and every
  committed evaluation artifact stays valid.
- Explicit, reviewed publisher mappings are the **only** way a Yahoo
  representation and an RSS representation of one publisher unify.
- Unmapped sources preserve the Yahoo/RSS separation. The fallback never
  infers that two spellings are the same publisher.
- `RawItem.source` is the sole outlet input. The projection chooses that one
  value and nothing else.
- The persisted `StoryMemberRecord.outlet` and `StoryRecord.outlet` are
  **copied from M2's actual result** — `ClusterMember.outlet` and
  `DeduplicatedCluster.outlet` — and are never recomputed by the projection.
- `provider_item_id` stays scheme-qualified (`yahoo:<id>`, `rss:<id>`) so
  provider-identity namespaces remain distinct even where the publisher has
  been unified.
- The publisher policy carries a version string.
- The actual mapping contents are determined by PR 2's live observation, not
  by inference from the repository.

**Why.** The same publisher reaches us as `yahoo:Reuters` and as
`rss:reuters.com`, and M2 counts distinct outlets to decide whether a story
is syndicated. M5 then spends 30% of its salience weight on outlet breadth.
Left uncorrected, one publisher reads as two and themes rank on a partly
fictional number.

Correcting it inside `nlp/dedup` would move the `text_policy` fingerprint,
which invalidates every committed M2/M3/M4 result and forces a full
re-baseline for a change that has nothing to do with the algorithms. The
projection is the right place: it is the layer that already translates
between the persisted world and the stage world.

### The fixed-point invariant

`canonical_publisher(source)` chooses `RawItem.source`. The emitted value
must satisfy:

```
normalize_source(value) == value
```

M2 derives its outlet from that exact string, and persistence copies M2's
outlet verbatim. **There is one outlet representation, not two.** The
property is established by *calling* `nlp.dedup.text.normalize_source` and
refusing any value that is not already its own image — never by predicting
what that function would do. `normalize_source` is idempotent, so the check
is sound.

A corollary closes the last gap: every emitted value is non-empty, so the
`or url_host(canonical_url) or url_host(url)` fallback inside
`NormalizedItem.outlet` can never fire. No hostname the projection did not
choose can become a persisted outlet.

There is deliberately **no** second display-versus-canonical outlet
representation. Two representations whose equality is assumed rather than
established is the failure this invariant exists to prevent.

### The fallback

An unmapped source keeps its scheme and contributes its remainder as a
single alphanumeric token, so `normalize_source` has no trailing token to
reinterpret. This is not public-suffix knowledge: nothing is classified,
dropped, or interpreted — the whole remainder is kept with its separators
removed.

The consequence is stronger separation than a space-preserving fallback
would give. `rss:example-news.com` and `rss:example-news.co` emit
`rss examplenewscom` and `rss examplenewsco` — distinct, where a
space-preserving form would have collapsed both to `rss examplenews`.

### Reserved namespace

Unmapped values always have the form `<scheme> <token>` or bare `<scheme>`.
A mapped publisher id therefore must not equal `yahoo` or `rss`, and must not
begin with `yahoo ` or `rss `. This is why the Yahoo Finance house byline maps
to `yahoofinance` and not to `yahoo finance`: an unmapped `yahoo:Finance`
would emit exactly `yahoo finance`, and the two would silently unify.

Mapped ids may otherwise be multi-token. `RawItem.source` also feeds
`analyze_title`, which uses `normalize_source(source)` when stripping
trailing publisher attribution from a headline, so a mapped multi-token id
can legitimately strip an attribution suffix that a squashed one cannot.
Unmapped sources are unaffected, because a scheme-prefixed key never matches
an attribution segment — which is exactly today's behaviour.

### Stop condition

The observed sources are recorded in
`docs/observations/i5-provider-observation-2026-08-23.md`: 18 distinct
`yahoo:<publisher>` strings, each one display name per publisher, and 2
`rss:<host>` strings. One spelling per publisher, so B's premise holds so far.
Three things in that record bear directly on the mapping:

- `content.provider.sourceId` is **not** a usable key. Its vocabulary is
  mixed — `motleyfool.com` and `wsj.com` beside `benzinga_79`,
  `24_7_wall_st__718`, and `us.finance.gurufocus` — and where it does look
  like a host it can still disagree with the article's: `ibd.com` against
  articles on `www.investors.com`.
- Most Yahoo articles canonicalize to `finance.yahoo.com` rather than to the
  publisher's own host, so a shared article URL is a weak bridge between the
  two schemes in practice.
- `yahoo:Yahoo Finance` is real, observed under META and NVDA. The reserved-
  namespace rule above is what that row needs, not a hypothesis about it.

No Yahoo↔RSS equivalence was observed at all in that window — neither
MarketWatch nor TechCrunch appeared as a Yahoo publisher — so the reviewed
mapping starts empty, which the fallback already makes safe.

If the sources observed in PR 2 cannot be represented by a small explicit
reviewed mapping — many spellings per publisher, or names that vary per
article — then decision B's premise does not hold for real data. **Revisit
B rather than introducing heuristics or growing the table without limit.**
Shipping with an empty mapping is a safe interim state: the fallback
preserves identity, so Yahoo and RSS copies simply count as separate
outlets, which is today's behaviour and a measured limitation rather than a
wrong answer.

---

## C — The evidence partition day stays UTC-derived for I5

**Decision.** I5 preserves I1's current partition day: the UTC date of
`COALESCE(published_at, fetched_at)`. No schema change, no change to day
derivation, no migration of existing rows.

**This is explicitly not yet a trustworthy US-market "daily" boundary**, and
must not be presented to a reader as one. A UTC date cuts the trading day at
20:00 America/New_York during EDT, so evening coverage lands on the
following "day".

Real-data review must measure after-hours and cross-boundary effects rather
than assuming they are negligible.

**Why.** The derivation is embedded in the SQL that reads evidence, in the
partition assertions that protect every write, and in every ingestion path.
Changing it is an I1 change with a data migration attached, and I5's purpose
is to connect what exists and find out what real evidence breaks. Making a
persistence-contract change first, on an assumption, is the opposite of
that.

Note that `pipeline.invocation_day` already uses America/New_York, but only
as a label for logs and CLI output — evidence takes its day from its own
timestamps. The two are deliberately different things, and neither is wrong
today; what is unresolved is which one a *product* "today" should mean.

**Revisit before** any user-facing daily experience is claimed to be
trustworthy. This is a blocking prerequisite for that claim, not a
nice-to-have.

---

## D — Cross-midnight duplicates are accepted and measured for I5

**Decision.** I5 preserves the current one-day story partition. A syndicated
pair straddling the UTC boundary becomes two canonical stories. We measure
how often that actually happens on real evidence before redesigning
anything.

**Why.** I1 confines every story to one trading day —
`_assert_members_in_partition` requires every member raw item to fall on the
reconciliation's day. M2's merge windows (72h content and exact-title, 72h
URL, 36h near-exact) and M3's ±36h window are all wider than that, so within
a single-day batch they never bind.

Widening the partition would contradict I1's story rule and is a large
change. Doing it speculatively, before we know whether the case is common or
rare, would be redesigning against an assumption.

**Revisit when** the measured frequency justifies it. The review rubric
counts near-twin stories on adjacent days, and the inspection tooling
reports stories published near the boundary.

---

## E — One raw item may participate in several ticker partitions

**Decision.** A single raw item can legitimately belong to more than one
ticker's stories and themes. `raw_item_tickers` is authoritative for that
membership. Per-ticker counts are therefore **not** a partition of unique
day-level evidence.

**Why.** One article about NVDA and AMD is ordinary, and the schema records
it: `_assert_raw_item_association` requires each run's own ticker to be
among an item's associations and ignores the others, so each ticker's
derived output is built independently and an AMD-only item stays out of
NVDA's.

The counting consequence matters and is easy to get wrong — see A1 below.

**Known asymmetry, to be measured rather than fixed in I5.** Yahoo evidence
reaches this state naturally: the same article fetched under two tickers is
one row with two `'source'` associations. RSS evidence does not, because the
relevance policy is `ambiguous_match_action: flag_do_not_assign` — an item
matching more than one ticker is flagged with candidates and match evidence
and given *no* association, so it is eligible for neither ticker. That is a
property of the current relevance policy, not of I5. How much real coverage
it costs is a question for real-data review.

---

## F — Evidence eligibility

**Decision.** A raw item enters downstream processing for ticker `T` on day
`D` if and only if all four hold:

1. `ingest_status == 'valid'`
2. `raw_item_tickers` authoritatively associates it with `T`
3. its derived evidence day equals `D`
4. it is minimally projectable — a non-empty title or a non-empty URL

`raw_item_candidates` and `raw_item_match_evidence` are **observability
only**. They may explain why ownership was withheld. They never confer
eligibility and never establish ticker ownership.

Persisted invalid, malformed, ambiguous, and excluded evidence remains
durable. It is kept, counted, and inspectable — it simply does not become a
canonical story.

**Why.** Rule 1 keeps unusable evidence out of the reader-facing output.
Both fetchers deliberately store records they could not parse, and M2 will
happily emit a title-less item as its own single-member cluster; without
this rule a failed provider payload becomes a canonical story with an empty
headline.

Rule 2 is I1's own rule, applied where it had not been. `raw_items.ticker`
is only the first claimant — it is `NULL` for all RSS evidence, and for a
multi-ticker Yahoo article it names whichever run inserted the row first.
The association table is the authority.

Rule 3 uses the same derivation as `raw_items_for_day` and the partition
assertions, so "which day is this on" has one definition.

Rule 4 is defence behind rule 1: the `raw_items` CHECK already guarantees a
valid row has both a title and a URL, so rule 4 can only fire if that
invariant is ever relaxed. It fails loudly rather than producing an
empty-headline story.

---

## G — Yahoo `external_id` requires investigation before any change

**Decision.** Investigate the real provider payload first. Use **only** a
stable, provider-issued article identifier. Never synthesize one from the
title, the URL, the canonical URL, a content hash, a timestamp, the response
position, or any combination of these. If stability or semantics cannot be
established, leave `external_id` absent and record the consequence.

This is a small, separately reviewed I2 correction, not part of the I5
wiring.

**Why.** `normalize_yahoo_item` currently returns no `external_id` for valid
items — only the invalid-evidence path sets one. So `provider_key` is `None`
for every usable Yahoo row and `MatchReason.PROVIDER_ITEM`, M2's one
authoritative and non-time-bounded signal, can never fire on Yahoo evidence.

That is worth fixing, but only with a real identifier. The provider tier is
the one signal M2's compatibility gate does not second-guess: an unstable or
mis-scoped id would produce authoritative merges that are wrong and that
nothing downstream can veto. A synthesized id is worse than none, because it
looks like provenance and is not.

The investigation must establish four things, in writing, with captured
payloads: presence (what fraction of valid items carry it), stability (the
same article carries the same value across fetches hours apart), semantics
(one value per *article*, and two tickers returning the same article return
the same value), and absence of collisions.

Partial presence is not a blocker — `external_id` is nullable and a missing
provider id already means "no authoritative signal". Failed stability or
failed semantics **is** a blocker.

**Evidence (I5 PR 2).** `docs/observations/i5-provider-observation-2026-08-23.json`,
collected by `tools/observe_phase0_providers.py` over four attempts spanning
20.65 hours against the five approved tickers. All four required findings are
met by the top-level `id` field:

- **Presence** — 200 of 200 valid items carried it. `content.id` carried the
  identical value on all 200; the legacy `uuid` did not appear at all.
- **Stability** — 86 distinct articles, 51 of them observed in more than one
  attempt across that span, and not one carried two values.
- **Semantics** — article-scoped. 10 articles appeared under more than one
  ticker and each kept one identifier; 43 appeared at more than one response
  position and each kept one identifier.
- **Collisions** — 86 identifiers for 86 articles. None was shared.

This clears the bar G sets. It does not by itself authorize the change: the
I2 correction that writes `external_id` on the valid path is still separately
reviewed, and the observation is one window, not a guarantee.

---

## H — M3 degradation must never look like a healthy day

**Decision.** An M3 failure must not silently produce ordinary M5 themes
from M2-only output. Specifically:

- The degraded M2 stories **may** be retained, recorded as
  `stories.stage = 'm2.exact'`.
- **No** theme set exists for that partition.
- A healthy recovery replaces the degraded rows.
- Consumers must be able to distinguish degraded and intermediate output
  **mechanically**, not by reading prose.

**Why.** The specification allows M5 to proceed on stage-1/2 dedup output as
a documented degradation, and the bridge supports it. But an automatic
fallback produces themes built over undeduplicated stories that are
*visually identical* to a healthy day's themes. A reader cannot tell, and
neither can a consumer.

Retaining the M2 stories is not that fallback. It keeps the dedup work,
records honestly which stage produced it, and ships no themes at all — which
is the opposite of an indistinguishable degradation, and gives a later retry
something to compare against.

**Recovery is clean, by construction.** M2's `cluster_fingerprint_for`
digests `("m2.cluster.v1", ticker, sorted member item ids)`; M3's
`story_fingerprint_for` digests `("m3.story.v1", ticker, sorted member
cluster keys)`. Different namespace, different inputs — so no M2 fingerprint
can be mistaken for an M3 one, and every `m2.exact` row is obsolete to a
later successful reconciliation. Because `_story_is_referenced` checks only
`theme_stories`, and under this decision no theme set exists, those rows are
deleted outright rather than left tombstoned. Recovery leaves no residue.

The mechanical test for a consumer is two fields: **serve a partition only
when a `theme_sets` row exists and every contributing story has
`stories.stage = 'm3.semantic'`.** No heuristics, no text parsing.

---

## A1 — Evidence-read accounting

Four counters were originally specified on one read. They cannot all live
there honestly: an item with no `raw_item_tickers` association belongs to no
ticker, so there is no `(ticker, day)` row it can be counted on without
either inventing ownership or double-counting. The unassociated population
is a property of the **day**, not of a partition.

### Ticker-partition read — `evidence_partitions()`

- **Grain:** ticker + trading_day.
- **Population (the denominator):** only items authoritatively associated
  with that ticker on that day.
- **Fields:** `associated_item_count`, `eligible_item_count`,
  `excluded_invalid`, `excluded_unprojectable`, `latest_fetched_at`.

This read carries **no** ticker-level `total_item_count` and **no**
`excluded_unassociated`. There is no honest ticker-scoped total — the day's
total includes items this ticker has no claim on — and `excluded_unassociated`
was a counter for a population that has no ticker.

**Invariant:**

```
eligible_item_count + excluded_invalid + excluded_unprojectable
    = associated_item_count
```

Counting is by distinct raw item. `raw_item_tickers` has primary key
`(raw_item_id, ticker, association_type)`, so one item legitimately carries
both a `'source'` and a `'relevance'` row for one ticker and a naive count
would count it twice.

### Day-level read — `evidence_days()`

- **Grain:** trading_day.
- **Population (the denominator):** every raw item whose derived day is that
  day, regardless of ownership.
- **Fields:** `total_item_count`, `associated_any_ticker`,
  `unassociated_item_count`, plus an explanatory breakdown of the
  unassociated bucket (ambiguous, invalid, has candidates, matched nothing).

**Invariant:**

```
associated_any_ticker + unassociated_item_count = total_item_count
```

### The identity that does not hold

```
SUM(ticker associated_item_count) != associated_any_ticker
```

A multi-ticker article is counted once in each ticker's partition and once
on the day. This is correct under decision E — the article really does
participate in both partitions independently — but it means **the
per-ticker rows are not a partition of the day's evidence and must never be
presented as one.** Inspection output shows the day block and the ticker
blocks separately, with the overlap named, rather than in one table that
invites subtraction.

### Row-level read — `unassociated_items()`

Row-level observability: the items themselves, with the tickers named by
their candidate rows and by their match evidence. It exists so "was any of
this real coverage we missed?" can be answered by reading. It never
establishes ticker ownership.

### Candidates and match evidence

`raw_item_candidates` and `raw_item_match_evidence` may explain withheld
ownership — for example, evidence recording `decision = 'matched'` for a
ticker that holds no association, which today means the item matched more
than one ticker and was flagged rather than assigned. Such a counter has its
own denominator (items whose evidence names that ticker) and must never be
summed with the eligibility counters. It confers no eligibility.

---

## A2 — Publisher policy, recorded

- An explicit, **versioned** publisher mapping, keyed on the exact stored
  source string.
- An identity-preserving, scheme-separated fallback for anything unmapped.
- Actual mapping entries come from PR 2's live observation. They cannot be
  written from the repository: Yahoo's source is a live provider display
  name and RSS's is each article's resolved hostname, and `config/feeds.yaml`
  names *feed* hosts, which are not the same thing.
- No public-suffix inference.
- No inferred Yahoo↔RSS equivalence.

**Final fixed-point contract**, for every value `canonical_publisher` emits:

```
RawItem.source == normalize_source(RawItem.source)
```

Persisted story and member outlet fields come from M2's resulting outlet
values verbatim.

Provider ids remain scheme-qualified — `yahoo:<provider-id>` and
`rss:<provider-id>` — even where the outlet itself has been unified. This is
load-bearing rather than defensive: unifying a publisher makes
`provider_namespace` identical for both copies, so the scheme qualifier on
the provider id is the only thing keeping the two id spaces apart.

Reserved-namespace constraints and the stop condition are stated under B.

---

## A3 — Previous-theme stability

### Mandatory runner ordering

1. Read eligible evidence.
2. **Read previous themes — before any story reconciliation.**
3. M2 exact dedup.
4. M1 encode, then M3 semantic dedup.
5. `reconcile_stories`.
6. Read back the persisted story ids.
7. M5, using the previous themes captured in step 2.
8. `reconcile_themes`.

**Why step 2 comes before step 5.** `reconcile_stories` invalidates the
partition's theme set whenever any story is obsolete or structurally changed,
and `themes` has no `invalidated_at` column — the rows are deleted. Reading
previous themes afterwards therefore finds nothing on exactly the runs where
story structure moved, which is the only case theme-identity continuity is
about. The stage would silently mint new identities on every meaningful
re-run, and no test that ran the stages in the wrong order would notice.

### Compatibility gate

A stored theme may be used for stability matching only when **every** one of
these matches the run about to happen:

- `ticker`
- `trading_day`
- `pipeline_version`
- `model_name`
- `model_revision`
- `embedding_dimension`
- `config_fingerprint`
- `algorithm_version`

and only when its centroid is present and deserializes cleanly to
`embedding_dimension`.

**Do not carry theme identity across incompatible embedding, model, or
configuration spaces.**

The partition triple is required because `theme_key` is unique only within
`(ticker, trading_day, pipeline_version)`; a key reused across partitions
either collides or names a different theme.

The model identity is required for a reason dimension alone does not cover.
M5's centroid comparison returns `-1.0` on a *length* mismatch, so it already
fails safe across dimensions — but two different encoders of the same width
produce perfectly plausible cosine values over incomparable spaces, and a
theme would inherit an identity matched in a space it was never embedded in.

`config_fingerprint` is the strongest single check: `ThemeConfig`'s
fingerprint already folds in the algorithm version, the model identity and
dimension, the stability threshold, both cohesion floors, the theme caps, the
salience weights, the ticker universe, and every static policy module.
`algorithm_version` is compared as well because it is a separate nullable
column, and a row that cannot state its own provenance is not reusable.

Rejections are counted and reported by reason, never silent: "theme identity
was not carried over" and "there were no previous themes" are different
facts and a reader must be able to tell them apart.

---

## A4 — Stable operational stage names

### The stage model

```
run_log.stage = "stories"
run_log.stage = "themes"
```

These name **units of work**, not algorithms and not outcomes. They join the
existing operational vocabulary — `yahoo`, `rss`, `rss_relevance_replay` —
rather than putting milestone names into an operational ledger.

Algorithm provenance lives on the row it describes:

```
stories.stage = "m3.semantic"     healthy
stories.stage = "m2.exact"        degraded
```

### Why stage names cannot vary by outcome

`latest_stage_status()` selects the newest `run_log` row **grouped by
stage**. A stage name that varied with the outcome would create a second
name whose newest row is a degradation, and nothing under the other name
would ever supersede it: `pipeline.py --status` would report a stale
degradation indefinitely. Stage names must be stable per unit of work.

Naming the run stage after M3 while it persists M2 output would also be a
false statement in the audit ledger, which is the more basic objection.

No I1 change is needed for this: `stage` is validated only as non-empty
text, and both `latest_stage_status()` and `pipeline_status()` are
stage-agnostic.

### Degraded state, in full

- The `stories` run settles with status `degraded`.
- The story rows carry `stage = 'm2.exact'`.
- No `themes` run is attempted from those rows, and none is produced.
- No `theme_sets` row exists for the partition.
- Inspection must distinguish this from a quiet day. "No themes because M3
  failed" and "no themes because there was little news" must never render
  alike.

---

## Related records

- `docs/PHASE_0_SPEC.md` — the approved Phase 0 specification, including the
  go/no-go gates these decisions are eventually measured against.
- `docs/PHASE0_DATA_PIPELINE.md` — the operational contract for
  `pipeline.py`, including run identity and replay scope.
- `nlp/README.md` — M1–M5 design notes and their honest limitations.
- `docs/observations/` — what the providers actually sent, on a stated day.
  Decisions B, G, and A2 rest on those observations; the artifacts are
  evidence rather than tests, and are expected to age.
