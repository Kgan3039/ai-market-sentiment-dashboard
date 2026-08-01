## Phase 0 embeddings

`nlp.embeddings` provides the shared local embedding API used by the Phase 0
deduplication, clustering, and theme-stability stages. It uses
`sentence-transformers/all-MiniLM-L6-v2` by default, loads it lazily, and
pins a model revision for replayability. It L2-normalizes float32 vectors.
Title and description whitespace is collapsed; when both are present, the
exact model input is `title + "\n\n" + description`.

Model files use the sentence-transformers/Hugging Face cache. Set
`TICKER_NARRATIVES_MODEL_CACHE` or pass `cache_location` to select a cache
outside the repository.

Persistence is intentionally behind the `EmbeddingRepository` protocol. Main
does not yet contain issue #57's write repository, so this module does not
create a parallel SQLite layer. The eventual adapter must atomically implement
`get_embedding()` and `upsert_embedding()` and persist all fields in
`PersistedEmbedding`. Unit tests recreate repository adapters over shared
in-memory backing state; they do not claim durable SQLite restart coverage.

Run the hardware-sensitive, fixture-derived warm-model benchmark separately:

```bash
python -m tools.benchmark_embeddings
```

Black-compatible Flake8 policy is committed in `.flake8`. Lint the M1 files
with:

```bash
python -m flake8 nlp/embeddings.py tests/test_embeddings.py tools/benchmark_embeddings.py
```
## Phase 0 exact / near-exact dedup core (M2, issue #64)

`nlp.dedup` is the **pure core** of issue #64: stage 1 canonical-URL and
normalized-title exact match, stage 2 MinHash over normalized title
shingles. It is a function, not a service — no database, no clock, no
network, no filesystem, no model, no state between calls.

```python
from nlp.dedup import DedupConfig, RawItem, deduplicate

config = DedupConfig(supported_tickers=["TSLA", "NVDA", "AMD", "AAPL", "META"])
result = deduplicate(raw_items, config=config)
result.clusters             # deterministically ordered
result.stats                # counters, including compatibility-gate vetoes
result.provider_conflicts   # quarantined provider identities
```

**This does not close issue #64.** The DoD requires the stage to run inside
`pipeline.py`, which is issue #68 (open, blocked by #61/#62); #64 also
declares itself blocked by #57 (open). AC-3's precision/recall half needs
M4's labelled set and M3. What is here is the deduplication logic those
will call.

### Precision-first, with one gate

The core merges only mechanically obvious copies. Anything that needs an
understanding that two differently worded headlines describe one event is
M3's job — false negatives here are expected, false positives are not.

| Signal | Window | Gated |
|---|---|---|
| provider namespace + provider item id | none | no — authoritative |
| URL identity key, corroborated, both records dated | 72 h | yes |
| identical normalized title **and** description | 72 h | yes |
| identical normalized title | 72 h | yes |
| MinHash candidate verified structurally identical | 36 h | yes |

Provider identity is the one *authoritative* signal: a feed asserting that
two records are its same item. It is not time-bounded and not gated — but
only while the identity is consistent (see quarantine below).

Every other signal is circumstantial, so every other edge passes **one**
compatibility gate before it is unioned, and the gate is applied to the
**whole prospective cluster**, not to the edge's two endpoints. Checking
endpoints alone let a sparse record bridge contradictory ones — A "profit
rose", B with no description, C "profit fell" — since A–B and B–C are each
individually fine. Each component now carries an `EvidenceSummary` of every
value its members assert, and a union is admissible only when the combined
summary still holds at most one known value per field. That is exactly
equivalent to comparing every member on one side against every member on
the other, at constant cost.

The gate covers: unparseable numeric notation, text-free versus
text-bearing, protected-expression disagreement (numbers, currencies,
units, ranges, signs, percentages, quarters, years, dates), negation
disagreement, differing titles, differing descriptions. The same primitive
backs provider-conflict quarantine, which adds URL, ticker, and timestamp
checks on top — provider quarantine may be stricter than ordinary
compatibility, never weaker.

The rule is asymmetric on purpose — **explicit disagreement vetoes, missing
information does not.** So this pair stays apart even though the headlines
normalize identically:

```
title:       "Quarterly results"
description: "Profit rose sharply."   vs   "Profit fell sharply."
```

while a Reuters original still merges with a Yahoo copy carrying the same
headline and no standfirst at all.

Windows apply to the **span of the resulting cluster**, so transitive
chaining cannot widen a merge. Merges never cross a ticker.

### Provider-conflict quarantine

For each `(provider namespace, provider_item_id)` the core compares every
pair with the shared evidence primitive, then adds URL, ticker, and
timestamp checks (beyond a 1-hour tolerance). Two titles the core could not
parse — "Profit ½ higher" and "Revenue ⅓ lower" — conflict rather than
merging on the accident that neither yields an ordinary title key. On any
disagreement **every item under that identity is quarantined**: it merges through no signal at all, and is emitted as its own
cluster so no coverage is lost. `result.provider_conflicts` reports the
namespace, id, affected item ids, and the fields that disagreed.

### MinHash: real, deterministic, and honestly scoped

Stage 2 shingles the normalized title into character 5-grams, computes a
128-permutation signature from a `blake2b`-seeded universal hash family
(never Python's randomized `hash()`), and proposes candidate pairs at a
permissive estimated-Jaccard floor inside one ticker and one 36-hour
window. Candidates are then verified exactly.

**It adds no unique merge today, and nothing here claims otherwise.**
Verification requires identical normalized titles, which is the exact-title
signal, so every verified candidate is a pair stage 1 also finds and
`MatchReason.NEAR_EXACT_TITLE` is never recorded. The stage is retained
because issue #64 specifies it and because it is the point where M4's
labelled data will widen matching with measured precision. A test asserts
this equality rather than leaving it to be discovered. No LSH banding: the
issue specifies MinHash, and banding is scope the authoritative sources do
not ask for.

### Capacity: fail fast, never partial

Candidate generation is quadratic (2,000 items in one window measured ~54 s
on the dev box). A ticker partition larger than `max_partition_items`
(default 250) raises `DedupCapacityError` **before any output exists**,
carrying the ticker, the item count, and the limit. The core never returns
a result that looks complete while having skipped work; splitting the batch
or raising the limit is the caller's deliberate choice.

The provider namespace is deliberately *not* the outlet key: it applies
case, accent, punctuation, and whitespace folding only. Stripping legal or
domain suffixes would give `Acme Inc` and `Acme LLC` one authoritative
identity, letting one feed's item id merge another company's article. Being
too strict costs at most a tier-1 merge the weaker signals can still make.

### Contracts the core insists on

- **`supported_tickers` is required.** There is no default, no fallback, and
  no file to read: whoever orchestrates the core knows the approved
  universe and passes it in. It participates in the configuration
  fingerprint.
- **Every item must carry a supported ticker.** Unassigned items are the
  caller's to filter — Phase 0's relevance filter already excludes them
  from processing (spec section 2) — so the core rejects them rather than
  inventing a nameless partition.
- **Naive timestamps are rejected;** no Phase 0 contract states their wall
  clock. Aware values convert to UTC, so DST is real elapsed time.
  Timestamps outside 1990–2100 are rejected as corrupt, on absolute bounds
  so replay does not depend on today's date.
- Windows must be positive, at most two weeks, and ordered
  `near_exact ≤ exact_title ≤ content`, `url ≤ content`. **Every** signal
  except authoritative provider identity is windowed, URL included: the
  same reusable slug 180 days apart does not merge, and a URL pair with no
  usable timestamps is not a URL match at all.
- `supported_tickers` is validated against the same syntax a record's
  ticker must satisfy, and rejects duplicates that differ only in case, so
  a configuration cannot hold a symbol no record could ever match.

### URL identity

`clean_url()` (display) and `url_identity_key()` (identity) are separate.
Identity applies only transformations safe at every publisher: lower-cased
scheme and host, IDNA/punycode, valid default-port removal, fragment
removal, and a narrow click/analytics allowlist. It does **not** strip
`www.`/`m.`/`amp.` hosts or `/amp` paths, collapse slashes, normalize
trailing slashes, equate `http` with `https`, or unwrap redirect wrappers.
The query is filtered by splitting on `&` and rejoining verbatim — never
parsed and re-encoded — so `a+b` stays distinct from `a%20b`, percent
escapes keep the publisher's spelling, and repeated parameters keep their
order. Ports outside 1–65535, credentials, IPv6 literals, and unparseable
values yield `None`, meaning "no URL evidence".

### Cluster fingerprints, not story IDs

The core assigns **no** durable identifier. Each cluster carries
`cluster_fingerprint`: a full-width SHA-256 digest of the ticker and the
sorted **unique** member set, length-prefix encoded so no item id can forge
a separator boundary. The helper rejects blank tickers, blank member ids,
duplicates, and empty member sets outright. It is order-independent and independent of which
member is canonical, and it **changes when membership changes** — that is
the change-detection signal a reconciler needs. Durable `stories.id`
belongs to issue #57, and joining a run's clusters to stored rows is done
on `member_ids`.

### What a future pipeline adapter must add

Row projection from `raw_items`, provider-identity extraction from the
persisted columns, `run_log` counters, per-row error isolation, retry
handling for capacity failures, `trading_day` assignment, and the
transaction that writes `stories`. All of that is orchestration and belongs
to #57/#68, not here.

Lint and test the core with:

```bash
python -m black --check nlp/dedup tests/test_dedup_core*.py
python -m flake8 nlp/dedup tests/test_dedup_core*.py
python -m pytest tests/test_dedup_core.py tests/test_dedup_core_signals.py
```

## Phase 0 dedup evaluation sets and evaluators (M4, issue #67)

`nlp.eval` holds the labelled deduplication datasets and the deterministic
evaluators that measure a stage against them. The evaluators call the public
stage APIs and only count what they return, so a reported number cannot
drift away from shipped behaviour.

```bash
python -m tools.eval_dedup --stage m2                    # isolated pairs
python -m tools.eval_dedup --stage m2 --scope clusters   # whole batches
python -m tools.eval_dedup --composition                 # dataset makeup
python -m tools.eval_dedup --stage m2 --json             # committed form
```

### Read this before reading any number below

**WARNING: Synthetic, single-author, unadjudicated development dataset.
Metrics are not valid for K3/G4 or final AC-3 acceptance.**

| | |
|---|---|
| `dataset_kind` | `synthetic_development` |
| `real_ingested_evidence` | `false` |
| `labeling_status` | `single_author_unadjudicated` |
| `reviewer_count` | 1 |
| `adjudicated` | `false` |
| `gate_eligible` | `false` |
| `metrics_purpose` | `development_regression_only` |

This block is **enforced, not documented**, and validation is *relational*
rather than per-field. `dataset_kind`, `labeling_status` and
`metrics_purpose` are enums with no unknown values accepted, and each
dataset kind's invariants live in one table, `DATASET_KIND_RULES`, which the
validator asserts covers every enum member — so a new kind cannot silently
inherit permissive behaviour.

| | `synthetic_development` | `sampled_production` |
|---|---|---|
| `real_ingested_evidence` | must be `false` | must be **`true`** |
| `metrics_purpose` | only `development_regression_only` | any |
| may be gate eligible | no | yes, subject to the rules below |
| `provenance.kind` | `synthetic` | `sampled` |
| `provenance.collection_method` | `authored` | `sampled` or `ingested` |
| `provenance.urls_are_synthetic` | `true` | `false` |
| extra provenance required | `why_synthetic`, `blocked_by` | `ingestion_source`, `sample_selection` |

A production sample must name what it was sampled *from*, and cannot
describe itself as authored — the combination `sampled_production` with
`real_ingested_evidence=false` used to load and no longer does.

On top of the per-kind table, kind-independent rules: a
`single_author_unadjudicated` set has exactly one reviewer, is not
adjudicated, and is not gate eligible; `adjudicated=true` needs at least two
reviewers *and* an adjudicated status; `gate_eligible=true` needs real
ingested evidence, adjudication, two or more reviewers, a
gate-or-acceptance purpose, and a kind that permits it; and a gate purpose
declared without `gate_eligible` is refused. The `labeling` block must agree
with the contract on all four shared fields.

### The banner is derived, never supplied

There is no `warning` field on a trust contract and a manifest that supplies
one is **rejected** — a banner a caller could write is a banner that could
contradict the fields beside it. `derive_trust_summary()` computes it from
`dataset_kind`, evidence status, `labeling_status`, `reviewer_count`,
`adjudicated`, `gate_eligible` and `metrics_purpose`:

| state | banner |
|---|---|
| synthetic | `WARNING: Synthetic, {labelling} development dataset.` / `Metrics are not valid for K3/G4 or final AC-3 acceptance.` |
| production, unadjudicated | `WARNING: Production-sampled evidence has not completed independent adjudication.` / `Metrics are development-only and not gate eligible.` |
| production, adjudicated, not gated | `NOTICE: Production-sampled, independently adjudicated evaluation dataset.` / `Metrics are not configured as a release gate.` |
| production, adjudicated, gated | `NOTICE: Production-sampled, independently adjudicated gate-eligible dataset.` |

The labelling phrase is derived too — `single-author, unadjudicated`,
`3-reviewer, unadjudicated`, `2-reviewer, independently adjudicated` — so
the wording tracks the fields rather than a constant. Branching on
`dataset_kind` first means **no production dataset can receive synthetic
wording and no synthetic dataset can receive a production or adjudicated
notice**; both directions are tested.

Every text renderer prints the derived summary above and below the numbers,
and every JSON payload carries **both** the structured `trust_contract`
block and the derived `trust_summary` (`level`, `headline`, `detail`,
`text`) — including the low-level sweep document.

Every loaded pair, item, and case is stamped `synthetic=True` from the
manifest.

Issue #67 asks for ~150 pairs **sampled from real ingested data**,
co-labelled with Kartik under the K3 (#60) guidelines. None of that is
possible on `main`: I2 (#61) and I3 (#62) are open, so there is no ingestion
and no populated `raw_items`; K3 is open, so the co-labelling protocol is
not written. Every headline, URL, outlet and timestamp is authored. Real
outlet names are used so the syndication and attribution cases exercise the
real publisher policy; invented names (`Wolfsberg Motors`, `Pacific Advanced
Packaging`, `Harbourline Media`, `Northfield Securities`, `Calder Bank
Markets`) are used wherever a label needs to name a third party.

### The two measurements are not interchangeable

| scope | what it does | what it can see |
|---|---|---|
| `isolated_pair_metrics` | one pair per invocation, two records | whether a **two-item call** merges a pair |
| `multi_item_cluster_metrics` | one whole group per invocation | cluster-wide compatibility, transitivity, provider quarantine, window-on-span |

The pairwise evaluator previously claimed its results were faithful to
production clustering because M2 guarantees a record's clusters do not
depend on batch companions. **That claim was wrong and has been removed.**
M2 makes the opposite guarantee explicitly: the compatibility gate is
applied to the whole prospective cluster, merges are transitive, quarantine
looks at every record under a provider id, and the window applies to a
cluster's span. All four mean a third record *can* change what happens to
two others. Isolated-pair metrics therefore cannot on their own validate
production clustering, the report is scoped `isolated_pairs`, the field is
named `isolated_pair_metrics` rather than `overall`, and
`ISOLATED_PAIR_LIMITATION` travels with every rendering.

### Composition (153 pairs)

| | count |
|---|---|
| duplicate / distinct / ambiguous | 78 / 73 / **2** |
| expected stage m2 / m3 / none | 48 / 30 / 75 |
| tickers AAPL / AMD / META / NVDA / TSLA | 32 / 27 / 29 / 32 / 33 |

`ambiguous` now means **the records do not contain enough to decide it**,
not "the author expects disagreement". Two pairs qualify: `P133`, one EU
fine reported in euros by one outlet and dollars by another, where a rounded
conversion explains the pair as well as two decisions do; and `P138`, a point
estimate of 62 billion inside a published 60-65 billion range, which is
exactly how a second outlet summarizes one disclosure. Both were confident
hard negatives and should not have been.

Five pairs went the other way. `P149`-`P153` (a live blog beside an article
about one announcement in it, a report beside a follow-up analysis, a rumour
beside its confirmation, a release beside an executive interview, an
announcement beside a hands-on) were labelled ambiguous. The article-level
contract settles all five: a canonical story is one event reported by
multiple outlets, and none of these is a copy of the other. Labelling them
ambiguous was class balancing. They are now `distinct`, in a new
`same_event_different_article` category.

### M2 isolated-pair baseline

`nlp/eval/data/results/m2_baseline.json`, 151 scored pairs:

| metric | value |
|---|---|
| precision | **1.0000** (48 merges, 0 false) |
| recall | 0.6154 |
| F1 | 0.7619 |
| tp / fp / tn / fn | 48 / 0 / **73** / 30 |
| recall on `expected_stage=m2` | **1.0000** (48/48) |
| recall on `expected_stage=m3` | 0.0000 (0/30) |
| ambiguous pairs merged | 0 |
| complete | true (0 failures) |

The true-negative count moved 70 → 73 because three pairs joined the scored
set. Precision, recall and F1 are unchanged: the relabelled pairs were all
ones M2 correctly refused, so they moved from "excluded" to "true negative"
without touching a numerator. M2 was not tuned.

### M2 multi-item cluster results

Nine authored cases, 30 records, covering a sparse bridge between
contradictory endpoints, a legitimate sparse bridge, a provider-conflict
group, a recycled-URL group, a repeated quarterly group, three-item semantic
transitivity, a mixed-stage group, a permutation-equivalence batch, and a
release-plus-interview group.

**Three expectations per case, kept apart on purpose:**

| field | what it is |
|---|---|
| `expected_partition` | ground truth — how a reader groups the *articles* |
| `indeterminate_item_ids` | items the records do not place; never scored |
| `exact_stage_partition` | what the exact stage alone should produce |

Recording an implementation's traversal order as human truth would make the
fixture agree with the code by construction. Three cases were corrected for
exactly that:

- **C001** previously asserted `[[1,2],[3]]` as ground truth. Item 2 carries
  the same headline and no standfirst — nothing in the records says which of
  the two contradicting articles it is a copy of. It is now
  `indeterminate`, ground truth asserts only the decidable part (1 and 3
  stay apart), and no pair involving item 2 is scored. M2's answer is kept
  in `exact_stage_partition`, where it belongs.
- **C003** previously made ground truth all-singleton because M2 quarantines
  a provider conflict. As articles, records 1 and 3 are one story and record
  2 is a different recall. Ground truth now says so, quarantine stays in
  `exact_stage_partition`, and the gap shows up as an under-merge — the
  honest cost of the policy rather than a definition that hides it.
- **C009** was `ambiguous` on the same question P152 answers. P152 is
  `distinct`; C009 now takes the same decision and declares the link, so the
  two fixtures cannot disagree silently.

**Cross-fixture claims.** A case may declare that a relationship it contains
is the same one a pair records. The loader checks the claim against the
pair's label *and* against the case's own partition, refuses a claim that
borrows authority from an `ambiguous` pair, and refuses a contradiction
unless the case states a `divergence_reason`. Eleven claims are committed.

Against `exact_stage_partition` (`m2_clusters_exact_stage.json`):

| metric | value |
|---|---|
| exact partition match | **9/9** (1.0000) |
| co-clustering precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 |
| over-merged / under-merged cases | none / none |
| permutation failures | none |
| accounting violations | none |

Against `expected_partition` (`m2_clusters_ground_truth.json`):

| metric | value |
|---|---|
| exact partition match | 6/9 (0.6667) |
| co-clustering precision / recall / F1 | 1.0000 / 0.5882 / 0.7407 |
| over-merged cases | **none** |
| under-merged cases | `C003` (quarantine), `C006`, `C007` (semantic) |

### Cluster-member accounting is checked before any metric

A predicted partition must contain exactly the case's item ids. Missing ids,
duplicate ids across groups, invented ids, empty groups, blank ids and
non-collection groups all raise `PartitionAccountingError`, and the case is
**failed** — excluded from every denominator, with `missing_item_ids`,
`duplicated_item_ids` and `unexpected_item_ids` reported per case and in
aggregate. A clusterer that returns the right answer plus one invented id
scores `exact_partition_matches=0` and undefined precision, not a perfect
run.

### Permutation coverage

Every case is re-run under **every ordering** while the factorial stays at or
below 120 — which covers all nine fixtures (6, 6, 6, 6, 6, 6, 24, 120, 6
orderings). Above that the set is the original, the reverse, every cyclic
rotation, and eight shuffles seeded on the case id, so it is documented,
deterministic and reproducible from the case alone. Each case reports
`permutation_count` and `unstable_permutation_count`.

### Dataset integrity, enforced at load

Refused: unknown vocabulary values; duplicate `pair_id` or `item_id`; a pair
that repeats another pair's **content**, including with the two sides
swapped; a non-duplicate pair whose two records are byte-identical; a
missing `canonical_url` key (it must be present even when null); a URL that
is not parseable http(s) with a host; a naive timestamp; a timestamp outside
the range `nlp.dedup` itself accepts, imported rather than restated; a label
contradicting its expected stage; rows out of `pair_id` order; and any
malformed trust, provenance, or labeling block. Sweep thresholds must be
finite numbers inside `[0, 1]` and distinct.

### Failures are reported, never swallowed

A stage that raises on one case does not end the run. The case is recorded
with its id, exception type and message, left out of **every** denominator,
and the report exposes `evaluated_case_count`, `failed_case_count`,
`failed_case_ids` and `complete`. The text renderer prints the failures and
states that they were excluded; the CLI exits 1 on an incomplete run.

### One validator for every gate value

`nlp/eval/validation.py` is the only place a threshold or floor becomes a
number. It rejects NaN, `inf`, `-inf`, values outside `[0, 1]`,
non-numbers, and booleans — `True` would otherwise be read as `1.0`. It is
used by the CLI's `--precision-floor`, `--recall-floor` and `--threshold`,
by the sweep, and by the report constructor.

This matters because **NaN loses every comparison silently**: a gate checked
against it does not fail loudly, it passes or fails depending on which way
the comparison happens to be written. `argparse`'s `type=float` accepts
`nan`, `inf` and `-inf` without complaint, so the check happens after
parsing and before any comparison. All six of `--precision-floor nan`,
`--recall-floor nan`, `--precision-floor inf`, `--recall-floor -inf`,
`--precision-floor -0.1` and `--recall-floor 1.1` exit 2 with a message
naming the field and the reason.

### The trust block reaches the raw rows, not only the rendering

Every public payload-producing function — `to_payload`, `cluster_payload`
and `sweep_payload` — returns a document carrying the trust contract, the
dataset id, a versioned `schema_version`, the scope, the limitation, and the
completeness counts. `sweep_payload` used to return a bare list of rows,
which is exactly the object somebody quotes; it now returns the document and
the rows live under `points`, each carrying its own scope and completeness.

### Evaluator conventions

- **Undefined is `None`, not zero.** Precision over zero predicted merges is
  unmeasured, not 0%. An undefined metric never clears a gate.
- **Ambiguous pairs and cases are excluded** from the headline numbers and
  reported separately.
- **Candidate recall is reported beside merge recall**, so a sweep can tell
  a missing candidate from a refused one.
- **Everything is deterministic.** Both committed baselines are byte-compared
  against a fresh run in the test suite, under three `PYTHONHASHSEED` values.

### This does not close issue #67

The DoD's "sampled from real ingested data" and "co-label with Kartik per
K3" halves remain blocked on #61, #62 and #60. The numbers here are
development regression signals. G4 needs the real sample, a second reviewer,
and adjudication.

Lint and test with:

```bash
python -m black --check nlp/eval tools/eval_dedup.py tests/test_dedup_eval.py
python -m flake8 nlp/eval tools/eval_dedup.py tests/test_dedup_eval.py
python -m pytest tests/test_dedup_eval.py
```

## Phase 0 semantic dedup (M3, issue #70)

`nlp.semdedup` merges canonical stories that describe one event in different
words, using M1 embeddings and a threshold selected on M4's labelled set. It
runs **after** M2 and never changes it.

```python
from nlp.dedup import DedupConfig, deduplicate
from nlp.embeddings import EmbeddingService
from nlp.semdedup import (
    SemanticDedupConfig, merge_semantic_duplicates, stories_from_dedup,
)

exact = deduplicate(raw_items, config=DedupConfig(supported_tickers=TICKERS))
result = merge_semantic_duplicates(
    stories_from_dedup(exact, raw_items),
    config=SemanticDedupConfig(supported_tickers=TICKERS),
    encoder=EmbeddingService(),
)
```

**This does not close issue #70.** The DoD requires the stage to be
"integrated in pipeline"; `pipeline.py` is issue #68 (open, blocked by
#61/#62). The AC-3 half of the DoD is met *on the committed labelled set* —
which is synthetic, so it is a design measurement, not the G4 gate result.

### The finding that shaped the design

Sweeping the labelled set with the Phase 0 encoder, cosine similarity alone
**cannot separate the classes at any threshold**:

| | cosine range |
|---|---|
| genuine same-story rewrites | 0.42 – 0.73 |
| hard negatives (date, magnitude, role, sign changes) | 0.97 – 0.996 |

The classes are *inverted* with respect to similarity, and for a structural
reason: a real rewrite shares almost no wording with its twin, while a
template negative that swaps one date or one number shares almost all of it.
A stage that merged on similarity would merge exactly the pairs it must not.

So M3 is not a threshold with guards bolted on. It is a set of guards with a
threshold behind them.

### The guards (`nlp/semdedup/evidence.py`)

Each is a static, versioned policy over M2's tokenizer, tried in this order
so a recorded veto names the most specific true objection:

| guard | refuses |
|---|---|
| `temporal_disagreement` | different quarters, fiscal years, months, dates |
| `numeric_disagreement` | different magnitudes, currencies, units, ranges, signs |
| `role_disagreement` | different named roles (CFO vs COO) |
| `subject_shift` | one story is about a supplier, reseller, or agency |
| `contrast_polarity` | opposing claims: raised/cut, approved/rejected, beat/missed, profit/loss, maintained/withdrawn |
| `negation` | one explicitly negates where the other does not |
| `same_frame_different_event` | heavy lexical overlap with a substituted content slot |

The last is the load-bearing one for "same template, different event": when
two headlines overlap heavily, the tokens they do *not* share are the story.
A rewrite has the opposite shape — low overlap, high similarity — so it never
triggers. A strict elaboration ("Apple opens a store" / "…a store in
Riyadh") adds detail rather than substituting an event and is not caught.

M2's asymmetry is preserved: **explicit disagreement vetoes, missing
information does not.** A story with no numbers does not contradict one that
has them. Numbers bind only to neighbours that change their meaning, so
"1,000 roles" and "1,000 under the plan" agree while "5 million" and
"5 billion" do not.

### Threshold selection, from evidence

Committed sweep: `nlp/eval/data/results/m3_threshold_sweep.json`.

| threshold | P | R | F1 | fp |
|---|---|---|---|---|
| 0.60 | 0.9861 | 0.9103 | **0.9467** | 1 |
| 0.67 | 0.9848 | 0.8333 | 0.9028 | 1 |
| 0.68 | 1.0000 | 0.8333 | 0.9091 | 0 |
| **0.70** | **1.0000** | **0.8077** | 0.8936 | **0** |
| 0.75 | 1.0000 | 0.7692 | 0.8696 | 0 |
| 0.85 | 1.0000 | 0.6667 | 0.8000 | 0 |

The highest-scoring false merge is 0.6753 — P079, two different
partnerships announced under one headline template — and precision reaches
1.0000 from 0.68 upward. **0.70 is the committed default.** 0.68 would sit
0.005 above a single observation; 0.70 keeps a real margin over the worst
false merge while recall still clears AC-3's floor with room. F1 peaks at
0.60 and that threshold is *not* used: it buys 0.05 of F1 with a false
merge, and a false merge is the failure the whole design exists to prevent.

### Result at the committed threshold

`nlp/eval/data/results/m2_m3_pipeline.json`, M2 then M3, 148 scored pairs:

| metric | value | AC-3 |
|---|---|---|
| precision | **1.0000** | ≥ 0.85 ✓ |
| recall | **0.8077** | ≥ 0.75 ✓ |
| F1 | 0.8936 | |
| tp / fp / tn / fn | 63 / 0 / 70 / 15 | |
| candidate recall | 1.0000 | |

**False positives: none.** Every one of the 70 hard negatives was refused,
including all 7 different-company pairs, all 10 same-template pairs, and all
20 magnitude/date/unit/sign pairs.

**False negatives: 15**, every one a semantic rewrite scoring below 0.70 —
0.42 (`P075`, "trims Reality Labs hiring plans" / "slows recruitment inside
its metaverse division") through 0.6957 (`P051`). No guard rejected a true
positive: all 15 are `below_threshold`. That is the price of the precision
margin and it is visible in the sweep — 0.60 would recover 8 of them and
cost one false merge.

**Ambiguous pairs:** 3 of 5 merged (`P150`, `P151`, `P153`). They are
excluded from the metrics by design, and which way they should go is exactly
the question K3 adjudication exists to settle.

### Cluster semantics

A story is a **clique**: every pair inside it independently cleared the
threshold, every guard, and the ±36 h window. Single-link chaining would let
A–B and B–C place A and C together without anything ever comparing them.
On top of that, each prospective story carries a combined evidence summary,
so a vague story cannot bridge two that contradict each other — the same
constant-cost trick M2 uses for its compatibility gate.

Canonical member is the earliest published, outlet then story key breaking
ties. Every member id, outlet, and source link is carried through, and
`outlet_count` unions the declared outlets with the retained links so it can
never undercount them. Merges never cross a ticker. Undated stories do not
merge by default — without a timestamp the window cannot be enforced.

No durable story id is invented: `story_fingerprint` is a change-detection
digest over the ticker and the sorted member set, exactly as M2 does.

### Boundaries

- **The encoder is injected**, never constructed. `EmbeddingService`
  satisfies `StoryEncoder` as-is. No test in this package loads a model or
  touches the network, and one asserts `sentence_transformers` is not in
  `sys.modules` after a run.
- **The embedded text is M1's composition** (`title + "\n\n" + description`),
  deliberately unlike M2, which owns its content rules: M2's identity keys
  must not move when the encoder's input changes, whereas M3's whole job is
  to ask the encoder a question and it must ask it the way everyone else
  does.
- **Encoder identity is in the fingerprint.** Same stories, same settings,
  different model is a different result.
- **`nlp/semdedup/bridge.py` is the only place M3 knows what an M2 cluster
  looks like.** Everything else takes `StoryInput`.
- An encoder failure, wrong vector count, ragged dimension, non-finite or
  zero vector raises rather than falling back to a lexical comparison.
  Silently changing which algorithm produced a merge would make the run
  unexplainable.

### Honest limitations

- **The dataset and the guards share an author.** The reported precision is
  optimistic for that reason alone: the guard families were written from the
  same failure taxonomy the negatives were written from. Real-sample
  numbers (blocked on #61/#62) should be expected to be worse, and the
  guards will need cases they have not seen.
- **The guard lexicons are English and finite.** An opposing claim phrased
  outside them ("greenlights"/"nixes") is invisible to `contrast_polarity`,
  and the threshold is the only thing left.
- **Recall is the sacrifice.** Half the semantic rewrites in the set are
  missed at the committed threshold. That is the deliberate trade, not an
  accident, and the sweep documents its price exactly.

Lint and test with:

```bash
python -m black --check nlp/semdedup tests/test_semantic_dedup.py
python -m flake8 nlp/semdedup tests/test_semantic_dedup.py
python -m pytest tests/test_semantic_dedup.py
python -m tools.eval_dedup --stage m2+m3 --precision-floor 0.85 --recall-floor 0.75
```
