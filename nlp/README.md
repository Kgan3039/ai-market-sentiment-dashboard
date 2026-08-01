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

Committed sweep: `nlp/eval/data/results/m3_threshold_sweep.json`, recomputed
from source after every guard change. Its `selection` block is generated
from the rows; nothing below is restated by hand in code.

| threshold | P | R | F1 | fp | false positives | guard-driven FN |
|---|---|---|---|---|---|---|
| 0.50 | 0.9747 | 0.9872 | **0.9809** | 2 | P079, P080 | none |
| 0.58 | 0.9867 | 0.9487 | 0.9673 | 1 | P079 | none |
| 0.65 | 0.9859 | 0.8974 | 0.9396 | 1 | P079 | none |
| **0.68** | **1.0000** | 0.8590 | 0.9241 | **0** | — | none |
| **0.70** | **1.0000** | **0.8333** | 0.9091 | **0** | — | none |
| 0.75 | 1.0000 | 0.7821 | 0.8777 | 0 | — | none |
| 0.90 | 1.0000 | 0.6282 | 0.7717 | 0 | — | none |

- **0.68 is the lowest tested threshold with zero false positives.**
  `selected_is_lowest_clean_threshold` is `false`.
- **0.70 is the provisional precision-first selection**, taken for margin.
- Highest surviving false-positive score: **0.675315** (P079).
- Margin at 0.70: **0.024685**.
- At 0.70 **all 13 false negatives are threshold-driven; none is
  guard-driven** — measured, and recorded per point as
  `guard_rejected_positive_ids` / `threshold_rejected_positive_ids`.

F1 peaks at 0.50 and is not used: it buys F1 with two false merges.
Provisional — synthetic, single-author, unadjudicated, not gate eligible.

### M2 quarantine is authoritative and survives the bridge

M2 quarantines every item under a provider identity that described two
different articles. That decision used to be dropped at the M2→M3 bridge,
so M3 could re-merge on cosine what M2 had isolated — and did, at 1.0000.

`StoryInput` now carries `quarantined_member_ids` and `provider_conflicts`,
read from the public `DedupResult` fields and never inferred. A quarantined
story is excluded from candidate generation, retained unchanged, and stamped
`semantic_skip_reason=provider_quarantine`.

**C003 is therefore not a semantic improvement.** Its two identical wire
stories are one story, but recovering them requires overruling an
authoritative conflict with a similarity score. C003 stays an under-merge
for M2+M3 exactly as for M2 alone.

### The nine guards

`article_type`, `temporal_disagreement`, `role_disagreement`,
`subject_shift`, `numeric_disagreement`, `contrast_polarity`, `negation`,
`entity_conflict`, `same_frame_different_event` — in that veto order, which
is itself a fingerprint component.

**`same_frame` and `subject_shift` read the headline only** (subject_shift
also strips a trailing attribution clause and lemma-folds its markers). A
frame is a headline template and a subject is a headline subject; reading
the standfirst as well rejected P054 and P068.

**`same_frame` compares canonicalized quantities, not raw number tokens.**
Each quantity span in the headline is replaced by one placeholder carrying
exactly the fields `quantities_conflict` treats as assertions — the counted
unit is left out, since it is unknown when absent and survives as its own
token. Without this the guard answered a question about spelling: "five
million units" and "5 million units" are one claim written two ways, but as
bare tokens `five` and `5` look like a substituted slot, and the guard
refused the pair after the quantity comparison had already accepted it. A
genuinely different quantity still yields a different placeholder, and
reaches `numeric_disagreement` first in any case — that guard is scanned
before `same_frame`, so the recorded reason stays the specific one.

### Article-type classification

Two match modes. *Anywhere* phrases identify a genre wherever they appear
("live updates", "hands on", "what to expect", "is said to"). *Anchored*
single words are genre labels only in headline position — at the start
before a delimiter, immediately before a delimiter, or at the end. Either
way a match adjacent to a corporate designator is discarded.

That combination handles Title Case without a capitalisation heuristic:

| classifies | as | | stays `report` |
|---|---|---|---|
| `Nvidia GTC: Live Updates` | live_blog | | `First Look Capital` |
| `What To Expect From Nvidia Keynote` | preview | | `Interview Corp` |
| `A First Look At Apple Headset` | hands_on | | `Preview Networks` |
| `Tesla Earnings Preview` | preview | | `Recap Media` |
| `Nvidia Interview: CEO Discusses AI Demand` | interview | | `Company confirms earnings date` |
| `Apple Product Review` | hands_on | | `CEO confirms guidance` |
| `Live Blog: Meta Developer Conference` | live_blog | | `the review board` |
| `AMD Launch Recap` | recap | | `analysts review results` |
| | | | `live operations` |

Each classifies identically in sentence case. `confirmation` requires a
confirming-a-prior-report shape, so a bare "confirms" is ordinary copy.

### Explicit entity evidence, context-anchored

A capitalised run counts **only** when a context puts it in a named slot:
an appointment verb, a counterparty preposition (`partnership with`,
`acquires`), a role word (`CEO X`), an analyst-action verb, or a corporate
designator of its own. Outlet names come from M2's versioned publisher list
and are excluded outright; headline scaffolding is filtered; a leading
possessive is stripped.

Roles are compared **separately** — an appointee conflicts with an
appointee, a counterparty with a counterparty, never across. Missing entity
evidence is unknown, not contradictory.

| vetoes | does not veto |
|---|---|
| `Alice Smith` vs `Bob Jones` appointed | `New York Times` vs `Wall Street Journal` reporting one event |
| `acquires Beta Corp` vs `Gamma Corp` | `Company Reports Strong Results` vs `Company Posts Strong Results` |
| `partnership with Company Alpha` vs `Company Beta` | `Apple's App Store` vs `App Store` |

**Documented limitation, unchanged:** single-token organisation names are
out of scope. `Acme acquires Beta` (no designator) is not caught; adding a
broad single-token heuristic would misread ordinary headline casing.

### Structured quantities

A quantity is decomposed into approximation, sign, value, range-ness,
magnitude, currency, percent kind and counted unit, and each field compares
on its own. Only the **unit** is unknown-if-absent — a record naming no unit
does not contradict one that does, which is what lets "495,000 vehicles" and
"495,000 cars" merge.

| equivalent | distinct |
|---|---|
| `eleven units` ≡ `11 units` | `about 5 units` ≠ `5 units` |
| `twenty-one vehicles` ≡ `21 vehicles` | `5 million units` ≠ `5 million dollars` |
| `dozen chips` ≡ `12 chips` | `5 million` ≠ `5 billion` |
| `one hundred users` ≡ `100 users` | `at least 5` ≠ `up to 5` |
| `five million units` ≡ `5 million units` | `5-10 units` ≠ `5 units` |
| `495,000 vehicles` ≡ `495,000 cars` | `$5 million` ≠ `5 million users` |
| | `5%` ≠ `5 basis points` |

Quarter and year context is carried by the temporal guard, so `Q1 5 million
units` and `Q2 5 million units` differ there. Names and identifiers stay
protected: `One Medical`, `Formula One`, `MI400`, `H100`, `Model 3`.

Approximation qualifiers match on **token boundaries**. They did not, and
`over` inside `handovers` turned an exact delivery figure into "more than",
rejecting P055.

### Result at the committed threshold

`m2_m3_pipeline.json`, M2 then M3, 151 scored pairs:

| metric | value | AC-3 |
|---|---|---|
| precision | **1.0000** | ≥ 0.85 ✓ |
| recall | **0.8333** | ≥ 0.75 ✓ |
| F1 | 0.9091 | |
| tp / fp / tn / fn | 65 / 0 / 73 / 13 | |
| complete / failed | true / 0 | |

**False positives: none.** **False negatives: 13**, all threshold-driven:
P049, P050, P051, P057, P058, P059, P060, P065, P067, P072, P073, P075,
P078.

### Multi-item cluster results

`m2_m3_clusters_ground_truth.json`, nine cases against ground truth:

| | M2 alone | M2 + M3 |
|---|---|---|
| exact partition match | 6/9 | **7/9** |
| co-clustering P / R / F1 | 1.0000 / 0.5882 / 0.7407 | **1.0000 / 0.7647 / 0.8667** |
| over-merged | none | **none** |
| under-merged | C003, C006, C007 | **C003, C006** |
| permutation failures | none | none |

M3 recovers C007. C003 is held by quarantine and C006 by the threshold.

### Reproducibility

Every M3 artifact carries a `semantic_metadata` block — model name,
revision, embedding dimension, semantic input composition, threshold, time
window, frame-overlap threshold, candidate capacity, guard ordering,
evidence policy version and fingerprint, the cluster-compatibility policy,
and the semantic config fingerprint the run actually used — beside the
trust contract and summary, dataset id and schema version, complete
confusion accounting with FP/FN ids, the guard-driven and threshold-driven
split, quarantine-skip ids, and complete/evaluated/failed counts.

Cluster-wide compatibility is its own fingerprint component
(`cluster_compatibility.*`: linkage, compatibility scope, evidence
combination, quarantine policy, edge order, window scope, candidate
generation), so changing the linkage rule moves the digest without touching
`ALGORITHM_VERSION`.

Scores serialize at six decimal places, four orders of magnitude finer than
the tightest margin in the sweep. Artifacts are equal to within that
precision across model executions — **not byte-identical**.

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

## Phase 0 theme clustering (M5, issue #72)

`nlp.themes` groups one ticker-day's canonical stories into 2-6
salience-ranked themes plus "Other coverage".

```python
from datetime import date
from nlp.embeddings import EmbeddingService
from nlp.themes import ThemeConfig, cluster_themes

themes = cluster_themes(
    stories,                       # canonical stories, after M2 and M3
    ticker="NVDA",
    trading_day=date(2026, 3, 5),
    config=ThemeConfig(supported_tickers=TICKERS),
    encoder=EmbeddingService(),
    previous_themes=last_run,      # optional; keeps unchanged themes' identity
)
```

**This does not close issue #72.** The DoD asks for AC-4 demonstrated on
three *real* ticker-days. Real days need I2 (#61), I3 (#62) and I1 (#57),
none of which are on main. The three committed days are authored to the same
shape and marked synthetic; they demonstrate the behaviour, not the DoD.

### No story disappears

Every input comes back in exactly one of three places — a theme,
`other_coverage`, or `excluded` with a stated reason — and
`cluster_themes` asserts that partition before returning. `excluded` is
currently only ever "no encodable text". A story that is in none of the
three is a bug the function raises on rather than ships.

### Algorithm, and when it gives way

HDBSCAN over story embeddings on a **precomputed cosine-distance matrix**,
with the agglomerative fallback issue #72 allows. "Unstable" is given a
checkable meaning rather than left to judgement — HDBSCAN gives way when:

1. it finds a number of clusters outside AC-4's 2-6 band, which at Phase 0
   volumes it often does; or
2. one of its clusters is looser than `min_theme_cohesion` (0.40).

The second case is the one that bites. On the committed TSLA day HDBSCAN
under-split 18 stories into a seven-story grab-bag at cohesion 0.351 holding
the robotaxi permit, the Berlin factory restart, the Nevada battery line and
the investor-day notice together. That is the "one giant catch-all cluster"
a reader must never be shown as a theme. The fallback picks a cluster count
in the band by silhouette (ties to the smaller count) and produced five
themes at 0.42-0.58.

Below four stories the day is not clustered at all (AC-4): stories are
listed individually under "Other coverage". A group of fewer than
`min_theme_stories` (2) is coverage, not a theme, and goes the same way — so
a day never shows a "theme" of one story.

### Coherence: exactly one contradiction transfers from M3

A theme is coarser than a story. Different quarters, different magnitudes
and different people legitimately share an earnings theme, so applying M3's
full guard set here would shred every theme into singletons. Exactly one
family transfers: **opposing claims**. A theme holding both "raised
guidance" and "cut guidance" cannot be summarized in 2-4 faithful sentences.
The minority side moves to "Other coverage" — majority by story count, ties
to the side holding the theme's leading story — and every move is reported
in `method_reason`.

### Salience and stability

`salience = f(story count, outlet diversity, recency)`, each component
normalized within the ticker-day so a quiet day's top theme is not penalised
against a busy one's. Recency decays with a 6-hour half-life measured
against **the day's own most recent story, never a clock** — replaying a
stored day (AC-8) must rank it the same way next year. Ranking ties break on
the earliest story, then the fingerprint, so equal-salience themes never
swap places between runs. `SalienceFeatures` reports every input, because
"why is this theme first" is a question a reviewer asks about a ranked list.

Theme identity survives across runs by centroid matching: one-to-one, greedy
from the most similar pair, at cosine ≥ 0.90. A theme that gained a story
keeps its `theme_key` while its `fingerprint` moves — which is exactly the
signal a reconciler needs. A genuinely new theme gets a new key.

### Measured on the three committed days

`nlp/themes/data/results/theme_quality.json`, real encoder:

| | AAPL 03-04 | NVDA 03-05 | TSLA 03-06 |
|---|---|---|---|
| stories | 3 | 9 | 18 |
| method | small-n fallback | hdbscan | agglomerative |
| themes | 0 | 3 | 5 |
| other coverage | 3 | 3 | 1 |
| excluded | 0 | 0 | 0 |
| theme coverage | n/a | 0.667 | 0.944 |
| mean cohesion | n/a | 0.729 | 0.479 |
| max inter-theme similarity | n/a | 0.583 | 0.617 |
| AC-4 shape | yes | yes | yes |
| no story lost | yes | yes | yes |
| permutation stable | yes | yes | yes |
| re-run keeps identities | yes | yes | yes |

### Honest limitations

- **Unsupervised clustering has no objective truth, and these numbers do not
  claim any.** They measure what is mechanically checkable: coverage, shape,
  cohesion versus separation, stability. A day can score perfectly here and
  still group two stories a reader would separate — which is what the K3
  (#60) human review exists for, and it is not written yet.
- **Two of the five TSLA themes are imperfect.** The top one puts the
  battery line and the supercharger corridor with the delivery report; the
  fourth puts the investor-day notice with the Berlin factory. Both are
  above the cohesion floor and both would survive to a reader. Tuning the
  parameters until this particular fixture looked clean would be fitting to
  a fixture I wrote, so it has not been done.
- **Membership is fragile under perturbation; identity is not.** Dropping
  one story from the TSLA day re-runs the silhouette choice and retains only
  20% of the exact memberships — but centroid matching retained 100% of the
  theme identities, which is the property AC-4 actually asks for. On the
  HDBSCAN days membership retention was 0.67-1.00.
- **`min_theme_cohesion` was calibrated on these three days.** Real days
  will need it re-measured.
- **The fixture, the guards, and the thresholds share an author**, exactly
  as in M4, so the results are optimistic.

### Boundaries

- **M5 calls no LLM and adds no retrieval framework.** No RAG, no LangChain,
  no LangGraph, no MCP. It prepares the closed evidence set the citation-safe
  summarizer (#65/#80) consumes and stops. A test asserts none of those
  modules is importable into the stage.
- `ThemeEvidence` is a projection carrying only what the citation contract
  resolves. `Theme.citable_item_ids` is what a citation may resolve to, and
  a test asserts the sets are disjoint across themes — a citation cannot
  reach outside its own theme.
- **`trading_day` is an argument, not a derivation.** Which trading day a
  story belongs to is a calendar question owned by #57/#68; guessing it here
  would file a story under the wrong day silently.
- No durable theme id is invented. `fingerprint` is a content digest;
  `themes.id` belongs to issue #57.
- The encoder is injected. No test in this package loads a model or touches
  the network.

Lint and test with:

```bash
python -m black --check nlp/themes tools/eval_themes.py tests/test_theme_clustering.py
python -m flake8 nlp/themes tools/eval_themes.py tests/test_theme_clustering.py
python -m pytest tests/test_theme_clustering.py
python -m tools.eval_themes
```
