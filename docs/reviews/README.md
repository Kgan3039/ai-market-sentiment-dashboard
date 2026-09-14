# Phase 0 review sheets (A4, issue #74)

Section 8 of [`PHASE_0_SPEC.md`](../PHASE_0_SPEC.md) reads two gates off
human review sheets: **G1** theme-assignment agreement (≥ 75% over ≥ 80
sampled assignments) and **G2** summary-sentence faithfulness (≥ 95%).
`tools/make_review_sheets.py` draws the samples and computes the numbers
from completed sheets.

This directory holds the sheets and their manifests once real rounds are run.
Nothing is committed here yet: no soak-window theme sets exist, K3 (#60) has
not ratified a review protocol, and A2/A3 do not yet persist summaries. The
tooling is built so that none of those facts can be papered over by a
good-looking number, an edited file, or a well-meaning claim.

**Scope today: G1 only (A4a).** G2 sampling waits for A2/A3 to persist the
summaries a reader actually sees; sampling summaries generated at review time
would measure the model, not the product, so the tool does not do it.

## What a G1 row is

One story's placement inside one **persisted** theme set — into a theme, into
"Other coverage", or excluded — read from the Phase 0 database in one
snapshot per partition (`Phase0Reader.theme_population`) exactly as the
`themes` stage stored it, over the *current* authoritative story generation.
The sampler never re-runs clustering.

A partition is reviewable only when its theme set is a valid view of the live
story generation: every live story is placed exactly once and every story is
healthy `m3.semantic` output. Anything else is recorded under
`selection.skipped_partitions` with a reason and is never reviewed as M5
output:

| reason | meaning |
|---|---|
| `m2_only_degraded` | stories exist at `m2.exact`, no theme set (decision H) |
| `themes_not_generated` | healthy stories, no theme set persisted |
| `mixed_story_stages` | stories of mixed stages, no theme set |
| `theme_set_without_live_stories` | a theme set persists but its stories are gone: stale |
| `theme_set_over_degraded_stories` | a theme set persists over `m2.exact` stories: stale |
| `theme_set_inconsistent_with_stories` | the set does not account for exactly the live stories, or recorded a different input count |
| `provenance_identifier_credential` | a provenance identifier carries credential-like text (field named, value withheld) |
| `no_story_output` | a requested ticker holds nothing on that day |

### The build binding is unknown, and said so

Nothing persisted records which story generation a theme set was built over:
`reconcile_themes` verifies the signature inside its transaction and does not
keep it, and `theme_sets.source_metadata` holds counts and model identity,
not a signature. So each theme set's provenance carries three fields:

| field | value for a database set |
|---|---|
| `story_generation_signature_current` | the generation the partition holds *now* |
| `theme_build_story_generation_signature` | `null` — unavailable |
| `generation_binding` | `unverified` |

The current signature is reported as current and proves nothing about the
build: stories mutated in place with the same ids, stages and memberships
change the current signature and nothing else, and the set is still not
claimed to have been built over them. A generation-unverified set may be
sampled for development review, but `unverified` is a release blocker on its
own, independent of origin and protocol. `verified` has no producer until
persistence keeps the build-time signature. A fixture set is clustered
in-process, so its binding is `in_process` (build and current are one act).

### Provenance identifiers are refused, not rewritten

Identifier fields — `pipeline_version`, `ticker`, `method`,
`config_fingerprint`, `algorithm_version`, `model_name`, `model_revision`,
`theme_key`, `label_source` — are compared and digested, so rewriting one
would corrupt the identity it names. They are checked with the project's
credential detector (`phase0.redaction.contains_credential`); a partition
whose identifier carries credential-like text is skipped with the *field*
named and the value withheld. Free-form context (titles, descriptions, URLs,
outlets, labels) continues to be redacted.

## Files

Every sample is two files, written together:

| file | mutability | purpose |
|---|---|---|
| `<name>.csv` | reviewers fill the four trailing columns | the sheet |
| `<name>.manifest.json` | immutable | the captured snapshot: what was drawn, from what, and what a review is *of* |

### The snapshot is the review boundary

The manifest's `snapshot` holds, for every sampled row, **every non-reviewer
column** the reviewer saw — identity, title, description, canonical URL,
outlets, stage, theme label and its source, theme story count, sibling
titles, placement reason — plus the theme-set provenance behind it (config
fingerprint, algorithm version, model name/revision/dimension, method, story
generation signature). `snapshot.sha256` is a full SHA-256 over the canonical
JSON of all of it.

- `read_manifest` recomputes every `row_id` from its identity and the snapshot
  digest from the snapshot, and refuses a manifest where either disagrees.
- A completed sheet is held to the snapshot **column by column**: editing a
  title, label, description, sibling list, or any identity field is refused.
- Scoring never consults the database. Changing the live data after a sample
  was drawn does not change what the review was of, and does not invalidate
  it.
- The SQLite main file's bytes are *not* the boundary — the database runs in
  WAL mode, so they are not even stable — and no file digest is recorded.

### Sheets are bound to the exact artifact they were cut from

Every row of a blank sheet carries two non-reviewer columns, `manifest_id`
(SHA-256 over the manifest's content, sheet and binding blocks excluded) and
`snapshot_sha256`. A completed sheet must carry the scored manifest's values
on every row. So a review completed against snapshot A cannot be handed in
against a re-authored manifest B — even one whose visible rows are identical
and whose digests were all recomputed — because B has a different identity
and the sheet says which one it belongs to. `read_manifest` also recomputes
`manifest_id`, so a manifest edited anywhere after its sheet was cut is
refused.

What this does not do: sign anything. A manifest re-authored *before* any
review, with its sheet recut, is a different artifact rather than a forgery
of this one; the scorecard names the manifest and snapshot digests it scored
so the substitution is visible in the record, and nothing re-authored can
gain eligibility (see below).

### The selection is recomputed, not read

`selection.digest` is one function
(`nlp.eval.review.selection_digest`) over the trading days, tickers,
pipeline versions, theme-set ids, source mode, theme-set provenance and
skipped partitions. `read_manifest` recomputes it and requires equality, and
then holds the snapshot to the selection: every snapshot row and theme set
must lie inside the selected days, tickers, versions and theme-set ids, and
the selection may not name theme sets the snapshot does not carry.

### Sheet columns

Identity (do not edit): `row_id`, `theme_set_id`, `pipeline_version`,
`ticker`, `trading_day`, `story_id`, `assignment_type`, `theme_key`,
`placement_reason`. `row_id` is `g1-` + SHA-256 over these.

Context (do not edit; captured in the snapshot): `story_title`,
`story_description`, `story_canonical_url`, `story_outlets`, `story_stage`,
`theme_label`, `theme_label_source`, `theme_story_count`,
`sibling_story_titles`.

Reviewer columns: `reviewer_id`, `reviewed_at`, `reviewer_verdict`,
`reviewer_notes`. One sheet is one reviewer's work; a second reviewer gets
their own copy of the blank sheet. Every verdict must carry a `reviewer_id`,
so K3/K4 can check the "not your own stage" rule — the tool records identity
and does not try to enforce independence it cannot verify.

All human-facing context passes through `phase0.redaction.redact_text`
before it is written, so a credential that reached a provider description
does not reach a sheet or a manifest.

### Adjudication sheet

When two reviewers disagree on a row, a third party records the final call in
a separate CSV: `row_id`, `final_verdict`, `adjudicator_id`, `adjudicated_at`,
`adjudication_notes`.

The **adjudication state** is derived from what happened, never from whether
a file was supplied:

| state | meaning |
|---|---|
| `not_applicable` | one reviewer |
| `unanimous` | two reviewers, no disagreement |
| `resolved` | every disagreement has a final verdict |
| `open` | a disagreement has no final verdict |

Which of these counts as *adjudicated* for the gate is a property of the
ratified protocol (`Protocol.adjudicated_states`), and the provisional
protocol counts none. An empty adjudication file changes nothing.

## Origin

A row that came out of SQLite is **not** thereby real ingested evidence.
`Phase0Admin.insert_raw_items` writes rows indistinguishable from fetched
ones, every test does so, and `raw_items` carries no link to the run that
fetched it. So origin is derived from *how the sample was read* and from
nothing anyone says:

| `origin.status` | source | `trust_contract` | can be gate eligible |
|---|---|---|---|
| `synthetic` | `--fixture` | `synthetic_development` | no |
| `unverified` | any `--database` | **none** — no dataset kind is truthful | no |
| `verified_live` | *nothing produces this today* | `sampled_production` | yes, with everything else |

`verified_live` needs persistence to link each raw item structurally to the
fetch run that wrote it, and then a reviewed change to
`nlp.eval.review.classify_origin`. Until then every database sample is
`unverified`, and the scorecard's banner says exactly that rather than
calling the data synthetic.

`--attested-by` / `--attestation` record an operator's statement on the
manifest under `operator_attestation`, with `effect: none`. It is audit
metadata. It does not change origin, dataset kind, eligibility, or result.

## Scorecard: four facts, kept apart

```
threshold_met    bool | None   rate >= threshold, over resolved rows; None if none resolved
review_complete  bool          every sampled row resolved; sheets match the snapshot;
                               >= 80 unique assignments across linked rounds; no shortfall
gate_eligible    bool          origin verified_live + ratified protocol + two reviewers
                               + an adjudication state the protocol counts + no overrides
gate_result      PASS | FAIL | INCOMPLETE | NOT_ELIGIBLE
```

Precedence: not eligible → `NOT_ELIGIBLE`; not complete → `INCOMPLETE`;
else `PASS` / `FAIL` on the threshold. There is no `meets_gate` boolean.

A4 reports one gate. The Phase 0 GO / NO-GO decision combines G1–G7 and
Q1–Q3 and is K4's (#77), not this tool's.

### What is believed, and what is not

Scoring takes **only** manifests and completed sheets (and adjudication
sheets). From them, and from code:

| fact | authority |
|---|---|
| origin | `classify_origin(manifest.source)` — the source *mode*, in code |
| protocol ratification and vocabulary | `nlp.eval.review.RATIFIED_PROTOCOLS`, in code; the manifest's `labeling_protocol.id` is only a lookup key, and an unknown id scores as unratified |
| reviewer count and ids | the sheets actually parsed |
| adjudication state | the disagreements actually found |
| population compatibility | manifest digests recomputed and compared |
| threshold and floor | `RELEASE_G1_THRESHOLD = 0.75`, `RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS = 80`, in code |

A `RoundResult` is a report of one round's scoring. Before it counts toward
a gate it is re-derived from the artifacts it names — the manifest re-read
and re-verified, the sheets re-parsed and re-scored — and must match field
for field. There is no way to read a round result back from disk as input.

### Release requirements are the spec's

`--development-threshold` and `--development-required-unique` exist for
development review. Using either makes the evaluation a *development* one:
`evaluation_mode: development`, a blocker naming the overrides, and
`NOT_ELIGIBLE` regardless of everything else. Values must be finite, the
threshold within `[0, 1]`, the count a positive integer.

### 40 per round, 80 to release, linked

Issue #74 asks for rounds of 40; section 8 releases G1 on ≥ 80 sampled
assignments. Rounds combine only when they are provably one review of one
population:

- same gate, same population digest (full context, every row), same selection
  provenance, same source mode, same protocol id;
- an explicit prior-round chain: each round after the first must have been
  drawn with `--exclude-manifest` naming every earlier round, and the
  manifest records those rounds' manifest and snapshot digests at draw time;
- no row appears in two rounds.

Two unlinked draws from one population are refused — they cannot show they
are not the same forty twice. A story re-placed, re-titled, or re-clustered
between rounds changes the population digest, so those rounds are
incompatible rather than "more unique rows". Fixture and persisted rounds
never combine.

## Commands

```
# draw round 1 from what the themes stage persisted for a day
python -m tools.make_review_sheets sample-assignments \
    --database "$PHASE0_DATABASE_PATH" --day 2026-09-10 \
    --seed phase0-g1-r1 --round-id r1 --out docs/reviews/g1/r1.csv

# round 2, non-overlapping and linked
python -m tools.make_review_sheets sample-assignments \
    --database "$PHASE0_DATABASE_PATH" --day 2026-09-10 \
    --seed phase0-g1-r2 --round-id r2 \
    --exclude-manifest docs/reviews/g1/r1.manifest.json --out docs/reviews/g1/r2.csv

# the gate, from the artifacts themselves
python -m tools.make_review_sheets score-assignments \
    --round docs/reviews/g1/r1.manifest.json docs/reviews/g1/r1.alice.csv docs/reviews/g1/r1.bob.csv \
    --adjudication docs/reviews/g1/r1.manifest.json docs/reviews/g1/r1.adjudicated.csv \
    --round docs/reviews/g1/r2.manifest.json docs/reviews/g1/r2.alice.csv docs/reviews/g1/r2.bob.csv \
    --report docs/reviews/g1/scorecard.json
```

`--fixture` is the one development substitute: it clusters the committed M5
fixture offline and the sample is classified `synthetic`. It is never used
unless asked for, and nothing drawn from it can reach `PASS`.

Exit status: `0` PASS, `1` FAIL, `2` usage or input error, `3` INCOMPLETE or
NOT_ELIGIBLE. The last two share a code: both mean no gate verdict is
available, the scorecard text says which, and a script that branches on
eligibility is making a decision that belongs to K4.

## What is still open

- **No soak-window theme sets exist.** `pipeline.py` does not yet register
  the `stories`/`themes` stages (`DOWNSTREAM_STAGES = ()`), so nothing
  scheduled writes what this tool samples.
- **Origin cannot be verified from persistence.** `verified_live` needs a
  structural fetch-run link on raw items. Until it exists, every database
  sample is `unverified` and no G1 release verdict is reachable.
- **K3 (#60)** owns the vocabulary, the adjudication rule, and the
  reviewer-independence rule; the provisional protocol counts nothing as
  adjudicated.
- **G2 (A4b)** waits on A2/A3 persisting summaries with per-sentence
  citations.
