# Provider observations

What the news providers actually sent, on a stated day, captured by
`tools/observe_phase0_providers.py`.

These files are **evidence, not tests**. They record one window and are
expected to be stale the moment they are committed; that is the point.
A decision that rests on provider behavior — I5's decision G on
`raw_items.external_id`, and its explicit publisher map — has to be able to
name the observations it rests on, and re-running the tool produces a new
dated pair rather than editing an old one.

Each observation is a pair:

- `i5-provider-observation-<date>.json` — the machine-readable record:
  window, code commit, library versions, feed-config hash, per-attempt
  counts, the minimal per-item observations, per-candidate identifier
  findings, the ranking that chose the field the verdict rests on, the exact
  source strings, and the equivalence table.
- `i5-provider-observation-<date>.md` — the same artifact rendered for a
  reviewer. It is generated from the JSON and holds no independent claims,
  so the two cannot drift apart.

## Re-running

```
python -m tools.observe_phase0_providers --attempts 4 --interval-seconds 2800
```

Space the attempts deliberately. Decision G's stability bar is measured per
article — the longest gap between two observations *of one canonical URL* —
so a long run of closely spaced attempts tests a short claim and earns
nothing. The artifact reports the run's own span separately, as context, and
never as evidence about an identifier.

The tool is diagnostic. It never opens a `Phase0Repository`, never writes a
row, never reads or mutates `source_state`, and nothing in `pipeline.py`
reaches it. It does issue live provider requests, so run it deliberately.

## Recomputing a verdict

`yahoo.observations` holds the minimal record each conclusion was computed
from: attempt, timestamp, ticker, response position, the candidate
identifiers, the canonical URL, the stored source, and the title. Every
number under `provider_id_candidates` follows from those rows plus the
article identity — the canonical URL — so a methodology error found later
can be re-run against the committed evidence instead of against providers
that have since moved on:

```bash
PYTHONPATH=. python tools/observe_phase0_providers.py \
  --recompute docs/observations/i5-provider-observation-2026-08-23.json
```

That touches no network. It re-derives the candidate findings, the
agreement table, and the `external_id` verdict from the retained rows,
rewrites the JSON and its markdown in place, and stamps the artifact with
a `recomputed` block naming when and at which commit. Anything the rows
cannot regenerate — the attempt log, the RSS side, the source strings, the
equivalence pairs — is carried through untouched, and a record set that no
longer accounts for every item the artifact says it observed is refused
rather than quietly recomputed against a smaller window.

## What is deliberately not here

Whole raw provider responses. Only the fields that establish a conclusion
are kept — identifiers, publisher names, URLs, titles, timestamps — each
passed through `phase0.redaction.redact_secrets` on the way in. Thumbnails,
image resolution ladders, premium-content flags, and storyline objects
establish nothing and are dropped.
