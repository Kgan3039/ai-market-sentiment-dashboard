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
  counts, per-candidate identifier findings, the exact source strings, and
  the equivalence table.
- `i5-provider-observation-<date>.md` — the same artifact rendered for a
  reviewer. It is generated from the JSON and holds no independent claims,
  so the two cannot drift apart.

## Re-running

```
python -m tools.observe_phase0_providers --attempts 4 --interval-seconds 300
```

The tool is diagnostic. It never opens a `Phase0Repository`, never writes a
row, never reads or mutates `source_state`, and nothing in `pipeline.py`
reaches it. It does issue live provider requests, so run it deliberately.

## What is deliberately not here

Whole raw provider responses. Only the fields that establish a conclusion
are kept — identifiers, publisher names, URLs, titles, timestamps — each
passed through `phase0.redaction.redact_secrets` on the way in. Thumbnails,
image resolution ladders, premium-content flags, and storyline objects
establish nothing and are dropped.
