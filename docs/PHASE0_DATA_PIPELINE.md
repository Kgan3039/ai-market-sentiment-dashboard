# Phase 0 Data Pipeline

The pipeline stores immutable Yahoo Finance and RSS input in SQLite before any
deduplication, clustering, or summarization. The default local database is
`data/phase0.sqlite3`; production should pass a persistent path with
`--database`.

## Local setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r data/requirements.txt
```

## Commands

```bash
.venv/bin/python pipeline.py
.venv/bin/python pipeline.py --status
.venv/bin/python pipeline.py --replay --date 2026-07-23
.venv/bin/python pipeline.py --database /var/lib/ticker-narratives/phase0.sqlite3
```

A live run treats Yahoo ticker failures and RSS feed failures independently.
Successful sources continue and errors are written to `run_log`. Re-running
does not duplicate raw input because `(source, canonical_url)` is unique. A
partial source failure produces a degraded stage record while allowing the
remaining sources to run. If an entire provider family fails, the stage is
marked failed and the process exits nonzero after the other stages finish.

Replay retains `raw_items`, clears only derived `stories` and `themes` for the
selected date, and records the replay preparation. Deduplication, clustering,
and summarization stages should be registered sequentially in `pipeline.py` as
their Phase 0 issues land.

## Scheduling

`deploy/phase0-pipeline.cron` is the production schedule template. It runs every
30 minutes from 09:00 through 16:30 America/New_York on weekdays and hourly
otherwise. Replace `/opt/ticker-narratives`, the Python path, database path, and
log destination for the deployment host.
