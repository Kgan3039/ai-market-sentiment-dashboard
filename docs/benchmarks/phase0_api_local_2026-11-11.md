# Phase 0 API Local SQLite Benchmark - 2026-11-11

This is B1's local performance evidence. It is not private-host evidence and
does not complete the final measurement blocked by deployment issue #76.

## Command

Run from the repository root:

```bash
env -u DEBUG backend/venv/bin/python tools/phase0_api_benchmark.py \
  --database /tmp/phase0-api-benchmark-20261111-verified.sqlite3 \
  --seed-representative \
  --end-date 2026-10-30 \
  --days 30 \
  --themes 4 \
  --stories-per-theme 6 \
  --other-stories 6 \
  --requests 200 \
  --warmup 10
```

## Environment

- Mode: in-process FastAPI `TestClient` through the isolated `phase0` profile
- Hardware: Apple arm64, 10 logical CPUs
- OS: macOS 26.7.3
- Python: 3.13.12
- SQLite: 3.50.4
- Recorded at: 2026-11-11T16:56:25Z

## Dataset

The tool generated deterministic synthetic records through the real Phase 0
SQLite schema, story/theme reconciliation and summary-artifact persistence
paths. No external provider or network was used.

- Database size: 6,868,992 bytes (6.55 MiB)
- Tickers: 5
- Trading days: 30
- Raw items: 4,500
- Stories: 4,500
- Themes: 600 across 150 theme sets
- Current summary artifacts: 20
- Run-log rows: 320
- Latest-day shape per ticker: 4 themes x 6 stories, plus 6 Other Coverage
  stories

## Results

There were 200 measured requests plus 10 warmups per endpoint group. The
themes group rotated across all five supported tickers. Total measured
requests: 600.

| Endpoint group | p50 | p95 | max | Target |
|---|---:|---:|---:|---:|
| `GET /api/v1/tickers` | 24.425 ms | 28.360 ms | 44.709 ms | < 300 ms |
| `GET /api/v1/tickers/{ticker}/themes` | 20.518 ms | 21.131 ms | 50.627 ms | < 300 ms |
| `GET /api/v1/meta/status` | 2.731 ms | 2.809 ms | 2.906 ms | < 300 ms |

All three endpoint groups passed locally. Re-run the tool with `--base-url`
against the private host and attach that JSON report to B1 once #76 exists.
