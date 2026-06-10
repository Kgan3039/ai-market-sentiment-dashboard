# Phase 0 Experiment: Is FinBERT-on-yfinance-headlines foundation-grade?

This experiment answers ONE question before any new dashboard features are
built: does FinBERT sentiment on yfinance headlines produce enough
**variation** (the number moves day to day), **freshness** (the headline feed
actually refreshes), and **trustworthiness** (humans agree with the scores) to
be the foundation of sentiment-history and watchlist features?

It is NOT testing predictive alpha. A signal can have zero alpha and still be
a product; a signal that fails these gates cannot be saved by anything.

No new endpoints. No frontend. Two scripts and this README.

## What it reuses (and what it deliberately avoids)

- Headlines come from `DataService._fetch_yfinance_headlines()` — the **raw**
  provider call. We intentionally do NOT use `DataService.get_headlines()`,
  because that method silently substitutes committed demo headlines and
  cached/stale results on failure, which would fake the freshness and
  turnover numbers. In this experiment, a provider failure is recorded as a
  failure — that's the platform-stability signal we want.
- Scoring uses `nlp/sentiment.py`'s classifier directly, recording **both**
  raw FinBERT probabilities and the repo's adjusted production scores
  (`_build_sentiment_result` applies neutral-boosting/confidence-cap
  heuristics on top of raw FinBERT). If the neutral gate fails, the raw-vs-
  adjusted comparison tells you whether FinBERT or our post-processing is to
  blame.
- **Scorer-mode trap:** `nlp/sentiment.py` silently falls back to a keyword
  heuristic if `transformers` isn't installed. Every record is stamped with
  `scorer_mode`; the report's first gate fails if <95% of records came from
  real FinBERT. If you see the keyword-fallback warning at collection time,
  fix the environment (`pip install transformers torch`) — don't keep
  collecting.

## How to run

**Daily collection** (once per day, at a consistent time — pick one, e.g.
3:50pm ET, and never vary it; freshness is measured against a 24h window):

    python experiments/collect_daily.py

Re-running the same day skips already-collected tickers (`--force` to redo).
Cron example (weekdays 3:50pm, from repo root):

    50 15 * * 1-5  cd /path/to/repo && python experiments/collect_daily.py >> experiments/data/collect.log 2>&1

If nobody has an always-on machine, a calendar reminder and a laptop is fine —
consistency of time-of-day matters more than automation. Commit
`experiments/data/` to the repo after each run; the dataset is the deliverable.

**Report** (any time; gates show INSUFFICIENT until ~5 collection days):

    python experiments/report.py

**Trust gate — manual, do this once around day 3–4:**

    python experiments/report.py --export-labels 150

This writes `data/labels_sheet.csv` (shuffled, no model answers). Two team
members independently fill in positive/negative/neutral WITHOUT looking at
model output (~1 hour each). Then compare to model labels by joining on
`headline_id` against `headlines.jsonl`.

The ten tickers (NVDA, TSLA, AAPL, AMD, PLTR, META, JPM, WMT, KO, XOM) are
fixed for the experiment. Do not add or remove tickers mid-run. Note the
deliberate split: the boring half (JPM, WMT, KO, XOM) is the real test —
the megacaps will always have headlines.

## Gates: success vs failure (pre-committed — do not tune after data arrives)

Run for **10 trading days**, then judge. Anything in the gray zone between
the pass and fail thresholds counts as FAIL for the "build now" decision.

| # | Gate | PASS | FAIL |
|---|------|------|------|
| 0 | Scorer integrity | >=95% of records scored by real FinBERT | any meaningful share of keyword-fallback records |
| 1 | Coverage | >=70% of ticker-days have >=3 fresh (<24h) headlines — check the JPM/WMT/KO/XOM cohort separately | <50% overall, or the boring cohort starves (<40%) |
| 2 | Turnover | median day-over-day headline-set overlap <=50% | >70% (deltas would be computed on a stale window) |
| 3 | Movement above noise | >=30% of day-over-day deltas exceed their bootstrap 95% noise band | <15% (the daily delta is sampling jitter, not signal) |
| 4 | No neutral collapse | <=60% of fresh headlines labeled neutral | >75% (scores huddle at 0; deltas are meaningless) |
| 5 | Trust (manual) | FinBERT agrees with human-majority label on >=70% of non-neutral headlines, AND the two humans agree with each other >=80% | <60% model-human agreement, or humans can't agree (task ill-posed) |

Soft check alongside gate 5: each week, the team reads the report's "biggest
movers" section. On >=7 of 10 days the arrows should make sense given the
headlines. If they don't, trust fails regardless of the numbers.

## What each outcome means (decided before day one)

- **All gates pass** → green-light the Sentiment Shift feature. The collector
  becomes the history service; the experiment's JSONL is day 1–10 of
  production sentiment history. Nothing is thrown away.
- **Coverage/turnover fail, model passes** → the NLP is fine; the *source*
  can't carry a daily product. Either restrict to a published list of
  high-coverage tickers, or make richer sourcing (news API / the stalled
  Reddit pipeline) the prerequisite milestone. Do not ship daily deltas on a
  feed that doesn't refresh daily.
- **Movement fails, everything else passes** → kill the numeric delta badge,
  not the thesis. Ship categories ("notably negative news day") and
  per-headline tags, and/or move to weekly deltas — which changes the
  retention design and must trigger a rethink.
- **Trust fails (gate 5) or neutral collapse (gate 4)** → scorer problem.
  Compare raw vs adjusted neutral shares in the report: if raw FinBERT is
  fine but adjusted collapses, fix `_build_sentiment_result`'s heuristics;
  if raw collapses, bake off alternatives against the same 150-headline
  sheet (different finance-tuned model, or LLM-scored headlines) and rerun
  only this gate.
- **Gate 0 fails or yfinance breaks mid-run** → environment/platform problem.
  A 10-day, 10-ticker run failing on platform stability is a free and very
  loud warning about building a product on an unofficial scraper.

## Files

    experiments/
      collect_daily.py        daily collector (run this)
      report.py               metrics, gates, movers, label-sheet export
      README.md               this file
      data/
        headlines.jsonl       one record per (day, ticker, deduped headline)
        daily_summary.jsonl   one record per (day, ticker)
        headline_index.json   headline_id -> first_seen_date
        labels_sheet.csv      blind labeling sheet (after --export-labels)
