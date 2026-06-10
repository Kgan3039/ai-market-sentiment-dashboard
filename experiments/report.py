#!/usr/bin/env python3
"""Phase 0 experiment: report generator for collected headline-sentiment data.

Reads experiments/data/headlines.jsonl (written by collect_daily.py) and
prints a human-readable report answering one question: is FinBERT sentiment
on yfinance headlines dynamic, fresh, and trustworthy enough to build
sentiment-history features on?

Computes, per ticker/day:
  - headline count and fresh-headline count
  - mean and variance of (adjusted) sentiment over fresh headlines
  - day-over-day change in mean sentiment
  - headline-set overlap (Jaccard) vs the previous collection day
  - bootstrap noise band for the daily mean, and whether the day-over-day
    delta exceeds it (i.e., is the movement real or sampling jitter?)

Then evaluates the pre-committed PASS/FAIL gates (see experiments/README.md).

Usage:
  python experiments/report.py                 # full report
  python experiments/report.py --export-labels 150
      # also writes data/labels_sheet.csv: a shuffled, model-answer-free CSV
      # for two humans to blind-label (trust gate)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
HEADLINES_PATH = DATA_DIR / "headlines.jsonl"
LABELS_PATH = DATA_DIR / "labels_sheet.csv"

# --- Pre-committed gate thresholds (do not tune after data starts arriving) ---
MIN_DAYS_FOR_GATES = 5          # turnover/movement gates need history
COVERAGE_MIN_FRESH = 3          # a "covered" ticker-day has >= this many fresh headlines
COVERAGE_PASS = 0.70            # >=70% of ticker-days covered
OVERLAP_PASS = 0.50             # median day-over-day Jaccard overlap <= 50%
MOVEMENT_PASS = 0.30            # >=30% of eligible deltas exceed their noise band
NEUTRAL_COLLAPSE_FAIL = 0.60    # >60% of fresh headlines labeled neutral = collapse
BOOTSTRAP_ITERS = 1000
RNG_SEED = 7


def load_records() -> list[dict]:
    if not HEADLINES_PATH.exists():
        raise SystemExit(f"No data at {HEADLINES_PATH}. Run collect_daily.py first.")
    records = []
    with HEADLINES_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bootstrap_halfwidth(scores: list[float], rng: random.Random) -> float:
    """95% bootstrap half-width of the mean of `scores`."""
    if len(scores) < 2:
        return float("inf")  # one headline -> the mean has no stability at all
    means = sorted(
        statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(BOOTSTRAP_ITERS)
    )
    lo = means[int(0.025 * BOOTSTRAP_ITERS)]
    hi = means[int(0.975 * BOOTSTRAP_ITERS)]
    return (hi - lo) / 2.0


def build_panel(records: list[dict]) -> dict:
    """Group records into panel[(ticker, date)] = {...metrics...}."""
    by_day = defaultdict(list)
    for rec in records:
        by_day[(rec["ticker"], rec["run_date"])].append(rec)

    rng = random.Random(RNG_SEED)
    panel = {}
    for (ticker, date), recs in by_day.items():
        fresh = [r for r in recs if r.get("is_fresh")]
        fresh_scores = [r["sentiment_score"] for r in fresh]
        panel[(ticker, date)] = {
            "ticker": ticker,
            "date": date,
            "scorer_modes": {r.get("scorer_mode", "unknown") for r in recs},
            "headline_ids": {r["headline_id"] for r in recs},
            "count": len(recs),
            "fresh_count": len(fresh),
            "mean": statistics.mean(fresh_scores) if fresh_scores else None,
            "std": statistics.stdev(fresh_scores) if len(fresh_scores) >= 2 else None,
            "noise_hw": bootstrap_halfwidth(fresh_scores, rng) if fresh_scores else None,
            "labels": [r["sentiment_label"] for r in fresh],
            "raw_neutrals": [
                r for r in fresh if r.get("raw_neutral") is not None
                and r["raw_neutral"] >= max(r["raw_positive"], r["raw_negative"])
            ],
            "extremes": sorted(fresh, key=lambda r: r["sentiment_score"]),
        }
    return panel


def attach_deltas(panel: dict) -> None:
    """Compute day-over-day delta, overlap, and noise-significance in place."""
    by_ticker = defaultdict(list)
    for key in panel:
        by_ticker[key[0]].append(key[1])
    for ticker, dates in by_ticker.items():
        dates.sort()
        for prev_date, date in zip(dates, dates[1:]):
            cur, prev = panel[(ticker, date)], panel[(ticker, prev_date)]
            inter = len(cur["headline_ids"] & prev["headline_ids"])
            union = len(cur["headline_ids"] | prev["headline_ids"])
            cur["overlap"] = inter / union if union else None
            if cur["mean"] is not None and prev["mean"] is not None:
                cur["delta"] = cur["mean"] - prev["mean"]
                if cur["noise_hw"] is not None and prev["noise_hw"] is not None:
                    pooled = (cur["noise_hw"] ** 2 + prev["noise_hw"] ** 2) ** 0.5
                    cur["pooled_noise"] = pooled
                    cur["significant"] = abs(cur["delta"]) > pooled


def evaluate_gates(panel: dict, records: list[dict]) -> list[tuple[str, str, str]]:
    """Return [(gate_name, status, detail)] with status PASS/FAIL/INSUFFICIENT."""
    gates = []
    days = sorted({d for _, d in panel})
    n_days = len(days)

    # Gate 0: scorer integrity -- everything else is meaningless without it
    finbert_recs = sum(1 for r in records if r.get("scorer_mode") == "finbert")
    share = finbert_recs / len(records) if records else 0.0
    gates.append((
        "Scorer integrity (FinBERT, not keyword fallback)",
        "PASS" if share >= 0.95 else "FAIL",
        f"{share:.0%} of {len(records)} records scored by real FinBERT (need >=95%)",
    ))

    # Gate 1: coverage
    cells = list(panel.values())
    covered = sum(1 for c in cells if c["fresh_count"] >= COVERAGE_MIN_FRESH)
    cov = covered / len(cells) if cells else 0.0
    gates.append((
        f"Coverage (>= {COVERAGE_MIN_FRESH} fresh headlines per ticker-day)",
        "PASS" if cov >= COVERAGE_PASS else "FAIL",
        f"{cov:.1%} of {len(cells)} ticker-days covered (need >={COVERAGE_PASS:.0%})",
    ))

    # Gate 2: turnover
    overlaps = [c["overlap"] for c in cells if c.get("overlap") is not None]
    if n_days < MIN_DAYS_FOR_GATES or not overlaps:
        gates.append(("Turnover (headline set refreshes daily)", "INSUFFICIENT",
                      f"need >={MIN_DAYS_FOR_GATES} collection days, have {n_days}"))
    else:
        med = statistics.median(overlaps)
        gates.append((
            "Turnover (headline set refreshes daily)",
            "PASS" if med <= OVERLAP_PASS else "FAIL",
            f"median day-over-day overlap {med:.0%} across {len(overlaps)} "
            f"ticker-days (need <={OVERLAP_PASS:.0%})",
        ))

    # Gate 3: movement above noise
    eligible = [c for c in cells if "significant" in c]
    if n_days < MIN_DAYS_FOR_GATES or not eligible:
        gates.append(("Movement above sampling noise", "INSUFFICIENT",
                      f"need >={MIN_DAYS_FOR_GATES} collection days, have {n_days}"))
    else:
        sig = sum(1 for c in eligible if c["significant"]) / len(eligible)
        gates.append((
            "Movement above sampling noise",
            "PASS" if sig >= MOVEMENT_PASS else "FAIL",
            f"{sig:.1%} of {len(eligible)} day-over-day deltas exceed their bootstrap "
            f"noise band (need >={MOVEMENT_PASS:.0%})",
        ))

    # Gate 4: neutral collapse (adjusted production labels, fresh only)
    fresh_labels = [lab for c in cells for lab in c["labels"]]
    if fresh_labels:
        neutral_share = fresh_labels.count("neutral") / len(fresh_labels)
        raw_neutral = sum(len(c["raw_neutrals"]) for c in cells)
        raw_total = sum(
            1 for r in records if r.get("is_fresh") and r.get("raw_neutral") is not None
        )
        raw_note = (
            f"; raw-FinBERT neutral share {raw_neutral / raw_total:.0%}" if raw_total else ""
        )
        gates.append((
            "No neutral collapse",
            "PASS" if neutral_share <= NEUTRAL_COLLAPSE_FAIL else "FAIL",
            f"adjusted neutral share {neutral_share:.0%} of {len(fresh_labels)} fresh "
            f"headlines (fail if >{NEUTRAL_COLLAPSE_FAIL:.0%}){raw_note}",
        ))
    else:
        gates.append(("No neutral collapse", "INSUFFICIENT", "no fresh headlines collected"))

    return gates


def print_report(panel: dict, records: list[dict]) -> None:
    days = sorted({d for _, d in panel})
    tickers = sorted({t for t, _ in panel})
    print("=" * 78)
    print("PHASE 0 REPORT -- FinBERT on yfinance headlines")
    print(f"Collection days: {len(days)} ({days[0]} .. {days[-1]})"
          if days else "Collection days: 0")
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 78)

    # Per-ticker/day table
    print(f"\n{'date':<12}{'ticker':<8}{'hl':>4}{'fresh':>6}{'mean':>8}{'std':>7}"
          f"{'delta':>8}{'noise':>7}{'sig':>5}{'ovlp':>6}")
    for ticker in tickers:
        for date in days:
            c = panel.get((ticker, date))
            if not c:
                continue
            fmt = lambda v, n=4: f"{v:.{n - 2}f}" if isinstance(v, float) else "-"
            print(f"{date:<12}{ticker:<8}{c['count']:>4}{c['fresh_count']:>6}"
                  f"{fmt(c['mean']):>8}{fmt(c['std']):>7}{fmt(c.get('delta')):>8}"
                  f"{fmt(c.get('pooled_noise')):>7}"
                  f"{('YES' if c.get('significant') else 'no') if 'significant' in c else '-':>5}"
                  f"{fmt(c.get('overlap'), 3):>6}")

    # Movers: biggest significant day-over-day shifts, with extreme headlines,
    # for the human face-validity review.
    movers = sorted(
        (c for c in panel.values() if c.get("significant")),
        key=lambda c: -abs(c["delta"]),
    )[:8]
    print("\n--- Biggest significant movers (read these; do the arrows make sense?) ---")
    if not movers:
        print("(none yet)")
    for c in movers:
        ex = c["extremes"]
        print(f"\n{c['ticker']} {c['date']}: delta {c['delta']:+.3f} "
              f"(mean {c['mean']:+.3f}, {c['fresh_count']} fresh headlines)")
        if ex:
            print(f"   most negative: [{ex[0]['sentiment_score']:+.2f}] {ex[0]['title']}")
            print(f"   most positive: [{ex[-1]['sentiment_score']:+.2f}] {ex[-1]['title']}")

    # Gates
    print("\n--- GATES (pre-committed; see experiments/README.md) ---")
    gates = evaluate_gates(panel, records)
    for name, status, detail in gates:
        print(f"[{status:>12}]  {name}\n{'':>16}{detail}")

    hard_fail = any(s == "FAIL" for _, s, _ in gates)
    insufficient = any(s == "INSUFFICIENT" for _, s, _ in gates)
    print("\n--- VERDICT ---")
    if hard_fail:
        print("FAIL: at least one gate failed. See README 'What each outcome means'.")
    elif insufficient:
        print("KEEP COLLECTING: no failures yet, but some gates need more days of data.")
    else:
        print("PASS: all gates passed. Headline sentiment is dynamic and trustworthy "
              "enough to proceed with sentiment-history/watchlist features.")
    print("Note: the trust gate (human vs FinBERT agreement) is manual -- "
          "run --export-labels and have two people blind-label the sheet.")


def export_labels(records: list[dict], n: int) -> None:
    """Write a shuffled, model-answer-free CSV for blind human labeling."""
    fresh = [r for r in records if r.get("is_fresh")]
    pool = fresh if len(fresh) >= n else records
    rng = random.Random(RNG_SEED)
    sample = rng.sample(pool, min(n, len(pool)))
    with LABELS_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["headline_id", "ticker", "title",
                         "human_label_1 (positive/negative/neutral)", "human_label_2"])
        for r in sample:
            writer.writerow([r["headline_id"], r["ticker"], r["title"], "", ""])
    print(f"Wrote {min(n, len(pool))} headlines to {LABELS_PATH} "
          "(no model answers included -- label blind, then compare via headline_id).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-labels", type=int, metavar="N", default=0,
                        help="also export N headlines as a blind labeling sheet")
    args = parser.parse_args()

    recs = load_records()
    pnl = build_panel(recs)
    attach_deltas(pnl)
    print_report(pnl, recs)
    if args.export_labels:
        export_labels(recs, args.export_labels)
