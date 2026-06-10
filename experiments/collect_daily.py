#!/usr/bin/env python3
"""Phase 0 experiment: daily collector for yfinance headlines + FinBERT scores.

Purpose
-------
Determine whether FinBERT-on-yfinance-headlines produces enough variation,
freshness, and trustworthiness to justify building sentiment-history and
watchlist features. This script ONLY collects data. It adds no endpoints,
no frontend, no product code.

Reused entry points (audited 2026-06-09)
----------------------------------------
- DataService._fetch_yfinance_headlines(ticker, limit)
    Raw provider call. Deliberately used INSTEAD of DataService.get_headlines()
    because get_headlines() silently substitutes committed demo headlines and
    cached/stale results on provider failure, which would contaminate the
    experiment's freshness and turnover measurements. Here, a fetch failure is
    recorded as a failure -- that is itself data (platform-stability gate).
- nlp.sentiment._get_classifier(), nlp.sentiment._build_sentiment_result()
    We call the classifier directly to capture RAW FinBERT probabilities,
    then pass the same probabilities through _build_sentiment_result() to get
    the ADJUSTED production scores (the repo applies neutral-boosting and
    confidence-cap heuristics on top of raw FinBERT). Recording both lets the
    report distinguish "FinBERT is neutral-collapsed" from "our post-processing
    neutral-collapses it".
    CAVEAT: nlp.sentiment falls back to a keyword heuristic when transformers
    is unavailable. Every record is stamped with scorer_mode so the report can
    reject keyword-mode data.

Outputs (append-only JSONL under experiments/data/)
---------------------------------------------------
- headlines.jsonl      one record per (run_date, ticker, deduped headline)
- daily_summary.jsonl  one record per (run_date, ticker)
- headline_index.json  headline_id -> first_seen_date (turnover tracking)

Run once per day at a consistent time:  python experiments/collect_daily.py
Re-running on the same date skips already-collected tickers (use --force).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import statistics
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo wiring: experiments/ sits at repo root, backend imports need backend/
# on sys.path (the app package), nlp imports need repo root or nlp/ itself.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.services.data_service import DataService  # noqa: E402

try:
    from nlp import sentiment as nlp_sentiment  # noqa: E402
except ImportError:  # nlp/ may not be a package; import the module directly
    sys.path.insert(0, str(REPO_ROOT / "nlp"))
    import sentiment as nlp_sentiment  # type: ignore # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration -- FIXED for the duration of the experiment.
# Do not add/remove tickers mid-run; it invalidates cross-day comparisons.
# ---------------------------------------------------------------------------
TICKERS = ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "META", "JPM", "WMT", "KO", "XOM"]
HEADLINE_LIMIT = 25          # ask for more than the dashboard's 6; provider caps anyway
FRESH_WINDOW_HOURS = 24      # "fresh" = published within this window before the run
NEAR_DUP_RATIO = 0.92        # difflib ratio above which two titles are the same story

DATA_DIR = Path(__file__).resolve().parent / "data"
HEADLINES_PATH = DATA_DIR / "headlines.jsonl"
SUMMARY_PATH = DATA_DIR / "daily_summary.jsonl"
INDEX_PATH = DATA_DIR / "headline_index.json"


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()


def _headline_id(title: str) -> str:
    return hashlib.sha1(_normalize_title(title).encode("utf-8")).hexdigest()[:16]


def _to_utc(dt: datetime | None) -> datetime | None:
    """Normalize any datetime to timezone-aware UTC.

    The repo's DataService._parse_publish_time returns MIXED forms: naive
    local datetimes for unix timestamps (datetime.fromtimestamp) and aware
    datetimes for ISO strings with offsets. astimezone() handles both
    correctly: naive values are interpreted as system local time (which is
    exactly what fromtimestamp produced), aware values are converted.
    """
    if dt is None:
        return None
    return dt.astimezone(timezone.utc)


def _is_near_duplicate(title: str, kept_titles: list[str]) -> bool:
    norm = _normalize_title(title)
    for kept in kept_titles:
        if difflib.SequenceMatcher(None, norm, _normalize_title(kept)).ratio() >= NEAR_DUP_RATIO:
            return True
    return False


def _detect_scorer_mode() -> str:
    """'finbert' if the real model loaded, else 'keyword_fallback'."""
    classifier = nlp_sentiment._get_classifier()
    return "finbert" if classifier else "keyword_fallback"


def _score_headlines(titles: list[str], scorer_mode: str) -> list[dict]:
    """Score titles, returning raw FinBERT probs AND adjusted production scores.

    One inference pass via the repo's classifier; the repo's own
    _build_sentiment_result() is then reused for the adjusted scores so the
    experiment measures exactly what the product would ship.
    """
    results = []
    if scorer_mode == "finbert":
        classifier = nlp_sentiment._get_classifier()
        batch = classifier(titles, top_k=None)
        # classifier(single_str) and classifier(list) shapes differ; normalize.
        if titles and isinstance(batch[0], dict):
            batch = [batch]
        for title, scores_list in zip(titles, batch):
            scores = {item["label"].lower(): item["score"] for item in scores_list}
            raw_pos = float(scores.get("positive", 0.0))
            raw_neg = float(scores.get("negative", 0.0))
            raw_neu = float(scores.get("neutral", 0.0))
            adjusted = nlp_sentiment._build_sentiment_result(title, raw_pos, raw_neg, raw_neu)
            results.append(
                {
                    "raw_positive": round(raw_pos, 4),
                    "raw_negative": round(raw_neg, 4),
                    "raw_neutral": round(raw_neu, 4),
                    "positive_prob": round(adjusted["positive_prob"], 4),
                    "negative_prob": round(adjusted["negative_prob"], 4),
                    "neutral_prob": round(adjusted["neutral_prob"], 4),
                    "sentiment_score": round(adjusted["sentiment_score"], 4),
                    "sentiment_label": adjusted["sentiment_label"],
                    "sentiment_confidence": round(adjusted["sentiment_confidence"], 4),
                }
            )
    else:
        # Keyword fallback path -- still recorded, but the report will refuse
        # to evaluate gates on it. Uses the repo's public function unchanged.
        for title in titles:
            adjusted = nlp_sentiment.get_sentiment_scores(title)
            results.append(
                {
                    "raw_positive": None,
                    "raw_negative": None,
                    "raw_neutral": None,
                    "positive_prob": round(adjusted["positive_prob"], 4),
                    "negative_prob": round(adjusted["negative_prob"], 4),
                    "neutral_prob": round(adjusted["neutral_prob"], 4),
                    "sentiment_score": round(adjusted["sentiment_score"], 4),
                    "sentiment_label": adjusted["sentiment_label"],
                    "sentiment_confidence": round(adjusted["sentiment_confidence"], 4),
                }
            )
    return results


def _load_index() -> dict:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {}


def _already_collected(run_date: str) -> set[str]:
    done = set()
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("run_date") == run_date:
                    done.add(rec.get("ticker"))
    return done


def collect(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now()          # human-facing day bucket (consistent with cron time)
    now_utc = datetime.now(timezone.utc)  # freshness clock: aware UTC only
    run_date = now_local.date().isoformat()
    run_ts = now_local.isoformat(timespec="seconds")
    fresh_cutoff = now_utc - timedelta(hours=FRESH_WINDOW_HOURS)  # aware UTC

    scorer_mode = _detect_scorer_mode()
    if scorer_mode != "finbert":
        print(
            "WARNING: transformers/FinBERT failed to load -- nlp.sentiment is in "
            "keyword-fallback mode. Records will be stamped keyword_fallback and "
            "the report will not count them toward any gate. Fix the environment "
            "(pip install transformers torch) before relying on this run.",
            file=sys.stderr,
        )

    skip = set() if force else _already_collected(run_date)
    index = _load_index()
    headline_fh = HEADLINES_PATH.open("a")
    summary_fh = SUMMARY_PATH.open("a")

    try:
        for ticker in TICKERS:
            if ticker in skip:
                print(f"{ticker}: already collected for {run_date}, skipping (--force to redo)")
                continue

            fetch_error = None
            items = []
            try:
                items = DataService._fetch_yfinance_headlines(ticker, HEADLINE_LIMIT)
            except Exception as exc:  # provider failure is data, not a crash
                fetch_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

            # Dedupe: exact normalized hash, then near-duplicate fuzzy match.
            kept, kept_titles, seen_ids = [], [], set()
            for item in items:
                title = (item.title or item.headline or "").strip()
                if not title:
                    continue
                hid = _headline_id(title)
                if hid in seen_ids or _is_near_duplicate(title, kept_titles):
                    continue
                seen_ids.add(hid)
                kept_titles.append(title)
                kept.append((hid, title, item))

            scores = _score_headlines([t for _, t, _ in kept], scorer_mode) if kept else []

            fresh_scores_adj, all_scores_adj, label_counts = [], [], {}
            for (hid, title, item), score in zip(kept, scores):
                published_utc = _to_utc(item.published_at)  # aware UTC or None
                is_fresh = bool(published_utc and published_utc >= fresh_cutoff)
                first_seen = index.setdefault(hid, run_date)
                record = {
                    "run_date": run_date,
                    "run_ts": run_ts,
                    "ticker": ticker,
                    "headline_id": hid,
                    "title": title,
                    "publisher": item.source,
                    "published_at": published_utc.isoformat() if published_utc else None,
                    "is_fresh": is_fresh,
                    "first_seen_date": first_seen,
                    "scorer_mode": scorer_mode,
                    **score,
                }
                headline_fh.write(json.dumps(record) + "\n")
                all_scores_adj.append(score["sentiment_score"])
                if is_fresh:
                    fresh_scores_adj.append(score["sentiment_score"])
                label_counts[score["sentiment_label"]] = (
                    label_counts.get(score["sentiment_label"], 0) + 1
                )

            summary = {
                "run_date": run_date,
                "run_ts": run_ts,
                "ticker": ticker,
                "scorer_mode": scorer_mode,
                "fetch_error": fetch_error,
                "headline_count": len(kept),
                "fresh_count": len(fresh_scores_adj),
                "mean_sentiment_fresh": (
                    round(statistics.mean(fresh_scores_adj), 4) if fresh_scores_adj else None
                ),
                "std_sentiment_fresh": (
                    round(statistics.stdev(fresh_scores_adj), 4)
                    if len(fresh_scores_adj) >= 2
                    else None
                ),
                "mean_sentiment_all": (
                    round(statistics.mean(all_scores_adj), 4) if all_scores_adj else None
                ),
                "label_counts": label_counts,
            }
            summary_fh.write(json.dumps(summary) + "\n")
            print(
                f"{ticker}: {len(kept)} headlines ({len(fresh_scores_adj)} fresh), "
                f"mean_fresh={summary['mean_sentiment_fresh']}"
                + (f"  FETCH ERROR: {fetch_error}" if fetch_error else "")
            )
    finally:
        headline_fh.close()
        summary_fh.close()
        INDEX_PATH.write_text(json.dumps(index, indent=0))

    print(f"\nDone. Data appended under {DATA_DIR}/ (run_date={run_date}, scorer={scorer_mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-collect tickers already done today"
    )
    collect(force=parser.parse_args().force)
