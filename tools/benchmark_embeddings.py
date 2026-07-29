"""Reproducible real-model benchmark for GitHub issue #63.

This script intentionally sits outside the ordinary test suite because model
availability and hardware performance vary.  It builds a representative
250-item day (50 items for each Phase 0 ticker) from committed story fixtures,
warms the model once, and times one batch encoding pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from nlp.embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "ai" / "fixtures" / "theme_fixtures.json"
TICKERS = ("TSLA", "NVDA", "AMD", "AAPL", "META")


def _fixture_records(items_per_ticker: int) -> list[tuple[str, str | None]]:
    payload: list[dict[str, Any]] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    by_ticker: dict[str, list[tuple[str, str | None]]] = {
        ticker: [] for ticker in TICKERS
    }
    for theme in payload:
        ticker = str(theme.get("ticker", "")).upper()
        if ticker not in by_ticker:
            continue
        for story in theme.get("member_stories", []):
            by_ticker[ticker].append(
                (str(story.get("title") or ""), story.get("description"))
            )
    missing = [ticker for ticker, records in by_ticker.items() if not records]
    if missing:
        raise RuntimeError(f"fixture has no stories for: {', '.join(missing)}")
    records = []
    for ticker in TICKERS:
        ticker_records = by_ticker[ticker]
        records.extend(
            ticker_records[index % len(ticker_records)]
            for index in range(items_per_ticker)
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--cache-location")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--items-per-ticker", type=int, default=50)
    args = parser.parse_args()

    service = EmbeddingService(
        model_name=args.model,
        model_revision=args.model_revision,
        cache_location=args.cache_location,
        batch_size=args.batch_size,
    )
    records = _fixture_records(args.items_per_ticker)
    service.embed_record("Benchmark warm-up", None)
    started = time.perf_counter()
    vectors = service.embed_records(records)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "condition": "warm",
                "elapsed_seconds": round(elapsed, 3),
                "item_count": len(records),
                "model": args.model,
                "model_revision": args.model_revision,
                "tickers": list(TICKERS),
                "vector_dimension": len(vectors[0]) if vectors else None,
                "within_60_seconds": elapsed < 60,
            },
            sort_keys=True,
        )
    )
    return 0 if elapsed < 60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
