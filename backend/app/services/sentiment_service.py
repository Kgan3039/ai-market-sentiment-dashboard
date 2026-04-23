"""Sentiment Analysis Service - backend integration layer for NLP scoring."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict

from app.models.schemas import SentimentScores
from app.services.data_service import DataService


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_nlp_module():
    """Load the shared NLP sentiment module dynamically."""
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    try:
        from nlp import sentiment as sentiment_module
        return sentiment_module
    except Exception:
        nlp_dir = os.path.join(repo_root, "nlp")
        if nlp_dir not in sys.path:
            sys.path.append(nlp_dir)
        try:
            import sentiment as sentiment_module
            return sentiment_module
        except Exception:
            return None


class SentimentService:
    """Service for managing sentiment analysis operations."""

    @staticmethod
    def get_sentiment_for_text(text: str) -> SentimentScores:
        """Analyze sentiment for a single text item."""
        if not text or not str(text).strip():
            raise ValueError("Text input is required")

        sentiment_module = _load_nlp_module()
        if sentiment_module is not None and hasattr(sentiment_module, "get_sentiment_scores"):
            raw = sentiment_module.get_sentiment_scores(str(text))
            return SentimentScores(**raw)

        positive_prob = 0.65
        negative_prob = 0.20
        neutral_prob = 0.15
        sentiment_score = positive_prob - negative_prob
        sentiment_confidence = max(positive_prob, negative_prob, neutral_prob)
        sentiment_label = (
            "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
        )

        return SentimentScores(
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            sentiment_confidence=sentiment_confidence,
        )

    @staticmethod
    def get_sentiment_for_ticker(ticker: str) -> Dict[str, Any]:
        """Aggregate sentiment for a ticker using pipeline posts/headlines."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        sentiment_module = _load_nlp_module()
        post_bundle = DataService.get_social_media_data(ticker)
        posts = post_bundle.get("posts", [])

        if not posts:
            fallback = SentimentScores(
                positive_prob=0.70,
                negative_prob=0.15,
                neutral_prob=0.15,
                sentiment_score=0.55,
                sentiment_label="positive",
                sentiment_confidence=0.70,
            )
            return {
                "ticker": ticker,
                "date": datetime.now().date().isoformat(),
                "overall_sentiment": fallback,
                "source_breakdown": {},
                "source": "fallback",
                "item_count": 0,
            }

        source_stats: Dict[str, Dict[str, float | int]] = {}
        total_score = 0.0
        total_confidence = 0.0
        total_positive = 0.0
        total_negative = 0.0
        total_neutral = 0.0

        for post in posts:
            text = post.get("text", "")
            if sentiment_module is not None and hasattr(sentiment_module, "get_sentiment_scores"):
                scored = sentiment_module.get_sentiment_scores(text)
                score_obj = SentimentScores(**scored)
            else:
                score_obj = SentimentService.get_sentiment_for_text(text)

            source = post.get("source", "unknown")
            source_bucket = source_stats.setdefault(
                source,
                {
                    "positive_prob": 0.0,
                    "negative_prob": 0.0,
                    "neutral_prob": 0.0,
                    "count": 0,
                },
            )
            source_bucket["positive_prob"] += score_obj.positive_prob
            source_bucket["negative_prob"] += score_obj.negative_prob
            source_bucket["neutral_prob"] += score_obj.neutral_prob
            source_bucket["count"] += 1

            total_positive += score_obj.positive_prob
            total_negative += score_obj.negative_prob
            total_neutral += score_obj.neutral_prob
            total_score += score_obj.sentiment_score
            total_confidence += score_obj.sentiment_confidence

        count = max(1, len(posts))
        overall_score = total_score / count
        overall_sentiment = SentimentScores(
            positive_prob=total_positive / count,
            negative_prob=total_negative / count,
            neutral_prob=total_neutral / count,
            sentiment_score=overall_score,
            sentiment_label="positive" if overall_score > 0.05 else ("negative" if overall_score < -0.05 else "neutral"),
            sentiment_confidence=total_confidence / count,
        )

        breakdown = {}
        for source, stats in source_stats.items():
            source_count = max(1, int(stats["count"]))
            breakdown[source] = {
                "positive_prob": float(stats["positive_prob"]) / source_count,
                "negative_prob": float(stats["negative_prob"]) / source_count,
                "neutral_prob": float(stats["neutral_prob"]) / source_count,
                "count": source_count,
            }

        return {
            "ticker": ticker,
            "date": post_bundle.get("date", datetime.now().date().isoformat()),
            "overall_sentiment": overall_sentiment,
            "source_breakdown": breakdown,
            "source": "finbert" if sentiment_module is not None else "fallback",
            "item_count": len(posts),
        }
