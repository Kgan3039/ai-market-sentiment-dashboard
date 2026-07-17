"""Sentiment Analysis Service - API interface to NLP sentiment model.

Author: Mihir (with integration from Matthew NLP module)
Responsibility: Provide REST API endpoints for sentiment analysis

Integration Points:
- Calls Matthew NLP sentiment analysis (../nlp/sentiment.py)
- Consumes social media data from Isaac data pipeline
- Returns results to Sentim ent routes for API endpoints

Current Status: Active integration with validated pipeline text and news headlines
"""

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


def _pipeline_file_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
    )


def _load_grouped_posts(ticker: str) -> list[dict[str, Any]]:
    """Load validated text for aggregate sentiment.

    Pipeline posts remain the preferred source, but the committed pipeline file can
    contain only placeholder text when Finnhub is not configured. In that case,
    reuse the existing Yahoo Finance headline provider instead of scoring mock
    content.
    """
    pipeline_path = _pipeline_file_path()
    flattened_posts = []

    if os.path.exists(pipeline_path):
        try:
            with open(pipeline_path, "r") as file:
                records = json.load(file)
        except Exception:
            records = []

        for record in records:
            if record.get("ticker", "").upper() != ticker:
                continue

            record_posts = record.get("posts")
            if isinstance(record_posts, list):
                candidate_posts = [
                    {
                        "ticker": ticker,
                        "date": record.get("date"),
                        "text": post.get("text", ""),
                        "source": post.get("source", "unknown"),
                        "post_score": post.get("post_score", 0),
                    }
                    for post in record_posts
                ]
            else:
                candidate_posts = [
                    {
                        "ticker": ticker,
                        "date": record.get("date"),
                        "text": record.get("text", ""),
                        "source": record.get("source", "unknown"),
                        "post_score": record.get("post_score", 0),
                    }
                ]

            flattened_posts.extend(
                post for post in candidate_posts if _is_valid_pipeline_text(post)
            )

    if flattened_posts:
        return flattened_posts

    try:
        from app.services.data_service import DataService

        headlines = DataService.get_headlines(ticker, limit=8)
    except Exception:
        headlines = []

    return [
        {
            "ticker": ticker,
            "date": (
                headline.published_at.date().isoformat()
                if headline.published_at
                else datetime.now().date().isoformat()
            ),
            "text": headline.headline,
            "source": headline.source or "Yahoo Finance via yfinance",
            "post_score": 1,
        }
        for headline in headlines
        if _is_valid_pipeline_text(
            {
                "text": headline.headline,
                "source": headline.source or "Yahoo Finance via yfinance",
            }
        )
    ]


def _is_valid_pipeline_text(post: dict[str, Any]) -> bool:
    text = str(post.get("text", "")).strip()
    source = str(post.get("source", "")).strip().lower()
    lowered_text = text.lower()

    if not text or source.startswith("mock"):
        return False

    placeholder_phrases = (
        "sample post while waiting for api approval",
        "sample fallback post",
        "discussion about",
    )
    return not any(phrase in lowered_text for phrase in placeholder_phrases)


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

        raise RuntimeError("Sentiment scorer is unavailable")

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
            return grouped

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
