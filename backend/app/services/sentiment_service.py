"""Sentiment Analysis Service - API interface to NLP sentiment model.

Author: Mihir (with integration from Matthew NLP module)
Responsibility: Provide REST API endpoints for sentiment analysis

Integration Points:
- Calls Matthew NLP sentiment analysis (../nlp/sentiment.py)
- Consumes social media data from Isaac data pipeline
- Returns results to Sentim ent routes for API endpoints

Current Status: Placeholder implementation - awaiting NLP integration
"""

import os
import sys
import json
from typing import Dict, Any
from datetime import datetime
from app.models.schemas import SentimentScores


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _load_nlp_module():
    """Load nlp/sentiment module dynamically for backend integration."""
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    try:
        from nlp import sentiment as sentiment_module
        return sentiment_module
    except Exception:
        # If PYTHONPATH is not set to project root, add relative nlp path
        nlp_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "nlp")
        )
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


def _aggregated_sentiment_file_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "aggregated_sentiment.json")
    )

class SentimentService:
    """Service for managing sentiment analysis operations."""

    @staticmethod
    def get_sentiment_for_text(text: str) -> SentimentScores:
        """
        Analyze sentiment of a given text.

        Args:
            text (str): Text to analyze

        Returns:
            SentimentScores: Sentiment analysis results with probabilities and labels
        """
        if not text or not str(text).strip():
            raise ValueError("Text input is required")

        sentiment_module = _load_nlp_module()
        if sentiment_module is not None and hasattr(sentiment_module, 'get_sentiment_scores'):
            raw = sentiment_module.get_sentiment_scores(str(text))
            return SentimentScores(**raw)

        # Fallback legacy placeholder
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
        """
        Aggregate sentiment for a ticker using pipeline data and FinBERT scoring.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Aggregated sentiment with source breakdown
        """
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        sentiment_module = _load_nlp_module()

        grouped = {
            'ticker': ticker,
            'overall_sentiment': None,
            'source_breakdown': {},
            'date': datetime.now().date().isoformat(),
        }

        aggregated_path = _aggregated_sentiment_file_path()
        if os.path.exists(aggregated_path):
            try:
                with open(aggregated_path, 'r') as f:
                    aggregated_rows = json.load(f)

                for row in aggregated_rows:
                    if row.get('ticker', '').upper() == ticker:
                        confidence = float(
                            row.get(
                                'avg_sentiment_confidence',
                                max(
                                    row.get('avg_positive_prob', 0.0),
                                    row.get('avg_negative_prob', 0.0),
                                    row.get('avg_neutral_prob', 0.0),
                                ),
                            )
                        )
                        score = float(row.get('avg_sentiment_score', 0.0))
                        grouped['overall_sentiment'] = SentimentScores(
                            positive_prob=float(row.get('avg_positive_prob', 0.0)),
                            negative_prob=float(row.get('avg_negative_prob', 0.0)),
                            neutral_prob=float(row.get('avg_neutral_prob', 0.0)),
                            sentiment_score=score,
                            sentiment_label='positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'),
                            sentiment_confidence=confidence,
                        ).model_dump()
                        grouped['date'] = row.get('date', grouped['date'])
                        return grouped
            except Exception:
                pass

        posts = []
        pipeline_path = _pipeline_file_path()
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'r') as f:
                all_posts = json.load(f)
            posts = [p for p in all_posts if p.get('ticker', '').upper() == ticker]

        if not posts:
            return {
                'ticker': ticker,
                'overall_sentiment': SentimentScores(
                    positive_prob=0.70,
                    negative_prob=0.15,
                    neutral_prob=0.15,
                    sentiment_score=0.55,
                    sentiment_label='positive',
                    sentiment_confidence=0.70,
                ).model_dump(),
                'source_breakdown': {},
                'date': grouped['date'],
            }

        source_stats = {}
        total_score = 0.0
        total_confidence = 0.0

        for post in posts:
            text = post.get('text', '')
            if sentiment_module is not None and hasattr(sentiment_module, 'get_sentiment_scores'):
                scored = sentiment_module.get_sentiment_scores(text)
                score_obj = SentimentScores(**scored)
            else:
                score_obj = SentimentService.get_sentiment_for_text(text)

            source = post.get('source', 'unknown')
            source_bucket = source_stats.setdefault(source, {
                'positive_prob': 0.0,
                'negative_prob': 0.0,
                'neutral_prob': 0.0,
                'count': 0,
            })
            source_bucket['positive_prob'] += score_obj.positive_prob
            source_bucket['negative_prob'] += score_obj.negative_prob
            source_bucket['neutral_prob'] += score_obj.neutral_prob
            source_bucket['count'] += 1

            total_score += score_obj.sentiment_score
            total_confidence += score_obj.sentiment_confidence

        count = max(1, len(posts))
        overall_score = total_score / count
        overall_confidence = total_confidence / count
        overall_positive = sum(s['positive_prob'] for s in source_stats.values()) / max(1, count)
        overall_negative = sum(s['negative_prob'] for s in source_stats.values()) / max(1, count)
        overall_neutral = sum(s['neutral_prob'] for s in source_stats.values()) / max(1, count)

        grouped['overall_sentiment'] = SentimentScores(
            positive_prob=overall_positive,
            negative_prob=overall_negative,
            neutral_prob=overall_neutral,
            sentiment_score=overall_score,
            sentiment_label='positive' if overall_score > 0.05 else ('negative' if overall_score < -0.05 else 'neutral'),
            sentiment_confidence=overall_confidence,
        ).model_dump()

        for source, stats in source_stats.items():
            count_src = stats['count']
            grouped['source_breakdown'][source] = {
                'positive_prob': stats['positive_prob'] / count_src,
                'negative_prob': stats['negative_prob'] / count_src,
                'neutral_prob': stats['neutral_prob'] / count_src,
                'count': count_src,
            }

        return grouped
