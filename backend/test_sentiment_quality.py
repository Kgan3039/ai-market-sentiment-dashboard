#!/usr/bin/env python3
"""Regression checks for headline sentiment calibration."""

from nlp import sentiment


def _score(text: str):
    sentiment.classifier = False
    sentiment._sentiment_cache.clear()
    return sentiment.get_sentiment_scores(text)


def test_generic_headline_stays_neutral() -> None:
    result = _score("Nvidia announces quarterly earnings date")

    assert result["sentiment_label"] == "neutral"
    assert result["sentiment_confidence"] <= 0.72


def test_short_market_move_is_not_overconfident() -> None:
    result = _score("Tesla shares rise ahead of investor day")

    assert result["sentiment_label"] in ("positive", "neutral")
    assert result["sentiment_confidence"] <= 0.76


def test_directional_headlines_keep_signal() -> None:
    positive = _score("Nvidia beats expectations as revenue surges")
    negative = _score("Tesla shares drop after weak delivery warning")

    assert positive["sentiment_label"] == "positive"
    assert positive["positive_prob"] > positive["negative_prob"]
    assert positive["sentiment_confidence"] <= 0.82

    assert negative["sentiment_label"] == "negative"
    assert negative["negative_prob"] > negative["positive_prob"]
    assert negative["sentiment_confidence"] <= 0.82


if __name__ == "__main__":
    test_generic_headline_stays_neutral()
    test_short_market_move_is_not_overconfident()
    test_directional_headlines_keep_signal()
    print("Sentiment quality tests passed.")
