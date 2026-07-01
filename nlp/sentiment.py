"""FinBERT Sentiment Module - Analyzes sentiment of financial text.

Author: Matthew
Responsibility: Process raw social media posts through FinBERT and generate sentiment scores

Dataset Format Contract:
- Input: Raw posts from Isaac pipeline (ticker, date, text, source, post_score, market data)
- Output: Aggregated sentiment features grouped by (ticker, date)
- Output fields: ticker, date, avg_sentiment_score, avg_positive_prob, avg_negative_prob,
                 avg_neutral_prob, avg_sentiment_confidence, avg_post_score, mention_count
- This data flows into Abhi prediction model

Model Used: FinBERT (ProsusAI/finbert) - Pre-trained on financial text
"""

import pandas as pd

classifier = None
_sentiment_cache = {}

POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "gain",
    "gains",
    "growth",
    "jump",
    "jumps",
    "outperform",
    "profit",
    "rally",
    "record",
    "rise",
    "rises",
    "strong",
    "surge",
    "surges",
    "upgrade",
    "upside",
}
NEGATIVE_TERMS = {
    "bearish",
    "cut",
    "decline",
    "declines",
    "downgrade",
    "drop",
    "drops",
    "fall",
    "falls",
    "loss",
    "miss",
    "misses",
    "recall",
    "risk",
    "selloff",
    "slump",
    "weak",
    "warning",
}

MIN_DIRECTIONAL_GAP = 0.12
NO_EVIDENCE_NEUTRAL_MARGIN = 0.08
SHORT_HEADLINE_WORDS = 12
SHORT_HEADLINE_CONFIDENCE_CAP = 0.76
GENERIC_HEADLINE_CONFIDENCE_CAP = 0.82


def _get_classifier():
    """Load FinBERT lazily so local API development still works without transformers."""
    global classifier

    if classifier is not None:
        return classifier

    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    except Exception:
        classifier = False

    return classifier


def _term_hits(text, terms):
    lower_text = text.lower()
    return sum(term in lower_text for term in terms)


def _normalize_probabilities(positive, negative, neutral):
    positive = max(0.0, float(positive))
    negative = max(0.0, float(negative))
    neutral = max(0.0, float(neutral))
    total = positive + negative + neutral
    if total <= 0:
        return 0.0, 0.0, 1.0
    return positive / total, negative / total, neutral / total


def _cap_winning_confidence(positive, negative, neutral, cap):
    scores = {"positive": positive, "negative": negative, "neutral": neutral}
    winning_label = max(scores, key=scores.get)
    winning_score = scores[winning_label]
    if winning_label == "neutral" or winning_score <= cap:
        return positive, negative, neutral

    shift = winning_score - cap
    if winning_label == "positive":
        positive -= shift
    else:
        negative -= shift
    neutral += shift
    return _normalize_probabilities(positive, negative, neutral)


def _build_sentiment_result(text, positive, negative, neutral):
    positive, negative, neutral = _normalize_probabilities(positive, negative, neutral)
    word_count = len(str(text).split())
    directional_gap = abs(positive - negative)
    positive_hits = _term_hits(text, POSITIVE_TERMS)
    negative_hits = _term_hits(text, NEGATIVE_TERMS)
    evidence_count = positive_hits + negative_hits

    if directional_gap < MIN_DIRECTIONAL_GAP:
        neutral += 0.10
        positive *= 0.95
        negative *= 0.95

    directional_neutral_margin = max(positive, negative) - neutral
    if (
        evidence_count == 0
        and 0 < directional_neutral_margin < NO_EVIDENCE_NEUTRAL_MARGIN
    ):
        neutral += 0.12
        positive *= 0.92
        negative *= 0.92

    if word_count <= SHORT_HEADLINE_WORDS and evidence_count <= 1:
        neutral += 0.08

    positive, negative, neutral = _normalize_probabilities(positive, negative, neutral)

    confidence_cap = (
        SHORT_HEADLINE_CONFIDENCE_CAP
        if word_count <= SHORT_HEADLINE_WORDS and evidence_count <= 1
        else GENERIC_HEADLINE_CONFIDENCE_CAP
    )
    positive, negative, neutral = _cap_winning_confidence(
        positive, negative, neutral, confidence_cap
    )

    if neutral >= max(positive, negative) or (
        abs(positive - negative) < MIN_DIRECTIONAL_GAP and neutral >= 0.30
    ):
        sentiment_label = "neutral"
    elif positive > negative:
        sentiment_label = "positive"
    else:
        sentiment_label = "negative"

    sentiment_score = positive - negative
    sentiment_confidence = max(positive, negative, neutral)

    return {
        "positive_prob": positive,
        "negative_prob": negative,
        "neutral_prob": neutral,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "sentiment_confidence": sentiment_confidence,
    }


def get_sentiment_scores(text):
    """
    Analyze sentiment of a single text input.

    Args:
        text (str): Input text

    Returns:
        dict: sentiment probabilities, confidence, score, and label
    """
    text = str(text)

    if text in _sentiment_cache:
        return _sentiment_cache[text]

    loaded_classifier = _get_classifier()

    if not loaded_classifier:
        positive_hits = _term_hits(text, POSITIVE_TERMS)
        negative_hits = _term_hits(text, NEGATIVE_TERMS)

        if positive_hits > negative_hits:
            positive = 0.70 if positive_hits >= 2 else 0.64
            negative = 0.10
            neutral = 1.0 - positive - negative
        elif negative_hits > positive_hits:
            negative = 0.70 if negative_hits >= 2 else 0.64
            positive = 0.10
            neutral = 1.0 - positive - negative
        else:
            positive, negative, neutral = 0.18, 0.18, 0.64

        result = _build_sentiment_result(text, positive, negative, neutral)
        _sentiment_cache[text] = result
        return result

    # Get all class scores
    results = loaded_classifier(text, top_k=None)

    # Handle pipeline output format
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        scores_list = results
    else:
        scores_list = results[0]

    # Convert to dictionary
    scores = {item["label"].lower(): item["score"] for item in scores_list}

    # Extract probabilities
    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    neutral = scores.get("neutral", 0.0)

    result = _build_sentiment_result(text, positive, negative, neutral)
    _sentiment_cache[text] = result
    return result


def batch_get_sentiment_scores(texts):
    """
    Analyze sentiment for a batch of texts.

    Args:
        texts (list): List of input texts

    Returns:
        list[dict]: List of sentiment result dictionaries
    """
    texts = [str(text) for text in texts]
    loaded_classifier = _get_classifier()
    uncached_texts = [text for text in texts if text not in _sentiment_cache]

    if uncached_texts:
        if not loaded_classifier:
            for text in uncached_texts:
                get_sentiment_scores(text)
        else:
            batch_results = loaded_classifier(uncached_texts, top_k=None)

            for text, results in zip(uncached_texts, batch_results):
                scores = {item["label"].lower(): item["score"] for item in results}
                positive = scores.get("positive", 0.0)
                negative = scores.get("negative", 0.0)
                neutral = scores.get("neutral", 0.0)

                _sentiment_cache[text] = _build_sentiment_result(
                    text, positive, negative, neutral
                )

    return [_sentiment_cache[text] for text in texts]


def score_dataframe(df, text_column="text", min_confidence=0.0):
    """
    Apply sentiment scoring to a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Column containing text
        min_confidence (float): Minimum confidence threshold (0.0 to 1.0)

    Returns:
        pd.DataFrame: Original dataframe + sentiment columns
    """
    texts = df[text_column].astype(str).tolist()
    scored = pd.DataFrame(batch_get_sentiment_scores(texts))
    result = pd.concat([df.reset_index(drop=True), scored], axis=1)

    if min_confidence > 0:
        result = result[result["sentiment_confidence"] >= min_confidence].reset_index(drop=True)

    return result


def flatten_grouped_pipeline_records(records):
    """Expand grouped ticker/date records into post-level rows for NLP scoring."""
    rows = []

    for record in records:
        ticker = record.get("ticker")
        date = record.get("date")
        for post in record.get("posts", []) or []:
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "text": post.get("text", ""),
                    "source": post.get("source", "unknown"),
                    "post_score": post.get("post_score", 0),
                }
            )

    return rows


def aggregate_sentiment(df, date_col="date", ticker_col="ticker"):
    """
    Aggregate sentiment scores by date and ticker.

    Args:
        df (pd.DataFrame): Data with sentiment columns
        date_col (str): Date column name
        ticker_col (str): Ticker column name

    Returns:
        pd.DataFrame: Aggregated sentiment features
    """
    if date_col not in df.columns or ticker_col not in df.columns:
        raise ValueError("DataFrame must contain date and ticker columns")

    return df.groupby([date_col, ticker_col], as_index=False).agg({
        "sentiment_score": "mean",
        "positive_prob": "mean",
        "negative_prob": "mean",
        "neutral_prob": "mean",
        "sentiment_confidence": "mean",  # Add sentiment_confidence aggregation
        "post_score": "mean",  # For avg_post_score
        "text": "count"  # For mention_count
    }).rename(columns={
        "sentiment_score": "avg_sentiment_score",
        "positive_prob": "avg_positive_prob",
        "negative_prob": "avg_negative_prob",
        "neutral_prob": "avg_neutral_prob",
        "sentiment_confidence": "avg_sentiment_confidence",
        "post_score": "avg_post_score",
        "text": "mention_count"
    })

# TODO (Mihir): Once NLP pipeline is finalized, integrate with backend API

# Quick test when running file directly - loads data from Isaac's pipeline and runs sentiment analysis
if __name__ == "__main__":
    import json
    import os

    # Load data from data pipeline
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.json")
    output_path = os.path.join(os.path.dirname(__file__), "sentiment_data.json")

    try:
        with open(data_path, 'r') as f:
            records = json.load(f)

        if records and isinstance(records[0], dict) and "posts" in records[0]:
            posts = flatten_grouped_pipeline_records(records)
            print(f"Loaded {len(posts)} posts from grouped data pipeline")
        else:
            posts = records
            print(f"Loaded {len(posts)} posts from data pipeline")

        # Convert to DataFrame for processing
        df = pd.DataFrame(posts)

        # Add sentiment scores
        df_with_sentiment = score_dataframe(df)

        # Aggregate by ticker and date
        aggregated_df = aggregate_sentiment(df_with_sentiment)

        # Save aggregated data for prediction model
        aggregated_df.to_json(output_path, orient='records', indent=2)

        print(f"Processed sentiment analysis and saved {len(aggregated_df)} aggregated records to {output_path}")

    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Running sample test instead...")
        sample_text = "Apple stock is rising after strong earnings"
        print(get_sentiment_scores(sample_text))
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Running sample test instead...")
        sample_text = "Apple stock is rising after strong earnings"
        print(get_sentiment_scores(sample_text))
