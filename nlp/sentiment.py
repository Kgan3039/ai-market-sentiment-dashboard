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


def get_sentiment_scores(text):
    """
    Analyze sentiment of a single text input.

    Args:
        text (str): Input text

    Returns:
        dict: sentiment probabilities, confidence, score, and label
    """
    text = str(text)

    loaded_classifier = _get_classifier()

    if not loaded_classifier:
        lower_text = text.lower()
        positive_terms = ["beat", "growth", "bullish", "surge", "strong", "gain", "up"]
        negative_terms = ["miss", "drop", "bearish", "weak", "fall", "down", "risk"]
        positive_hits = sum(term in lower_text for term in positive_terms)
        negative_hits = sum(term in lower_text for term in negative_terms)

        if positive_hits > negative_hits:
            positive, negative, neutral = 0.72, 0.10, 0.18
            sentiment_label = "positive"
        elif negative_hits > positive_hits:
            positive, negative, neutral = 0.10, 0.72, 0.18
            sentiment_label = "negative"
        else:
            positive, negative, neutral = 0.20, 0.20, 0.60
            sentiment_label = "neutral"

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

    probabilities = [positive, negative, neutral]
    sentiment_confidence = max(probabilities)

    # Create sentiment score
    sentiment_score = positive - negative

    # Choose label based on highest probability
    sentiment_label = max(scores, key=scores.get)

    # Add sentiment confidence (highest probability)
    sentiment_confidence = max(positive, negative, neutral)

    return {
        "positive_prob": positive,
        "negative_prob": negative,
        "neutral_prob": neutral,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "sentiment_confidence": sentiment_confidence
    }


def score_dataframe(df, text_column="text"):
    """
    Apply sentiment scoring to a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Column containing text

    Returns:
        pd.DataFrame: Original dataframe + sentiment columns
    """
    scored = df[text_column].apply(lambda x: pd.Series(get_sentiment_scores(x)))
    return pd.concat([df, scored], axis=1)


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


# TODO (Matthew): Add confidence thresholds - skip low-confidence predictions
# TODO (Matthew): Implement caching for repeated text analysis
# TODO (Matthew): Add batch processing optimization for large datasets
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
            posts = json.load(f)

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
