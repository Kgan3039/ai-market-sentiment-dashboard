"""
FinBERT Sentiment Module

This module uses FinBERT to generate sentiment scores for financial text.
Outputs:
- positive / negative / neutral probabilities
- sentiment_score = positive - negative
- sentiment_label = highest probability class
"""

from transformers import pipeline
import pandas as pd

# Load model once (important for performance)
classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)


def get_sentiment_scores(text):
    """
    Analyze sentiment of a single text input.

    Args:
        text (str): Input text

    Returns:
        dict: sentiment probabilities, score, and label
    """
    text = str(text)

    # Get all class scores
    results = classifier(text, top_k=None)

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

    # Create sentiment score
    sentiment_score = positive - negative

    # Choose label based on highest probability
    sentiment_label = max(scores, key=scores.get)

    return {
        "positive_prob": positive,
        "negative_prob": negative,
        "neutral_prob": neutral,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
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
        "neutral_prob": "mean"
    })


# Optional: quick test when running file directly
if __name__ == "__main__":
    sample_text = "Apple stock is rising after strong earnings"
    print(get_sentiment_scores(sample_text))
