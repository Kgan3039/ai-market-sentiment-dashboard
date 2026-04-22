# Dataset Format (Pipeline Contract)

This document defines the shared schema across:
Data → NLP → Prediction

---

## 1. Raw Data (Data → NLP)

Each row = one grouped ticker/date record

Fields:
- ticker (string) — e.g. "NVDA"
- date (string) — ISO format
- posts (array)
- market_data (object)

Each post contains:
- text (string)
- source (string)
- post_score (int)

market_data contains:
- price (float)
- price_delta_24h (float)
- percent_change_24h (float)
- volume (int)

Example:
{
  "ticker": "NVDA",
  "date": "2026-03-27",
  "posts": [
    {
      "text": "NVDA is going crazy after earnings",
      "source": "reddit",
      "post_score": 42
    }
  ],
  "market_data": {
    "price": 910.5,
    "price_delta_24h": 12.4,
    "percent_change_24h": 1.38,
    "volume": 1250000
  }
}

---

## 2. Sentiment Output (NLP → Prediction)

Adds sentiment fields to each row

Fields:
- positive_prob (float)
- negative_prob (float)
- neutral_prob (float)
- sentiment_score = positive - negative
- sentiment_label (string)
- sentiment_confidence = max(probabilities)

Example:
{
  "ticker": "NVDA",
  "date": "2026-03-27",
  "text": "NVDA is going crazy",
  "positive_prob": 0.72,
  "negative_prob": 0.10,
  "neutral_prob": 0.18,
  "sentiment_score": 0.62,
  "sentiment_label": "positive",
  "sentiment_confidence": 0.72
}

---

## 3. Aggregated Features (NLP → Prediction)

Grouped by (ticker, date)

Fields:
- avg_sentiment_score
- avg_positive_prob
- avg_negative_prob
- avg_neutral_prob
- mention_count
- avg_post_score

Example:
{
  "ticker": "NVDA",
  "date": "2026-03-27",
  "avg_sentiment_score": 0.41,
  "avg_positive_prob": 0.58,
  "avg_negative_prob": 0.19,
  "avg_neutral_prob": 0.23,
  "mention_count": 17,
  "avg_post_score": 21.4
}

---

## 4. Final Model Input (Aggregated Data → Prediction)

The prediction model consumes aggregated features from the NLP pipeline along with market data.

Each row represents a grouped time window (by ticker and date).

### Required Fields

- avg_sentiment_score (float)
- avg_positive_prob (float)
- avg_negative_prob (float)
- avg_neutral_prob (float)
- price_delta_24h (float)
- volume_delta (float)
- label (int, required for training, optional for inference)

---

### Derived Features (Computed Inside Model)

The model internally converts aggregated sentiment into final features:

- sentiment_score = avg_sentiment_score

- sentiment_confidence = max(
    avg_positive_prob,
    avg_negative_prob,
    avg_neutral_prob
  )

These derived features are NOT required from upstream pipelines.

---

### Example Final Input Row

{
  "ticker": "NVDA",
  "date": "2026-03-27",
  "avg_sentiment_score": 0.41,
  "avg_positive_prob": 0.58,
  "avg_negative_prob": 0.19,
  "avg_neutral_prob": 0.23,
  "price_delta_24h": 0.012,
  "volume_delta": 0.10,
  "label": 1
}

---

## Notes

- Isaac (Data) → produces raw + market features
- Matthew (NLP) → produces sentiment features
- Abhi (Prediction) → consumes aggregated dataset
