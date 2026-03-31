# Dataset Format (Pipeline Contract)

This document defines the shared schema across:
Data → NLP → Prediction

---

## 1. Raw Data (Data → NLP)

Each row = one social media post

Fields:
- ticker (string) — e.g. "NVDA"
- date (string) — ISO format
- text (string)
- source (string) — e.g. "reddit"
- post_score (int)

Example:
{
  "ticker": "NVDA",
  "date": "2026-03-27",
  "text": "NVDA is going crazy after earnings",
  "source": "reddit",
  "post_score": 42
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

## 4. Final Model Input (Data + NLP → Prediction)

Fields:
- sentiment_score
- sentiment_confidence
- price_delta_24h
- volume_delta
- label (target)

---

## Notes

- Isaac (Data) → produces raw + market features
- Matthew (NLP) → produces sentiment features
- Abhi (Prediction) → consumes aggregated dataset
