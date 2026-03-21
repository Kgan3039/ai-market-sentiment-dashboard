# Dataset Format

## Raw Data
{
  id: string,
  stock: string,
  text: string,
  timestamp: string,
  source: string
}

## After Sentiment
+ sentiment_score (float)
+ sentiment_label (string)

## Final Output
{
  stock: string,
  avg_sentiment: float,
  probability_up: float,
  risk_level: string
}
