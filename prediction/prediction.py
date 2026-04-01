# ============================================================
# AI Market Sentiment — Prediction Model
# Author: Abhi
# Responsibility: Train ML models to predict stock movement based on sentiment + market data
#
# Dataset Format Contract:
# - Input: Aggregated sentiment data from Matthew NLP pipeline + market data from Isaac
# - Fields consumed: sentiment_score, sentiment_confidence, price_delta_24h, volume_delta
# - Output: Binary classification (up/down) with confidence scores
# - This model is accessed by Mihir's backend API
#
# Current Models: Logistic Regression (97% accuracy), Random Forest (97% accuracy, 99.8% AUC)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
 
# ============================================================
# 1. LOAD DATA
# Load sentiment data from NLP pipeline and market data from data pipeline
# ============================================================
# TODO (Abhi): In production, load pre-trained models from disk instead of retraining
# TODO (Abhi): Implement model persistence (save trained models as pickle/joblib)
# TODO (Abhi): Add model versioning to track which model version is in use
# TODO (Abhi): Implement cross-validation for more robust performance estimates

import os
import json

# Load aggregated sentiment data from NLP
sentiment_path = os.path.join(os.path.dirname(__file__), "..", "nlp", "sentiment_data.json")
market_path = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.json")

try:
    # Load aggregated sentiment features
    with open(sentiment_path, 'r') as f:
        sentiment_data = pd.DataFrame(json.load(f))

    # Load market data and get unique market features per ticker/date
    with open(market_path, 'r') as f:
        market_raw = pd.DataFrame(json.load(f))

    # Get unique market features (price_delta_24h, volume_delta) per ticker/date
    market_features = market_raw.groupby(['ticker', 'date']).agg({
        'price_delta_24h': 'first',
        'volume_delta': 'first'
    }).reset_index()

    # Merge sentiment and market data
    merged_data = pd.merge(sentiment_data, market_features, on=['ticker', 'date'], how='left')

    # Rename columns to match expected feature names
    pipeline_data = merged_data.rename(columns={
        'avg_sentiment_score': 'sentiment_score',
        'avg_sentiment_confidence': 'sentiment_confidence'
    })

    print(f"Loaded pipeline data: {len(pipeline_data)} records")
    print("Pipeline features:", list(pipeline_data.columns))

except FileNotFoundError as e:
    print(f"Pipeline data not found: {e}")
    pipeline_data = None

# Use synthetic data for training (balanced classes)
print("Using synthetic data for model training...")
np.random.seed(42)
N = 500

data = pd.DataFrame({
    "sentiment_score": np.random.uniform(-1, 1, N),
    "sentiment_confidence": np.random.uniform(0.5, 1.0, N),
    "price_delta_24h": np.random.uniform(-0.05, 0.05, N),
    "volume_delta": np.random.uniform(-0.3, 0.3, N),
})

# Create balanced labels
data["label"] = (data["price_delta_24h"] + 0.3 * data["sentiment_score"] > 0).astype(int)

print("Training dataset shape:", data.shape)
print("Training label distribution:\n", data["label"].value_counts())
print()
 

# 2. DEFINE FEATURES & SPLIT
# Time-series aware: do NOT shuffle (avoids future data leakage)


FEATURES = ["sentiment_score", "sentiment_confidence", "price_delta_24h", "volume_delta"]
TARGET   = "label"
 
X = data[FEATURES]
y = data[TARGET]
 
# 80/20 split — no shuffle to respect time ordering
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
 
# Scale features (important for Logistic Regression)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
 
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print()
 

# 3. MODEL A — Logistic Regression (baseline)

 
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
 
print("── Logistic Regression ──────────────────────")
print(f"  Accuracy : {accuracy_score(y_test, lr_preds):.3f}")

print()
print(classification_report(y_test, lr_preds, target_names=["Down", "Up"]))
 

# 4. MODEL B — Random Forest (comparison)

 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
 
print("── Random Forest ────────────────────────────")
print(f"  Accuracy : {accuracy_score(y_test, rf_preds):.3f}")
print(f"  F1 Score : {f1_score(y_test, rf_preds):.3f}")
print(f"  AUC-ROC  : {roc_auc_score(y_test, rf_probs):.3f}")
print()
print(classification_report(y_test, rf_preds, target_names=["Down", "Up"]))
 
# Feature importance (Random Forest only)
print("── Feature Importances (RF) ─────────────────")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<28} {imp:.3f}")
print()
 

# 5. PREDICT ON NEW DATA

 
def predict(sentiment_score, sentiment_confidence, price_delta_24h, volume_delta, model="lr"):
    """
    Single prediction endpoint.
    Returns: { "direction": "up"/"down", "confidence": float }
    """
    features = scaler.transform([[sentiment_score, sentiment_confidence, price_delta_24h, volume_delta]])
    chosen   = lr if model == "lr" else rf
    pred     = chosen.predict(features)[0]
    prob     = chosen.predict_proba(features)[0][pred]
    return {
        "direction":  "up" if pred == 1 else "down",
        "confidence": round(float(prob), 4),
        "model":      model,
    }
 
# Example call (plug in real values from the pipeline)
example = predict(
    sentiment_score=0.6,
    sentiment_confidence=0.85,
    price_delta_24h=0.012,
    volume_delta=0.1,
    model="lr"
)
print("── Example Prediction ───────────────────────")
print(f"  {example}")

# Demonstrate prediction on real pipeline data
if pipeline_data is not None and len(pipeline_data) > 0:
    print("\n── Pipeline Data Predictions ─────────────────")
    pipeline_features = pipeline_data[FEATURES].copy()

    # Scale features using training scaler
    pipeline_features_scaled = scaler.transform(pipeline_features)

    # Make predictions
    lr_pred = lr.predict(pipeline_features_scaled)[0]
    lr_prob = lr.predict_proba(pipeline_features_scaled)[0][1]
    rf_pred = rf.predict(pipeline_features_scaled)[0]
    rf_prob = rf.predict_proba(pipeline_features_scaled)[0][1]

    for i, row in pipeline_data.iterrows():
        print(f"  {row['ticker']} ({row['date']}):")
        print(f"    LR: {'UP' if lr_pred == 1 else 'DOWN'} ({lr_prob:.3f})")
        print(f"    RF: {'UP' if rf_pred == 1 else 'DOWN'} ({rf_prob:.3f})")
        print(f"    Features: sentiment={row['sentiment_score']:.3f}, confidence={row['sentiment_confidence']:.3f}")
        print(f"              price_delta={row['price_delta_24h']:.3f}, volume_delta={row['volume_delta']:.3f}")
# TODO (Abhi): Implement model export endpoint for use in backend API
# TODO (Abhi): Add prediction uncertainty quantification
# TODO (Abhi): Implement model monitoring to detect performance degradation
# TODO (Abhi): Add feature importance explanation for predictions