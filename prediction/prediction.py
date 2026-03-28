# ============================================================
# AI Market Sentiment — Prediction Model (Issue #3)
# Author: Abhi

 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
 
# ============================================================
# 1. LOAD DATA
# Replace this dummy data with Isaac's real pipeline output
# Expected columns from Matthew: sentiment_score, sentiment_confidence
# Expected columns from Isaac:   price_delta_24h, volume_delta
# ============================================================
 
np.random.seed(42)
N = 500
 
data = pd.DataFrame({
    "sentiment_score":      np.random.uniform(-1, 1, N),       # From Matthew (NLP)
    "sentiment_confidence": np.random.uniform(0.5, 1.0, N),    # From Matthew (NLP)
    "price_delta_24h":      np.random.uniform(-0.05, 0.05, N), # From Isaac (Data)
    "volume_delta":         np.random.uniform(-0.3, 0.3, N),   # From Isaac (Data)
})
 
# Label: 1 = price went up, 0 = price went down/flat
# TODO: Replace with real historical labels from Isaac
data["label"] = (data["price_delta_24h"] + 0.3 * data["sentiment_score"] > 0).astype(int)
 
print("Dataset shape:", data.shape)
print("Label distribution:\n", data["label"].value_counts())
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