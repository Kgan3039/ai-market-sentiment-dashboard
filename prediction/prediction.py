#Prediction

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


# These are the columns we expect from the pipeline based on dataset_format.md
# The pipeline gives us aggregated sentiment features plus market features.
REQUIRED_COLUMNS = [
    "avg_sentiment_score",
    "avg_positive_prob",
    "avg_negative_prob",
    "avg_neutral_prob",
    "price_delta_24h",
    "volume_delta",
    "label"
]

# These are the actual features the model will train on
FEATURES = [
    "sentiment_score",
    "sentiment_confidence",
    "price_delta_24h",
    "volume_delta"
]

TARGET = "label"
_MODEL_CACHE = None


def prepare_data(df):
    """
    Takes in the real dataframe from the pipeline and makes sure
    it has everything needed for the model.

    We also create the final model features here.

    Input:
        df = dataframe from the real pipeline

    Output:
        dataframe with model-ready columns added
    """

    # Make a copy so we do not accidentally change the original dataframe
    df = df.copy()

    # Check that all required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Based on the pipeline contract:
    # avg_sentiment_score becomes the model's sentiment_score
    df["sentiment_score"] = df["avg_sentiment_score"]

    # Sentiment confidence is the highest of the three probabilities
    # Example:
    # positive = 0.70, negative = 0.10, neutral = 0.20
    # confidence = 0.70
    df["sentiment_confidence"] = df[
        ["avg_positive_prob", "avg_negative_prob", "avg_neutral_prob"]
    ].max(axis=1)

    return df


def train_models(df):
    """
    Trains both a Logistic Regression model and a Random Forest model.

    Returns:
        lr      = trained logistic regression model
        rf      = trained random forest model
        scaler  = fitted scaler for feature normalization
        results = dictionary with evaluation metrics
    """

    # First, clean and format the data
    df = prepare_data(df)

    # Separate input features (X) and the target label (y)
    X = df[FEATURES]
    y = df[TARGET]

    # We do NOT shuffle because this is supposed to respect time order.
    # Earlier rows should train the model, later rows should test it.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale the features.
    # This is especially important for Logistic Regression.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Dataset shape:", df.shape)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print()

    # ------------------------------------------------------------
    # Model 1: Logistic Regression
    # ------------------------------------------------------------
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    print("── Logistic Regression ──────────────────────")
    print(f"Accuracy : {accuracy_score(y_test, lr_preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, lr_preds):.3f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, lr_probs):.3f}")
    print()
    print(classification_report(y_test, lr_preds, target_names=["Down", "Up"]))

    # ------------------------------------------------------------
    # Model 2: Random Forest
    # ------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    rf_preds = rf.predict(X_test_scaled)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    print("── Random Forest ────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_test, rf_preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, rf_preds):.3f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, rf_probs):.3f}")
    print()
    print(classification_report(y_test, rf_preds, target_names=["Down", "Up"]))

    # Show feature importance for Random Forest
    print("── Feature Importances (RF) ─────────────────")
    for feature, importance in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"{feature:<24} {importance:.3f}")
    print()

    # Save results in case we want to use them later
    results = {
        "lr_accuracy": accuracy_score(y_test, lr_preds),
        "lr_f1": f1_score(y_test, lr_preds),
        "lr_auc": roc_auc_score(y_test, lr_probs),
        "rf_accuracy": accuracy_score(y_test, rf_preds),
        "rf_f1": f1_score(y_test, rf_preds),
        "rf_auc": roc_auc_score(y_test, rf_probs),
    }

    return lr, rf, scaler, results


def predict_single(row, lr, rf, scaler, model="lr"):
    """
    Makes a prediction for one new example.

    The input row should be in the same format as the pipeline output,
    meaning it should have:
        avg_sentiment_score
        avg_positive_prob
        avg_negative_prob
        avg_neutral_prob
        price_delta_24h
        volume_delta

    Example input:
        {
            "avg_sentiment_score": 0.41,
            "avg_positive_prob": 0.58,
            "avg_negative_prob": 0.19,
            "avg_neutral_prob": 0.23,
            "price_delta_24h": 0.012,
            "volume_delta": 0.10,
            "label": 1
        }

    The label is not needed for prediction, but it is okay if it's there.
    """

    # Turn the single dictionary into a dataframe with one row
    one_row = pd.DataFrame([row])

    # Prepare it the same way we prepared the training data
    one_row = prepare_data(one_row)

    # Pull out just the model features
    X_new = one_row[FEATURES]

    # Scale them using the same scaler from training
    X_new_scaled = scaler.transform(X_new)

    # Choose which model to use
    chosen_model = lr if model == "lr" else rf

    # Predict class (0 or 1)
    pred = chosen_model.predict(X_new_scaled)[0]

    # Get prediction probabilities
    probs = chosen_model.predict_proba(X_new_scaled)[0]
    confidence = probs[pred]

    return {
        "direction": "up" if pred == 1 else "down",
        "confidence": round(float(confidence), 4),
        "model": model
    }


def predict_batch(df, lr, rf, scaler, model="lr"):
    """
    Makes predictions for a whole dataframe at once.
    Useful if we want to score many rows from the pipeline.
    """

    df = prepare_data(df)

    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    chosen_model = lr if model == "lr" else rf

    df["prediction"] = chosen_model.predict(X_scaled)
    df["confidence"] = chosen_model.predict_proba(X_scaled).max(axis=1)

    return df


def _synthetic_training_frame():
    """Create a lightweight training set so backend inference can run locally."""
    rows = []
    for sentiment_score in (-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8):
        for sentiment_confidence in (0.55, 0.7, 0.85, 0.95):
            for price_delta in (-0.04, -0.015, 0.015, 0.04):
                for volume_delta in (-0.2, 0.0, 0.2):
                    if sentiment_score > 0.05:
                        avg_positive_prob = sentiment_confidence
                        avg_negative_prob = max(0.02, 0.25 - sentiment_score / 4)
                        avg_neutral_prob = max(0.02, 1 - avg_positive_prob)
                    elif sentiment_score < -0.05:
                        avg_negative_prob = sentiment_confidence
                        avg_positive_prob = max(0.02, 0.25 + sentiment_score / 4)
                        avg_neutral_prob = max(0.02, 1 - avg_negative_prob)
                    else:
                        avg_neutral_prob = sentiment_confidence
                        avg_positive_prob = max(0.02, (1 - avg_neutral_prob) / 2)
                        avg_negative_prob = max(0.02, (1 - avg_neutral_prob) / 2)

                    rows.append({
                        "avg_sentiment_score": sentiment_score,
                        "avg_positive_prob": min(avg_positive_prob, 0.98),
                        "avg_negative_prob": min(avg_negative_prob, 0.98),
                        "avg_neutral_prob": min(avg_neutral_prob, 0.98),
                        "price_delta_24h": price_delta,
                        "volume_delta": volume_delta,
                        "label": int(price_delta + (0.35 * sentiment_score) > 0),
                    })

    return pd.DataFrame(rows)


def _ensure_models():
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        lr, rf, scaler, _ = train_models(_synthetic_training_frame())
        _MODEL_CACHE = {
            "lr": lr,
            "rf": rf,
            "scaler": scaler,
        }

    return _MODEL_CACHE


def _row_from_features(sentiment_score, sentiment_confidence, price_delta_24h, volume_delta):
    if sentiment_score > 0.05:
        return {
            "avg_sentiment_score": sentiment_score,
            "avg_positive_prob": sentiment_confidence,
            "avg_negative_prob": max(0.02, 0.2 - (sentiment_score / 4)),
            "avg_neutral_prob": max(0.02, 1 - sentiment_confidence),
            "price_delta_24h": price_delta_24h,
            "volume_delta": volume_delta,
            "label": int(price_delta_24h + (0.35 * sentiment_score) > 0),
        }

    if sentiment_score < -0.05:
        return {
            "avg_sentiment_score": sentiment_score,
            "avg_positive_prob": max(0.02, 0.2 + (sentiment_score / 4)),
            "avg_negative_prob": sentiment_confidence,
            "avg_neutral_prob": max(0.02, 1 - sentiment_confidence),
            "price_delta_24h": price_delta_24h,
            "volume_delta": volume_delta,
            "label": int(price_delta_24h + (0.35 * sentiment_score) > 0),
        }

    neutral_prob = max(sentiment_confidence, 0.34)
    side_prob = max(0.02, (1 - neutral_prob) / 2)
    return {
        "avg_sentiment_score": sentiment_score,
        "avg_positive_prob": side_prob,
        "avg_negative_prob": side_prob,
        "avg_neutral_prob": neutral_prob,
        "price_delta_24h": price_delta_24h,
        "volume_delta": volume_delta,
        "label": int(price_delta_24h + (0.35 * sentiment_score) > 0),
    }


def predict(sentiment_score, sentiment_confidence, price_delta_24h, volume_delta, model="lr"):
    """
    Backend-compatible prediction entrypoint.

    Returns:
        { "direction": "up"/"down", "confidence": float, "model": str }
    """
    models = _ensure_models()
    row = _row_from_features(
        sentiment_score=sentiment_score,
        sentiment_confidence=sentiment_confidence,
        price_delta_24h=price_delta_24h,
        volume_delta=volume_delta,
    )
    return predict_single(
        row=row,
        lr=models["lr"],
        rf=models["rf"],
        scaler=models["scaler"],
        model=model,
    )

