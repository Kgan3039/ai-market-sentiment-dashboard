"""Prediction utilities for the market sentiment pipeline.

This module keeps the project aligned with ``docs/dataset_format.md``:
- training/inference inputs use aggregated ``avg_*`` sentiment fields
- model features are derived internally from those fields
- backend can call ``predict(...)`` directly for single inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_FEATURE_COLUMNS = [
    "avg_sentiment_score",
    "avg_positive_prob",
    "avg_negative_prob",
    "avg_neutral_prob",
    "price_delta_24h",
    "volume_delta",
]

REQUIRED_TRAINING_COLUMNS = REQUIRED_FEATURE_COLUMNS + ["label"]

FEATURES = [
    "sentiment_score",
    "sentiment_confidence",
    "price_delta_24h",
    "volume_delta",
]

TARGET = "label"


@dataclass
class ModelArtifacts:
    lr: LogisticRegression
    rf: RandomForestClassifier
    scaler: StandardScaler
    results: Dict[str, float]


_MODEL_CACHE: Optional[ModelArtifacts] = None


def _build_synthetic_training_data(size: int = 500) -> pd.DataFrame:
    """Create deterministic fallback training data for local/demo inference."""
    np.random.seed(42)

    avg_positive_prob = np.random.uniform(0.2, 0.9, size)
    avg_negative_prob = np.random.uniform(0.0, 0.6, size)
    avg_neutral_prob = np.clip(
        1.0 - avg_positive_prob - avg_negative_prob,
        0.0,
        1.0,
    )

    df = pd.DataFrame(
        {
            "avg_sentiment_score": avg_positive_prob - avg_negative_prob,
            "avg_positive_prob": avg_positive_prob,
            "avg_negative_prob": avg_negative_prob,
            "avg_neutral_prob": avg_neutral_prob,
            "price_delta_24h": np.random.uniform(-0.05, 0.05, size),
            "volume_delta": np.random.uniform(-0.3, 0.3, size),
        }
    )

    signal = df["price_delta_24h"] + 0.3 * df["avg_sentiment_score"]
    df["label"] = (signal > 0).astype(int)
    return df


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def prepare_data(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    """Convert aggregated pipeline rows into model-ready features."""
    df = df.copy()
    required = REQUIRED_TRAINING_COLUMNS if require_label else REQUIRED_FEATURE_COLUMNS
    _validate_columns(df, required)

    df["sentiment_score"] = df["avg_sentiment_score"]
    df["sentiment_confidence"] = df[
        ["avg_positive_prob", "avg_negative_prob", "avg_neutral_prob"]
    ].max(axis=1)

    return df


def train_models(df: pd.DataFrame) -> Tuple[LogisticRegression, RandomForestClassifier, StandardScaler, Dict[str, float]]:
    """Train logistic regression and random forest models."""
    df = prepare_data(df, require_label=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    rf_preds = rf.predict(X_test_scaled)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    results = {
        "lr_accuracy": accuracy_score(y_test, lr_preds),
        "lr_f1": f1_score(y_test, lr_preds),
        "lr_auc": roc_auc_score(y_test, lr_probs),
        "rf_accuracy": accuracy_score(y_test, rf_preds),
        "rf_f1": f1_score(y_test, rf_preds),
        "rf_auc": roc_auc_score(y_test, rf_probs),
    }

    return lr, rf, scaler, results


def get_model_artifacts() -> ModelArtifacts:
    """Lazily train local demo models once and cache them."""
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        training_df = _build_synthetic_training_data()
        lr, rf, scaler, results = train_models(training_df)
        _MODEL_CACHE = ModelArtifacts(lr=lr, rf=rf, scaler=scaler, results=results)

    return _MODEL_CACHE


CONFIDENCE_MIN = 0.52
CONFIDENCE_MAX = 0.84

def predict_single(
    row: Dict[str, float],
    lr: LogisticRegression,
    rf: RandomForestClassifier,
    scaler: StandardScaler,
    model: str = "lr",
) -> Dict[str, float | str]:
    """Predict one aggregated pipeline row."""
    one_row = pd.DataFrame([row])
    one_row = prepare_data(one_row, require_label=False)
    X_new = one_row[FEATURES]
    X_new_scaled = scaler.transform(X_new)

    chosen_model = lr if model == "lr" else rf
    pred = chosen_model.predict(X_new_scaled)[0]
    probs = chosen_model.predict_proba(X_new_scaled)[0]
    confidence = probs[pred]

    return {
        "predicted_movement": "up" if pred == 1 else "down",
        "probability": round(float(np.clip(probs[1], 0.18, 0.82)), 4),
        "confidence": round(float(np.clip(confidence, CONFIDENCE_MIN, CONFIDENCE_MAX)), 4),
    }


def predict(
    sentiment_score: float,
    sentiment_confidence: float,
    price_delta_24h: float,
    volume_delta: float,
    model: str = "lr",
) -> Dict[str, float | str]:
    """Backend-friendly prediction wrapper.

    Assumptions (Abhi):
    - sentiment_score: float in [-1, 1], derived as avg_positive_prob - avg_negative_prob
    - sentiment_confidence: max(positive, negative, neutral) prob from NLP model
    - price_delta_24h: (close - open) / open for last 24h; clipped to ±8% pre-inference
    - volume_delta: (today_vol - avg_vol) / avg_vol; clipped to ±50% pre-inference
    - Confidence output soft-clipped to [0.52, 0.84] for demo plausibility
    - Model trained on synthetic data; label = sign(price_delta + 0.3 * sentiment)
    """
    artifacts = get_model_artifacts()

    # Clip extremes before inference
    price_delta_24h = float(np.clip(price_delta_24h, -0.08, 0.08))
    volume_delta = float(np.clip(volume_delta, -0.5, 0.5))

    # Synthesize avg_* columns more realistically
    pos_weight = max(0.0, 0.5 + sentiment_score / 2)
    neg_weight = max(0.0, 0.5 - sentiment_score / 2)
    row = {
        "avg_sentiment_score": sentiment_score,
        "avg_positive_prob": round(sentiment_confidence * pos_weight, 4),
        "avg_negative_prob": round(sentiment_confidence * neg_weight, 4),
        "avg_neutral_prob": round(max(0.0, 1.0 - sentiment_confidence), 4),
        "price_delta_24h": price_delta_24h,
        "volume_delta": volume_delta,
    }

    return predict_single(
        row=row,
        lr=artifacts.lr,
        rf=artifacts.rf,
        scaler=artifacts.scaler,
        model=model,
    )


def predict_batch(
    df: pd.DataFrame,
    lr: LogisticRegression,
    rf: RandomForestClassifier,
    scaler: StandardScaler,
    model: str = "lr",
) -> pd.DataFrame:
    """Predict a batch of aggregated pipeline rows."""
    prepared = prepare_data(df, require_label=False)
    X = prepared[FEATURES]
    X_scaled = scaler.transform(X)

    chosen_model = lr if model == "lr" else rf
    preds = chosen_model.predict(X_scaled)
    probs = chosen_model.predict_proba(X_scaled)

    return pd.DataFrame(
        {
            "predicted_movement": ["up" if p == 1 else "down" for p in preds],
            "probability": probs[:, 1].round(4),
            "confidence": probs[np.arange(len(preds)), preds].round(4),
        }
    )


if __name__ == "__main__":
    artifacts = get_model_artifacts()
    print("Dataset shape:", _build_synthetic_training_data().shape)
    print("Train metrics:", artifacts.results)
    print(
        classification_report(
            [0, 1],
            [0, 1],
            target_names=["Down", "Up"],
            zero_division=0,
        )
    )
    print(
        predict(
            sentiment_score=0.41,
            sentiment_confidence=0.58,
            price_delta_24h=0.012,
            volume_delta=0.10,
            model="rf",
        )
    )

    result = predict(
        sentiment_score=0.41,
        sentiment_confidence=0.58,
        price_delta_24h=0.012,
        volume_delta=0.10,
        model="lr",
    )
    assert set(result.keys()) == {"predicted_movement", "probability", "confidence"}, \
        f"Contract mismatch: {result.keys()}"
    assert result["predicted_movement"] in ("up", "down")
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    print("predict() contract verification passed:", result)

    batch_input = pd.DataFrame(
        [
            {
                "avg_sentiment_score": 0.41,
                "avg_positive_prob": 0.58,
                "avg_negative_prob": 0.19,
                "avg_neutral_prob": 0.23,
                "price_delta_24h": 0.012,
                "volume_delta": 0.10,
            }
        ]
    )
    batch_result = predict_batch(
        batch_input,
        artifacts.lr,
        artifacts.rf,
        artifacts.scaler,
        model="lr",
    )
    assert list(batch_result.columns) == ["predicted_movement", "probability", "confidence"], \
        f"Batch contract mismatch: {batch_result.columns.tolist()}"
    print("predict_batch() contract verification passed:", batch_result.to_dict(orient="records"))
