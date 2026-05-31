"""Prediction utilities for the market sentiment pipeline.

This module keeps the project aligned with ``docs/dataset_format.md``:
- training/inference inputs use aggregated ``avg_*`` sentiment fields
- model features are derived internally from those fields
- backend can call ``predict(...)`` directly for single inference
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
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
ARTIFACT_VERSION = "synthetic-demo-v1"
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("models") / "model_artifacts.joblib"
RUNTIME_CALIBRATION_VERSION = "demo-runtime-calibration-v2"
PROBABILITY_CLIP = (0.18, 0.82)
MODEL_PROBABILITY_WEIGHT = 0.60
SIGNAL_PROBABILITY_WEIGHT = 1.0 - MODEL_PROBABILITY_WEIGHT
SIGNAL_SENTIMENT_WEIGHT = 0.35
SIGNAL_PRICE_WEIGHT = 1.50
SIGNAL_SCALE = 8.0
NEUTRAL_BAND = 0.02
CONFIDENCE_MIN = 0.52
CONFIDENCE_MAX = 0.78


@dataclass
class ModelArtifacts:
    lr: LogisticRegression
    rf: RandomForestClassifier
    scaler: StandardScaler
    results: Dict[str, float]
    provenance: Dict[str, object]


_MODEL_CACHE: Optional[ModelArtifacts] = None


def _artifact_path() -> Path:
    configured = os.getenv("PREDICTION_MODEL_ARTIFACT_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_ARTIFACT_PATH


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


def _build_model_artifacts() -> ModelArtifacts:
    training_df = _build_synthetic_training_data()
    lr, rf, scaler, results = train_models(training_df)
    now = datetime.now(timezone.utc).isoformat()
    provenance = {
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "trained_at": now,
        "training_data": "deterministic_synthetic_v1",
        "training_rows": len(training_df),
        "features": FEATURES,
        "calibration": {
            "confidence_min": CONFIDENCE_MIN,
            "confidence_max": CONFIDENCE_MAX,
            "probability_clip": list(PROBABILITY_CLIP),
            "model_probability_weight": MODEL_PROBABILITY_WEIGHT,
            "signal_probability_weight": SIGNAL_PROBABILITY_WEIGHT,
            "neutral_band": NEUTRAL_BAND,
            "runtime_calibration_version": RUNTIME_CALIBRATION_VERSION,
        },
        "metrics": results,
    }
    return ModelArtifacts(lr=lr, rf=rf, scaler=scaler, results=results, provenance=provenance)


def save_model_artifacts(artifacts: ModelArtifacts, artifact_path: Optional[Path] = None) -> Path:
    """Persist trained models, scaler, metrics, and provenance to disk."""
    path = artifact_path or _artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "lr": artifacts.lr,
        "rf": artifacts.rf,
        "scaler": artifacts.scaler,
        "results": artifacts.results,
        "provenance": artifacts.provenance,
    }
    joblib.dump(payload, path)
    return path


def load_model_artifacts(artifact_path: Optional[Path] = None) -> ModelArtifacts:
    """Load persisted model artifacts from disk."""
    path = artifact_path or _artifact_path()
    payload = joblib.load(path)

    required_keys = {"schema_version", "artifact_version", "lr", "rf", "scaler", "results", "provenance"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Model artifact missing keys: {sorted(missing)}")
    if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model artifact schema: {payload['schema_version']}"
        )

    provenance = dict(payload["provenance"])
    provenance.update(
        {
            "artifact_source": "disk",
            "artifact_path": str(path),
        }
    )
    return ModelArtifacts(
        lr=payload["lr"],
        rf=payload["rf"],
        scaler=payload["scaler"],
        results=payload["results"],
        provenance=provenance,
    )


def get_model_artifacts() -> ModelArtifacts:
    """Load persisted demo models, training and saving only when artifacts are absent."""
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        path = _artifact_path()
        if path.exists():
            _MODEL_CACHE = load_model_artifacts(path)
        else:
            artifacts = _build_model_artifacts()
            try:
                saved_path = save_model_artifacts(artifacts, path)
                artifacts.provenance.update(
                    {
                        "artifact_source": "trained_and_persisted",
                        "artifact_path": str(saved_path),
                    }
                )
            except Exception:
                artifacts.provenance.update(
                    {
                        "artifact_source": "trained_in_memory",
                        "artifact_path": str(path),
                    }
                )
            _MODEL_CACHE = artifacts

    return _MODEL_CACHE


def bootstrap_model_artifacts(force_retrain: bool = False) -> ModelArtifacts:
    """Ensure a persisted artifact exists and return the loaded artifacts."""
    global _MODEL_CACHE
    if force_retrain:
        artifacts = _build_model_artifacts()
        saved_path = save_model_artifacts(artifacts)
        _MODEL_CACHE = None
        loaded = load_model_artifacts(saved_path)
        _MODEL_CACHE = loaded
        return loaded

    return get_model_artifacts()


def get_model_provenance(model: str = "rf") -> Dict[str, object]:
    """Return lightweight metadata about the serving prediction model."""
    artifacts = get_model_artifacts()
    model_name = "LogisticRegression" if model == "lr" else "RandomForestClassifier"
    metric_prefix = "lr" if model == "lr" else "rf"
    metrics = {
        key: value
        for key, value in artifacts.results.items()
        if key.startswith(metric_prefix)
    }

    return {
        "name": model_name,
        "version": artifacts.provenance.get("artifact_version", ARTIFACT_VERSION),
        "artifact_source": artifacts.provenance.get("artifact_source"),
        "artifact_path": artifacts.provenance.get("artifact_path"),
        "trained_at": artifacts.provenance.get("trained_at"),
        "training_data": artifacts.provenance.get("training_data"),
        "features_used": FEATURES,
        "calibration": {
            **(artifacts.provenance.get("calibration") or {}),
            "confidence_min": CONFIDENCE_MIN,
            "confidence_max": CONFIDENCE_MAX,
            "probability_clip": list(PROBABILITY_CLIP),
            "model_probability_weight": MODEL_PROBABILITY_WEIGHT,
            "signal_probability_weight": SIGNAL_PROBABILITY_WEIGHT,
            "neutral_band": NEUTRAL_BAND,
            "runtime_calibration_version": RUNTIME_CALIBRATION_VERSION,
        },
        "metrics": metrics,
    }


def _bounded_signal_probability(row: Dict[str, float]) -> float:
    signal = (
        SIGNAL_SENTIMENT_WEIGHT * float(row.get("sentiment_score", 0.0))
        + SIGNAL_PRICE_WEIGHT * float(row.get("price_delta_24h", 0.0))
    )
    return float(1.0 / (1.0 + np.exp(-SIGNAL_SCALE * signal)))


def _calibrate_probability(row: Dict[str, float], model_probability: float) -> float:
    clipped_model_probability = float(np.clip(model_probability, *PROBABILITY_CLIP))
    signal_probability = _bounded_signal_probability(row)
    blended_probability = (
        MODEL_PROBABILITY_WEIGHT * clipped_model_probability
        + SIGNAL_PROBABILITY_WEIGHT * signal_probability
    )
    return round(float(np.clip(blended_probability, *PROBABILITY_CLIP)), 4)


def _calibrate_confidence(row: Dict[str, float], probability: float) -> float:
    price_evidence = min(abs(float(row.get("price_delta_24h", 0.0))) * 2.0, 0.04)
    sentiment_evidence = min(abs(float(row.get("sentiment_score", 0.0))) * 0.06, 0.04)
    volume_evidence = min(abs(float(row.get("volume_delta", 0.0))) * 0.02, 0.02)
    confidence = max(probability, 1.0 - probability)
    confidence += price_evidence + sentiment_evidence + volume_evidence
    return round(float(np.clip(confidence, CONFIDENCE_MIN, CONFIDENCE_MAX)), 4)

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
    probs = chosen_model.predict_proba(X_new_scaled)[0]
    probability = _calibrate_probability(row, float(probs[1]))
    confidence = _calibrate_confidence(row, probability)
    if probability > 0.5 + NEUTRAL_BAND:
        movement = "up"
    elif probability < 0.5 - NEUTRAL_BAND:
        movement = "down"
    else:
        movement = "neutral"

    return {
        "predicted_movement": movement,
        "probability": probability,
        "confidence": confidence,
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
