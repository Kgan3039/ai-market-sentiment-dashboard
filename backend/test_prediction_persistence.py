"""Regression checks for persisted prediction model artifacts."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prediction")))

import prediction as prediction_module


def test_prediction_artifacts_persist_and_reload(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "model_artifacts.joblib"
    monkeypatch.setenv("PREDICTION_MODEL_ARTIFACT_PATH", str(artifact_path))

    prediction_module._MODEL_CACHE = None
    first_artifacts = prediction_module.get_model_artifacts()
    first_prediction = prediction_module.predict(
        sentiment_score=0.41,
        sentiment_confidence=0.58,
        price_delta_24h=0.012,
        volume_delta=0.10,
        model="rf",
    )

    assert artifact_path.exists()
    assert first_artifacts.provenance["artifact_source"] == "trained_and_persisted"

    prediction_module._MODEL_CACHE = None
    second_artifacts = prediction_module.get_model_artifacts()
    second_prediction = prediction_module.predict(
        sentiment_score=0.41,
        sentiment_confidence=0.58,
        price_delta_24h=0.012,
        volume_delta=0.10,
        model="rf",
    )
    provenance = prediction_module.get_model_provenance("rf")

    assert second_artifacts.provenance["artifact_source"] == "disk"
    assert second_artifacts.provenance["trained_at"] == first_artifacts.provenance["trained_at"]
    assert second_prediction == first_prediction
    assert provenance["name"] == "RandomForestClassifier"
    assert provenance["version"] == prediction_module.ARTIFACT_VERSION
    assert provenance["artifact_path"] == str(artifact_path)
