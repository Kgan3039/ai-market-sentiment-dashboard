"""Regression checks for backend experimental signal artifact prewarming."""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services import prediction_service
from app.services.prediction_service import PredictionService
import main as backend_main


def test_prediction_prewarm_uses_bootstrap_artifacts(monkeypatch) -> None:
    calls = {"bootstrap": 0}

    class FakeArtifacts:
        provenance = {
            "artifact_source": "disk",
            "artifact_path": "/tmp/model_artifacts.joblib",
            "artifact_version": "experimental-synthetic-v1",
            "trained_at": "2026-05-28T03:49:52+00:00",
        }

    class FakePredictionModule:
        @staticmethod
        def bootstrap_model_artifacts(force_retrain=False):
            calls["bootstrap"] += 1
            assert force_retrain is False
            return FakeArtifacts()

    monkeypatch.setattr(
        prediction_service,
        "_load_prediction_module",
        lambda: FakePredictionModule,
    )

    prewarm_info = PredictionService.prewarm_model_artifacts()

    assert calls["bootstrap"] == 1
    assert prewarm_info == {
        "status": "ready",
        "artifact_source": "disk",
        "artifact_path": "/tmp/model_artifacts.joblib",
        "version": "experimental-synthetic-v1",
        "trained_at": "2026-05-28T03:49:52+00:00",
    }


def test_startup_event_handles_signal_prewarm_failure(monkeypatch, caplog) -> None:
    def fail_prewarm():
        raise RuntimeError("artifact file is unreadable")

    monkeypatch.setattr(
        backend_main.PredictionService,
        "prewarm_model_artifacts",
        staticmethod(fail_prewarm),
    )

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        asyncio.run(backend_main.startup_event())

    assert "Experimental signal artifact preload failed" in caplog.text
    assert "Runtime signal loading will remain lazy" in caplog.text
