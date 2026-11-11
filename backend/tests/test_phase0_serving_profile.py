"""B1 serving-surface isolation for the Phase 0 private pilot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import main as backend_main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.services.prediction_service import PredictionService  # noqa: E402

PHASE0_PATHS = {
    "/test",
    "/api/v1/tickers",
    "/api/v1/tickers/{ticker}/themes",
    "/api/v1/meta/status",
}
LEGACY_PATHS = {
    "/sentiment/{ticker}",
    "/prediction/{ticker}",
    "/market/{ticker}",
    "/dashboard/summary/{ticker}",
}


def test_phase0_profile_serves_exact_surface_without_prewarming(monkeypatch):
    calls = 0

    def refuse_prewarm():
        nonlocal calls
        calls += 1
        raise AssertionError("Phase 0 startup reached prediction prewarming")

    monkeypatch.setattr(
        PredictionService,
        "prewarm_model_artifacts",
        staticmethod(refuse_prewarm),
    )
    app = backend_main.create_app(serving_profile="phase0")

    with TestClient(app) as client:
        paths = set(client.get("/openapi.json").json()["paths"])
        assert paths == PHASE0_PATHS
        assert client.get("/test").status_code == 200
        assert client.get("/api/v1/tickers").status_code == 200

    assert calls == 0
    assert app.state.serving_profile == "phase0"


def test_full_profile_preserves_legacy_surface_and_startup_handler():
    app = backend_main.create_app(serving_profile="full")
    paths = set(app.openapi()["paths"])

    assert PHASE0_PATHS <= paths
    assert LEGACY_PATHS <= paths
    assert backend_main.startup_event in app.router.on_startup
    assert app.state.serving_profile == "full"


def test_phase0_process_does_not_import_legacy_or_prediction_modules():
    code = """
import json, sys
import main
print(json.dumps({
    "paths": sorted(main.app.openapi()["paths"]),
    "legacy_modules": sorted(
        name for name in sys.modules
        if name in {
            "app.routes.sentiment",
            "app.routes.prediction",
            "app.routes.market",
            "app.routes.dashboard",
            "app.services.prediction_service",
        }
    ),
    "startup_handlers": len(main.app.router.on_startup),
}))
"""
    environment = os.environ.copy()
    environment["API_SERVING_PROFILE"] = "phase0"
    environment["PYTHONPATH"] = os.pathsep.join([str(BACKEND_ROOT), str(PROJECT_ROOT)])
    environment.pop("DEBUG", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert set(payload["paths"]) == PHASE0_PATHS
    assert payload["legacy_modules"] == []
    assert payload["startup_handlers"] == 0


def test_serving_profile_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("API_SERVING_PROFILE", raising=False)
    assert Settings(_env_file=None).API_SERVING_PROFILE == "full"
    assert (
        Settings(_env_file=None, API_SERVING_PROFILE="phase0").API_SERVING_PROFILE
        == "phase0"
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None, API_SERVING_PROFILE="other")
