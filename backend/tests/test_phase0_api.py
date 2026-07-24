"""Contract tests for Mihir's Phase 0 fixture-first read API."""

import json
import sys
from pathlib import Path
from statistics import quantiles
from time import perf_counter

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from main import app


client = TestClient(app)
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_phase0_copy_rules import detected_categories, load_rules


def test_tickers_endpoint_returns_the_fixed_phase0_universe():
    response = client.get("/api/v1/tickers")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"data_as_of", "tickers"}
    assert [item["ticker"] for item in payload["tickers"]] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "META",
    ]
    assert all({"ticker", "company_name", "data_as_of", "theme_count", "is_stale"} == set(item) for item in payload["tickers"])


def test_themes_endpoint_resolves_every_summary_citation():
    response = client.get("/api/v1/tickers/NVDA/themes?date=2026-07-15")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"ticker", "date", "data_as_of", "themes", "other_coverage"}
    theme = payload["themes"][0]
    assert set(theme) == {
        "id",
        "label",
        "rank",
        "sentences",
        "citations",
        "stories",
        "outlet_count",
        "story_count",
        "degraded",
    }
    citation_ids = {citation["id"] for citation in theme["citations"]}
    assert not theme["degraded"]
    assert 2 <= len(theme["sentences"]) <= 4
    assert all(set(sentence["citation_ids"]).issubset(citation_ids) for sentence in theme["sentences"])


def test_degraded_and_empty_states_are_contractual_not_errors():
    degraded = client.get("/api/v1/tickers/TSLA/themes").json()["themes"][0]
    empty = client.get("/api/v1/tickers/AMD/themes?date=2026-07-14")

    assert degraded["degraded"] is True
    assert degraded["sentences"] == []
    assert degraded["stories"]
    assert empty.status_code == 200
    assert empty.json()["themes"] == []
    assert empty.json()["other_coverage"]["stories"] == []


def test_meta_status_and_legacy_product_routes():
    status = client.get("/api/v1/meta/status")

    assert status.status_code == 200
    assert {"data_as_of", "is_stale", "last_runs"} == set(status.json())
    assert {entry["stage"] for entry in status.json()["last_runs"]} == {
        "fetch",
        "relevance",
        "dedup",
        "cluster",
        "summarize",
    }
    assert client.get("/sentiment/NVDA").status_code == 404
    assert client.get("/prediction/NVDA").status_code == 404
    assert client.get("/dashboard/summary/NVDA").status_code == 404


def test_fixture_read_p95_is_under_300ms():
    samples = []
    for _ in range(25):
        started_at = perf_counter()
        response = client.get("/api/v1/tickers/NVDA/themes")
        samples.append(perf_counter() - started_at)
        assert response.status_code == 200

    p95 = quantiles(samples, n=20, method="inclusive")[18]
    assert p95 < 0.3


def test_ui_uses_approved_copy_and_phase0_endpoint_contract():
    app_source = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    required_copy = (
        "Ticker Narratives",
        "Themes dominating current coverage",
        "Key narratives around today’s move",
        "Summary unavailable — source stories are still available",
        "Loading current coverage…",
        "No current coverage for",
        "Check back after the next update.",
        "Coverage is temporarily unavailable. Please try again shortly.",
        "AI-generated from cited sources. Informational only — not investment advice.",
    )
    forbidden_legacy_surfaces = ("/sentiment/", "/prediction/", "/dashboard/summary")

    assert all(copy in app_source for copy in required_copy)
    assert all(surface not in app_source for surface in forbidden_legacy_surfaces)
    assert "/api/v1/tickers" in app_source


def test_fixture_generated_content_and_ui_copy_pass_banned_language_rules():
    fixture_path = PROJECT_ROOT / "backend" / "app" / "phase0" / "fixtures" / "phase0_narratives.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rules = load_rules()
    app_source = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")

    for ticker_days in fixture["ticker_days"].values():
        for day in ticker_days.values():
            for theme in day["themes"]:
                assert not detected_categories(theme["label"], rules, content_scope="generated_label")
                for sentence in theme["sentences"]:
                    assert not detected_categories(
                        sentence["text"], rules, content_scope="generated_summary"
                    )

    assert not detected_categories(app_source, rules, content_scope="product_ui_copy")
