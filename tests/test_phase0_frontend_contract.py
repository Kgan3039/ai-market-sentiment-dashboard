"""B2: the frontend contract artifact, captured from the real B1 API.

One persisted world is written through the real Phase 0 write paths (story
and theme reconciliation, and A3's ``ensure_summary`` with B1's fake
provider), read by the real ``SqliteNarrativeRepository`` and serialized by
the real FastAPI routes.  The captured responses are the checked-in
artifact the React contract tests render
(``frontend/src/App.contract.test.jsx``).

Normal runs fail when the artifact drifts from what the backend produces.
Regenerate it only deliberately::

    PHASE0_WRITE_FRONTEND_CONTRACT=1 .venv/bin/python -m pytest \\
        -p no:cacheprovider -q tests/test_phase0_frontend_contract.py

The artifact is acceptance evidence for the frontend; it says nothing about
which source a deployment serves.  Generated sentences come from the fake
provider, passed through real A3 persistence and currentness.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

WRITE_ENV = "PHASE0_WRITE_FRONTEND_CONTRACT"
#: The explicit write opt-in, captured before any isolation strips PHASE0_*.
WRITE_MODE = os.environ.get(WRITE_ENV) == "1"
ISOLATED_PREFIXES = ("GEMINI_", "PHASE0_")
#: Set in the hostile-environment child so it never spawns another child.
SUBPROCESS_SENTINEL = "PHASE0_FRONTEND_CONTRACT_CHILD"


@contextmanager
def _without_ambient_config() -> Iterator[None]:
    """Hide GEMINI_* and PHASE0_* for the block, then restore them exactly.

    Importing the B1 helpers imports the FastAPI app, which builds
    ``Settings()`` from the environment at import time -- before any pytest
    fixture can run.  A hostile exported value must neither break collection
    nor shape the app; the caller's environment is put back afterwards.
    """

    saved = {k: v for k, v in os.environ.items() if k.startswith(ISOLATED_PREFIXES)}
    for name in saved:
        del os.environ[name]
    try:
        yield
    finally:
        for name in [k for k in os.environ if k.startswith(ISOLATED_PREFIXES)]:
            del os.environ[name]
        os.environ.update(saved)


with _without_ambient_config():
    from test_phase0_narrative_sqlite_api import (  # noqa: F401  (fixtures by name)
        DAY,
        GENERATED_LABEL,
        NOW,
        UNIVERSE,
        VERSION,
        Clock,
        Phase0Repository,
        World,
        app,
        assert_contract,
        generation_forbidden,
        narrative,
        network_forbidden,
        read_api,
        routes,
        seed,
        seed_m2_only,
        summarize,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "src"
    / "test"
    / "fixtures"
    / "phase0_sqlite_contract.json"
)
REGENERATE = (
    f"{WRITE_ENV}=1 .venv/bin/python -m pytest -p no:cacheprovider -q "
    "tests/test_phase0_frontend_contract.py"
)

#: A real-shaped M5 label: a canonical headline, Unicode, over 120 chars.
LONG_UNICODE_LABEL = (
    "Tesla’s Grünheide Gigafactory expansion — and Shanghai 上海 output "
    "guidance — draw renewed analyst scrutiny across Europe and Asia this week"
)
#: AAPL's runs complete here: over an hour before NOW, inside the session.
STALE_AT = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No exported GEMINI_* or PHASE0_* variable may shape the artifact.

    The API resolves the production summary policy from GEMINI_*; a
    developer's model setting would change the policy fingerprint and turn
    current summaries degraded.  The write switch is read at import.
    """

    for name in list(os.environ):
        if name.startswith(ISOLATED_PREFIXES):
            monkeypatch.delenv(name, raising=False)


def _seed_world(path: Path) -> World:
    clock = Clock()
    repository = Phase0Repository(path, clock=clock)
    repository.migrate()
    world = World(path, repository, clock)

    # AAPL first, at an old instant: its partition is stale while the
    # global freshness anchor (the newest run, below) stays fresh.
    fresh_at = clock.now
    clock.now = STALE_AT
    seed(world, "AAPL", themes=(("Services demand", 2, 1),))
    clock.now = fresh_at

    tsla = seed(
        world,
        "TSLA",
        themes=(("Alpha", 3, 1), (LONG_UNICODE_LABEL, 2, 2)),
        other=2,
        excluded=1,
    )
    summarize(world, tsla.theme_ids["Alpha"], ticker="TSLA")

    seed_m2_only(world, "NVDA")

    meta = seed(
        world,
        "META",
        themes=(("Gamma", 2, 1),),
        other=2,
        published_at={
            f"META {DAY} t0 story 0": None,
            f"META {DAY} other 0": None,
        },
    )
    summarize(world, meta.theme_ids["Gamma"], ticker="META")
    # AMD: nothing persisted.
    return world


def _capture(client: TestClient, path: str) -> dict:
    response = client.get(path)
    return {"path": path, "status": response.status_code, "body": response.json()}


def _capture_bundle(world: World) -> dict:
    """Every response the frontend reads, through the real routes."""

    missing = narrative.build_narrative_repository(
        "sqlite",
        database_path=world.path.parent / "missing.db",
        pipeline_version=VERSION,
    )

    responses: dict[str, dict] = {}
    try:
        app.dependency_overrides[routes.repository_dependency] = lambda: read_api(world)
        client = TestClient(app)
        responses["tickers"] = _capture(client, "/api/v1/tickers")
        responses["meta_status"] = _capture(client, "/api/v1/meta/status")
        for ticker in UNIVERSE:
            responses[f"themes_{ticker}"] = _capture(
                client, f"/api/v1/tickers/{ticker}/themes"
            )

        app.dependency_overrides[routes.repository_dependency] = lambda: missing
        unavailable = [
            _capture(client, path)
            for path in (
                "/api/v1/tickers",
                "/api/v1/meta/status",
                "/api/v1/tickers/NVDA/themes",
            )
        ]
    finally:
        app.dependency_overrides.clear()

    # One fixed public failure whichever endpoint fails.
    assert {(r["status"], json.dumps(r["body"])) for r in unavailable} == {
        (503, json.dumps({"detail": routes.UNAVAILABLE_DETAIL}))
    }
    responses["unavailable"] = unavailable[-1]
    assert not missing.reader._database_path.exists()

    return {
        "_generated_by": "tests/test_phase0_frontend_contract.py",
        "_regenerate": REGENERATE,
        "_reference_time": NOW.isoformat().replace("+00:00", "Z"),
        "responses": responses,
    }


def _serialize(bundle: dict) -> str:
    # sort_keys=False: objects keep FastAPI's response-model field order
    # and the bundle keeps capture order; both are deterministic.
    return json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"


def _themes(bundle: dict, ticker: str) -> dict:
    captured = bundle["responses"][f"themes_{ticker}"]
    assert captured["status"] == 200
    return captured["body"]


def _assert_acceptance_states(bundle: dict) -> None:
    responses = bundle["responses"]
    for key, captured in responses.items():
        if key != "unavailable":
            assert captured["status"] == 200, key

    # TSLA: one current theme, one degraded theme, Other Coverage.
    tsla = _themes(bundle, "TSLA")
    assert_contract(tsla)
    current = [t for t in tsla["themes"] if not t["degraded"]]
    degraded = [t for t in tsla["themes"] if t["degraded"]]
    assert len(current) == 1 and len(degraded) == 1
    [alpha], [beta] = current, degraded
    assert alpha["label"] == GENERATED_LABEL
    assert len(alpha["sentences"]) >= 2
    assert any(len(s["citation_ids"]) > 1 for s in alpha["sentences"])
    cited = [i for s in alpha["sentences"] for i in s["citation_ids"]]
    assert any(cited.count(i) > 1 for i in cited), "no story cited twice"
    urls = {c["id"]: c["url"] for c in alpha["citations"]}
    assert set(cited) <= set(urls)
    assert beta["sentences"] == []
    assert beta["label"] == LONG_UNICODE_LABEL and len(beta["label"]) > 120
    assert beta["story_count"] == 2 == len(beta["stories"])
    other = tsla["other_coverage"]
    assert other["story_count"] == 3 == len(other["stories"])  # 2 other + 1 excluded
    placed = [s["id"] for t in tsla["themes"] for s in t["stories"]]
    placed += [s["id"] for s in other["stories"]]
    assert len(placed) == len(set(placed)) == 3 + 2 + 3

    # NVDA: an M2-only day -- no themes, every live story in Other Coverage.
    nvda = _themes(bundle, "NVDA")
    assert_contract(nvda)
    assert nvda["themes"] == []
    assert nvda["other_coverage"]["story_count"] == 3

    # AMD: a known ticker with nothing persisted.
    amd = _themes(bundle, "AMD")
    assert amd["themes"] == []
    assert amd["other_coverage"] == {"outlet_count": 0, "story_count": 0, "stories": []}

    # META: missing publication times stay null in both projections.
    meta = _themes(bundle, "META")
    assert_contract(meta)
    [gamma] = meta["themes"]
    assert gamma["degraded"] is False
    assert None in [s["published_at"] for s in gamma["stories"]]
    assert None in [c["published_at"] for c in gamma["citations"]]
    assert None in [s["published_at"] for s in meta["other_coverage"]["stories"]]
    assert all(
        "published_at" in s
        for s in gamma["stories"] + meta["other_coverage"]["stories"]
    )

    # AAPL: its own partition is stale; TSLA and the global status are fresh.
    aapl = _themes(bundle, "AAPL")
    assert_contract(aapl)
    assert len(aapl["themes"]) == 1
    tickers = {t["ticker"]: t for t in responses["tickers"]["body"]["tickers"]}
    assert list(tickers) == UNIVERSE
    assert tickers["AAPL"]["is_stale"] is True
    assert tickers["TSLA"]["is_stale"] is False
    assert responses["meta_status"]["body"]["is_stale"] is False
    assert aapl["data_as_of"] == STALE_AT.isoformat().replace("+00:00", "Z")
    assert {t: tickers[t]["theme_count"] for t in UNIVERSE} == {
        "TSLA": 2,
        "NVDA": 0,
        "AMD": 0,
        "AAPL": 1,
        "META": 1,
    }

    assert responses["unavailable"]["status"] == 503
    assert responses["unavailable"]["body"] == {"detail": routes.UNAVAILABLE_DETAIL}


def _seed_then_read_only(request, *paths: Path) -> list[World]:
    """Seed each world with no network, then forbid every write and generation."""

    request.getfixturevalue("network_forbidden")
    worlds = []
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        worlds.append(_seed_world(path))
    # From here on only reads: generation and writable connections explode.
    request.getfixturevalue("generation_forbidden")
    return worlds


def test_frontend_contract_artifact_matches_the_real_sqlite_api(tmp_path, request):
    [world] = _seed_then_read_only(request, tmp_path / "phase0.db")
    bundle = _capture_bundle(world)
    _assert_acceptance_states(bundle)
    generated = _serialize(bundle)

    if WRITE_MODE:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(generated, encoding="utf-8")

    assert ARTIFACT_PATH.exists(), f"artifact missing; regenerate with: {REGENERATE}"
    checked_in = ARTIFACT_PATH.read_text(encoding="utf-8")
    assert (
        json.loads(checked_in) == bundle
    ), f"frontend contract artifact drifted; regenerate with: {REGENERATE}"
    assert checked_in == generated, f"artifact formatting drifted; run: {REGENERATE}"


def test_generation_is_repeatable(tmp_path, request):
    """Two fresh worlds in one session serialize byte-identically."""

    first, second = _seed_then_read_only(
        request, tmp_path / "a" / "phase0.db", tmp_path / "b" / "phase0.db"
    )
    assert _serialize(_capture_bundle(first)) == _serialize(_capture_bundle(second))


#: Values that would break collection or change the artifact if they
#: reached the app's import-time ``Settings()`` or the summary policy.
HOSTILE_ENVIRONMENT = {
    "PHASE0_NARRATIVE_SOURCE": "invalid",
    "PHASE0_PIPELINE_VERSION": "hostile-version",
    "PHASE0_DATABASE_PATH": "/nonexistent/hostile.db",
    "GEMINI_MODEL": "gemini-hostile",
    "GEMINI_MAX_OUTPUT_TOKENS": "not-a-number",
    "GEMINI_API_KEY": "hostile-key",
}


def _child_environment(**extra: str) -> dict[str, str]:
    """A fresh process environment: no inherited GEMINI_*/PHASE0_* at all."""

    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(ISOLATED_PREFIXES)
    }
    environment.update(HOSTILE_ENVIRONMENT, **extra)
    environment[SUBPROCESS_SENTINEL] = "1"
    return environment


def _artifact_digest() -> str:
    return hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()


@pytest.mark.skipif(
    os.environ.get(SUBPROCESS_SENTINEL) == "1",
    reason="already the hostile-environment child",
)
def test_hostile_import_time_configuration_cannot_break_or_shape_the_artifact():
    """A fresh pytest collects and passes under hostile exported config.

    The child runs only the artifact test node, in drift mode, so it never
    recurses and never writes; the artifact's bytes are checked around it.
    """

    before = _artifact_digest()
    target = test_frontend_contract_artifact_matches_the_real_sqlite_api.__name__
    node = f"{Path(__file__).name}::{target}"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", node],
        cwd=Path(__file__).parent,
        env=_child_environment(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "1 passed" in child.stdout, child.stdout
    assert _artifact_digest() == before


@pytest.mark.skipif(
    os.environ.get(SUBPROCESS_SENTINEL) == "1",
    reason="already the hostile-environment child",
)
def test_import_restores_the_callers_environment_and_keeps_the_write_switch():
    """Importing this module hides hostile config only while it imports."""

    probe = (
        "import os, sys\n"
        "sys.path.insert(0, '.')\n"
        "import test_phase0_frontend_contract as contract\n"
        "assert contract.WRITE_MODE is True, 'write switch lost'\n"
        "assert contract.narrative is not None\n"
        "for name, value in contract.HOSTILE_ENVIRONMENT.items():\n"
        "    assert os.environ[name] == value, name\n"
        "assert os.environ[contract.WRITE_ENV] == '1'\n"
        "print('restored')\n"
    )
    before = _artifact_digest()
    child = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).parent,
        env=_child_environment(**{WRITE_ENV: "1"}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == "restored"
    # Importing, even with the switch on, writes nothing: only the test does.
    assert _artifact_digest() == before
