"""B1 representative SQLite API benchmark tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from tools import phase0_api_benchmark as benchmark  # noqa: E402


def test_representative_database_and_all_endpoint_groups(tmp_path):
    database = tmp_path / "benchmark.sqlite3"
    seed = benchmark.build_representative_database(
        database,
        days=2,
        themes=2,
        stories_per_theme=2,
        other_stories=1,
    )

    report = benchmark.benchmark_local(
        database,
        pipeline_version=benchmark.DEFAULT_VERSION,
        requests=5,
        warmup=1,
        max_p95_ms=300,
    )

    assert seed["tickers"] == 5
    assert report["passed"] is True
    assert set(report["endpoints"]) == {"tickers", "themes", "status"}
    assert report["total_measured_requests"] == 15
    assert report["dataset"]["sqlite"]
    assert report["dataset"]["rows"]["stories"] == 50
    assert report["dataset"]["rows"]["themes"] == 20
    assert report["dataset"]["rows"]["summary_artifacts"] == 10
    assert all(result["requests"] == 5 for result in report["endpoints"].values())


def test_representative_builder_refuses_to_replace_a_database(tmp_path):
    database = tmp_path / "existing.sqlite3"
    database.write_bytes(b"already here")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        benchmark.build_representative_database(database, days=1)


def test_cli_prints_machine_readable_evidence(tmp_path, capsys, monkeypatch):
    database = tmp_path / "cli.sqlite3"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase0_api_benchmark.py",
            "--database",
            str(database),
            "--seed-representative",
            "--days",
            "1",
            "--themes",
            "1",
            "--stories-per-theme",
            "1",
            "--other-stories",
            "1",
            "--requests",
            "2",
            "--warmup",
            "0",
        ],
    )

    assert benchmark.main(sys.argv[1:]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["seed"]["trading_days"] == 1
    assert payload["environment"]["mode"].startswith("in-process FastAPI")
    assert payload["command"].startswith("python")
