from pathlib import Path


def test_cron_covers_market_hours_and_off_hours_in_new_york():
    cron = Path("deploy/phase0-pipeline.cron").read_text(encoding="utf-8")

    assert "CRON_TZ=America/New_York" in cron
    assert "0,30 9-16 * * 1-5" in cron
    assert "0 0-8,17-23 * * 1-5" in cron
    assert "0 * * * 0,6" in cron
    assert cron.count(".venv/bin/python pipeline.py") == 3
