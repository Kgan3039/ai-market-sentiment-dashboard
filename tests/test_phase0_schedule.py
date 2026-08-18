"""Scheduling semantics for the Phase 0 pipeline (#68).

Two separate claims are checked here, because a cron file proves neither
on its own: that the schedule means what it says in America/New_York, and
that a run which outlasts its interval cannot be joined by a second copy.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import multiprocessing
import re

import pytest

from pipeline import MARKET_TIMEZONE, invocation_day, single_instance


CRON_PATH = Path("deploy/phase0-pipeline.cron")


@pytest.fixture(scope="module")
def cron() -> str:
    return CRON_PATH.read_text(encoding="utf-8")


def schedule_lines(cron: str) -> list[str]:
    """The cron entries, without comments or environment assignments."""

    return [
        line
        for line in cron.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not re.match(r"^[A-Z_][A-Z0-9_]*=", line.strip())
    ]


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------


def test_schedule_declares_new_york_rather_than_host_local_time(cron):
    """The zone is stated, not inherited.

    Without ``CRON_TZ`` these expressions run in whatever the host is set
    to, and "09:30" silently becomes a different instant -- the exact
    failure that looks like a working schedule.
    """

    assert "CRON_TZ=America/New_York" in cron
    assert schedule_lines(cron), "no schedule entries found"


def test_schedule_documents_that_cron_tz_is_not_universal(cron):
    """An assumption that can be wrong has to say so.

    ``CRON_TZ`` is a Vixie/cronie extension. On an implementation that
    ignores it the file still installs and still runs -- in the wrong zone.
    """

    assert "Vixie" in cron and "cronie" in cron
    assert "host local time" in cron


def test_schedule_covers_market_hours_and_off_hours(cron):
    assert "0,30 9-16 * * 1-5" in cron  # 09:00-16:30 ET, every 30 minutes
    assert "0 0-8,17-23 * * 1-5" in cron  # weekday off-hours, hourly
    assert "0 * * * 0,6" in cron  # weekends, hourly


def test_every_scheduled_invocation_runs_the_pipeline(cron):
    lines = schedule_lines(cron)
    assert len(lines) == 3
    assert all("pipeline.py" in line for line in lines)


# ---------------------------------------------------------------------------
# The expressions, evaluated rather than string-matched
# ---------------------------------------------------------------------------


def _field(spec: str, span: range) -> set[int]:
    if spec == "*":
        return set(span)
    values: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            low, high = (int(bound) for bound in part.split("-"))
            values.update(range(low, high + 1))
        else:
            values.add(int(part))
    return values


def matches(line: str, moment: datetime) -> bool:
    """Whether one crontab line fires at *moment*, read as Eastern wall time.

    Substring assertions prove an expression is present, not that it covers
    the hours it claims to. Only the fields these schedules actually use
    are supported -- day-of-month and month are ``*`` throughout.
    """

    minute, hour, dom, month, dow = line.split()[:5]
    assert dom == "*" and month == "*", "unsupported field in schedule"
    # cron counts Sunday as 0; datetime counts Monday as 0.
    cron_dow = (moment.weekday() + 1) % 7
    return (
        moment.minute in _field(minute, range(60))
        and moment.hour in _field(hour, range(24))
        and cron_dow in _field(dow, range(7))
    )


def eastern(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=MARKET_TIMEZONE)


@pytest.mark.parametrize(
    "moment,expected",
    [
        ("2026-08-18T09:00", True),  # Tuesday, the open
        ("2026-08-18T09:30", True),
        ("2026-08-18T16:00", True),
        ("2026-08-18T16:30", True),  # the last market-hours firing
        ("2026-08-18T17:00", True),  # off-hours takes over on the hour
        ("2026-08-18T03:00", True),  # weekday overnight
        ("2026-08-16T03:00", True),  # Sunday
        ("2026-08-15T22:00", True),  # Saturday
        ("2026-08-18T09:15", False),  # nothing runs off the half hour
        ("2026-08-18T17:30", False),  # off-hours is hourly, not half-hourly
    ],
)
def test_the_schedule_fires_when_it_claims_to(cron, moment, expected):
    lines = schedule_lines(cron)
    assert any(matches(line, eastern(moment)) for line in lines) is expected


def test_no_hour_of_the_week_is_left_unscheduled(cron):
    """Every hour has an invocation; an unscheduled window is a data gap."""

    lines = schedule_lines(cron)
    uncovered = [
        (day, hour)
        for day in range(16, 23)  # 2026-08-16 Sunday .. 2026-08-22 Saturday
        for hour in range(24)
        if not any(
            matches(line, eastern(f"2026-08-{day:02d}T{hour:02d}:00")) for line in lines
        )
    ]
    assert uncovered == []


def test_no_instant_is_scheduled_twice(cron):
    """Two lines firing together would start two invocations at once.

    The lock would refuse the second, but relying on it to paper over an
    overlapping schedule turns a silent duplicate into a silent skip.
    """

    lines = schedule_lines(cron)
    doubled = [
        (day, hour, minute)
        for day in range(16, 23)
        for hour in range(24)
        for minute in (0, 30)
        if sum(
            matches(line, eastern(f"2026-08-{day:02d}T{hour:02d}:{minute:02d}"))
            for line in lines
        )
        > 1
    ]
    assert doubled == []


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_every_scheduled_invocation_is_guarded_against_overlap(cron):
    """A cron file alone is not overlap prevention.

    Each entry takes the same non-blocking lock, so a run that outlasts its
    interval refuses its successor immediately rather than letting two live
    fetchers race for the same feeds and provider slots.
    """

    lines = schedule_lines(cron)
    assert all("flock -n" in line for line in lines)
    locks = {re.search(r"flock -n (\S+)", line).group(1) for line in lines}
    assert len(locks) == 1, f"schedules take different locks: {locks}"


def test_schedule_documents_exit_codes_rather_than_implying_green_or_red(cron):
    assert "0 success" in cron and "1 degraded" in cron and "2 failed" in cron


def _hold_lock(path, acquired, release):
    with single_instance(Path(path)) as got:
        acquired.put(got)
        release.wait(10)


def test_a_second_invocation_is_refused_while_the_first_holds_the_lock(tmp_path):
    """The in-process guard, proved across real processes.

    ``flock`` is per open file description, so a same-process check would
    pass trivially. The holder here is a separate process, which is what
    cron actually produces.
    """

    lock = tmp_path / "phase0.lock"
    acquired: multiprocessing.Queue = multiprocessing.Queue()
    release = multiprocessing.Event()
    holder = multiprocessing.Process(
        target=_hold_lock, args=(str(lock), acquired, release)
    )
    holder.start()
    try:
        assert acquired.get(timeout=10) is True
        with single_instance(lock) as second:
            assert second is False, "a second invocation acquired a held lock"
    finally:
        release.set()
        holder.join(10)

    # The lock is released with the holder, not leaked past it.
    with single_instance(lock) as third:
        assert third is True


def test_the_lock_is_released_when_an_invocation_raises(tmp_path):
    lock = tmp_path / "phase0.lock"
    with pytest.raises(RuntimeError):
        with single_instance(lock) as acquired:
            assert acquired is True
            raise RuntimeError("invocation blew up")

    with single_instance(lock) as again:
        assert again is True


def test_the_lock_file_directory_is_created_on_demand(tmp_path):
    lock = tmp_path / "missing" / "phase0.lock"
    with single_instance(lock) as acquired:
        assert acquired is True
    assert lock.exists()


# ---------------------------------------------------------------------------
# Trading day: DST and the UTC boundary
# ---------------------------------------------------------------------------


def at(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


@pytest.mark.parametrize(
    "moment,expected,zone",
    [
        # Spring forward 2026-03-08: 02:00 EST becomes 03:00 EDT.
        ("2026-03-08T06:30:00+00:00", "2026-03-08", "EST"),
        ("2026-03-08T07:30:00+00:00", "2026-03-08", "EDT"),
        # Fall back 2026-11-01: 02:00 EDT becomes 01:00 EST, so 01:30 ET
        # happens twice. Both instants still name the same date.
        ("2026-11-01T05:30:00+00:00", "2026-11-01", "EDT"),
        ("2026-11-01T06:30:00+00:00", "2026-11-01", "EST"),
    ],
)
def test_invocation_day_follows_eastern_offsets_across_dst(moment, expected, zone):
    instant = at(moment)
    assert invocation_day(instant) == expected
    assert instant.astimezone(MARKET_TIMEZONE).tzname() == zone


@pytest.mark.parametrize(
    "moment,utc_date,eastern_date",
    [
        ("2026-08-18T02:00:00+00:00", "2026-08-18", "2026-08-17"),  # EDT, -4
        ("2026-01-05T03:00:00+00:00", "2026-01-05", "2026-01-04"),  # EST, -5
    ],
)
def test_invocation_day_is_not_the_utc_date(moment, utc_date, eastern_date):
    """Late-evening ET belongs to the previous UTC date, and vice versa.

    Taking the UTC date would relabel every evening invocation as the next
    day's -- the specific mistake that makes an off-hours schedule look
    like it ran tomorrow.
    """

    instant = at(moment)
    assert instant.date().isoformat() == utc_date
    assert invocation_day(instant) == eastern_date


def test_invocation_day_ignores_the_host_timezone(monkeypatch):
    """Same instant, same answer, whatever the host thinks the time is."""

    instant = at("2026-08-18T02:00:00+00:00")
    answers = {
        invocation_day(instant.astimezone(timezone(timedelta(hours=offset))))
        for offset in (-8, 0, 5, 9)
    }
    assert answers == {"2026-08-17"}


def test_invocation_day_refuses_a_naive_datetime():
    """A naive stamp has no answer; guessing one is how host time leaks in."""

    with pytest.raises(ValueError, match="aware datetime"):
        invocation_day(datetime(2026, 8, 18, 2, 0, 0))


def test_invocation_day_without_an_argument_uses_now():
    assert invocation_day() == invocation_day(datetime.now(timezone.utc))
