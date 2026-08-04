"""Loading the committed ticker-day fixture, strictly.

Same posture as the M4 loader: a fixture that cannot be vouched for is not
quietly clustered around.  An unknown key, a naive timestamp, or a repeated
story key is a defect in the fixture, and the evaluation refuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nlp.eval.trust import TrustContract, TrustSummary, derive_trust_summary
from nlp.eval.trust import parse_trust_contract

from .errors import ThemeInputError
from .models import ThemeStory

SUPPORTED_SCHEMA_VERSION = "phase0.theme_eval.v1"
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "ticker_days.json"

_STORY_KEYS = frozenset(
    {
        "story_key",
        "title",
        "description",
        "published_at",
        "outlets",
        "item_ids",
    }
)
_DAY_KEYS = frozenset({"ticker", "trading_day", "volume", "expectation", "stories"})
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "issue",
        "acceptance_criteria",
        "trust_contract",
        "trust_summary",
        "stage_specific_trust_summary",
        "known_limitations",
        "provenance",
        "shape_notes",
        "ticker_days",
    }
)


@dataclass(frozen=True)
class TickerDay:
    """One fixture ticker-day, ready to cluster."""

    ticker: str
    trading_day: date
    volume: str
    expectation: str
    stories: tuple[ThemeStory, ...]


@dataclass(frozen=True)
class TickerDaySet:
    """The committed fixture: three days of differing volume."""

    dataset_id: str
    metadata: Mapping[str, Any]
    days: tuple[TickerDay, ...]
    #: Validated provenance, parsed by M4's contract so M5 states its
    #: trust the same way M3 and M4 do rather than inventing a second
    #: vocabulary for the same warning.
    trust_contract: TrustContract = None  # type: ignore[assignment]

    @property
    def trust_summary(self) -> TrustSummary:
        """The banner, derived from the validated fields, never supplied."""

        return derive_trust_summary(self.trust_contract)

    @property
    def known_limitations(self) -> tuple[str, ...]:
        """What this fixture cannot show, as the manifest states it."""

        return tuple(self.metadata.get("known_limitations", ()))


def _require(payload: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in payload:
        raise ThemeInputError(f"{where}: missing {key!r}")
    return payload[key]


def _timestamp(value: Any, where: str) -> datetime:
    if not isinstance(value, str):
        raise ThemeInputError(f"{where}: published_at must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ThemeInputError(
            f"{where}: published_at is not ISO-8601: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ThemeInputError(f"{where}: published_at needs a timezone offset")
    return parsed


def _story(payload: Any, ticker: str, where: str) -> ThemeStory:
    if not isinstance(payload, dict):
        raise ThemeInputError(f"{where}: each story must be an object")
    unknown = sorted(set(payload) - _STORY_KEYS)
    if unknown:
        raise ThemeInputError(f"{where}: unknown field(s) {unknown}")
    item_ids = tuple(payload.get("item_ids", ()))
    outlets = tuple(payload.get("outlets", ()))
    return ThemeStory(
        story_key=str(_require(payload, "story_key", where)),
        ticker=ticker,
        title=str(_require(payload, "title", where)),
        description=payload.get("description"),
        published_at=_timestamp(_require(payload, "published_at", where), where),
        outlets=outlets,
        item_ids=item_ids,
        source_links=tuple(
            (item_id, outlets[index % len(outlets)] if outlets else "", None)
            for index, item_id in enumerate(item_ids)
        ),
    )


def load_ticker_days(path: str | Path = DEFAULT_FIXTURE_PATH) -> TickerDaySet:
    """Load and validate the committed ticker-day fixture."""

    fixture = Path(path).resolve()
    try:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ThemeInputError(f"{fixture}: fixture not found") from exc
    except json.JSONDecodeError as exc:
        raise ThemeInputError(f"{fixture}: not valid JSON: {exc}") from exc
    unknown = sorted(set(payload) - _MANIFEST_KEYS)
    if unknown:
        raise ThemeInputError(f"{fixture}: unknown manifest field(s) {unknown}")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ThemeInputError(
            f"{fixture}: unsupported schema_version "
            f"{payload.get('schema_version')!r}; this build reads "
            f"{SUPPORTED_SCHEMA_VERSION!r}"
        )
    days: list[TickerDay] = []
    seen: set[str] = set()
    for entry in _require(payload, "ticker_days", str(fixture)):
        unknown = sorted(set(entry) - _DAY_KEYS)
        if unknown:
            raise ThemeInputError(f"{fixture}: unknown ticker-day field(s) {unknown}")
        ticker = str(_require(entry, "ticker", str(fixture)))
        where = f"{ticker} {entry.get('trading_day')}"
        stories = tuple(
            _story(story, ticker, where) for story in _require(entry, "stories", where)
        )
        for story in stories:
            if story.story_key in seen:
                raise ThemeInputError(f"duplicate story_key: {story.story_key}")
            seen.add(story.story_key)
        days.append(
            TickerDay(
                ticker=ticker,
                trading_day=date.fromisoformat(
                    str(_require(entry, "trading_day", where))
                ),
                volume=str(_require(entry, "volume", where)),
                expectation=str(_require(entry, "expectation", where)),
                stories=stories,
            )
        )
    if not days:
        raise ThemeInputError(f"{fixture}: holds no ticker-days")
    try:
        contract = parse_trust_contract(payload, where=str(fixture))
    except ValueError as exc:
        # A fixture that cannot say what it is worth is not clustered
        # around; an unstated provenance is the one a reader assumes away.
        raise ThemeInputError(f"{fixture}: {exc}") from exc
    return TickerDaySet(
        dataset_id=str(payload.get("dataset_id", "")),
        metadata=payload,
        days=tuple(days),
        trust_contract=contract,
    )


def tickers_of(day_set: TickerDaySet) -> tuple[str, ...]:
    """Every ticker the fixture covers, sorted."""

    return tuple(sorted({day.ticker for day in day_set.days}))


def stories_of(days: Sequence[TickerDay]) -> tuple[ThemeStory, ...]:
    """Flatten a set of days into one story sequence."""

    return tuple(story for day in days for story in day.stories)
