"""Error hierarchy for the M5 theme clustering stage."""

from __future__ import annotations


class ThemeError(RuntimeError):
    """Base error for theme clustering failures."""


class ThemeInputError(ValueError, ThemeError):
    """A story is structurally invalid and cannot be clustered."""


class ThemeConfigError(ValueError, ThemeError):
    """The theme clustering configuration is not usable."""


class ThemeEncodingError(ThemeError):
    """The encoder returned something the stage cannot trust."""


class ThemeCapacityError(ThemeError):
    """A ticker-day holds more stories than the stage will cluster.

    Raised *before* any output exists.  A partial theme set is worse than
    no theme set: a reader cannot tell a day with three themes from a day
    whose clustering gave up after three.
    """

    def __init__(self, ticker: str, story_count: int, limit: int) -> None:
        super().__init__(
            f"ticker-day {ticker!r} holds {story_count} stories, above the "
            f"configured max_stories_per_day={limit}"
        )
        self.ticker = ticker
        self.story_count = story_count
        self.limit = limit
