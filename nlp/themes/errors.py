"""Error hierarchy for the M5 theme clustering stage."""

from __future__ import annotations


class ThemeError(RuntimeError):
    """Base error for theme clustering failures."""


class ThemeInputError(ValueError, ThemeError):
    """A story is structurally invalid and cannot be clustered."""


class ThemeInvariantError(ThemeInputError):
    """A theme set cannot describe its own membership honestly.

    Raised at construction rather than reported as a diagnostic, because
    these are the failures no diagnostic could describe: a story in two
    themes, an empty theme, evidence that does not match membership, one
    raw item citable from two themes, two themes sharing one identity.  A
    ``ThemeSet`` that got past this is one whose ``complete`` can be
    believed.
    """

    def __init__(
        self, message: str, *, duplicate_theme_keys: tuple[str, ...] = ()
    ) -> None:
        super().__init__(message)
        self.duplicate_theme_keys = duplicate_theme_keys


class ThemePartitionError(ThemeInputError):
    """The stories handed in do not form a partition of the day's coverage.

    Raised at the public boundary, before anything is clustered.  M5's
    whole contract is that one raw item is citable from exactly one theme,
    and that guarantee is only as good as the input: two canonical stories
    claiming the same raw item make it citable from two themes no matter
    how carefully the clustering behaves afterwards.  The bridge checks
    this too, but a caller may build ``ThemeStory`` objects by hand and the
    guarantee cannot depend on which door they came through.
    """

    def __init__(
        self,
        message: str,
        *,
        overlapping_story_keys: tuple[str, ...] = (),
        overlapping_item_ids: tuple[str, ...] = (),
        affected_story_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.overlapping_story_keys = overlapping_story_keys
        self.overlapping_item_ids = overlapping_item_ids
        self.affected_story_keys = affected_story_keys


class ThemeConfigError(ValueError, ThemeError):
    """The theme clustering configuration is not usable."""


class ThemeEncodingError(ThemeError):
    """The encoder returned something the stage cannot trust."""


class ThemeClusteringError(ThemeError):
    """The clustering library failed on input the stage had accepted.

    Wrapped rather than propagated raw: a caller handling one ticker-day
    among five needs to tell "sklearn raised" apart from "M5 refused", and
    an unwrapped library exception makes that a string comparison.
    """

    def __init__(self, method: str, cause: BaseException) -> None:
        super().__init__(f"the {method} clustering library failed: {cause}")
        self.method = method
        self.cause = cause


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


class ThemeNarrativeCapacityError(ThemeCapacityError):
    """A candidate cluster is larger than the exact narrative search allows.

    Raised *before* any theme is returned, and never softened into an
    approximate answer.  The selection contract is "the largest mutually
    compatible subset"; a search that ran out of budget does not know
    whether what it found is the largest, and returning it anyway would
    make the contract a claim nobody checked.

    Carries what a caller needs to act: which stories were involved, how
    many there were, which limit was hit, and what to do about it.
    """

    def __init__(
        self,
        story_keys: tuple[str, ...],
        item_count: int,
        *,
        limit: int | None = None,
        budget: int | None = None,
        states: int | None = None,
    ) -> None:
        if limit is not None:
            detail = (
                f"{item_count} stories exceed max_narrative_selection_items=" f"{limit}"
            )
            guidance = (
                "raise max_narrative_selection_items, or split the ticker-day "
                "so no candidate cluster is this large"
            )
        else:
            detail = (
                f"the exact search over {item_count} stories passed "
                f"max_narrative_search_states={budget} after {states} states"
            )
            guidance = (
                "raise max_narrative_search_states, or lower "
                "max_narrative_selection_items so this cluster is refused "
                "before the search rather than during it"
            )
        ThemeError.__init__(
            self,
            f"narrative selection refused: {detail}; no approximate subset is "
            f"returned because the contract is the largest compatible subset. "
            f"To proceed: {guidance}. Stories: {list(story_keys)}",
        )
        self.story_keys = story_keys
        self.item_count = item_count
        self.story_count = item_count
        self.limit = limit
        self.budget = budget
        self.states = states
        self.ticker = None
