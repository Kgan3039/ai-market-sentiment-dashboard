"""Cross-stage ordering, and nothing else.

Two stages already know how to settle a partition: :mod:`phase0.stories`
and :mod:`phase0.themes`.  Neither knows about the other, and neither
should.  What is left over is an ordering constraint that belongs to
neither of them, and this module is the smallest thing that can hold it.

**The constraint.**  Story reconciliation invalidates a partition's theme
set whenever any authoritative story representation changes, and
``themes`` has no ``invalidated_at`` column, so invalidation means
deletion.  A theme runner that read its own continuity identities would
therefore find nothing on precisely the runs continuity is about — the
ones where story structure moved — and would mint fresh identities every
time, renaming themes a reader would have called the same theme.  So the
identities are captured *before* the story stage runs, and handed in.

**Why a coordinator rather than a hook.**  Giving ``StoryReconciler`` a
pre-write callback would invert the dependency: the story stage would have
to know that themes exist and that something wants warning before it
writes.  It would also put a theme read inside the *stories* run, so a
failure reading themes would be recorded as a story failure.  Capture is a
read, and a read needs no hook — it needs only to happen first.

**No stage logic lives here.**  M2, M3, and M5 are not called from this
module; the two runners are, in order, and their outcomes are reported
whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from .repository import Phase0Repository, _normalize_day
from .stories import PartitionOutcome, StoryReconciler
from .themes import ThemePartitionOutcome, ThemeReconciler
from .tickers import normalize_ticker


@dataclass(frozen=True)
class PartitionResult:
    """Both stages' outcomes for one partition, plus what was captured.

    The captured identity count is kept even when the theme stage never
    ran: "there were previous themes and we never got to use them" is what
    an upstream story failure actually did, and a result that reported
    only the failure would not say it.
    """

    ticker: str
    trading_day: str
    stories: PartitionOutcome
    themes: ThemePartitionOutcome
    previous_captured: int = 0

    @property
    def themes_attempted(self) -> bool:
        return self.themes.attempted


def _not_attempted(ticker: str, trading_day: str, reason: str) -> ThemePartitionOutcome:
    """A theme outcome for a partition whose themes were never opened."""

    return ThemePartitionOutcome(
        ticker=ticker,
        trading_day=trading_day,
        status="not_attempted",
        generation=None,
        theme_count=0,
        cleared=False,
        error={
            "type": "themes_not_attempted",
            "ticker": ticker,
            "trading_day": trading_day,
            "reason": reason,
        },
    )


class PartitionCoordinator:
    """Runs ``stories`` then ``themes`` for a day, in that order.

    Owns the ordering and the partition enumeration.  Everything else is
    delegated to the two runners it holds.
    """

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        pipeline_version: str,
        stories: StoryReconciler | None = None,
        themes: ThemeReconciler | None = None,
        encoder: Any | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline_version = str(pipeline_version).strip()
        if not self.pipeline_version:
            raise ValueError("pipeline_version is required")
        self.stories = stories or StoryReconciler(
            repository, pipeline_version=self.pipeline_version, encoder=encoder
        )
        self.themes = themes or ThemeReconciler(
            repository, pipeline_version=self.pipeline_version, encoder=encoder
        )

    # -- Enumeration ------------------------------------------------------

    def partitions(self, trading_day: str | date) -> list[str]:
        """Every ticker this day is answerable for, from all three sources.

        Evidence alone is not enough, and stories alone are not either.  A
        partition can hold themes whose stories have gone, or stories
        whose evidence has gone; either one left unvisited keeps a
        generation that looks authoritative and describes something that
        no longer exists.  The union is what makes the sweep complete.

        None of the three reads projects anything: association state, one
        ticker column from ``stories``, one from ``theme_sets``.
        Classification and clustering wait until a partition's own run is
        open.
        """

        day = _normalize_day(trading_day)
        reader = self.repository.read
        return sorted(
            set(reader.evidence_partition_tickers(day))
            | set(reader.story_partitions(day, pipeline_version=self.pipeline_version))
            | set(reader.theme_partitions(day, pipeline_version=self.pipeline_version))
        )

    # -- The ordering ------------------------------------------------------

    def run(
        self, trading_day: str | date, *, run_id: str
    ) -> tuple[dict[str, Any], list[Any]]:
        """Settle every partition of ``trading_day``; report what happened."""

        day = _normalize_day(trading_day)
        results = [
            self.run_partition(ticker, day, base_run_id=run_id)
            for ticker in self.partitions(day)
        ]
        return summarize(results)

    def run_partition(
        self, ticker: str, trading_day: str | date, *, base_run_id: str
    ) -> PartitionResult:
        """Capture, then stories, then themes -- in that order, always.

        The capture happens outside both runs.  It belongs to neither
        stage, and attributing it to one would put a foreign failure in
        that stage's ledger.

        **An upstream story failure stops here.**  When the story stage
        fails, the partition's previous story generation is intact and its
        themes still describe it, so opening a themes run could only make
        things worse: it would either rebuild themes from a generation the
        story stage just refused to vouch for, or clear a theme set that
        is still consistent with what is stored.  Doing nothing leaves the
        partition exactly as it was, and the story stage's own ``failed``
        row already records why.
        """

        symbol = normalize_ticker(ticker)
        day = _normalize_day(trading_day)

        # 1. Before any story write can delete it.  The whole generation,
        #    not only its themes: a set with no themes in it still names
        #    the model and configuration that built it.
        previous = self.themes.capture_previous(symbol, day)
        captured = 0 if previous is None else len(previous.identities)

        # 2. The story stage, which may invalidate what we just captured.
        story_outcome = self.stories.run_partition(symbol, day, base_run_id=base_run_id)

        # 3. Themes, unless the stories could not be settled.
        if story_outcome.status == "failed":
            return PartitionResult(
                ticker=symbol,
                trading_day=day,
                stories=story_outcome,
                themes=_not_attempted(symbol, day, "upstream_story_failure"),
                previous_captured=captured,
            )

        theme_outcome = self.themes.run_partition(
            symbol, day, base_run_id=base_run_id, previous=previous
        )
        return PartitionResult(
            ticker=symbol,
            trading_day=day,
            stories=story_outcome,
            themes=theme_outcome,
            previous_captured=captured,
        )


_STORY_COUNTER = {
    "success": "succeeded",
    "degraded": "degraded",
    "failed": "failed",
}
_THEME_COUNTER = {
    "success": "succeeded",
    "degraded": "degraded",
    "failed": "failed",
    "not_attempted": "not_attempted",
}


def summarize(
    results: Sequence[PartitionResult],
) -> tuple[dict[str, Any], list[Any]]:
    """Fold per-partition results into one component report.

    The two stages are counted apart.  "Four partitions, one of which
    produced stories but no themes" is the fact an operator needs, and a
    single total cannot say it.
    """

    counts: dict[str, Any] = {
        "partitions": len(results),
        "stories_succeeded": 0,
        "stories_degraded": 0,
        "stories_failed": 0,
        "themes_succeeded": 0,
        "themes_degraded": 0,
        "themes_failed": 0,
        "themes_not_attempted": 0,
        "themes_persisted": 0,
        "theme_sets_cleared": 0,
        "theme_rows_cleared": 0,
        "previous_themes_captured": 0,
    }
    errors: list[Any] = []
    for result in results:
        counts[f"stories_{_STORY_COUNTER[result.stories.status]}"] += 1
        counts[f"themes_{_THEME_COUNTER[result.themes.status]}"] += 1
        counts["themes_persisted"] += result.themes.theme_count
        counts["theme_sets_cleared"] += 1 if result.themes.cleared else 0
        # Committed work survives a later failure in the same attempt, so
        # the aggregate has to survive it too.
        counts["theme_rows_cleared"] += result.themes.counts.get("cleared_rows", 0)
        counts["previous_themes_captured"] += result.previous_captured
        if result.stories.error is not None:
            errors.append(dict(result.stories.error))
        if (
            result.themes.error is not None
            # A partition whose themes were never opened is already
            # explained by the story error folded in above; reporting it
            # again would count one failure twice.  The
            # ``themes_not_attempted`` counter is where it shows.
            and result.themes.error.get("type") != "themes_not_attempted"
        ):
            errors.append(dict(result.themes.error))
        if result.themes.degraded:
            errors.append(
                {
                    "type": "stage_degraded",
                    "stage": "themes",
                    "ticker": result.ticker,
                    "trading_day": result.trading_day,
                    "reason": result.themes.degradation_reason,
                }
            )
    return counts, errors


__all__ = [
    "PartitionCoordinator",
    "PartitionResult",
    "summarize",
]
