"""The story stage: persisted evidence in, one authoritative generation out.

This is the port between I5's evidence read boundary and the ``stories``
table.  It reads a partition's eligible evidence, runs M2 exact dedup, runs
M3 semantic dedup, and settles the result through ``reconcile_stories`` --
one partition, one run, one write.

**The operational stage name is ``stories``** (decision A4).  It names a
unit of work, not the algorithm that happened to do it: ``latest_stage_status``
groups by stage, so a name that varied with the outcome would create a
second name whose newest row is a degradation and which nothing under the
other name could ever supersede.  Which algorithm produced a generation is
recorded on the rows it produced, as ``stories.stage``.

**The run encloses the computation, not just the write.**  ``stage_run``
opens before the evidence read and closes after the reconcile, so an M2
capacity failure or a broken encoder is a *recorded* failed attempt rather
than an exception that the operational ledger never saw.  Nothing is
mutated on the way: M2 and M3 are pure, the reads take no locks, and
``reconcile_stories`` is the single write and the run's terminal operation.
A failure anywhere before it therefore leaves the partition's previous
committed generation exactly as it was.

**Degradation is narrow and explicit** (decision H).  Four expected
failures -- a model that will not load, a model that will not encode, an
encoder whose output M3 refuses, and a partition too large for exhaustive
comparison -- fall back to M2's clusters, recorded as ``m2.exact`` with the
model columns left NULL and an explicit ``stage_degraded`` marker in the
run log.  Everything else fails the partition.  In particular
``SemanticDedupInputError`` and ``SemanticDedupConfigError`` are *not*
caught: they mean the input this module built or the configuration it was
given is wrong, and shipping M2's answer instead would bury a defect that
also casts doubt on the M2 half.  No parent exception is caught anywhere
near M3 for the same reason.

**No themes here.**  M5 is not run, registered, or referenced.  A degraded
generation ships no theme set at all, which is the opposite of a
degradation that renders like a healthy day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from nlp.dedup import (
    DedupConfig,
    DedupResult,
    DeduplicatedCluster,
    deduplicate,
)
from nlp.embeddings import (
    EmbeddingEncodingError,
    EmbeddingModelLoadError,
    get_default_service,
)
from nlp.semdedup import (
    SemanticDedupCapacityError,
    SemanticDedupConfig,
    SemanticDedupEncodingError,
    SemanticDedupResult,
    SemanticStory,
    StoryEncoder,
    merge_semantic_duplicates,
    stories_from_dedup,
)

from .evidence import ExcludedEvidence
from .models import (
    ProviderConflictRecord,
    SemanticMergeRecord,
    StoryMemberRecord,
    StoryRecord,
)
from .repository import Phase0Repository, _normalize_day
from .scalars import sanitize_diagnostic_scalar
from .tickers import SUPPORTED_TICKERS, normalize_ticker

#: The unit of work, stable whatever the outcome (decision A4).
STAGE = "stories"

#: Recorded through ``record_degradation`` when M3 could not run.  One
#: reason, because from a consumer's side the four causes are one fact:
#: this generation is M2's, and no theme set may be built over it.
DEGRADATION_REASON = "m3_semantic_unavailable"

#: The four expected M3 failures, named individually.
#:
#: Their common parents are ``SemanticDedupError`` and ``EmbeddingError``,
#: and catching either would also swallow ``SemanticDedupInputError``,
#: ``SemanticDedupConfigError``, and ``EmbeddingInputError`` -- a malformed
#: bridge projection, an unusable configuration, and text this module
#: composed that the encoder rejected.  Those are defects in this stage,
#: not conditions in the world, and a fallback that hid them would report a
#: degradation every run while the bug stayed invisible.
_RECOVERABLE_M3: tuple[type[BaseException], ...] = (
    SemanticDedupCapacityError,
    SemanticDedupEncodingError,
    EmbeddingModelLoadError,
    EmbeddingEncodingError,
)


def partition_run_id(base_run_id: str, ticker: str, trading_day: str) -> str:
    """The run identity for one ticker/day partition.

    ``run_log`` is ``UNIQUE(run_id, stage)`` and I1 refuses to record one
    identity against a second partition, so the identity has to carry the
    partition it speaks for.  Spelled exactly as the ingestion stages spell
    it, because an operator reading the ledger should not have to learn two
    conventions.
    """

    return f"{base_run_id}:{ticker}:{trading_day}"


@dataclass(frozen=True)
class PartitionOutcome:
    """What one partition's run did, after that run has closed.

    Frozen, and built from values copied out of the run while it was still
    open.  The :class:`~phase0.repository.StageRunContext` itself never
    reaches here: it authorizes mutations, and an authorization that
    outlives the run it belongs to is the thing that class is designed to
    make impossible.  Nothing on this record can reach back into a run's
    state.
    """

    ticker: str
    trading_day: str
    #: ``success``, ``degraded``, or ``failed``.
    status: str
    #: ``m3.semantic``, ``m2.exact``, or ``None`` when nothing was written.
    story_stage: str | None
    #: :data:`DEGRADATION_REASON` when M3 was unavailable, else ``None``.
    degradation_reason: str | None
    story_count: int
    counts: Mapping[str, int] = field(default_factory=dict)
    excluded: tuple[ExcludedEvidence, ...] = ()
    error: Mapping[str, str] | None = None

    @property
    def degraded(self) -> bool:
        """True when this partition shipped M2 output on purpose."""

        return self.degradation_reason is not None


def _conflict_records(
    result: DedupResult, keys: Sequence[tuple[str, str]]
) -> tuple[ProviderConflictRecord, ...]:
    """M2's conflicts named by ``keys``, with their full payload.

    M3 carries conflicts forward as ``(namespace, provider_item_id)``
    pairs, which is enough to say *that* a story is disputed but not which
    items disagreed about what.  The payload is still on M2's result, so it
    is read from there rather than recorded as an emptier version of a fact
    the run already has.
    """

    by_key = {
        (conflict.provider_namespace, conflict.provider_item_id): conflict
        for conflict in result.provider_conflicts
    }
    records = []
    for key in sorted(set(keys)):
        conflict = by_key.get(key)
        if conflict is None:
            # M3 named a conflict M2 did not report.  Impossible through
            # the bridge, which reads both from the same result; recorded
            # honestly rather than dropped if it ever happens.
            records.append(
                ProviderConflictRecord(
                    provider_namespace=key[0], provider_item_id=key[1]
                )
            )
            continue
        records.append(
            ProviderConflictRecord(
                provider_namespace=conflict.provider_namespace,
                provider_item_id=conflict.provider_item_id,
                item_ids=tuple(conflict.item_ids),
                fields=tuple(conflict.fields),
            )
        )
    return tuple(records)


def _member_records(
    clusters: Sequence[DeduplicatedCluster], quarantined: frozenset[str]
) -> tuple[StoryMemberRecord, ...]:
    """Every retained source link of ``clusters``, in the order given.

    ``position`` is left unset so the repository numbers them by this
    order, which is canonical-cluster-first.  M2 partitions the items, so
    two clusters cannot name the same raw item and the union needs no
    de-duplication -- but a repeat would be refused by ``_prepare_story``
    rather than silently collapsed, which is the right failure.
    """

    return tuple(
        StoryMemberRecord(
            raw_item_id=int(member.item_id),
            outlet=member.outlet,
            url=member.url,
            canonical_url=member.canonical_url,
            match_reason=member.match_reason.value,
            quarantined=member.item_id in quarantined,
        )
        for cluster in clusters
        for member in cluster.members
    )


def exact_story_records(result: DedupResult) -> tuple[StoryRecord, ...]:
    """M2's clusters as persistable stories -- the degraded generation.

    Everything M2 knows is kept: the canonical item, the outlet count, each
    member's outlet, link, and match reason, and the provider conflicts
    that touched it.  What is *not* here is anything M3 would have supplied
    -- the model identity, the member story keys, the merge evidence --
    because M3 did not run, and a NULL that says so is worth more than a
    plausible value that does not.
    """

    quarantined = frozenset(result.quarantined_item_ids)
    conflicts_for_item: dict[str, list[tuple[str, str]]] = {}
    for conflict in result.provider_conflicts:
        for item_id in conflict.item_ids:
            conflicts_for_item.setdefault(item_id, []).append(
                (conflict.provider_namespace, conflict.provider_item_id)
            )
    return tuple(
        StoryRecord(
            cluster_fingerprint=cluster.cluster_fingerprint,
            canonical_title=cluster.canonical_title,
            members=_member_records([cluster], quarantined),
            canonical_item_id=int(cluster.canonical_item_id),
            outlet_count=cluster.outlet_count,
            published_at=cluster.published_at,
            canonical_url=cluster.canonical_url,
            source=cluster.source,
            outlet=cluster.outlet,
            content_hash=cluster.content_hash,
            algorithm_version=result.algorithm_version,
            config_fingerprint=result.config_fingerprint,
            stage="m2.exact",
            # Decision A: vectors stay in memory; nothing is keyed to a
            # story id that does not exist yet.
            embedding=None,
            model_name=None,
            model_revision=None,
            embedding_dimension=None,
            quarantined=bool(cluster.member_ids)
            and set(cluster.member_ids) <= quarantined,
            semantic_skip_reason=None,
            member_story_keys=(),
            provider_conflicts=_conflict_records(
                result,
                [
                    entry
                    for item_id in cluster.member_ids
                    for entry in conflicts_for_item.get(item_id, ())
                ],
            ),
            semantic_merges=(),
        )
        for cluster in result.clusters
    )


def _semantic_story_record(
    story: SemanticStory,
    semantic: SemanticDedupResult,
    exact: DedupResult,
    clusters: Mapping[str, DeduplicatedCluster],
) -> StoryRecord:
    """One M3 story as a persistable row.

    M3's result is authoritative for everything M3 decided -- the member
    set, the outlet count, the canonical title, the skip reason, the merge
    evidence.  It carries no canonical *raw item*, no link fields, and no
    provider-conflict payload, because those describe articles and M3
    reasons about stories.  Those come back from the M2 clusters this story
    was assembled from, which is where they were all along.

    The model identity is stamped here too: a stored story whose encoder
    cannot be named cannot be invalidated when the model moves, and A3's
    compatibility gate reads exactly these columns before it lets a
    previous theme's identity carry over.
    """

    members = [clusters[key] for key in story.member_story_keys if key in clusters]
    canonical = clusters.get(story.canonical_story_key)
    quarantined = frozenset(story.quarantined_member_ids)
    return StoryRecord(
        cluster_fingerprint=story.story_fingerprint,
        canonical_title=story.canonical_title,
        members=_member_records(members, quarantined),
        canonical_item_id=(
            None if canonical is None else int(canonical.canonical_item_id)
        ),
        outlet_count=story.outlet_count,
        published_at=story.published_at,
        canonical_url=None if canonical is None else canonical.canonical_url,
        source=None if canonical is None else canonical.source,
        outlet=None if canonical is None else canonical.outlet,
        content_hash=story.content_hash,
        algorithm_version=story.algorithm_version,
        config_fingerprint=semantic.config_fingerprint,
        stage="m3.semantic",
        # Decision A again: the vector that decided this merge is not kept.
        embedding=None,
        model_name=semantic.model_name,
        model_revision=semantic.model_revision,
        embedding_dimension=semantic.embedding_dimension,
        quarantined=story.is_quarantined,
        semantic_skip_reason=(
            None
            if story.semantic_skip_reason is None
            else story.semantic_skip_reason.value
        ),
        member_story_keys=tuple(story.member_story_keys),
        provider_conflicts=_conflict_records(exact, story.provider_conflicts),
        semantic_merges=tuple(
            SemanticMergeRecord(
                left_story_key=merge.left_story_key,
                right_story_key=merge.right_story_key,
                similarity=float(merge.similarity),
                reason=getattr(merge.reason, "value", str(merge.reason)),
            )
            for merge in story.merges
        ),
    )


def semantic_story_records(
    semantic: SemanticDedupResult, exact: DedupResult
) -> tuple[StoryRecord, ...]:
    """M3's stories as persistable rows -- the healthy generation."""

    clusters = {cluster.cluster_fingerprint: cluster for cluster in exact.clusters}
    return tuple(
        _semantic_story_record(story, semantic, exact, clusters)
        for story in semantic.stories
    )


class StoryReconciler:
    """Runs the ``stories`` stage over one trading day.

    Construction does no work and touches no model: the encoder resolves on
    first use, so a day with nothing to encode never loads one.
    """

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        pipeline_version: str,
        encoder: StoryEncoder | None = None,
        dedup_config: DedupConfig | None = None,
        semantic_config: SemanticDedupConfig | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline_version = str(pipeline_version).strip()
        if not self.pipeline_version:
            raise ValueError("pipeline_version is required")
        self._encoder = encoder
        self.dedup_config = dedup_config or DedupConfig(
            supported_tickers=SUPPORTED_TICKERS
        )
        self.semantic_config = semantic_config or SemanticDedupConfig(
            supported_tickers=SUPPORTED_TICKERS
        )

    @property
    def encoder(self) -> StoryEncoder:
        """The injected encoder, or M1's default service on first ask."""

        if self._encoder is None:
            self._encoder = get_default_service()
        return self._encoder

    # -- Partition enumeration -------------------------------------------

    def partitions(self, trading_day: str | date) -> list[str]:
        """Every ticker this day is answerable for, evidence or not.

        The union matters.  Driving the stage from evidence alone leaves a
        partition that *had* stories and no longer has evidence unvisited,
        and an unvisited partition keeps its previous generation --
        authoritative-looking, and derived from evidence that is gone.
        Including the persisted partitions means such a day is reconciled
        to an empty set and cleared, which is the honest answer.

        **Discovery projects nothing.**  Both reads here are association
        and row identity only; ``classify_evidence``, the publisher policy,
        and normalization all wait until a partition's own ``stage_run`` is
        open.  Enumerating through :meth:`~phase0.repository.Phase0Reader.
        evidence_partitions` instead would put projection *before* every
        run, where one unprojectable partition takes the whole day down --
        no stories written, and no failed attempt recorded either, so the
        ledger cannot even say the day was tried.
        """

        day = _normalize_day(trading_day)
        reader = self.repository.read
        evidence = set(reader.evidence_partition_tickers(day))
        persisted = set(
            reader.story_partitions(day, pipeline_version=self.pipeline_version)
        )
        return sorted(evidence | persisted)

    # -- The stage --------------------------------------------------------

    def run(
        self, trading_day: str | date, *, run_id: str
    ) -> tuple[dict[str, Any], list[Any]]:
        """Reconcile every partition of ``trading_day``; report what happened.

        Returns the ``(counts, errors)`` pair the orchestrator's component
        protocol expects.  Registering this stage in ``pipeline.py`` is a
        later change; the shape is here so that change is a line rather
        than a rewrite.
        """

        day = _normalize_day(trading_day)
        outcomes = [
            self.run_partition(ticker, day, base_run_id=run_id)
            for ticker in self.partitions(day)
        ]
        return summarize(outcomes)

    def run_partition(
        self, ticker: str, trading_day: str | date, *, base_run_id: str
    ) -> PartitionOutcome:
        """Reconcile one partition, and never raise.

        Isolation is the contract: a partition that fails has already
        settled its own ``failed`` run-log row inside ``stage_run``'s
        ``finally``, so catching the exception out here loses nothing from
        the ledger and lets the remaining tickers run.  Evidence and
        stories another partition already committed are durable; there is
        no cross-partition transaction to unwind.

        The message is redacted on the way into the outcome.  The run log
        redacts what it stores, but this value is *returned* -- to the
        orchestrator, to a log line, to whatever reads the report -- and an
        exception raised somewhere near an HTTP client carries whatever
        that client was holding.  Sanitizing here means the returned report
        and the durable row say the same safe thing rather than the report
        being the loose copy.
        """

        symbol = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        try:
            return self._settle_partition(symbol, day, base_run_id=base_run_id)
        except Exception as exc:  # noqa: BLE001 - isolation is the contract
            return PartitionOutcome(
                ticker=symbol,
                trading_day=day,
                status="failed",
                story_stage=None,
                degradation_reason=None,
                story_count=0,
                error={
                    "type": "partition_error",
                    "ticker": symbol,
                    "trading_day": day,
                    # The type name is kept whole -- it is the useful half,
                    # and a class name is not a place a secret lives.
                    "error": sanitize_diagnostic_scalar(
                        f"{type(exc).__name__}: {exc}", "partition error"
                    ),
                },
            )

    def _settle_partition(
        self, ticker: str, trading_day: str, *, base_run_id: str
    ) -> PartitionOutcome:
        """One partition, one run, one write.

        The whole computation sits inside the run.  That is what makes an
        M2 capacity failure or a dead encoder a recorded ``failed``
        ``stories`` attempt instead of a traceback the ledger never saw,
        and it costs nothing: no lock is held across the model call,
        because the reads finish before it and the only write is the
        terminal reconcile at the end.
        """

        with self.repository.stage_run(
            run_id=partition_run_id(base_run_id, ticker, trading_day),
            stage=STAGE,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=self.pipeline_version,
        ) as run:
            evidence = self.repository.read.partition_evidence(ticker, trading_day)
            exact = deduplicate(evidence.items, config=self.dedup_config)

            reason: str | None = None
            try:
                semantic = merge_semantic_duplicates(
                    stories_from_dedup(exact, evidence.items),
                    config=self.semantic_config,
                    encoder=self.encoder,
                )
            except _RECOVERABLE_M3 as exc:
                # Before the terminal mutation, always.  Recorded after it,
                # the marker lands in a list nothing reads while the
                # committed row still says the run succeeded.
                reason = DEGRADATION_REASON
                run.record_degradation(reason, detail=f"{type(exc).__name__}: {exc}")
                records = exact_story_records(exact)
                stage = "m2.exact"
            else:
                records = semantic_story_records(semantic, exact)
                stage = "m3.semantic"

            # The only story mutation of this run, and its last statement.
            report = self.repository.reconcile_stories(
                run=run,
                ticker=ticker,
                trading_day=trading_day,
                pipeline_version=self.pipeline_version,
                stories=records,
                delete_obsolete=True,
                terminal=True,
            )
            # Copied while the run is open; the context stays behind.
            counts = {
                str(key): int(value)
                for key, value in report.counts.items()
                if isinstance(value, int)
            }

        return PartitionOutcome(
            ticker=ticker,
            trading_day=trading_day,
            status="degraded" if reason is not None else "success",
            story_stage=stage,
            degradation_reason=reason,
            story_count=len(records),
            counts=counts,
            excluded=evidence.excluded,
        )


#: Outcome status to the counter that records it.
_STATUS_COUNTER = {
    "success": "succeeded",
    "degraded": "degraded",
    "failed": "failed",
}


def summarize(
    outcomes: Sequence[PartitionOutcome],
) -> tuple[dict[str, Any], list[Any]]:
    """Fold per-partition outcomes into one component report.

    Degraded partitions are counted apart from successful ones rather than
    folded into them: "four tickers, one of them without semantic dedup" is
    the fact an operator needs, and a single total cannot say it.
    """

    counts: dict[str, Any] = {
        "partitions": len(outcomes),
        "partitions_succeeded": 0,
        "partitions_degraded": 0,
        "partitions_failed": 0,
        "stories_inserted": 0,
        "stories_updated": 0,
        "stories_unchanged": 0,
        "stories_deleted": 0,
        "stories_invalidated": 0,
        "themes_invalidated": 0,
        "evidence_excluded": 0,
    }
    errors: list[Any] = []
    for outcome in outcomes:
        counts[f"partitions_{_STATUS_COUNTER[outcome.status]}"] += 1
        counts["stories_inserted"] += outcome.counts.get("inserted", 0)
        counts["stories_updated"] += outcome.counts.get("updated", 0)
        counts["stories_unchanged"] += outcome.counts.get("unchanged", 0)
        counts["stories_deleted"] += outcome.counts.get("deleted", 0)
        counts["stories_invalidated"] += outcome.counts.get("invalidated", 0)
        counts["themes_invalidated"] += outcome.counts.get("invalidated_themes", 0)
        counts["evidence_excluded"] += len(outcome.excluded)
        if outcome.error is not None:
            errors.append(dict(outcome.error))
        elif outcome.degraded:
            errors.append(
                {
                    "type": "stage_degraded",
                    "ticker": outcome.ticker,
                    "trading_day": outcome.trading_day,
                    "reason": outcome.degradation_reason,
                }
            )
    return counts, errors


__all__ = [
    "DEGRADATION_REASON",
    "PartitionOutcome",
    "STAGE",
    "StoryReconciler",
    "exact_story_records",
    "partition_run_id",
    "semantic_story_records",
    "summarize",
]
