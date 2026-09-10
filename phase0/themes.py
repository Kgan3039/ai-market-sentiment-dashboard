"""The theme stage: persisted stories in, one authoritative theme set out.

M5 clusters what the story stage *stored*, not what it happened to hold in
memory on the way there.  The two ought to agree; if they ever do not, the
stored generation is the authoritative one, and reading it back is what
makes that true rather than merely intended.

**The operational stage name is ``themes``** (decision A4), stable whatever
the outcome, with the algorithm's identity recorded on the rows it produced.

**There is no fallback tier.**  M3 could fall back to M2 because M2's
clusters are a real, complete, lower-tier answer that already existed.  M5
has no such neighbour: ``small_n_fallback`` and ``agglomerative`` are
choices *inside* ``cluster_themes``, made by the algorithm about the data,
not by a caller recovering from a failure.  A partial theme set is refused
upstream for the same reason — a reader cannot tell a day with three themes
from a day whose clustering gave up after three.  So every M5 failure fails
the partition, and the only deliberate degradation here is an all-M2 story
generation, which is a statement about the *stories*.

**Previous-theme identity arrives from outside.**  Story reconciliation
deletes a partition's theme set whenever any story representation changes,
and ``themes`` has no ``invalidated_at`` column, so by the time this stage
opens there may be nothing left to read.  The coordinator captures the
identities before the story stage runs and hands them in;
:class:`ThemeReconciler` never tries to read them itself, because the read
would silently return nothing on exactly the runs continuity is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from nlp.dedup.errors import DedupInputError
from nlp.dedup.selection import cluster_fingerprint_for
from nlp.embeddings import EmbeddingStorageError, deserialize_vector
from nlp.themes import (
    PreviousTheme,
    ThemeConfig,
    ThemeSet,
    ThemeSourceMetadata,
    ThemeStory,
    cluster_themes,
    encoder_identity,
)

from .errors import Phase0IntegrityError
from .models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from .repository import (
    Phase0Repository,
    PersistedStory,
    PersistedStoryMember,
    PreviousThemeGeneration,
    StoryGeneration,
    ThemeIdentity,
    _normalize_day,
)
from .scalars import sanitize_diagnostic_scalar
from .tickers import SUPPORTED_TICKERS, normalize_ticker

#: The unit of work, stable whatever the outcome (decision A4).
STAGE = "themes"

#: The one deliberate degradation this stage records.  It is a statement
#: about the stories, not about M5: the clustering was never attempted.
DEGRADATION_REASON = "m5_requires_semantic_stories"

#: How a persisted story's standfirst is recovered for embedding.
#:
#: M3 chose a story's description as "the canonical member's, else the
#: first member that has one, in cluster-member order".  Both halves
#: survive in the database: ``stories.canonical_item_id`` names the
#: canonical raw item, and ``story_members.position`` records the cluster
#: order M2 assembled — canonical first, then by publication time.  So the
#: reconstruction is the original selection rather than a new rule that
#: merely happens to be deterministic.
#:
#: Ordering by ``raw_item_id`` would *also* be deterministic and would
#: differ: ids are insertion order, and one provider response routinely
#: spans several publication days.  The description feeds
#: ``compose_embedding_text``, so choosing differently here changes the
#: vector, the clustering, and the themes.
#:
#: Changing this policy requires a ``pipeline_version`` bump unless a
#: persisted policy fingerprint is introduced first.
DESCRIPTION_POLICY = (
    "canonical_m2_cluster: canonical_item_description_then_"
    "first_non_empty_by_member_position_within_that_cluster"
)

#: Why a captured theme identity could not be carried into this run.  Each
#: is counted separately: "the model moved" and "the centroid was corrupt"
#: are different facts about a partition, and a single total cannot say
#: which happened.
REJECTED_ALGORITHM = "algorithm"
REJECTED_CONFIG = "config"
REJECTED_MODEL = "model"
REJECTED_REVISION = "revision"
REJECTED_DIMENSION = "dimension"
REJECTED_CENTROID = "centroid"
#: A child row whose provenance disagrees with the set that holds it.  The
#: whole generation is refused: a set assembled by one run cannot have been
#: half-produced by another, so the disagreement is corruption, and reusing
#: the agreeing half would carry identities out of a generation that never
#: existed.
REJECTED_INCONSISTENT = "inconsistent"

_REJECTION_REASONS = (
    REJECTED_ALGORITHM,
    REJECTED_CONFIG,
    REJECTED_MODEL,
    REJECTED_REVISION,
    REJECTED_DIMENSION,
    REJECTED_CENTROID,
    REJECTED_INCONSISTENT,
)

#: Story generation classifications; see :func:`classify_generation`.
GENERATION_EMPTY = "empty"
GENERATION_SEMANTIC = "semantic"
GENERATION_EXACT = "exact"


class ThemeGenerationError(Phase0IntegrityError):
    """The persisted story generation cannot be clustered as it stands.

    Not a degradation.  Every case that reaches here — two generations in
    one partition, a story whose stage nothing set, a partition whose rows
    disagree about which model produced them — means something wrote
    outside the story runner, and clustering it anyway would build themes
    over a set nobody can vouch for.
    """


class CanonicalClusterUnrecoverable(ThemeGenerationError):
    """A merged story's canonical M2 cluster cannot be identified.

    Its own kind because it is not a statement about the *stage* of the
    stories or about which model produced them -- those are visible in the
    rows.  This is the persisted ordering contract between
    ``member_story_keys`` and ``story_members.position`` having come
    apart, which is only discoverable by trying to prove it.
    """


def partition_run_id(base_run_id: str, ticker: str, trading_day: str) -> str:
    """The run identity for one ticker/day partition.

    ``run_log`` is ``UNIQUE(run_id, stage)`` and I1 refuses to record one
    identity against a second partition, so the identity carries the
    partition.  Spelled as the other stages spell it.
    """

    return f"{base_run_id}:{ticker}:{trading_day}"


@dataclass(frozen=True)
class ProvenanceExpectation:
    """The space this run's themes will live in.

    Assembled before clustering — the encoder can be asked its identity
    without embedding anything — so a captured identity can be judged
    against the run that is about to happen rather than against the run
    that just did.
    """

    algorithm_version: str
    config_fingerprint: str
    model_name: str
    model_revision: str | None
    embedding_dimension: int | None


@dataclass(frozen=True)
class PreviousThemeCapture:
    """Captured identities, and what became of each.

    The counts are the point as much as the identities are: "identity was
    not carried over" and "there were no previous themes" are different
    facts, and A3 requires a reader be able to tell them apart.
    """

    identities: tuple[ThemeIdentity, ...] = ()
    compatible: tuple[PreviousTheme, ...] = ()
    rejected: Mapping[str, int] = field(default_factory=dict)
    #: True when a ``theme_sets`` row was stored at all, themes or not.
    generation_present: bool = False
    #: Why the whole generation was refused, or ``None`` when it stands.
    generation_rejected: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        """Flat counters, ready to merge into a stage report."""

        counts = {
            "previous_themes_seen": len(self.identities),
            "previous_themes_compatible": len(self.compatible),
            "previous_themes_rejected": sum(self.rejected.values()),
            "previous_theme_generation_seen": 1 if self.generation_present else 0,
            "previous_theme_generation_rejected": (
                1 if self.generation_rejected is not None else 0
            ),
        }
        for reason in _REJECTION_REASONS:
            counts[f"previous_themes_rejected_{reason}"] = self.rejected.get(reason, 0)
            # Named separately from the identity counters above, which
            # count *rows*.  A refused generation holding no themes has no
            # rows to count, and reporting only that something was refused
            # without saying what moved -- the model, the configuration --
            # is the half of the fact nobody can act on.
            counts[f"previous_theme_generation_rejected_{reason}"] = (
                1 if self.generation_rejected == reason else 0
            )
        return counts

    @property
    def has_incompatible(self) -> bool:
        """True when a stored generation may not be reused at all.

        Keyed on the *generation*, not on how many identities survived.  A
        compatible generation whose only centroid was corrupt is still a
        generation this run could have produced, and ``reconcile_themes``
        replaces it wholesale anyway; a generation from another model is
        not, and has to go before M5 writes over the top of it.
        """

        return self.generation_rejected is not None


@dataclass(frozen=True)
class ThemePartitionOutcome:
    """What one partition's theme run did, after that run has closed.

    Frozen, and built from values copied while the run was still open.  No
    :class:`~phase0.repository.StageRunContext` reaches here: it authorizes
    mutations, and an authorization outliving its run is what that class
    exists to make impossible.
    """

    ticker: str
    trading_day: str
    #: ``success``, ``degraded``, ``failed``, or ``not_attempted``.
    status: str
    #: How the story generation was classified, or ``None`` when the run
    #: never got far enough to look.
    generation: str | None
    theme_count: int
    cleared: bool
    degradation_reason: str | None = None
    #: Why the stored theme generation could not be reused, when one was
    #: stored and was refused.  Not a degradation: a model or
    #: configuration transition is an ordinary thing for a healthy run to
    #: report having noticed.
    previous_generation_rejected: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    error: Mapping[str, str] | None = None

    @property
    def degraded(self) -> bool:
        return self.degradation_reason is not None

    @property
    def attempted(self) -> bool:
        """False when an upstream failure meant no run was opened."""

        return self.status != "not_attempted"


def classify_generation(generation: StoryGeneration) -> str:
    """Name what the persisted partition holds, or refuse it.

    Four answers and three refusals.  The refusals are deliberate: each
    describes a partition that one whole-partition reconciliation could
    not have produced, so treating it as ordinary would be treating
    evidence of a bug as a data condition.
    """

    if generation.is_empty:
        return GENERATION_EMPTY

    stages = generation.stages
    if "" in stages:
        raise ThemeGenerationError(
            f"{generation.ticker}/{generation.trading_day} holds a story whose "
            f"stage was never set; a story that cannot say which algorithm "
            f"produced it cannot be clustered"
        )
    unknown = stages - {"m2.exact", "m3.semantic"}
    if unknown:
        raise ThemeGenerationError(
            f"{generation.ticker}/{generation.trading_day} holds unknown story "
            f"stage(s) {sorted(unknown)}"
        )
    if len(stages) > 1:
        raise ThemeGenerationError(
            f"{generation.ticker}/{generation.trading_day} mixes story stages "
            f"{sorted(stages)}; one reconciliation writes one generation, so "
            f"two of them in one partition means something wrote around it"
        )
    if stages == {"m2.exact"}:
        return GENERATION_EXACT

    if len(generation.model_identities) > 1:
        raise ThemeGenerationError(
            f"{generation.ticker}/{generation.trading_day} holds semantic "
            f"stories from {len(generation.model_identities)} different model "
            f"identities; one generation speaks with one voice"
        )
    return GENERATION_SEMANTIC


def assert_encoder_matches(generation: StoryGeneration, encoder: Any) -> None:
    """Refuse to cluster stories an encoder did not produce.

    M5 embeds afresh, so the vectors it compares must come from the model
    the stories were merged under.  A different encoder of the same width
    produces perfectly plausible cosine values over an incomparable space:
    the themes would look ordinary and mean nothing, and a previous
    identity matched inside them would be inherited from a space it was
    never measured in.
    """

    persisted = next(iter(generation.model_identities))
    name, revision, dimension = encoder_identity(encoder)
    if persisted != (name, revision, dimension):
        raise ThemeGenerationError(
            f"{generation.ticker}/{generation.trading_day} stories were merged "
            f"under {persisted!r} but the configured encoder reports "
            f"{(name, revision, dimension)!r}; clustering across two embedding "
            f"spaces produces plausible numbers about nothing"
        )


def expected_provenance(config: ThemeConfig, encoder: Any) -> ProvenanceExpectation:
    """The identity this run's theme set will carry, before it runs."""

    from nlp.themes import ALGORITHM_VERSION

    name, revision, dimension = encoder_identity(encoder)
    return ProvenanceExpectation(
        algorithm_version=ALGORITHM_VERSION,
        config_fingerprint=config.fingerprint(
            model_name=name,
            model_revision=revision,
            embedding_dimension=dimension,
        ),
        model_name=name,
        model_revision=revision,
        embedding_dimension=dimension,
    )


def _generation_mismatch(
    previous: PreviousThemeGeneration, expected: ProvenanceExpectation
) -> str | None:
    """Which provenance dimension refuses this whole generation, if any.

    Checked in declaration order and reported as one reason: a generation
    built by a different model under a different configuration is refused
    once, not twice.
    """

    if previous.algorithm_version != expected.algorithm_version:
        return REJECTED_ALGORITHM
    if previous.config_fingerprint != expected.config_fingerprint:
        return REJECTED_CONFIG
    if previous.model_name != expected.model_name:
        return REJECTED_MODEL
    # NULL-safe by construction: both sides are ``str | None`` and
    # ``None == None`` is the answer we want, where SQL's would not be.
    if previous.model_revision != expected.model_revision:
        return REJECTED_REVISION
    if previous.embedding_dimension != expected.embedding_dimension:
        return REJECTED_DIMENSION
    return None


def _child_disagrees(
    identity: ThemeIdentity, previous: PreviousThemeGeneration
) -> bool:
    """Does this theme claim a provenance its own set does not?"""

    return (
        identity.algorithm_version != previous.algorithm_version
        or identity.config_fingerprint != previous.config_fingerprint
        or identity.model_name != previous.model_name
        or identity.model_revision != previous.model_revision
        or identity.embedding_dimension != previous.embedding_dimension
    )


def evaluate_previous_themes(
    previous: PreviousThemeGeneration | None, expected: ProvenanceExpectation
) -> PreviousThemeCapture:
    """Decide whether this run may reuse the stored theme generation.

    A3's gate, applied here rather than in SQL: whether a stored identity
    is reusable is a question about the *upcoming* run, and no row knows
    what that run will be.

    **The generation is the unit.**  Provenance is read off the
    ``theme_sets`` row, so a set holding no themes is still a generation
    that can be incompatible -- a day below the clustering floor stores
    exactly that, and judging it by its children would find nothing to
    judge.  ``config_fingerprint`` is the strongest single check, folding
    the algorithm version, the model identity and width, the thresholds,
    the caps and every static policy; the rest are compared anyway,
    because a nullable column that cannot state its own provenance is not
    evidence that it matches.

    Three outcomes, in order:

    * the set disagrees with the upcoming run -- refuse the whole
      generation, reuse nothing, and let the caller clear it;
    * a child disagrees with its own set -- refuse the whole generation
      too.  Reusing the agreeing subset would carry identities out of a
      generation that never existed as a whole;
    * otherwise judge the centroids one at a time.  A corrupt centroid
      costs its own identity and nothing else: the generation is still one
      this run could have produced.
    """

    if previous is None:
        return PreviousThemeCapture()

    identities = tuple(previous.identities)
    rejected: dict[str, int] = {}

    def refuse_generation(reason: str) -> PreviousThemeCapture:
        return PreviousThemeCapture(
            identities=identities,
            compatible=(),
            rejected={reason: len(identities)} if identities else {},
            generation_present=True,
            generation_rejected=reason,
        )

    mismatch = _generation_mismatch(previous, expected)
    if mismatch is not None:
        return refuse_generation(mismatch)
    if any(_child_disagrees(identity, previous) for identity in identities):
        return refuse_generation(REJECTED_INCONSISTENT)

    compatible: list[PreviousTheme] = []
    for identity in identities:
        centroid = _centroid_of(identity, expected.embedding_dimension)
        if centroid is None:
            rejected[REJECTED_CENTROID] = rejected.get(REJECTED_CENTROID, 0) + 1
            continue
        compatible.append(
            PreviousTheme(theme_key=identity.theme_key, centroid=centroid)
        )

    return PreviousThemeCapture(
        identities=identities,
        compatible=tuple(compatible),
        rejected=dict(rejected),
        generation_present=True,
        generation_rejected=None,
    )


def _centroid_of(
    identity: ThemeIdentity, dimension: int | None
) -> tuple[float, ...] | None:
    """Decode a stored centroid, or report that it cannot be used.

    A centroid that is missing, truncated, the wrong width, or not finite
    is a rejection, never an exception: one corrupt row must not cost a
    partition its whole capture, and it must certainly not reach M5, where
    it would silently claim an identity by comparing against nonsense.

    ``EmbeddingStorageError`` is the specific class the deserializer
    raises for every one of those; nothing broader is caught, so a genuine
    programming error still surfaces.
    """

    if identity.centroid is None or dimension is None:
        return None
    try:
        vector = deserialize_vector(identity.centroid, expected_dimension=dimension)
    except EmbeddingStorageError:
        return None
    return tuple(float(value) for value in vector)


def _published_at(value: str | None) -> datetime | None:
    """Parse a persisted timestamp into the aware datetime M5 requires.

    Phase 0 stores offset-bearing ISO-8601 and validates it on the way in,
    so a value that will not parse — or parses naive — is corruption
    rather than an ordinary absence, and is refused instead of quietly
    becoming ``None``.  A silently undated story cannot merge on time and
    scores zero recency, which is a different day's news presented as
    this one's.
    """

    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ThemeGenerationError(
            f"persisted published_at {value!r} is not an ISO-8601 timestamp"
        ) from exc
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise ThemeGenerationError(
            f"persisted published_at {value!r} carries no UTC offset"
        )
    return stamp


def _proves(ticker: str, identifiers: Sequence[str], key: str) -> bool:
    """Does this member set digest to ``key``?

    ``cluster_fingerprint_for`` refuses an empty, blank, or duplicated
    member set rather than returning a digest for it.  A set it will not
    fingerprint has not proved anything, so the refusal is an answer here
    rather than an exception: the caller's next line raises the error that
    describes the actual problem.
    """

    try:
        return cluster_fingerprint_for(ticker, list(identifiers)) == key
    except DedupInputError:
        return False


def canonical_cluster_members(
    story: PersistedStory,
) -> tuple[PersistedStoryMember, ...]:
    """The members of the M2 cluster this story is named after.

    M3 embedded one ``StoryInput`` per M2 cluster and then merged them; the
    canonical semantic story's text came from *one* of those clusters, and
    the others contributed members but never text.  A persisted story
    flattens all of them into one member list, so which members carried
    the text has to be recovered before the text can be.

    **Every claim is proved before it is used.**  ``member_story_keys[0]``
    is the canonical M2 cluster's fingerprint, and that fingerprint is a
    digest of the ticker and the cluster's sorted member ids
    (:func:`nlp.dedup.selection.cluster_fingerprint_for`).  So the record
    can be asked to demonstrate what it asserts:

    * **One key** claims the whole persisted member set is that single
      cluster.  The digest of the whole set must equal it.  Believing the
      claim because there is only one key is the trap: a story that really
      merged two clusters, with one key lost, still holds both clusters'
      members, and the surviving key does not describe them.
    * **Several keys** claim a merge, and members are persisted cluster by
      cluster in key order, so the canonical cluster is a prefix.  Exactly
      one prefix must digest to it.
    * **No keys at all** is not a claim M3 can make: it builds
      ``member_story_keys`` from a non-empty ordered list, so every
      semantic story names at least the cluster it is called after.

    Anything unproved fails closed.  There is no safe reading of a broken
    boundary -- stopping early drops a standfirst that belonged to the
    canonical cluster, reading on borrows one that did not -- and either
    way M5 would embed text no run ever encoded while the day's themes
    looked ordinary.  The reconstruction contract is that it reproduces
    M3's input, so when it cannot, it says so and the partition fails.
    """

    ordered = tuple(sorted(story.members, key=lambda entry: entry.position))
    keys = story.member_story_keys
    identifiers = [str(member.raw_item_id) for member in ordered]

    if not keys:
        raise CanonicalClusterUnrecoverable(
            f"story {story.cluster_fingerprint} names no M2 cluster at all; a "
            f"semantic story always records the cluster it is called after, so "
            f"the text M3 embedded cannot be identified"
        )

    canonical_key = keys[0]

    if len(keys) == 1:
        if _proves(story.ticker, identifiers, canonical_key):
            return ordered
        raise CanonicalClusterUnrecoverable(
            f"story {story.cluster_fingerprint} claims its {len(ordered)} "
            f"persisted members are the single M2 cluster {canonical_key!r}, but "
            f"their fingerprint is not that key; the membership and the key "
            f"describe different clusters, and which of them M3 embedded is "
            f"unknowable"
        )

    matches = [
        size
        for size in range(1, len(ordered) + 1)
        if _proves(story.ticker, identifiers[:size], canonical_key)
    ]
    if len(matches) == 1:
        return ordered[: matches[0]]
    if not matches:
        raise CanonicalClusterUnrecoverable(
            f"story {story.cluster_fingerprint} claims {len(keys)} merged "
            f"clusters, but no prefix of its {len(ordered)} persisted members "
            f"digests to the canonical cluster key {canonical_key!r}; the text M3 "
            f"embedded cannot be identified, and inventing one is not a "
            f"degradation this stage is allowed to make"
        )
    raise CanonicalClusterUnrecoverable(
        f"story {story.cluster_fingerprint} has {len(matches)} member prefixes "
        f"digesting to the canonical cluster key {canonical_key!r}; which members "
        f"M3 embedded is ambiguous, and a guess between them would be a guess about "
        f"what the day's themes are built from"
    )


def story_description(story: PersistedStory) -> str | None:
    """The standfirst M5 embeds for one persisted story.

    :data:`DESCRIPTION_POLICY`: the canonical member's, else the first
    member of the *canonical M2 cluster* that has one, in persisted
    ``position`` order.

    The cluster restriction is the whole point.  M3 embedded the canonical
    cluster's text and nothing else, so a story that merged two clusters
    was encoded from the canonical one's title and description -- often a
    title alone.  Falling through to the other cluster's standfirst would
    hand M5 a title-and-description pair that no run ever encoded, and
    call it a reconstruction.
    """

    members = canonical_cluster_members(story)
    by_item = {member.raw_item_id: member for member in members}
    canonical = by_item.get(story.canonical_item_id or -1)
    if canonical is not None and (canonical.description or "").strip():
        return canonical.description
    for member in members:
        if (member.description or "").strip():
            return member.description
    return None


def theme_story(story: PersistedStory) -> ThemeStory:
    """Project one persisted story onto M5's input type.

    Everything trust-bearing travels: the quarantine M2 raised, the
    conflicts it raised them over, M3's skip reason and its accepted merge
    evidence.  A projection that carried only the title and the timestamps
    would launder all of it, and M5 would cluster a disputed story as an
    ordinary one.
    """

    ordered = sorted(story.members, key=lambda member: member.position)
    return ThemeStory(
        story_key=story.cluster_fingerprint,
        ticker=story.ticker,
        title=story.canonical_title,
        description=story_description(story),
        published_at=_published_at(story.published_at),
        outlets=tuple(sorted({member.outlet for member in ordered if member.outlet})),
        item_ids=tuple(str(member.raw_item_id) for member in ordered),
        source_links=tuple(
            (
                str(member.raw_item_id),
                member.outlet or "",
                member.canonical_url or member.url,
            )
            for member in ordered
        ),
        outlet_count=story.outlet_count,
        member_story_keys=tuple(story.member_story_keys),
        quarantined_member_ids=tuple(
            str(member.raw_item_id) for member in ordered if member.quarantined
        ),
        provider_conflicts=tuple(story.provider_conflicts),
        semantic_skip_reason=story.semantic_skip_reason,
        merge_evidence=tuple(story.semantic_merges),
        content_hash=story.content_hash,
    )


def source_metadata(generation: StoryGeneration) -> ThemeSourceMetadata:
    """Record which story generation these themes were built from."""

    stage = next(iter(generation.stages))
    model_name, model_revision, dimension = next(iter(generation.model_identities))
    first = generation.stories[0]
    return ThemeSourceMetadata(
        stage=stage,
        algorithm_version=first.algorithm_version or "",
        config_fingerprint=first.config_fingerprint or "",
        model_name=model_name or "",
        model_revision=model_revision,
        embedding_dimension=dimension,
        story_count=len(generation.stories),
        quarantined_story_count=sum(
            1 for story in generation.stories if story.quarantined
        ),
        semantically_skipped_story_count=sum(
            1 for story in generation.stories if story.semantic_skip_reason is not None
        ),
        merged_story_count=sum(
            1 for story in generation.stories if len(story.member_story_keys) > 1
        ),
    )


def theme_records(
    theme_set: ThemeSet, story_ids: Mapping[str, int], item_ids: Mapping[str, int]
) -> tuple[
    ThemeSetRecord,
    list[ThemeRecord],
    list[OtherCoverageRecord],
    list[ExcludedStoryRecord],
]:
    """Project one :class:`~nlp.themes.ThemeSet` onto persistence records.

    ``story_ids`` maps a story key onto its ``stories.id``; the theme
    tables key on the durable row, and M5 reasons in story keys, so the
    translation happens once, here.
    """

    set_record = ThemeSetRecord(
        method=theme_set.method.value,
        method_reason=theme_set.method_reason,
        quality=_quality(theme_set),
        source_metadata=(
            None
            if theme_set.source_metadata is None
            else _source_metadata_mapping(theme_set.source_metadata)
        ),
        config_fingerprint=theme_set.config_fingerprint,
        algorithm_version=theme_set.algorithm_version,
        model_name=theme_set.model_name,
        model_revision=theme_set.model_revision,
        embedding_dimension=theme_set.embedding_dimension,
    )

    themes = [
        ThemeRecord(
            fingerprint=theme.fingerprint,
            theme_key=theme.theme_key,
            label=theme.label,
            label_source=theme.label_source,
            story_ids=tuple(story_ids[key] for key in theme.member_story_keys),
            citation_item_ids=tuple(
                item_ids[item] for item in theme.citable_item_ids if item in item_ids
            ),
            status="ready",
            salience=theme.salience,
            salience_rank=theme.salience_rank,
            cohesion=theme.cohesion,
            min_pairwise_cohesion=theme.min_pairwise_cohesion,
            story_count=theme.story_count,
            outlet_count=theme.outlet_count,
            latest_published_at=theme.salience_features.latest_published_at,
            salience_story_component=theme.salience_features.story_component,
            salience_outlet_component=theme.salience_features.outlet_component,
            salience_recency_component=theme.salience_features.recency_component,
            centroid=_centroid_blob(theme.centroid),
            matched_previous_key=theme.matched_previous_key,
            method=theme.method.value,
            content_hash=theme.fingerprint,
            algorithm_version=theme_set.algorithm_version,
            config_fingerprint=theme_set.config_fingerprint,
            model_name=theme_set.model_name,
            model_revision=theme_set.model_revision,
            embedding_dimension=theme_set.embedding_dimension,
        )
        for theme in theme_set.themes
    ]

    other = [
        OtherCoverageRecord(
            story_id=story_ids[entry.evidence.story_key],
            reason=entry.reason.value,
            position=position,
        )
        for position, entry in enumerate(theme_set.other_coverage)
    ]
    excluded = [
        ExcludedStoryRecord(
            story_id=story_ids[entry.story_key], reason=entry.reason.value
        )
        for entry in theme_set.excluded
    ]
    return set_record, themes, other, excluded


def _centroid_blob(centroid: Sequence[float]) -> bytes | None:
    from nlp.embeddings import serialize_vector

    return serialize_vector(list(centroid)) if centroid else None


def _quality(theme_set: ThemeSet) -> dict[str, Any]:
    quality = theme_set.quality
    return {
        "story_count": quality.story_count,
        "theme_count": quality.theme_count,
        "other_coverage_count": quality.other_coverage_count,
        "excluded_count": quality.excluded_count,
        "singleton_theme_count": quality.singleton_theme_count,
        "mean_cohesion": quality.mean_cohesion,
        "min_pairwise_cohesion": quality.min_pairwise_cohesion,
        "max_inter_theme_similarity": quality.max_inter_theme_similarity,
        "theme_coverage": quality.theme_coverage,
        "meets_ac4_shape": quality.meets_ac4_shape,
        "ac4_shape_detail": quality.ac4_shape_detail,
    }


def _source_metadata_mapping(metadata: ThemeSourceMetadata) -> dict[str, Any]:
    return {
        "stage": metadata.stage,
        "algorithm_version": metadata.algorithm_version,
        "config_fingerprint": metadata.config_fingerprint,
        "model_name": metadata.model_name,
        "model_revision": metadata.model_revision,
        "embedding_dimension": metadata.embedding_dimension,
        "story_count": metadata.story_count,
        "quarantined_story_count": metadata.quarantined_story_count,
        "semantically_skipped_story_count": (metadata.semantically_skipped_story_count),
        "merged_story_count": metadata.merged_story_count,
    }


class ThemeReconciler:
    """Runs the ``themes`` stage over one trading day's partitions."""

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        pipeline_version: str,
        encoder: Any | None = None,
        config: ThemeConfig | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline_version = str(pipeline_version).strip()
        if not self.pipeline_version:
            raise ValueError("pipeline_version is required")
        self._encoder = encoder
        self.config = config or ThemeConfig(supported_tickers=SUPPORTED_TICKERS)

    @property
    def encoder(self) -> Any:
        """The injected encoder, or M1's default service on first ask."""

        if self._encoder is None:
            from nlp.embeddings import get_default_service

            self._encoder = get_default_service()
        return self._encoder

    # -- Continuity capture ----------------------------------------------

    def capture_previous(
        self, ticker: str, trading_day: str | date
    ) -> PreviousThemeGeneration | None:
        """Read one partition's theme identities, before anything writes.

        A plain read, deliberately not a decision: the coordinator calls
        it *before* the story stage, where the provenance this run will
        carry is not yet the interesting question — keeping the row alive
        is.  Judging it happens later, in
        :func:`evaluate_previous_themes`.
        """

        return self.repository.read.previous_theme_generation(
            normalize_ticker(ticker),
            _normalize_day(trading_day),
            self.pipeline_version,
        )

    # -- The stage --------------------------------------------------------

    def run_partition(
        self,
        ticker: str,
        trading_day: str | date,
        *,
        base_run_id: str,
        previous: PreviousThemeGeneration | None = None,
    ) -> ThemePartitionOutcome:
        """Settle one partition's themes, and never raise.

        Isolation: a partition that fails has already written its own
        ``failed`` run-log row inside ``stage_run``'s ``finally``, so
        catching out here costs the ledger nothing and lets the remaining
        tickers run.  The diagnostic is redacted on the way into the
        outcome because the outcome is *returned* — an exception raised
        near a model client carries whatever that client was holding.
        """

        symbol = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        # What this attempt committed before it failed, if it failed.  A
        # clear is committed on its own transaction and is not rolled back
        # by a later failure, so an outcome reporting ``cleared=False``
        # would contradict both the database and the run log.
        committed: dict[str, int] = {}
        try:
            return self._settle_partition(
                symbol,
                day,
                base_run_id=base_run_id,
                previous=previous,
                committed=committed,
            )
        except Exception as exc:  # noqa: BLE001 - isolation is the contract
            return ThemePartitionOutcome(
                ticker=symbol,
                trading_day=day,
                status="failed",
                generation=None,
                theme_count=0,
                cleared=bool(committed.get("cleared_rows")),
                counts=dict(committed),
                error={
                    "type": "theme_partition_error",
                    "ticker": symbol,
                    "trading_day": day,
                    "error": sanitize_diagnostic_scalar(
                        f"{type(exc).__name__}: {exc}", "theme partition error"
                    ),
                },
            )

    def _settle_partition(
        self,
        ticker: str,
        trading_day: str,
        *,
        base_run_id: str,
        previous: PreviousThemeGeneration | None,
        committed: dict[str, int],
    ) -> ThemePartitionOutcome:
        """One partition, one run, and at most one theme mutation.

        The whole computation sits inside the run — the story read, the
        guard, the encoder check, and M5 itself — so a failure anywhere is
        a *recorded* attempt rather than a traceback the ledger never saw.
        """

        with self.repository.stage_run(
            run_id=partition_run_id(base_run_id, ticker, trading_day),
            stage=STAGE,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=self.pipeline_version,
        ) as run:
            generation = self.repository.story_generation(
                ticker, trading_day, self.pipeline_version
            )
            kind = classify_generation(generation)

            if kind == GENERATION_EMPTY:
                # Healthy absence.  There is nothing to cluster and
                # nothing wrong; whatever theme set is here describes
                # stories that no longer exist.
                removed = self.repository.clear_theme_set(
                    run=run,
                    ticker=ticker,
                    trading_day=trading_day,
                    pipeline_version=self.pipeline_version,
                    terminal=True,
                )
                committed["cleared_rows"] = removed
                return ThemePartitionOutcome(
                    ticker=ticker,
                    trading_day=trading_day,
                    status="success",
                    generation=kind,
                    theme_count=0,
                    cleared=bool(removed),
                    counts=dict(run.counts),
                )

            if kind == GENERATION_EXACT:
                # Decision H: M2-only output ships no themes at all, and
                # says so mechanically rather than by looking sparse.
                run.record_degradation(
                    DEGRADATION_REASON,
                    detail=(
                        f"{len(generation.stories)} stories are m2.exact; "
                        f"semantic dedup did not produce this generation"
                    ),
                )
                removed = self.repository.clear_theme_set(
                    run=run,
                    ticker=ticker,
                    trading_day=trading_day,
                    pipeline_version=self.pipeline_version,
                    terminal=True,
                )
                committed["cleared_rows"] = removed
                return ThemePartitionOutcome(
                    ticker=ticker,
                    trading_day=trading_day,
                    status="degraded",
                    generation=kind,
                    theme_count=0,
                    cleared=bool(removed),
                    degradation_reason=DEGRADATION_REASON,
                    counts=dict(run.counts),
                )

            assert_encoder_matches(generation, self.encoder)
            expected = expected_provenance(self.config, self.encoder)
            capture = evaluate_previous_themes(previous, expected)

            cleared = False
            if capture.has_incompatible:
                # Everything stored belongs to a space this run cannot
                # vouch for.  It goes now, as an ordinary step rather than
                # as error handling, so the failure path below needs no
                # special case: if M5 raises after this, the run settles
                # failed and the incompatible set stays gone.
                committed["cleared_rows"] = self.repository.clear_theme_set(
                    run=run,
                    ticker=ticker,
                    trading_day=trading_day,
                    pipeline_version=self.pipeline_version,
                    terminal=False,
                )
                committed.update(capture.counts)
                cleared = True

            stories = [theme_story(story) for story in generation.stories]
            theme_set = cluster_themes(
                stories,
                ticker=ticker,
                trading_day=date.fromisoformat(trading_day),
                config=self.config,
                encoder=self.encoder,
                previous_themes=capture.compatible,
                source_metadata=source_metadata(generation),
            )

            story_ids = {
                story.cluster_fingerprint: story.story_id
                for story in generation.stories
            }
            item_ids = {
                str(member.raw_item_id): member.raw_item_id
                for story in generation.stories
                for member in story.members
            }
            set_record, themes, other, excluded = theme_records(
                theme_set, story_ids, item_ids
            )
            self.repository.reconcile_themes(
                run=run,
                ticker=ticker,
                trading_day=trading_day,
                pipeline_version=self.pipeline_version,
                theme_set=set_record,
                themes=themes,
                other_coverage=other,
                excluded=excluded,
                expected_story_signature=generation.signature,
                terminal=True,
            )
            counts = dict(run.counts)
            counts.update(capture.counts)
            theme_count = len(themes)
            rejected = capture.generation_rejected

        return ThemePartitionOutcome(
            ticker=ticker,
            trading_day=trading_day,
            status="success",
            generation=kind,
            theme_count=theme_count,
            cleared=cleared,
            counts=counts,
            previous_generation_rejected=rejected,
        )


__all__ = [
    "CanonicalClusterUnrecoverable",
    "DEGRADATION_REASON",
    "DESCRIPTION_POLICY",
    "GENERATION_EMPTY",
    "GENERATION_EXACT",
    "GENERATION_SEMANTIC",
    "PreviousThemeCapture",
    "ProvenanceExpectation",
    "STAGE",
    "ThemeGenerationError",
    "ThemePartitionOutcome",
    "ThemeReconciler",
    "assert_encoder_matches",
    "canonical_cluster_members",
    "classify_generation",
    "evaluate_previous_themes",
    "expected_provenance",
    "partition_run_id",
    "source_metadata",
    "story_description",
    "theme_records",
    "theme_story",
]
