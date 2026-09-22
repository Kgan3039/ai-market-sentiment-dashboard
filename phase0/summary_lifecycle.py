"""A3: the durable lifecycle of one theme's guarded summary.

A2 (:mod:`ai.guarded_summary`) turns one frozen input into one typed
result and forgets it.  This module is what remembers: it decides whether
a stored artifact may stand in for a new generation, calls A2 only when
nothing may, and hands the result to the repository to be recorded --
activated as the theme's artifact if the input is still current, or kept
as accounting if it is not.

**Zero provider calls, exactly when.**  A stored artifact is reused only
if *all* of the following hold, in this order:

1. the persisted :class:`~phase0.repository.ThemePopulation` is healthy
   and the theme is in it (:func:`phase0.summaries.build_generation_input`
   refuses otherwise, with its code);
2. an ``accepted`` artifact exists for the exact
   ``(theme_id, input_fingerprint, policy_fingerprint)`` -- the frozen
   input's own fingerprint and the fingerprint of the *resolved* policy
   this invocation would generate under;
3. the stored row passes :func:`validate_persisted_artifact` -- identity,
   structure as stored, content digest -- and
4. A2's pure :func:`~ai.guarded_summary.validate_candidate` accepts the
   reconstruction against the frozen input under the policy's rules.

Matching fingerprints alone are not trusted.  A stored row that is missing
a sentence (trailing or middle), carries a citation the input does not
hold or out of position, whose identity columns disagree with its key, or
that would trip a copy rule is a miss, is never returned as current, and
causes no write on the read path.

**One policy, resolved once.**  The policy is resolved from the client by
A2's own :func:`~ai.guarded_summary.resolve_generation_policy` at the
start of an invocation and carried, unchanged, through the lookup, the
generation, and the write.  The rules the fingerprint was computed over
are the rules the validator runs -- a rules file reloaded between the two
cannot make the lookup and the generation disagree.

**Reads never call the provider.**  Only :func:`ensure_summary` may; the
current-artifact reader and every repository reader are pure.

**Currentness is derived, never stored, and always from the database.**
``current_summary_artifact`` and ``ensure_summary`` read their own fresh
:meth:`~phase0.repository.Phase0Reader.theme_population` snapshot and
build the frozen input from it; neither accepts a population from the
caller, so a retained snapshot cannot call a historical artifact current
after its theme moved or vanished.  The policy half of the key is a
:class:`~ai.guarded_summary.GenerationPolicy`, whose fingerprint is
intrinsic to its fields.  A caller cannot make an old artifact current by
handing over an old hash or an old population: there is no parameter that
takes either.

**One definition of a valid stored row.**  :func:`validate_persisted_artifact`
checks the stored identity columns against the input and policy, the
stored ordinals and positions as stored, the content digest recomputed
from the stored rows, and only then runs A2's validator.  The read path
and the write path's existing-holder check both use it; there is no
second definition of "corrupt".

**What this module does not do.**  It registers no stage, constructs no
provider client, holds no transaction across the provider call, and
claims nothing about semantic faithfulness (G2, A4b).  A retry cadence
after an unavailable outcome is not chosen here: :class:`RetryPolicy`
exists so a caller *can* suppress retries, and does nothing unless one is
supplied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Sequence

from ai.guarded_summary import (
    CITATION_CONVENTION,
    MAX_ATTEMPTS,
    MAX_SENTENCES,
    MIN_SENTENCES,
    PROMPT_VERSION,
    GenerationPolicy,
    GuardedSummaryError,
    SummaryGenerationInput,
    SummaryGenerationResult,
    citation_id_for,
    generate_guarded_summary,
    resolve_generation_policy,
    validate_candidate,
)

from .errors import Phase0ValidationError
from .repository import (
    SUMMARY_ARTIFACT_ACCEPTED,
    SUMMARY_GENERATION_ACCEPTED,
    SUMMARY_GENERATION_DISCARDED_DUPLICATE,
    SUMMARY_GENERATION_DISCARDED_STALE,
    SUMMARY_GENERATION_UNAVAILABLE,
    PersistedSummaryArtifact,
    PersistedSummaryGeneration,
    Phase0Reader,
    Phase0Repository,
    summary_artifact_candidate,
    summary_artifact_digest_of,
)
from .summaries import SummaryInputError, build_generation_input

#: The stage name a caller opens a run under to persist summaries.  Not
#: registered anywhere by this module.
STAGE = "summaries"

#: How an invocation ended.  ``provider_calls`` on the outcome says what it
#: cost; these say why.
SOURCE_REFUSED = "refused"  # population unhealthy or theme unknown
SOURCE_CACHE_HIT = "cache_hit"  # a stored artifact stood in
SOURCE_COOLDOWN = "cooldown"  # an explicit RetryPolicy suppressed the call
SOURCE_EXHAUSTED = "exhausted"  # an explicit RetryPolicy cap was reached
SOURCE_GENERATED = "generated"  # A2 accepted and the artifact was activated
SOURCE_UNAVAILABLE = "unavailable"  # A2 could not accept; recorded
SOURCE_DISCARDED_STALE = "discarded_stale"  # accepted, but no longer current
SOURCE_DISCARDED_DUPLICATE = "discarded_duplicate"  # another worker won

SOURCES: tuple[str, ...] = (
    SOURCE_REFUSED,
    SOURCE_CACHE_HIT,
    SOURCE_COOLDOWN,
    SOURCE_EXHAUSTED,
    SOURCE_GENERATED,
    SOURCE_UNAVAILABLE,
    SOURCE_DISCARDED_STALE,
    SOURCE_DISCARDED_DUPLICATE,
)

_OUTCOME_SOURCES = {
    SUMMARY_GENERATION_ACCEPTED: SOURCE_GENERATED,
    SUMMARY_GENERATION_UNAVAILABLE: SOURCE_UNAVAILABLE,
    SUMMARY_GENERATION_DISCARDED_STALE: SOURCE_DISCARDED_STALE,
    SUMMARY_GENERATION_DISCARDED_DUPLICATE: SOURCE_DISCARDED_DUPLICATE,
}


@dataclass(frozen=True)
class RetryPolicy:
    """Explicit, caller-owned suppression of repeat provider calls.

    Nothing here has a default that suppresses anything: a policy with
    both fields unset changes nothing, and :func:`ensure_summary` without
    a policy retries an unavailable key on every invocation.  Production
    cadence is a scheduling decision (A3b), not a constant in this module.

    ``cooldown``: after an ``unavailable`` generation for the exact key,
    make no provider call until this much time has passed since it
    completed.  Applies to every unavailable reason alike --
    ``provider_unconfigured`` included, because an authentication
    rejection is a network call too.

    ``max_generations``: make no provider call once this many generations
    have been recorded for the exact key, whatever their outcomes.
    """

    cooldown: Optional[timedelta] = None
    max_generations: Optional[int] = None

    def __post_init__(self) -> None:
        if self.cooldown is not None:
            if not isinstance(self.cooldown, timedelta):
                raise Phase0ValidationError("cooldown must be a timedelta")
            if self.cooldown < timedelta(0):
                raise Phase0ValidationError("cooldown cannot be negative")
        if self.max_generations is not None:
            if (
                isinstance(self.max_generations, bool)
                or not isinstance(self.max_generations, int)
                or self.max_generations < 1
            ):
                raise Phase0ValidationError("max_generations must be a positive int")

    def suppression(
        self, generations: Sequence[PersistedSummaryGeneration], *, now: datetime
    ) -> Optional[str]:
        """Why the next call is suppressed, or ``None``.

        ``generations`` are the recorded generations for one exact key,
        newest first.  The cap is checked before the cooldown so a key
        that is both exhausted and cooling reports the durable reason.
        """

        if self.max_generations is not None and len(generations) >= (
            self.max_generations
        ):
            return SOURCE_EXHAUSTED
        if self.cooldown is not None and generations:
            newest = generations[0]
            if newest.outcome == SUMMARY_GENERATION_UNAVAILABLE:
                completed = datetime.fromisoformat(newest.completed_at)
                if completed.tzinfo is None:
                    raise Phase0ValidationError(
                        "a generation's completed_at carries no UTC offset"
                    )
                if now - completed < self.cooldown:
                    return SOURCE_COOLDOWN
        return None


@dataclass(frozen=True)
class CurrentSummary:
    """A current artifact and the frozen input it is current *for*.

    ``generation_input.evidence`` carries each cited story's title, outlet,
    timestamp, raw item ids and URLs, so ``artifact -> sentence -> story_id
    -> story:<id> -> EvidenceStory -> raw_item_ids / urls`` resolves from
    this value alone.  That is structural traceability: it says which
    persisted story a sentence cited, not that the story supports it.
    """

    artifact: PersistedSummaryArtifact
    generation_input: SummaryGenerationInput
    policy: GenerationPolicy

    def evidence_for(self, story_id: int) -> Any:
        """The frozen :class:`~ai.guarded_summary.EvidenceStory` a citation names."""

        wanted = citation_id_for(story_id)
        for story in self.generation_input.evidence:
            if story.citation_id == wanted:
                return story
        raise GuardedSummaryError(
            f"{wanted} is not in the frozen input this artifact is current for"
        )


@dataclass(frozen=True)
class CacheVerdict:
    """What the reuse check found for one exact key."""

    artifact: Optional[PersistedSummaryArtifact]
    #: Why a stored accepted row was refused
    #: (:func:`validate_persisted_artifact`), when one existed.  Empty on a
    #: hit or a miss with no row.
    rejection_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.artifact is not None


@dataclass(frozen=True)
class SummaryLifecycleOutcome:
    """One :func:`ensure_summary` invocation, typed."""

    source: str
    ticker: str
    trading_day: str
    pipeline_version: str
    theme_id: int
    #: Exactly the provider calls this invocation made.
    provider_calls: int
    policy_fingerprint: str
    input_fingerprint: Optional[str] = None
    artifact: Optional[PersistedSummaryArtifact] = None
    generation_input: Optional[SummaryGenerationInput] = None
    result: Optional[SummaryGenerationResult] = None
    generation: Optional[PersistedSummaryGeneration] = None
    #: :mod:`phase0.summaries` refusal code, for ``refused``.
    refusal_code: Optional[str] = None
    #: Why a stored accepted row was not reused, when one existed.
    cache_rejection_codes: tuple[str, ...] = ()

    @property
    def current(self) -> Optional[CurrentSummary]:
        """The current artifact this invocation established or reused."""

        if self.artifact is None or self.generation_input is None:
            return None
        if self.source not in (
            SOURCE_CACHE_HIT,
            SOURCE_GENERATED,
            SOURCE_DISCARDED_DUPLICATE,
        ):
            return None
        return (
            None
            if self._policy is None
            else CurrentSummary(self.artifact, self.generation_input, self._policy)
        )

    # Kept off the public field list: a policy is an input, not an outcome.
    _policy: Optional[GenerationPolicy] = None


# ----------------------------------------------------------------------
# The one definition of a valid stored artifact
# ----------------------------------------------------------------------

#: Stable codes a stored artifact is refused with, before A2's validator
#: gets a say.  A2's own codes follow when it does.
REJECT_NOT_ACCEPTED = "artifact_not_accepted"
REJECT_IDENTITY = "artifact_identity_mismatch"
REJECT_STRUCTURE = "artifact_structure_invalid"
REJECT_DIGEST = "artifact_digest_mismatch"


@dataclass(frozen=True)
class ArtifactVerdict:
    """Whether one stored artifact may stand for one input under one policy."""

    valid: bool
    #: Why not: this module's structural codes and/or A2's validator codes,
    #: in the order they were found.  Empty when valid.
    codes: tuple[str, ...] = ()


def validate_persisted_artifact(
    artifact: PersistedSummaryArtifact,
    generation_input: SummaryGenerationInput,
    policy: GenerationPolicy,
) -> ArtifactVerdict:
    """Judge a stored artifact against a frozen input and a resolved policy.

    The single definition of "this stored row may be reused", shared by
    the read path (:func:`current_summary_artifact`, :func:`ensure_summary`)
    and the write path (an accepted holder of the key a replacement is
    about to take).  Pure: no database, no provider.  A verdict here says
    the artifact is valid *for this input*; whether the input is the live
    one is the caller's burden, and the public lifecycle functions carry
    it by reading a fresh snapshot themselves.

    Four checks, in order, and every failure is reported:

    1. **Identity.**  The row is ``accepted`` and every stored identity
       column agrees with the input and the policy -- theme id and key,
       ticker, day, pipeline version, both fingerprints, model, and A2's
       current citation convention and prompt version.  A row found by
       its lookup key is not trusted to be what the key says.
    2. **Structure, as stored.**  Ordinals are exactly ``1..N`` with
       ``N`` inside A2's sentence bounds; every sentence has at least one
       citation; positions are exactly ``0..M-1``; no story is cited
       twice by one sentence; every cited story is in the frozen input.
       Nothing is renumbered or normalized on the way to a verdict.
    3. **Content digest.**  :func:`~phase0.repository.summary_artifact_digest`
       recomputed from the stored rows equals the stored digest, so a
       deletion, reorder, or edit that still *looks* well-formed -- a
       three-sentence artifact quietly down to two -- is refused.
    4. **A2's validator**, against the reconstruction, under the policy's
       rules.
    """

    if not isinstance(artifact, PersistedSummaryArtifact):
        raise GuardedSummaryError("artifact must be a PersistedSummaryArtifact")
    if not isinstance(generation_input, SummaryGenerationInput):
        raise GuardedSummaryError("generation_input must be a SummaryGenerationInput")
    if not isinstance(policy, GenerationPolicy):
        raise GuardedSummaryError("policy must be a GenerationPolicy")

    codes: list[str] = []
    theme = generation_input.theme
    if artifact.status != SUMMARY_ARTIFACT_ACCEPTED:
        codes.append(REJECT_NOT_ACCEPTED)
    expected_identity = (
        theme.theme_id,
        theme.theme_key,
        generation_input.ticker,
        generation_input.trading_day,
        theme.pipeline_version,
        generation_input.input_fingerprint,
        policy.fingerprint,
        policy.model,
        CITATION_CONVENTION,
        PROMPT_VERSION,
    )
    stored_identity = (
        artifact.theme_id,
        artifact.theme_key,
        artifact.ticker,
        artifact.trading_day,
        artifact.pipeline_version,
        artifact.input_fingerprint,
        artifact.policy_fingerprint,
        artifact.model,
        artifact.citation_convention,
        artifact.prompt_version,
    )
    if stored_identity != expected_identity:
        codes.append(REJECT_IDENTITY)

    structural = False
    sentences = artifact.sentences
    count = len(sentences)
    if [sentence.ordinal for sentence in sentences] != list(range(1, count + 1)):
        structural = True
    if count < MIN_SENTENCES or count > MAX_SENTENCES:
        structural = True
    for sentence in sentences:
        positions = [citation.position for citation in sentence.citations]
        if not positions or positions != list(range(len(positions))):
            structural = True
        story_ids = [citation.story_id for citation in sentence.citations]
        if len(story_ids) != len(set(story_ids)):
            structural = True
        if any(
            citation_id_for(story_id) not in generation_input.evidence_ids
            for story_id in story_ids
            if story_id > 0
        ) or any(story_id <= 0 for story_id in story_ids):
            structural = True
        if not sentence.text.strip():
            structural = True
    if structural:
        codes.append(REJECT_STRUCTURE)

    if summary_artifact_digest_of(artifact) != artifact.content_digest:
        codes.append(REJECT_DIGEST)

    verdict = validate_candidate(
        summary_artifact_candidate(artifact, citation_id_for),
        generation_input,
        rules=policy.rules,
    )
    if not verdict.accepted:
        codes.extend(code for code in verdict.codes if code not in codes)
    return ArtifactVerdict(valid=not codes, codes=tuple(codes))


# ----------------------------------------------------------------------
# Reuse
# ----------------------------------------------------------------------


def _reusable_artifact(
    reader: Phase0Reader,
    generation_input: SummaryGenerationInput,
    policy: GenerationPolicy,
) -> CacheVerdict:
    """The stored artifact for this exact key, if it validates for this input.

    Private: ``generation_input`` must be the one just built from a fresh
    snapshot, which the public functions guarantee.  One read, no write,
    no provider.
    """

    artifact = reader.summary_artifact(
        generation_input.theme.theme_id,
        generation_input.input_fingerprint,
        policy.fingerprint,
    )
    if artifact is None:
        return CacheVerdict(None)
    verdict = validate_persisted_artifact(artifact, generation_input, policy)
    if not verdict.valid:
        return CacheVerdict(None, verdict.codes)
    return CacheVerdict(artifact)


def current_summary_artifact(
    reader: Phase0Reader,
    ticker: str,
    trading_day: str,
    pipeline_version: str,
    theme_id: int,
    policy: GenerationPolicy,
) -> Optional[CurrentSummary]:
    """The one artifact a reader may show for this theme right now, or ``None``.

    "Right now" is read here: one fresh
    :meth:`~phase0.repository.Phase0Reader.theme_population` snapshot,
    projected onto the frozen A2 input.  No population is accepted from
    the caller, so a stale snapshot cannot be used to call a historical
    artifact current after its theme moved or vanished.  The policy half
    of the key comes from ``policy`` -- a resolved
    :class:`~ai.guarded_summary.GenerationPolicy`, whose fingerprint is
    intrinsic and cannot be set by hand.

    Stale, superseded, invalidated and unavailable are all ``None`` here,
    as is a stored row that :func:`validate_persisted_artifact` refuses,
    and a population the A2 gate refuses.  Never calls a provider; never
    writes.
    """

    if not isinstance(policy, GenerationPolicy):
        raise GuardedSummaryError(
            "current_summary_artifact takes a resolved GenerationPolicy, not a "
            "fingerprint"
        )
    population = reader.theme_population(ticker, trading_day, pipeline_version)
    try:
        generation_input = build_generation_input(population, theme_id)
    except SummaryInputError:
        return None
    verdict = _reusable_artifact(reader, generation_input, policy)
    if verdict.artifact is None:
        return None
    return CurrentSummary(verdict.artifact, generation_input, policy)


# ----------------------------------------------------------------------
# The lifecycle
# ----------------------------------------------------------------------


def ensure_summary(
    repository: Phase0Repository,
    *,
    run: Any,
    ticker: str,
    trading_day: str,
    pipeline_version: str,
    theme_id: int,
    client: Any,
    max_attempts: int = MAX_ATTEMPTS,
    rules: Optional[Sequence[tuple[str, str, str]]] = None,
    retry: Optional[RetryPolicy] = None,
    clock: Callable[[], float] = time.perf_counter,
    terminal: bool = False,
) -> SummaryLifecycleOutcome:
    """Make one theme's summary current, calling the provider only if it must.

    1. resolve the policy once from ``client`` (A2's helper);
    2. read one fresh :meth:`~phase0.repository.Phase0Reader.theme_population`
       snapshot and build the frozen input from it (A2's gate; a refusal
       is a typed outcome, not an exception) -- the same live input then
       serves the lookup and the generation;
    3. look for a reusable artifact -- exact key, identity, structure,
       digest, A2's validator -- and return it with zero calls;
    4. if a :class:`RetryPolicy` was supplied, honour it against the
       recorded generations for the exact key;
    5. call A2 with the same policy object -- no database handle is held;
    6. hand the result to
       :meth:`~phase0.repository.Phase0Repository.persist_summary_generation`,
       which re-proves the result and currentness inside its own write
       transaction and activates the artifact only if the input is still
       the theme's.

    ``run`` is the :meth:`~phase0.repository.Phase0Repository.stage_run`
    context the write is recorded under; a cache hit, refusal, or
    suppression writes nothing and leaves the run as it was.
    """

    policy = resolve_generation_policy(client, max_attempts=max_attempts, rules=rules)
    if retry is not None and not isinstance(retry, RetryPolicy):
        raise Phase0ValidationError("retry must be a RetryPolicy")
    reader = repository.read
    population = reader.theme_population(ticker, trading_day, pipeline_version)

    def outcome(source: str, **fields: Any) -> SummaryLifecycleOutcome:
        return SummaryLifecycleOutcome(
            source=source,
            ticker=population.ticker,
            trading_day=population.trading_day,
            pipeline_version=population.pipeline_version,
            theme_id=int(theme_id),
            policy_fingerprint=policy.fingerprint,
            _policy=policy,
            **fields,
        )

    try:
        generation_input = build_generation_input(population, theme_id)
    except SummaryInputError as exc:
        return outcome(SOURCE_REFUSED, provider_calls=0, refusal_code=exc.code)

    verdict = _reusable_artifact(reader, generation_input, policy)
    if verdict.hit:
        return outcome(
            SOURCE_CACHE_HIT,
            provider_calls=0,
            input_fingerprint=generation_input.input_fingerprint,
            artifact=verdict.artifact,
            generation_input=generation_input,
        )

    if retry is not None:
        recorded = [
            generation
            for generation in reader.summary_generations(
                population.ticker,
                population.trading_day,
                population.pipeline_version,
                theme_id=generation_input.theme.theme_id,
            )
            if generation.input_fingerprint == generation_input.input_fingerprint
            and generation.policy_fingerprint == policy.fingerprint
        ]
        suppressed = retry.suppression(recorded, now=repository.now())
        if suppressed is not None:
            return outcome(
                suppressed,
                provider_calls=0,
                input_fingerprint=generation_input.input_fingerprint,
                generation_input=generation_input,
                cache_rejection_codes=verdict.rejection_codes,
            )

    # No repository read is open here and no transaction exists: the
    # readers above each closed their own connection before returning.
    result = generate_guarded_summary(
        generation_input,
        client=client,
        max_attempts=policy.max_attempts,
        clock=clock,
        policy=policy,
    )

    generation = repository.persist_summary_generation(
        run=run,
        result=result,
        generation_input=generation_input,
        policy=policy,
        terminal=terminal,
    )
    artifact = None
    if generation.artifact_id is not None:
        artifact = reader.summary_artifact(
            generation_input.theme.theme_id,
            generation_input.input_fingerprint,
            policy.fingerprint,
        )
    return outcome(
        _OUTCOME_SOURCES[generation.outcome],
        provider_calls=result.provider_calls,
        input_fingerprint=generation_input.input_fingerprint,
        artifact=artifact,
        generation_input=generation_input,
        result=result,
        generation=generation,
        cache_rejection_codes=verdict.rejection_codes,
    )


__all__ = [
    "ArtifactVerdict",
    "CacheVerdict",
    "CurrentSummary",
    "REJECT_DIGEST",
    "REJECT_IDENTITY",
    "REJECT_NOT_ACCEPTED",
    "REJECT_STRUCTURE",
    "RetryPolicy",
    "SOURCES",
    "SOURCE_CACHE_HIT",
    "SOURCE_COOLDOWN",
    "SOURCE_DISCARDED_DUPLICATE",
    "SOURCE_DISCARDED_STALE",
    "SOURCE_EXHAUSTED",
    "SOURCE_GENERATED",
    "SOURCE_REFUSED",
    "SOURCE_UNAVAILABLE",
    "STAGE",
    "SummaryLifecycleOutcome",
    "current_summary_artifact",
    "ensure_summary",
    "validate_persisted_artifact",
]
