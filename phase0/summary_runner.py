"""A3b: scheduled summary generation over persisted theme populations.

A2 (:mod:`ai.guarded_summary`) generates one guarded summary; A3
(:mod:`phase0.summary_lifecycle`) makes one theme's summary current,
reusing a stored artifact when it may and recording what the provider
did when it may not.  Neither decides *which* themes to visit, *when*,
or *how much* provider work one invocation may spend.  This module does,
and nothing else: it holds no cache of its own, keeps no accounting of
its own, and writes nothing except through
:func:`~phase0.summary_lifecycle.ensure_summary`.

**Off unless asked for.**  ``PHASE0_SUMMARIES_ENABLED`` gates the whole
component.  Absent or false, ``pipeline.py`` builds nothing, constructs
no client, and reads no summary table -- the scheduled pipeline is what
it was before this module existed.

**Configuration is checked before any work.**  Enabled without a usable
provider configuration, the component reports ``summaries_unconfigured``
and stops: zero provider calls, zero summary rows, zero runs.  A2's
``ProviderConfigurationError`` still exists for a key that goes missing
mid-invocation; it is not the ordinary missing-key path.

**Which partitions.**  Every ticker with persisted stories or a theme set
on a day a ``themes`` run completed within the caller's horizon, days
ascending, tickers ascending.  Story partitions are included so an
M2-only day -- which by decision H holds stories and no theme set -- is
counted as refused rather than passing unseen.  The sweep is independent
of the intelligence component's own selection and retry episodes: it
reads only the ledger's ``themes`` completions and the persisted rows.

**Zero calls, exactly when.**  A partition whose population the A2/A3
health gate refuses is counted and left alone.  A theme with a current
artifact (:func:`~phase0.summary_lifecycle.current_summary_artifact`) is a
cache hit.  A theme whose exact key the production
:class:`~phase0.summary_lifecycle.RetryPolicy` suppresses is counted as
``cooldown`` or ``exhausted``.  None of those opens a run.

**One run per partition, and why.**  A typed ``unavailable`` generation
is an ordinary logged mutation that counts as partial work, so one
``summaries`` run can record theme A unavailable and then theme B and C
accepted.  A logged mutation that *raises* settles its run ``failed``
and every later mutation on that run is refused
(``Phase0RunContextError``); what earlier mutations committed stays
durable.  So an unexpected exception ends that partition's run -- the
smallest unit the run contract lets continue safely -- and the sweep
moves on to the next partition.  Remaining themes of that partition are
counted ``themes_not_attempted`` and are ordinary candidates next time.

**The provider-call budget is a strict cap.**  Before a theme is started,
the policy's ``max_attempts`` (the most provider calls one
:func:`~phase0.summary_lifecycle.ensure_summary` can make) is reserved
from what is left.  If that much is not left, the theme -- and every
theme after it -- is deferred: no call, no row, a candidate on the next
invocation.  After the call the reservation is settled at the real
``provider_calls`` the outcome reports; a call that raised keeps its
whole reservation, because how many attempts it made is not known.
Real calls therefore never exceed the budget.

**An execution identity is used once.**  A partition's run id is
``<base>:<ticker>:<day>``.  If a ``summaries`` run with that id is already
in the ledger -- a caller reused an invocation id -- the partition does no
provider work under it and reports ``summary_run_identity_reused``:
reopening the run would rewrite its recorded outcome, and A3's per-run
idempotency key would absorb the new attempts without recording them.

**The configured key never becomes durable text.**  When the scheduled
entry point runs with a configured key, the provider client is wrapped so
a provider error echoing the bare key is re-raised as the same error class
with the key replaced -- before A2 records it as an attempt and A3
persists it -- and an unexpected exception carrying it is re-raised
scrubbed before ``stage_run`` records it in ``run_log.errors``.
Classification and retry behaviour are unchanged.

**Replay never reaches this module.**  ``pipeline.run_replay`` does not
build downstream stages at all.

Nothing here claims semantic faithfulness: an accepted summary is
structurally grounded and copy-policy valid, as A2 defines it.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from typing import Any, Callable, Mapping, Optional

from ai.guarded_summary import (
    MAX_ATTEMPTS,
    GenerationPolicy,
    resolve_generation_policy,
)
from ai.summarization import (
    GeminiClient,
    ProviderConfigurationError,
    SummarizationError,
)

from .redaction import redact_secrets
from .repository import Phase0Repository
from .summaries import SummaryInputError, assess_population, build_generation_input
from .summary_lifecycle import (
    SOURCE_CACHE_HIT,
    SOURCE_COOLDOWN,
    SOURCE_DISCARDED_DUPLICATE,
    SOURCE_DISCARDED_STALE,
    SOURCE_EXHAUSTED,
    SOURCE_GENERATED,
    SOURCE_REFUSED,
    SOURCE_UNAVAILABLE,
    STAGE,
    RetryPolicy,
    current_summary_artifact,
    ensure_summary,
)
from .themes import STAGE as THEMES_STAGE
from .themes import partition_run_id

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

#: The feature flag.  Absent means off.
ENABLED_ENV = "PHASE0_SUMMARIES_ENABLED"
#: The invocation-level cap on real provider calls.
BUDGET_ENV = "PHASE0_SUMMARIES_MAX_PROVIDER_CALLS"
#: The one provider credential A2's production client needs.
API_KEY_ENV = "GEMINI_API_KEY"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Ten themes' worth of worst-case calls per invocation.  Five tickers
#: with a few themes each fit comfortably on an ordinary run; a day whose
#: theme ids were all re-minted by a story change is spread over the next
#: few invocations instead of one burst.
DEFAULT_PROVIDER_CALL_BUDGET = 20

#: One regeneration with validation feedback, as A2 allows.  Part of the
#: policy fingerprint, so changing it is a new key for every theme.
PRODUCTION_MAX_ATTEMPTS = MAX_ATTEMPTS

#: Production cadence for an exact key that could not be accepted.  The
#: schedule fires every 30 minutes in market hours, so a one-hour
#: cooldown skips at least one tick after an ``unavailable`` result; three
#: generations per exact key bounds what a key the model cannot satisfy
#: costs.  A changed input or policy is a new key and starts fresh.
PRODUCTION_RETRY_POLICY = RetryPolicy(cooldown=timedelta(hours=1), max_generations=3)


class SummaryWorkError(RuntimeError):
    """An unexpected failure inside one partition's summary run, scrubbed."""


class SummaryConfigError(ValueError):
    """A summary setting is present but not one this module understands.

    The message names the variable and never echoes its value.
    """


def _environ(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def summaries_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Is the scheduled summary component switched on?

    Absent, blank, ``0``/``false``/``no``/``off`` mean off; ``1``/``true``/
    ``yes``/``on`` mean on, case-insensitively.  Anything else is refused
    rather than guessed at.
    """

    raw = _environ(environ).get(ENABLED_ENV)
    value = "" if raw is None else raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise SummaryConfigError(f"{ENABLED_ENV} must be one of true/false/1/0/yes/no")


def provider_call_budget(environ: Optional[Mapping[str, str]] = None) -> int:
    """The invocation's provider-call cap: a non-negative integer."""

    raw = _environ(environ).get(BUDGET_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_PROVIDER_CALL_BUDGET
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise SummaryConfigError(f"{BUDGET_ENV} must be an integer") from exc
    if value < 0:
        raise SummaryConfigError(f"{BUDGET_ENV} must not be negative")
    return value


def provider_configuration_problem(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Why the production provider cannot be called, or ``None``.

    A stable reason code, never the credential.  Checked before any run
    opens, so a missing key costs no call and writes no row.
    """

    key = _environ(environ).get(API_KEY_ENV)
    if key is None or not key.strip():
        return "provider_api_key_missing"
    if importlib.util.find_spec("google.genai") is None:
        return "provider_sdk_missing"
    return None


# ----------------------------------------------------------------------
# The one production policy boundary (shared with the future read API)
# ----------------------------------------------------------------------


def production_summary_client() -> GeminiClient:
    """The production provider client, configured from the environment.

    Construction makes no request: ``GeminiClient`` connects lazily on the
    first ``generate``.  Raises ``ProviderConfigurationError`` for a
    malformed numeric setting.
    """

    return GeminiClient()


def production_generation_policy(client: Any | None = None) -> GenerationPolicy:
    """The policy production generates under, resolved without generating.

    Model and output cap come from the production client's configuration,
    the attempt bound from :data:`PRODUCTION_MAX_ATTEMPTS`, and the copy
    rules from their source of truth.  A reader that must decide
    currentness -- the scheduler's pre-check here, and a read-only API
    later -- resolves the same policy through this function, so the two
    cannot disagree about which artifact is current.  No provider is
    called and the policy carries no credential.
    """

    return resolve_generation_policy(
        production_summary_client() if client is None else client,
        max_attempts=PRODUCTION_MAX_ATTEMPTS,
    )


# ----------------------------------------------------------------------
# Accounting
# ----------------------------------------------------------------------

#: Counters every invocation reports, zero or not, so a reader can tell
#: "nothing happened" from "not reported".
COUNTERS: tuple[str, ...] = (
    "days_considered",
    "partitions_considered",
    "partitions_refused",
    "partitions_current",
    "partitions_with_work",
    "partitions_settled",
    "partitions_failed",
    "themes_considered",
    "themes_refused",
    "cache_hits",
    "generated",
    "unavailable",
    "discarded_stale",
    "discarded_duplicate",
    "cooldown",
    "exhausted",
    "deferred_budget",
    "themes_not_attempted",
    "partitions_identity_reused",
    "provider_calls",
    "provider_call_budget",
)

_OUTCOME_COUNTER = {
    SOURCE_CACHE_HIT: "cache_hits",
    SOURCE_GENERATED: "generated",
    SOURCE_UNAVAILABLE: "unavailable",
    SOURCE_DISCARDED_STALE: "discarded_stale",
    SOURCE_DISCARDED_DUPLICATE: "discarded_duplicate",
    SOURCE_COOLDOWN: "cooldown",
    SOURCE_EXHAUSTED: "exhausted",
    SOURCE_REFUSED: "themes_refused",
}


def empty_counts() -> dict[str, int]:
    return {name: 0 for name in COUNTERS}


@dataclass
class ProviderBudget:
    """A strict cap on real provider calls, reserved before each theme."""

    limit: int
    spent: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.spent

    def can_reserve(self, calls: int) -> bool:
        return self.remaining >= calls


@dataclass
class _Sweep:
    counts: dict[str, int] = field(default_factory=empty_counts)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, value: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + value


# ----------------------------------------------------------------------
# The configured credential never becomes durable text
# ----------------------------------------------------------------------

REDACTED = "[REDACTED]"


def scrub_literal_secret(value: Any, secret: Optional[str]) -> Any:
    """Replace every literal occurrence of ``secret`` inside ``value``.

    ``redact_secrets`` removes a credential that a name introduces
    (``api_key=...``); this removes the configured value itself wherever
    it appears, bare.  A blank or absent ``secret`` changes nothing.
    """

    if not secret or not secret.strip():
        return value
    secret = secret.strip()
    if isinstance(value, str):
        return value.replace(secret, REDACTED)
    if isinstance(value, Mapping):
        return {key: scrub_literal_secret(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scrub_literal_secret(item, secret) for item in value)
    return value


class _ScrubbingClient:
    """The provider client, with the configured key removed from its errors.

    A2 records a provider error's text as the attempt's ``error`` and A3
    persists it.  ``SummarizationError`` already redacts named
    credentials on construction; a transport error that echoes the bare
    key does not name it.  This boundary re-raises such an error as the
    *same class* with the key replaced, so timeouts stay timeouts,
    authentication failures stay configuration failures, and A2's retry
    rules are unchanged.  Everything else -- ``model``,
    ``max_output_tokens``, ``last_usage`` -- is the wrapped client's, so the
    resolved policy and its fingerprint are identical.  The key is held
    only here and never appears in ``repr``.
    """

    __slots__ = ("_client", "_secret")

    def __init__(self, client: Any, secret: str) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_secret", secret)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"<credential-scrubbing {type(self._client).__name__}>"

    def generate(self, system_prompt: str, user_prompt: str, response_schema: Any):
        try:
            return self._client.generate(system_prompt, user_prompt, response_schema)
        except SummarizationError as exc:
            text = str(exc)
            scrubbed = scrub_literal_secret(text, self._secret)
            if scrubbed == text:
                raise
            # ``from None``: the original, unscrubbed exception must not
            # travel on as this one's cause.
            raise type(exc)(scrubbed) from None


# ----------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------


class SummaryRunner:
    """Visits recent theme partitions and makes their summaries current."""

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        pipeline_version: str,
        client: Any,
        budget: int,
        horizon: timedelta,
        retry: RetryPolicy = PRODUCTION_RETRY_POLICY,
        secret: Optional[str] = None,
    ) -> None:
        self.repository = repository
        self.pipeline_version = str(pipeline_version).strip()
        if not self.pipeline_version:
            raise ValueError("pipeline_version is required")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise SummaryConfigError("the provider-call budget is a non-negative int")
        if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
            raise SummaryConfigError("the horizon is a positive timedelta")
        if not isinstance(retry, RetryPolicy):
            raise SummaryConfigError("retry must be a RetryPolicy")
        secret = (secret or "").strip() or None
        self._secret = secret
        self.client = client if secret is None else _ScrubbingClient(client, secret)
        self.budget = ProviderBudget(budget)
        self.horizon = horizon
        self.retry = retry
        # Resolved once, and handed to ensure_summary as its rules, so the
        # pre-check and the generation agree on one fingerprint.
        self.policy = production_generation_policy(client)

    # -- Selection ------------------------------------------------------

    def partitions(self) -> list[tuple[str, str]]:
        """``(trading_day, ticker)`` pairs to visit, days then tickers ascending."""

        reader = self.repository.read
        since = self.repository.now().astimezone(timezone.utc) - self.horizon
        selected: list[tuple[str, str]] = []
        for day in sorted(
            reader.recent_run_days((THEMES_STAGE,), completed_since=since)
        ):
            tickers = set(
                reader.theme_partitions(day, pipeline_version=self.pipeline_version)
            ) | set(
                reader.story_partitions(day, pipeline_version=self.pipeline_version)
            )
            selected.extend((day, ticker) for ticker in sorted(tickers))
        return selected

    # -- The sweep --------------------------------------------------------

    def run(self, *, base_run_id: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
        sweep = _Sweep()
        sweep.counts["provider_call_budget"] = self.budget.limit
        pairs = self.partitions()
        sweep.counts["days_considered"] = len({day for day, _ in pairs})
        for day, ticker in pairs:
            sweep.add("partitions_considered")
            try:
                self._run_partition(sweep, ticker, day, base_run_id=base_run_id)
            except Exception as exc:  # noqa: BLE001 - isolation is the contract
                sweep.add("partitions_failed")
                sweep.errors.append(
                    {
                        "type": "summary_partition_failed",
                        "ticker": ticker,
                        "trading_day": day,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        sweep.counts["provider_calls"] = self.budget.spent
        if sweep.counts["deferred_budget"]:
            sweep.errors.append(
                {
                    "type": "summaries_budget_exhausted",
                    "deferred": sweep.counts["deferred_budget"],
                    "budget": self.budget.limit,
                }
            )
        suppressed = sweep.counts["cooldown"] + sweep.counts["exhausted"]
        if suppressed:
            sweep.errors.append(
                {
                    "type": "summaries_retry_suppressed",
                    "cooldown": sweep.counts["cooldown"],
                    "exhausted": sweep.counts["exhausted"],
                }
            )
        errors = scrub_literal_secret(list(redact_secrets(sweep.errors)), self._secret)
        return sweep.counts, errors

    def _run_partition(
        self, sweep: _Sweep, ticker: str, day: str, *, base_run_id: str
    ) -> None:
        reader = self.repository.read
        population = reader.theme_population(ticker, day, self.pipeline_version)
        refusal = assess_population(population)
        if refusal is not None:
            sweep.add("partitions_refused")
            sweep.add(f"refused_{refusal[0]}")
            sweep.add("partitions_settled")
            return

        pending: list[int] = []
        ordered = sorted(
            population.themes, key=lambda theme: (theme.salience_rank, theme.theme_id)
        )
        for theme in ordered:
            sweep.add("themes_considered")
            verdict = self._precheck(population, theme.theme_id, ticker, day)
            if verdict is None:
                pending.append(theme.theme_id)
            else:
                sweep.add(verdict)
        if not pending:
            sweep.add("partitions_current")
            sweep.add("partitions_settled")
            return

        max_calls = self.policy.max_attempts
        if not self.budget.can_reserve(max_calls):
            # Nothing can start, so nothing is opened.
            sweep.add("deferred_budget", len(pending))
            sweep.add("partitions_settled")
            return

        run_id = partition_run_id(base_run_id, ticker, day)
        if self.repository.read.run_log_rows(run_id=run_id, stage=STAGE):
            # This execution identity already has a durable run.  Opening it
            # again would rewrite that run's outcome and let A3's per-run
            # idempotency key swallow new provider attempts, so no work is
            # done under it: a fresh invocation id is the way to retry.
            sweep.add("partitions_identity_reused")
            sweep.errors.append(
                {
                    "type": "summary_run_identity_reused",
                    "ticker": ticker,
                    "trading_day": day,
                    "pending": len(pending),
                }
            )
            return

        sweep.add("partitions_with_work")
        remaining = list(pending)
        with self.repository.stage_run(
            run_id=run_id,
            stage=STAGE,
            ticker=ticker,
            trading_day=day,
            pipeline_version=self.pipeline_version,
        ) as run:
            deferred = 0
            while remaining:
                theme_id = remaining[0]
                if not self.budget.can_reserve(max_calls):
                    deferred = len(remaining)
                    remaining = []
                    break
                # Reserved in full; a raise below keeps the reservation.
                self.budget.spent += max_calls
                try:
                    outcome = ensure_summary(
                        self.repository,
                        run=run,
                        ticker=ticker,
                        trading_day=day,
                        pipeline_version=self.pipeline_version,
                        theme_id=theme_id,
                        client=self.client,
                        max_attempts=self.policy.max_attempts,
                        rules=self.policy.rules,
                        retry=self.retry,
                    )
                except BaseException as exc:
                    remaining.pop(0)
                    sweep.add("themes_not_attempted", len(remaining))
                    text = str(exc)
                    scrubbed = scrub_literal_secret(text, self._secret)
                    if isinstance(exc, Exception) and scrubbed != text:
                        # ``stage_run`` records the escaping exception's text
                        # in ``run_log.errors``; it must not carry the key.
                        raise SummaryWorkError(
                            f"{type(exc).__name__}: {scrubbed}"
                        ) from None
                    raise
                remaining.pop(0)
                # Settle the reservation at what was really spent.
                self.budget.spent -= max_calls - outcome.provider_calls
                sweep.add(_OUTCOME_COUNTER[outcome.source])
                if outcome.source == SOURCE_REFUSED and outcome.refusal_code:
                    sweep.add(f"refused_{outcome.refusal_code}")
                if outcome.source == SOURCE_UNAVAILABLE:
                    reason = (
                        outcome.generation.reason
                        if outcome.generation is not None
                        else None
                    )
                    sweep.errors.append(
                        {
                            "type": "summary_unavailable",
                            "ticker": ticker,
                            "trading_day": day,
                            "theme_id": theme_id,
                            "reason": reason,
                        }
                    )
            if deferred:
                sweep.add("deferred_budget", deferred)
                run.record_degradation(
                    "summary_budget_deferred",
                    detail=f"{deferred} themes deferred by the provider-call budget",
                )
        sweep.add("partitions_settled")

    def _precheck(
        self, population: Any, theme_id: int, ticker: str, day: str
    ) -> Optional[str]:
        """A zero-call verdict for one theme, or ``None`` if it needs work.

        Read-only: currentness through A3's own reader, the exact key
        through A2's own projection, and suppression through the
        production RetryPolicy's own rule.  Nothing is written and no
        provider is called.
        """

        reader = self.repository.read
        current = current_summary_artifact(
            reader, ticker, day, self.pipeline_version, theme_id, self.policy
        )
        if current is not None:
            return "cache_hits"
        try:
            generation_input = build_generation_input(population, theme_id)
        except SummaryInputError:
            return "themes_refused"
        recorded = [
            generation
            for generation in reader.summary_generations(
                ticker, day, self.pipeline_version, theme_id=theme_id
            )
            if generation.input_fingerprint == generation_input.input_fingerprint
            and generation.policy_fingerprint == self.policy.fingerprint
        ]
        suppressed = self.retry.suppression(recorded, now=self.repository.now())
        if suppressed == SOURCE_COOLDOWN:
            return "cooldown"
        if suppressed == SOURCE_EXHAUSTED:
            return "exhausted"
        return None


# ----------------------------------------------------------------------
# The scheduled entry point
# ----------------------------------------------------------------------


def run_scheduled_summaries(
    repository: Phase0Repository,
    *,
    pipeline_version: str,
    base_run_id: str,
    horizon: timedelta,
    environ: Optional[Mapping[str, str]] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    retry: RetryPolicy = PRODUCTION_RETRY_POLICY,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """One scheduled summary sweep: check configuration, then run.

    Configuration problems return a component-level error with every
    counter at zero -- no client is constructed for a missing key, no run
    is opened, and nothing is written.
    """

    counts = empty_counts()
    key = _environ(environ).get(API_KEY_ENV, "").strip()
    try:
        budget = provider_call_budget(environ)
    except SummaryConfigError as exc:
        return counts, [{"type": "summaries_misconfigured", "error": str(exc)}]
    counts["provider_call_budget"] = budget
    problem = provider_configuration_problem(environ)
    if problem is not None:
        return counts, [{"type": "summaries_unconfigured", "reason": problem}]
    try:
        client = (client_factory or production_summary_client)()
    except ProviderConfigurationError as exc:
        return counts, scrub_literal_secret(
            redact_secrets(
                [
                    {
                        "type": "summaries_unconfigured",
                        "reason": "provider_config_invalid",
                        "error": str(exc),
                    }
                ]
            ),
            key,
        )
    runner = SummaryRunner(
        repository,
        pipeline_version=pipeline_version,
        client=client,
        budget=budget,
        horizon=horizon,
        retry=retry,
        secret=key,
    )
    return runner.run(base_run_id=base_run_id)


__all__ = [
    "API_KEY_ENV",
    "BUDGET_ENV",
    "COUNTERS",
    "DEFAULT_PROVIDER_CALL_BUDGET",
    "ENABLED_ENV",
    "PRODUCTION_MAX_ATTEMPTS",
    "PRODUCTION_RETRY_POLICY",
    "ProviderBudget",
    "STAGE",
    "SummaryConfigError",
    "SummaryRunner",
    "SummaryWorkError",
    "empty_counts",
    "production_generation_policy",
    "production_summary_client",
    "provider_call_budget",
    "provider_configuration_problem",
    "run_scheduled_summaries",
    "scrub_literal_secret",
    "summaries_enabled",
]
