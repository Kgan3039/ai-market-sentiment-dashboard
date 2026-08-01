"""The trust contract every evaluation artefact has to carry.

A number produced from a dataset is only interpretable with the dataset's
provenance attached to it.  Putting that provenance in the README is not
enough: somebody reads ``m2_baseline.json``, or the CLI output pasted into
a ticket, and the README is not there.  So every loaded object, every text
report, every JSON payload, and every committed result file carries the
same seven fields, and the loader refuses a manifest that omits them or
contradicts itself.

The fields answer the questions a reader of a metric actually has:

``dataset_kind``            where the records came from
``real_ingested_evidence``  whether anything here was really fetched
``labeling_status``         how the labels were produced
``reviewer_count``          how many people looked at them
``adjudicated``             whether disagreements were resolved
``gate_eligible``           whether these numbers may clear a go/no-go gate
``metrics_purpose``         what the numbers are for

Validation is **relational**, not per-field.  Each field on its own can be
well-formed while the combination says something untrue: a synthetic set
declaring ``final_acceptance``, one reviewer declaring adjudication, a
gate-eligible set with no real evidence behind it.  Those are the
combinations that would let a development number be read as a go/no-go
result, so each is named and refused here rather than left to a reviewer to
notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DatasetKind(str, Enum):
    """Where the records came from."""

    #: Authored for development. Nothing was fetched.
    SYNTHETIC_DEVELOPMENT = "synthetic_development"
    #: Sampled from real ingested items.
    SAMPLED_PRODUCTION = "sampled_production"


class LabelingStatus(str, Enum):
    """How the labels were produced, including who resolved disagreements."""

    SINGLE_AUTHOR_UNADJUDICATED = "single_author_unadjudicated"
    MULTI_REVIEWER_UNADJUDICATED = "multi_reviewer_unadjudicated"
    MULTI_REVIEWER_ADJUDICATED = "multi_reviewer_adjudicated"


class MetricsPurpose(str, Enum):
    """What the numbers derived from this dataset may be used for."""

    #: Regression signal only. Says nothing about real coverage.
    DEVELOPMENT_REGRESSION_ONLY = "development_regression_only"
    #: May be quoted at a go/no-go gate (G1/G4).
    GATE_ACCEPTANCE = "gate_acceptance"
    #: May be quoted as final acceptance (AC-3/AC-4).
    FINAL_ACCEPTANCE = "final_acceptance"


#: Purposes that assert the numbers can settle something. Only a dataset
#: that clears every gate invariant may declare one.
GATE_PURPOSES = frozenset(
    {MetricsPurpose.GATE_ACCEPTANCE, MetricsPurpose.FINAL_ACCEPTANCE}
)

#: Labeling states in which disagreements have been resolved.
ADJUDICATED_STATUSES = frozenset({LabelingStatus.MULTI_REVIEWER_ADJUDICATED})

#: Labeling states that require more than one reviewer.
MULTI_REVIEWER_STATUSES = frozenset(
    {
        LabelingStatus.MULTI_REVIEWER_UNADJUDICATED,
        LabelingStatus.MULTI_REVIEWER_ADJUDICATED,
    }
)


@dataclass(frozen=True)
class DatasetKindRules:
    """The invariants one dataset kind must satisfy.

    A matrix rather than a chain of ``if`` statements, so a new
    :class:`DatasetKind` cannot silently inherit permissive behaviour: the
    validator asserts that every enum member has an entry and refuses a
    kind that does not.
    """

    kind: DatasetKind
    #: What ``real_ingested_evidence`` must be for this kind.  Both values
    #: are wrong for exactly one kind, which is the point: a production
    #: sample with no real evidence behind it is not a production sample.
    real_ingested_evidence: bool
    #: Purposes this kind may declare at all.
    allowed_metrics_purposes: frozenset[MetricsPurpose]
    #: Whether this kind may ever be gate eligible.  The reviewer and
    #: adjudication requirements are checked separately and apply on top.
    may_be_gate_eligible: bool
    #: Accepted ``provenance.kind`` values.
    provenance_kinds: frozenset[str]
    #: Accepted ``provenance.collection_method`` values.
    collection_methods: frozenset[str]
    #: What ``provenance.urls_are_synthetic`` must be.
    urls_are_synthetic: bool
    #: Provenance fields this kind must supply beyond the common ones.
    extra_provenance_fields: tuple[str, ...]
    #: One line naming what this kind is, used in error messages.
    description: str


#: The invariant matrix.  Every :class:`DatasetKind` must appear.
DATASET_KIND_RULES: dict[DatasetKind, DatasetKindRules] = {
    DatasetKind.SYNTHETIC_DEVELOPMENT: DatasetKindRules(
        kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
        real_ingested_evidence=False,
        allowed_metrics_purposes=frozenset(
            {MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY}
        ),
        may_be_gate_eligible=False,
        provenance_kinds=frozenset({"synthetic"}),
        collection_methods=frozenset({"authored"}),
        urls_are_synthetic=True,
        extra_provenance_fields=("why_synthetic", "blocked_by"),
        description="authored for development; nothing in it was fetched",
    ),
    DatasetKind.SAMPLED_PRODUCTION: DatasetKindRules(
        kind=DatasetKind.SAMPLED_PRODUCTION,
        real_ingested_evidence=True,
        allowed_metrics_purposes=frozenset(MetricsPurpose),
        may_be_gate_eligible=True,
        provenance_kinds=frozenset({"sampled"}),
        collection_methods=frozenset({"sampled", "ingested"}),
        urls_are_synthetic=False,
        # A production set has to say what it was sampled *from*, or its
        # numbers cannot be traced back to anything.
        extra_provenance_fields=("ingestion_source", "sample_selection"),
        description="sampled from real ingested items",
    ),
}

#: Fields a manifest's ``trust_contract`` may carry beyond the required
#: seven.  ``warning`` is deliberately absent: the banner is derived from
#: the validated metadata, and a supplied one could contradict it.
OPTIONAL_TRUST_FIELDS = ("why",)


def rules_for(kind: DatasetKind) -> DatasetKindRules:
    """Return the invariants for a dataset kind, or refuse to guess."""

    try:
        return DATASET_KIND_RULES[kind]
    except KeyError as exc:  # pragma: no cover - guarded by a test
        raise TrustContractError(
            f"dataset_kind={kind.value!r} has no entry in DATASET_KIND_RULES; "
            "a new kind must declare its invariants rather than inherit "
            "permissive behaviour"
        ) from exc


#: Every field a manifest must declare.  There is no default for any of
#: them: a dataset that does not state its own provenance is not loadable.
REQUIRED_TRUST_FIELDS = (
    "dataset_kind",
    "real_ingested_evidence",
    "labeling_status",
    "reviewer_count",
    "adjudicated",
    "gate_eligible",
    "metrics_purpose",
)

#: Required of every dataset kind.  Kind-specific extras live in
#: :data:`DATASET_KIND_RULES`.
REQUIRED_PROVENANCE_FIELDS = (
    "kind",
    "collection_method",
    "statement",
    "urls_are_synthetic",
    "uses_real_outlet_names",
)

REQUIRED_LABELING_FIELDS = (
    "status",
    "reviewer_count",
    "reviewers",
    "adjudicated",
    "gate_eligible",
    "protocol",
)


class TrustContractError(ValueError):
    """The dataset's provenance is missing, malformed, or self-contradictory."""


@dataclass(frozen=True)
class TrustSummary:
    """The one-line judgement a reader needs, derived from the metadata.

    Never supplied by a caller.  A banner that could be written into a
    manifest could contradict the fields beside it, which is the exact
    failure the banner exists to prevent.
    """

    #: ``WARNING`` when the numbers cannot settle anything, ``NOTICE`` when
    #: they can be read at face value.
    level: str
    headline: str
    detail: str

    @property
    def text(self) -> str:
        """The banner line(s), level included."""

        body = f"{self.level}: {self.headline}"
        return f"{body}\n{self.detail}" if self.detail else body

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "headline": self.headline,
            "detail": self.detail,
            "text": self.text,
        }


def _labeling_phrase(
    status: LabelingStatus, reviewer_count: int, adjudicated: bool
) -> str:
    """Describe how the labels were produced, in a reader's words."""

    if status is LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED:
        return "single-author, unadjudicated"
    if adjudicated:
        return f"{reviewer_count}-reviewer, independently adjudicated"
    return f"{reviewer_count}-reviewer, unadjudicated"


def derive_trust_summary(contract: "TrustContract") -> TrustSummary:
    """Derive the banner from the validated contract, and only from it.

    Four states, one per combination that changes what a reader may do
    with the numbers.  A production dataset never receives synthetic
    wording and a synthetic dataset never receives a production or
    adjudicated notice, because the branch is on ``dataset_kind`` first.
    """

    labeling = _labeling_phrase(
        contract.labeling_status, contract.reviewer_count, contract.adjudicated
    )
    if contract.dataset_kind is DatasetKind.SYNTHETIC_DEVELOPMENT:
        return TrustSummary(
            level="WARNING",
            headline=f"Synthetic, {labeling} development dataset.",
            detail=("Metrics are not valid for K3/G4 or final AC-3 acceptance."),
        )
    if not contract.adjudicated:
        return TrustSummary(
            level="WARNING",
            headline=(
                "Production-sampled evidence has not completed independent "
                "adjudication."
            ),
            detail="Metrics are development-only and not gate eligible.",
        )
    if not contract.gate_eligible:
        return TrustSummary(
            level="NOTICE",
            headline=(
                "Production-sampled, independently adjudicated evaluation " "dataset."
            ),
            detail="Metrics are not configured as a release gate.",
        )
    return TrustSummary(
        level="NOTICE",
        headline=(
            "Production-sampled, independently adjudicated gate-eligible " "dataset."
        ),
        detail="",
    )


@dataclass(frozen=True)
class TrustContract:
    """What a consumer of these metrics is entitled to know.

    There is no ``warning`` field.  The banner is a *function* of the seven
    validated fields (:func:`derive_trust_summary`), so no manifest can
    describe itself as something the metadata says it is not.
    """

    dataset_kind: DatasetKind
    real_ingested_evidence: bool
    labeling_status: LabelingStatus
    reviewer_count: int
    adjudicated: bool
    gate_eligible: bool
    metrics_purpose: MetricsPurpose

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind is DatasetKind.SYNTHETIC_DEVELOPMENT

    @property
    def summary(self) -> TrustSummary:
        """The derived trust summary for this contract."""

        return derive_trust_summary(self)

    @property
    def warning(self) -> str:
        """The derived banner text."""

        return self.summary.text

    def as_dict(self) -> dict[str, Any]:
        """The structured block that goes into every JSON report."""

        return {
            "dataset_kind": self.dataset_kind.value,
            "real_ingested_evidence": self.real_ingested_evidence,
            "labeling_status": self.labeling_status.value,
            "reviewer_count": self.reviewer_count,
            "adjudicated": self.adjudicated,
            "gate_eligible": self.gate_eligible,
            "metrics_purpose": self.metrics_purpose.value,
            "warning": self.warning,
        }

    def banner(self) -> str:
        """The derived summary followed by the fields it was derived from."""

        return (
            f"{self.summary.text}\n"
            f"  dataset_kind           {self.dataset_kind.value}\n"
            f"  real_ingested_evidence {str(self.real_ingested_evidence).lower()}\n"
            f"  labeling_status        {self.labeling_status.value}\n"
            f"  reviewer_count         {self.reviewer_count}\n"
            f"  adjudicated            {str(self.adjudicated).lower()}\n"
            f"  gate_eligible          {str(self.gate_eligible).lower()}\n"
            f"  metrics_purpose        {self.metrics_purpose.value}"
        )


def _require_mapping(value: Any, field: str, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustContractError(f"{where}: {field} must be a mapping")
    return value


def _require_bool(value: Any, field: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise TrustContractError(f"{where}: {field} must be a boolean, not {value!r}")
    return value


def _require_positive_int(value: Any, field: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrustContractError(
            f"{where}: {field} must be a positive integer, not {value!r}"
        )
    return value


def _require_text(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrustContractError(f"{where}: {field} must be a non-blank string")
    return value


def _require_enum(value: Any, enum: type[Enum], field: str, where: str) -> Any:
    text = _require_text(value, field, where)
    try:
        return enum(text)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum)
        raise TrustContractError(
            f"{where}: {field}={text!r} is not one of {allowed}"
        ) from exc


def _check_invariants(contract: TrustContract, where: str) -> None:
    """Refuse every combination that would misrepresent the dataset.

    The per-kind half comes from :data:`DATASET_KIND_RULES`, so a kind
    added without an entry fails rather than inheriting whatever the last
    ``if`` happened to allow.  The reviewer, adjudication and gate rules
    are kind-independent and apply on top.
    """

    rules = rules_for(contract.dataset_kind)
    kind = contract.dataset_kind.value
    status = contract.labeling_status
    purpose = contract.metrics_purpose

    if contract.real_ingested_evidence != rules.real_ingested_evidence:
        wanted = str(rules.real_ingested_evidence).lower()
        raise TrustContractError(
            f"{where}: dataset_kind={kind} requires "
            f"real_ingested_evidence={wanted}; it is {rules.description}"
        )
    if purpose not in rules.allowed_metrics_purposes:
        allowed = sorted(member.value for member in rules.allowed_metrics_purposes)
        raise TrustContractError(
            f"{where}: dataset_kind={kind} must declare a metrics_purpose in "
            f"{allowed}, not {purpose.value!r}"
        )
    if contract.gate_eligible and not rules.may_be_gate_eligible:
        raise TrustContractError(
            f"{where}: dataset_kind={kind} cannot claim gate_eligible; its "
            "numbers cannot clear K3/G4 or final AC-3"
        )

    if status is LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED:
        if contract.reviewer_count != 1:
            raise TrustContractError(
                f"{where}: labeling_status=single_author_unadjudicated "
                f"requires reviewer_count=1, got {contract.reviewer_count}"
            )
        if contract.adjudicated:
            raise TrustContractError(
                f"{where}: labeling_status=single_author_unadjudicated cannot "
                "be adjudicated; one author has nobody to disagree with"
            )
        if contract.gate_eligible:
            raise TrustContractError(
                f"{where}: labeling_status=single_author_unadjudicated cannot "
                "be gate_eligible"
            )
    elif status in MULTI_REVIEWER_STATUSES and contract.reviewer_count < 2:
        raise TrustContractError(
            f"{where}: labeling_status={status.value} requires "
            f"reviewer_count >= 2, got {contract.reviewer_count}"
        )

    if contract.adjudicated:
        if contract.reviewer_count < 2:
            raise TrustContractError(
                f"{where}: adjudicated=true requires reviewer_count >= 2, got "
                f"{contract.reviewer_count}; adjudication resolves a "
                "disagreement between reviewers"
            )
        if status not in ADJUDICATED_STATUSES:
            raise TrustContractError(
                f"{where}: adjudicated=true requires an adjudicated "
                f"labeling_status, got {status.value!r}"
            )
    elif status in ADJUDICATED_STATUSES:
        raise TrustContractError(
            f"{where}: labeling_status={status.value} but adjudicated=false"
        )

    if contract.gate_eligible:
        if not contract.real_ingested_evidence:
            raise TrustContractError(
                f"{where}: gate_eligible=true requires "
                "real_ingested_evidence=true; a gate cannot be cleared on "
                "records nobody fetched"
            )
        if not contract.adjudicated:
            raise TrustContractError(
                f"{where}: gate_eligible=true requires adjudicated=true"
            )
        if contract.reviewer_count < 2:
            raise TrustContractError(
                f"{where}: gate_eligible=true requires reviewer_count >= 2, "
                f"got {contract.reviewer_count}"
            )
        if purpose not in GATE_PURPOSES:
            allowed = sorted(member.value for member in GATE_PURPOSES)
            raise TrustContractError(
                f"{where}: gate_eligible=true requires a metrics_purpose in "
                f"{allowed}, got {purpose.value!r}"
            )
    elif purpose in GATE_PURPOSES:
        raise TrustContractError(
            f"{where}: metrics_purpose={purpose.value!r} claims these numbers "
            "may settle a gate, but gate_eligible is false"
        )


def parse_trust_contract(metadata: Mapping[str, Any], *, where: str) -> TrustContract:
    """Validate a manifest's trust block, values and relationships, and return it."""

    block = _require_mapping(metadata.get("trust_contract"), "trust_contract", where)
    missing = sorted(set(REQUIRED_TRUST_FIELDS) - set(block))
    if missing:
        raise TrustContractError(f"{where}: trust_contract is missing {missing}")
    if "warning" in block:
        raise TrustContractError(
            f"{where}: trust_contract may not supply a warning; the banner is "
            "derived from the validated fields so it cannot contradict them"
        )
    unknown = sorted(
        set(block) - set(REQUIRED_TRUST_FIELDS) - set(OPTIONAL_TRUST_FIELDS)
    )
    if unknown:
        raise TrustContractError(
            f"{where}: trust_contract has unknown field(s) {unknown}"
        )

    contract = TrustContract(
        dataset_kind=_require_enum(
            block["dataset_kind"], DatasetKind, "dataset_kind", where
        ),
        real_ingested_evidence=_require_bool(
            block["real_ingested_evidence"], "real_ingested_evidence", where
        ),
        labeling_status=_require_enum(
            block["labeling_status"], LabelingStatus, "labeling_status", where
        ),
        reviewer_count=_require_positive_int(
            block["reviewer_count"], "reviewer_count", where
        ),
        adjudicated=_require_bool(block["adjudicated"], "adjudicated", where),
        gate_eligible=_require_bool(block["gate_eligible"], "gate_eligible", where),
        metrics_purpose=_require_enum(
            block["metrics_purpose"], MetricsPurpose, "metrics_purpose", where
        ),
    )
    _check_invariants(contract, where)
    return contract


def validate_provenance(
    metadata: Mapping[str, Any], contract: TrustContract, *, where: str
) -> Mapping[str, Any]:
    """Validate the structured provenance block against the trust contract.

    The common fields are required of every kind; the rest comes from
    :data:`DATASET_KIND_RULES`, so a production set has to name what it was
    sampled from and cannot describe itself as authored.
    """

    rules = rules_for(contract.dataset_kind)
    kind_name = contract.dataset_kind.value
    block = _require_mapping(metadata.get("provenance"), "provenance", where)
    required = set(REQUIRED_PROVENANCE_FIELDS) | set(rules.extra_provenance_fields)
    missing = sorted(required - set(block))
    if missing:
        raise TrustContractError(
            f"{where}: provenance is missing {missing} "
            f"(required for dataset_kind={kind_name})"
        )
    _require_text(block["statement"], "provenance.statement", where)
    for field in rules.extra_provenance_fields:
        value = block[field]
        if isinstance(value, Mapping):
            if not value:
                raise TrustContractError(
                    f"{where}: provenance.{field} must not be empty"
                )
        else:
            _require_text(value, f"provenance.{field}", where)
    kind = _require_text(block["kind"], "provenance.kind", where)
    method = _require_text(
        block["collection_method"], "provenance.collection_method", where
    )
    urls_synthetic = _require_bool(
        block["urls_are_synthetic"], "provenance.urls_are_synthetic", where
    )
    _require_bool(
        block["uses_real_outlet_names"], "provenance.uses_real_outlet_names", where
    )
    if kind not in rules.provenance_kinds:
        allowed = sorted(rules.provenance_kinds)
        raise TrustContractError(
            f"{where}: dataset_kind={kind_name} requires a provenance.kind in "
            f"{allowed}, got {kind!r}"
        )
    if method not in rules.collection_methods:
        allowed = sorted(rules.collection_methods)
        raise TrustContractError(
            f"{where}: dataset_kind={kind_name} requires a "
            f"provenance.collection_method in {allowed}, got {method!r}; "
            f"it is {rules.description}"
        )
    if urls_synthetic != rules.urls_are_synthetic:
        wanted = str(rules.urls_are_synthetic).lower()
        raise TrustContractError(
            f"{where}: dataset_kind={kind_name} requires "
            f"provenance.urls_are_synthetic={wanted}"
        )
    return block


def validate_labeling(
    metadata: Mapping[str, Any], contract: TrustContract, *, where: str
) -> Mapping[str, Any]:
    """Validate the structured labeling block against the trust contract."""

    block = _require_mapping(metadata.get("labeling"), "labeling", where)
    missing = sorted(set(REQUIRED_LABELING_FIELDS) - set(block))
    if missing:
        raise TrustContractError(f"{where}: labeling is missing {missing}")
    status = _require_enum(block["status"], LabelingStatus, "labeling.status", where)
    _require_text(block["protocol"], "labeling.protocol", where)
    reviewer_count = _require_positive_int(
        block["reviewer_count"], "labeling.reviewer_count", where
    )
    adjudicated = _require_bool(block["adjudicated"], "labeling.adjudicated", where)
    gate_eligible = _require_bool(
        block["gate_eligible"], "labeling.gate_eligible", where
    )
    reviewers = block["reviewers"]
    if not isinstance(reviewers, list) or any(
        not isinstance(name, str) or not name.strip() for name in reviewers
    ):
        raise TrustContractError(
            f"{where}: labeling.reviewers must be a list of non-blank names"
        )
    if len(reviewers) > reviewer_count:
        raise TrustContractError(
            f"{where}: labeling names {len(reviewers)} reviewers but declares "
            f"reviewer_count={reviewer_count}"
        )
    for field, declared, expected in (
        ("status", status, contract.labeling_status),
        ("reviewer_count", reviewer_count, contract.reviewer_count),
        ("adjudicated", adjudicated, contract.adjudicated),
        ("gate_eligible", gate_eligible, contract.gate_eligible),
    ):
        if declared != expected:
            shown = declared.value if isinstance(declared, Enum) else declared
            wanted = expected.value if isinstance(expected, Enum) else expected
            raise TrustContractError(
                f"{where}: labeling.{field}={shown!r} contradicts "
                f"trust_contract.{field}={wanted!r}"
            )
    return block
