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

#: Printed above every metric, in every renderer.  One line, unmissable.
WARNING_BANNER = (
    "WARNING: Synthetic, single-author, unadjudicated development dataset.\n"
    "Metrics are not valid for K3/G4 or final AC-3 acceptance."
)


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
#: that clears every gate invariant below may declare one.
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

REQUIRED_PROVENANCE_FIELDS = (
    "kind",
    "collection_method",
    "statement",
    "urls_are_synthetic",
    "uses_real_outlet_names",
    "why_synthetic",
    "blocked_by",
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
class TrustContract:
    """What a consumer of these metrics is entitled to know."""

    dataset_kind: DatasetKind
    real_ingested_evidence: bool
    labeling_status: LabelingStatus
    reviewer_count: int
    adjudicated: bool
    gate_eligible: bool
    metrics_purpose: MetricsPurpose
    warning: str = WARNING_BANNER

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind is DatasetKind.SYNTHETIC_DEVELOPMENT

    def as_dict(self) -> dict[str, Any]:
        """The exact block that goes into every JSON report."""

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
        """The multi-line warning a text renderer prints before any metric."""

        return (
            f"{self.warning}\n"
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

    Each rule names a specific untrue claim a manifest could otherwise
    make, so a failure tells the author what they asserted rather than
    which assertion tripped a type check.
    """

    kind = contract.dataset_kind
    status = contract.labeling_status
    purpose = contract.metrics_purpose

    if kind is DatasetKind.SYNTHETIC_DEVELOPMENT:
        if contract.real_ingested_evidence:
            raise TrustContractError(
                f"{where}: dataset_kind=synthetic_development cannot claim "
                "real_ingested_evidence; nothing in it was fetched"
            )
        if contract.gate_eligible:
            raise TrustContractError(
                f"{where}: dataset_kind=synthetic_development cannot claim "
                "gate_eligible; its numbers cannot clear K3/G4 or final AC-3"
            )
        if purpose is not MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY:
            raise TrustContractError(
                f"{where}: dataset_kind=synthetic_development must declare "
                "metrics_purpose=development_regression_only, not "
                f"{purpose.value!r}"
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
        if kind is DatasetKind.SYNTHETIC_DEVELOPMENT:
            raise TrustContractError(
                f"{where}: gate_eligible=true is not available to a "
                "synthetic_development dataset"
            )
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
        warning=str(block.get("warning") or WARNING_BANNER),
    )
    _check_invariants(contract, where)
    return contract


def validate_provenance(
    metadata: Mapping[str, Any], contract: TrustContract, *, where: str
) -> Mapping[str, Any]:
    """Validate the structured provenance block against the trust contract."""

    block = _require_mapping(metadata.get("provenance"), "provenance", where)
    missing = sorted(set(REQUIRED_PROVENANCE_FIELDS) - set(block))
    if missing:
        raise TrustContractError(f"{where}: provenance is missing {missing}")
    _require_text(block["statement"], "provenance.statement", where)
    _require_text(block["why_synthetic"], "provenance.why_synthetic", where)
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
    _require_mapping(block["blocked_by"], "provenance.blocked_by", where)
    if contract.is_synthetic:
        if kind != "synthetic":
            raise TrustContractError(
                f"{where}: dataset_kind is synthetic but provenance.kind is {kind!r}"
            )
        if method != "authored":
            raise TrustContractError(
                f"{where}: a synthetic dataset cannot claim "
                f"collection_method={method!r}; nothing here was collected"
            )
        if not urls_synthetic:
            raise TrustContractError(
                f"{where}: a synthetic dataset cannot claim its URLs are real"
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
