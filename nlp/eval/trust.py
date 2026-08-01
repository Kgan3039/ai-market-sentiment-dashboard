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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: Printed above every metric, in every renderer.  One line, unmissable.
WARNING_BANNER = (
    "WARNING: Synthetic, single-author, unadjudicated development dataset.\n"
    "Metrics are not valid for K3/G4 or final AC-3 acceptance."
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

_DATASET_KINDS = frozenset({"synthetic_development", "sampled_production"})
_SYNTHETIC_KINDS = frozenset({"synthetic_development"})


class TrustContractError(ValueError):
    """The dataset's provenance is missing, malformed, or self-contradictory."""


@dataclass(frozen=True)
class TrustContract:
    """What a consumer of these metrics is entitled to know."""

    dataset_kind: str
    real_ingested_evidence: bool
    labeling_status: str
    reviewer_count: int
    adjudicated: bool
    gate_eligible: bool
    metrics_purpose: str
    warning: str = WARNING_BANNER

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind in _SYNTHETIC_KINDS

    def as_dict(self) -> dict[str, Any]:
        """The exact block that goes into every JSON report."""

        return {
            "dataset_kind": self.dataset_kind,
            "real_ingested_evidence": self.real_ingested_evidence,
            "labeling_status": self.labeling_status,
            "reviewer_count": self.reviewer_count,
            "adjudicated": self.adjudicated,
            "gate_eligible": self.gate_eligible,
            "metrics_purpose": self.metrics_purpose,
            "warning": self.warning,
        }

    def banner(self) -> str:
        """The multi-line warning a text renderer prints before any metric."""

        return (
            f"{self.warning}\n"
            f"  dataset_kind           {self.dataset_kind}\n"
            f"  real_ingested_evidence {str(self.real_ingested_evidence).lower()}\n"
            f"  labeling_status        {self.labeling_status}\n"
            f"  reviewer_count         {self.reviewer_count}\n"
            f"  adjudicated            {str(self.adjudicated).lower()}\n"
            f"  gate_eligible          {str(self.gate_eligible).lower()}\n"
            f"  metrics_purpose        {self.metrics_purpose}"
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


def parse_trust_contract(metadata: Mapping[str, Any], *, where: str) -> TrustContract:
    """Validate a manifest's trust block and return it.

    Beyond presence and type, three consistency rules are enforced, because
    each of them is a way a dataset could tell a reader something untrue:

    * synthetic data may not claim real ingested evidence;
    * synthetic data may not claim to be gate eligible;
    * a set with fewer than two reviewers may not claim to be adjudicated.
    """

    block = _require_mapping(metadata.get("trust_contract"), "trust_contract", where)
    missing = sorted(set(REQUIRED_TRUST_FIELDS) - set(block))
    if missing:
        raise TrustContractError(f"{where}: trust_contract is missing {missing}")

    dataset_kind = _require_text(block["dataset_kind"], "dataset_kind", where)
    if dataset_kind not in _DATASET_KINDS:
        raise TrustContractError(
            f"{where}: dataset_kind={dataset_kind!r} is not one of "
            f"{sorted(_DATASET_KINDS)}"
        )
    contract = TrustContract(
        dataset_kind=dataset_kind,
        real_ingested_evidence=_require_bool(
            block["real_ingested_evidence"], "real_ingested_evidence", where
        ),
        labeling_status=_require_text(
            block["labeling_status"], "labeling_status", where
        ),
        reviewer_count=_require_positive_int(
            block["reviewer_count"], "reviewer_count", where
        ),
        adjudicated=_require_bool(block["adjudicated"], "adjudicated", where),
        gate_eligible=_require_bool(block["gate_eligible"], "gate_eligible", where),
        metrics_purpose=_require_text(
            block["metrics_purpose"], "metrics_purpose", where
        ),
        warning=str(block.get("warning") or WARNING_BANNER),
    )

    if contract.is_synthetic and contract.real_ingested_evidence:
        raise TrustContractError(
            f"{where}: a synthetic dataset cannot claim real_ingested_evidence"
        )
    if contract.is_synthetic and contract.gate_eligible:
        raise TrustContractError(
            f"{where}: a synthetic dataset cannot claim gate_eligible; its "
            "numbers cannot clear K3/G4 or final AC-3"
        )
    if contract.adjudicated and contract.reviewer_count < 2:
        raise TrustContractError(
            f"{where}: adjudicated is true with reviewer_count="
            f"{contract.reviewer_count}; adjudication needs at least two reviewers"
        )
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
    _require_text(block["status"], "labeling.status", where)
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
        ("reviewer_count", reviewer_count, contract.reviewer_count),
        ("adjudicated", adjudicated, contract.adjudicated),
        ("gate_eligible", gate_eligible, contract.gate_eligible),
    ):
        if declared != expected:
            raise TrustContractError(
                f"{where}: labeling.{field}={declared!r} contradicts "
                f"trust_contract.{field}={expected!r}"
            )
    return block
