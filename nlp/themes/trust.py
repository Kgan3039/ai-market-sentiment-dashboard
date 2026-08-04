"""M5's own trust notice, beside the shared one.

The dataset contract in :mod:`nlp.eval.trust` is shared with M3 and M4, and
its derived banner names the gates *those* stages answer to — K3, G4, AC-3.
A reader looking at theme output and seeing "not valid for K3/G4 or final
AC-3" learns that some gate is unmet and nothing about the one M5 is
measured against, which is G1 (theme-assignment agreement) and AC-4.

So a stage notice is derived here from the same validated fields and
reported alongside the shared summary, never instead of it.  Nothing in it
is supplied by a manifest: like the shared banner it is a function of the
contract, so it cannot describe the fixture as something the fields say it
is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp.eval.trust import TrustContract

#: Bumped when the notice's wording contract changes.
STAGE_TRUST_VERSION = "m5.stage_trust.v2"

#: What M5's numbers are measured against, named so the notice cannot drift
#: onto another stage's gates.
STAGE_GATES = "AC-4 (clustering) and G1 (theme-assignment agreement)"


@dataclass(frozen=True)
class StageTrustSummary:
    """The M5-specific judgement, derived from the validated contract."""

    level: str
    stage: str
    issue: str
    headline: str
    detail: str
    gates: str
    version: str = STAGE_TRUST_VERSION

    @property
    def text(self) -> str:
        return f"{self.level} [{self.stage}]: {self.headline}\n{self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "stage": self.stage,
            "issue": self.issue,
            "headline": self.headline,
            "detail": self.detail,
            "gates": self.gates,
            "version": self.version,
            "text": self.text,
        }


def derive_stage_trust_summary(contract: TrustContract) -> StageTrustSummary:
    """Return M5's notice for a dataset, from the same seven fields."""

    synthetic = not contract.real_ingested_evidence
    single_author = contract.reviewer_count <= 1 and not contract.adjudicated
    parts: list[str] = []
    if synthetic:
        parts.append(
            "These are synthetic authored ticker-days, not ingested "
            "coverage: no theme here was produced from a real trading day, "
            "so there is no real ticker-day validation of AC-4 behind any "
            "number in this file"
        )
    else:
        parts.append("These ticker-days carry real ingested evidence")
    if single_author:
        parts.append(
            "the expected groupings are single-author, unadjudicated and "
            "have no second reviewer, so a clustering that reproduces them "
            "has agreed with its author and nothing more"
        )
    if not contract.gate_eligible:
        parts.append(
            f"the dataset is non-gate-eligible and nothing here validates "
            f"{STAGE_GATES}, which need real ingested ticker-days "
            "(#57/#61/#62) and the two-reviewer assignment review in #60"
        )
    level = "WARNING" if (synthetic or not contract.gate_eligible) else "NOTICE"
    headline = (
        "Issue #72 (M5 theme clustering) on a synthetic authored development "
        "fixture: not AC-4 or G1 validation."
        if synthetic
        else "Issue #72 (M5 theme clustering) on real ingested ticker-days."
    )
    return StageTrustSummary(
        level=level,
        stage="M5 theme clustering (issue #72)",
        issue="#72",
        headline=headline,
        detail="; ".join(parts) + ".",
        gates=STAGE_GATES,
    )
