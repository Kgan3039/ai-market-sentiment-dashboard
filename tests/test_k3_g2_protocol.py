"""K3a: the ratified G2 faithfulness protocol, ``k3-g2-v1``.

What a reviewer *decides* is human and lives in
``docs/reviews/K3_G2_PROTOCOL.md``.  What is tested here is what the code
holds that protocol to: it is registered and pinned to its text, a round
drawn under it can reach PASS / FAIL through the real write, sample, sheet
and score paths, and its reviewer, adjudicator, reason-code and timestamp
rules are refused when broken -- while G1 and the provisional protocol
read and score exactly as before.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import ai.summarization as summarization
import nlp.eval.faithfulness as g2
import nlp.eval.review as review
import test_review_sampling_g2 as a4b
from test_review_sampling_g2 import (
    CODE,
    D1,
    D2,
    D3,
    _twenty_sentence_world,
    population,
    read_rows,
    write_rows,
)
from tools import make_review_sheets

# A4b's persisted world and its fixtures, reused rather than rebuilt.
no_ambient_provider_config = a4b.no_ambient_provider_config
world = a4b.world
sampled = a4b.sampled

REPO = Path(__file__).resolve().parents[1]
K3 = g2.K3_G2_V1
GateResult = review.GateResult
State = review.AdjudicationState
#: The fingerprint of k3-g2-v1 as ratified.  It covers every rule the code
#: carries and the guidelines' digest; a change here is a protocol change,
#: and a material one is ``k3-g2-v2``, not an edit of this value.
K3_G2_V1_FINGERPRINT = (
    "cd1508e8bcb86e8e6c385d1cbcd00e65e85c4463e0128e8c2b4d7ac27b9b6f55"
)


# ----------------------------------------------------------------------
# Helpers: a verified two-day, twenty-sentence round under a protocol
# ----------------------------------------------------------------------


def draw(world, tmp_path, protocol=g2.K3_G2_V1_ID):
    """Twenty sentences on exactly two days, every provenance fact verified."""

    _twenty_sentence_world(world, logged_ingest=True, bound_themes=True)
    drawn = g2.sample_sentences(population(world, days=(D1, D2)), seed="k3")
    manifest = g2.build_manifest(
        drawn, csv_name="k3.csv", protocol_id=protocol, code=CODE
    )
    return g2.write_sample(drawn, tmp_path / "k3.csv", manifest=manifest)


def fill(blank, out, reviewer, verdicts, *, notes=None, reviewed_at="2026-07-26"):
    """One reviewer's sheet; ``unsupported`` gets a valid reason code by default."""

    rows = read_rows(blank)
    for index, row in enumerate(rows):
        verdict = verdicts(index)
        if notes is not None:
            note = notes(index)
        else:
            note = "NUMBER: test" if verdict == "unsupported" else ""
        row.update(
            reviewer_id=reviewer if verdict else "",
            reviewed_at=reviewed_at if verdict else "",
            reviewer_verdict=verdict,
            reviewer_notes=note,
        )
    return write_rows(out, rows)


def adjudicate(path, entries):
    """An adjudication sheet: ``entries`` are dicts over its columns."""

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=review.ADJUDICATION_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "final_verdict": "supported",
                    "adjudicator_id": "carol",
                    "adjudicated_at": "2026-07-27",
                    "adjudication_notes": "cited [1] states the claim",
                    **entry,
                }
            )
    return Path(path)


def unsupported_first(count):
    return lambda i: "unsupported" if i < count else "supported"


def all_supported(i):
    return "supported"


def score(manifest_path, sheets, adjudication=None):
    result = g2.score_sentence_round(
        g2.read_manifest(manifest_path), sheets, adjudicated=adjudication
    )
    return result, g2.score_g2(result)


@pytest.fixture
def k3_round(world, tmp_path):
    return draw(world, tmp_path)


def two_reviewers(k3_round, tmp_path, a, b, **kwargs):
    csv_path, _ = k3_round
    return [
        fill(csv_path, tmp_path / "alice.csv", "alice", a, **kwargs),
        fill(csv_path, tmp_path / "bob.csv", "bob", b, **kwargs),
    ]


def disputed(k3_round, tmp_path):
    """Alice supports everything; Bob marks row 0 unsupported."""

    csv_path, _ = k3_round
    sheets = two_reviewers(k3_round, tmp_path, all_supported, unsupported_first(1))
    return read_rows(csv_path)[0]["row_id"], sheets


# ----------------------------------------------------------------------
# Protocol identity and binding
# ----------------------------------------------------------------------


def test_k3_g2_v1_is_the_ratified_g2_protocol():
    assert g2.RATIFIED_G2_PROTOCOLS == {"k3-g2-v1": K3}
    assert g2.resolve_g2_protocol("k3-g2-v1") == (K3, True)
    assert g2.require_known_g2_protocol("k3-g2-v1") is K3
    assert K3.vocabulary == {"supported", "unsupported"}
    assert K3.adjudicated_states == {State.UNANIMOUS, State.RESOLVED}
    assert K3.strict_review is True
    assert K3.reason_codes == (
        "ADDITION",
        "CONTRADICTION",
        "NUMBER",
        "ENTITY",
        "TEMPORAL",
        "CAUSAL",
        "CERTAINTY",
        "SYNTHESIS",
        "MISCITATION",
        "UNVERIFIABLE",
    )


@pytest.mark.parametrize("identifier", ["k3-g2-v2", "K3-G2-V1", "k3-g2-v1 x", ""])
def test_an_unknown_g2_protocol_is_refused_and_scores_unratified(identifier):
    assert g2.resolve_g2_protocol(identifier) == (g2.PROVISIONAL_G2_PROTOCOL, False)
    with pytest.raises(review.ReviewSamplingError):
        g2.require_known_g2_protocol(identifier)


def test_k3_g2_v1_is_not_a_g1_protocol_and_g1_ratifies_nothing():
    assert review.RATIFIED_PROTOCOLS == {}
    assert review.resolve_protocol("k3-g2-v1") == (review.PROVISIONAL_PROTOCOL, False)
    with pytest.raises(review.ReviewSamplingError, match="unknown labeling protocol"):
        review.require_known_protocol("k3-g2-v1")


def test_the_guidelines_document_is_pinned_byte_for_byte():
    document = REPO / K3.document
    assert K3.document == "docs/reviews/K3_G2_PROTOCOL.md"
    assert hashlib.sha256(document.read_bytes()).hexdigest() == K3.document_sha256, (
        "docs/reviews/K3_G2_PROTOCOL.md changed. A material change is a new "
        "protocol id (k3-g2-v2); a non-material one needs a deliberate re-pin."
    )


def test_the_protocol_fingerprint_is_pinned_and_covers_every_rule():
    assert K3.fingerprint() == K3_G2_V1_FINGERPRINT
    variants = [
        dataclasses.replace(K3, reason_codes=K3.reason_codes[:-1]),
        dataclasses.replace(K3, exempt_framings=K3.exempt_framings + ("Today ",)),
        dataclasses.replace(K3, adjudicated_states=frozenset({State.UNANIMOUS})),
        dataclasses.replace(K3, strict_review=False),
        dataclasses.replace(K3, document_sha256="0" * 64),
        dataclasses.replace(K3, negative_verdict="not_supported"),
    ]
    assert len({v.fingerprint() for v in variants} | {K3.fingerprint()}) == 7


def test_the_guidelines_state_the_rules_the_code_enforces():
    text = (REPO / K3.document).read_text(encoding="utf-8")
    assert "`k3-g2-v1`" in text
    for code in K3.reason_codes:
        assert f"| `{code}` |" in text
    for framing in K3.exempt_framings:
        assert f'`"{framing}"`' in text
    # K3a closes the G2 portion only.
    assert "does **not** close K3" in text
    assert "K3b" in text and "K3c" in text


# ----------------------------------------------------------------------
# The gate, end to end
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsupported, gate_result", [(1, GateResult.PASS), (2, GateResult.FAIL)]
)
def test_a_unanimous_k3_round_reaches_pass_or_fail(
    k3_round, tmp_path, unsupported, gate_result
):
    _, manifest_path = k3_round
    verdicts = unsupported_first(unsupported)
    result, card = score(
        manifest_path, two_reviewers(k3_round, tmp_path, verdicts, verdicts)
    )
    assert result.adjudication_state is State.UNANIMOUS
    assert card.sentence_count == 20 and card.resolved_count == 20
    assert card.positive_count == 20 - unsupported
    assert card.eligibility_blockers == () and card.incompleteness == ()
    assert card.gate_eligible is True and card.review_complete is True
    assert card.gate_result is gate_result
    assert card.protocol_id == "k3-g2-v1" and card.protocol_ratified is True
    assert card.as_dict()["protocol"]["fingerprint"] == K3.fingerprint()


def test_an_open_disagreement_is_incomplete(k3_round, tmp_path):
    _, manifest_path = k3_round
    _, sheets = disputed(k3_round, tmp_path)
    result, card = score(manifest_path, sheets)
    assert result.adjudication_state is State.OPEN
    assert card.eligibility_blockers == ()
    assert card.unresolved_count == 1 and card.review_complete is False
    assert card.gate_result is GateResult.INCOMPLETE


def test_a_disagreement_resolved_by_a_third_identity_is_eligible(k3_round, tmp_path):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(tmp_path / "adj.csv", [{"row_id": row_id}])
    result, card = score(manifest_path, sheets, adjudication)
    assert result.adjudication_state is State.RESOLVED
    assert card.adjudicator_ids == ("carol",)
    assert card.gate_eligible is True and card.positive_count == 20
    assert card.gate_result is GateResult.PASS


def test_one_reviewer_is_never_eligible(k3_round, tmp_path):
    csv_path, manifest_path = k3_round
    sheet = fill(csv_path, tmp_path / "alice.csv", "alice", all_supported)
    _, card = score(manifest_path, [sheet])
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert card.eligibility_blockers == (
        "fewer than two reviewers; section 8 requires two, adjudicated",
    )


def test_a_provisional_round_on_verified_data_stays_not_eligible(world, tmp_path):
    csv_path, manifest_path = draw(world, tmp_path, protocol="unratified")
    sheets = [
        fill(csv_path, tmp_path / "alice.csv", "alice", all_supported),
        fill(csv_path, tmp_path / "bob.csv", "bob", all_supported),
    ]
    _, card = score(manifest_path, sheets)
    assert card.origin_status is review.OriginStatus.VERIFIED_LIVE
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any(
        "not in the ratified G2 registry" in b for b in card.eligibility_blockers
    )


def test_an_old_round_cannot_be_relabelled_as_k3(world, tmp_path):
    """Rewriting an ``unratified`` manifest's protocol breaks its identity."""

    csv_path, manifest_path = draw(world, tmp_path, protocol="unratified")
    payload = json.loads(manifest_path.read_text())
    payload["labeling_protocol"]["id"] = "k3-g2-v1"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(review.ReviewSamplingError):
        g2.read_manifest(manifest_path)


def test_a_development_draw_under_k3_is_never_eligible(world, tmp_path):
    _twenty_sentence_world(world, logged_ingest=True, bound_themes=True)
    drawn = g2.sample_sentences(
        population(world, days=(D1, D2)), seed="k3", draw_size=2
    )
    manifest = g2.build_manifest(
        drawn, csv_name="dev.csv", protocol_id="k3-g2-v1", code=CODE
    )
    csv_path, manifest_path = g2.write_sample(
        drawn, tmp_path / "dev.csv", manifest=manifest
    )
    sheets = [
        fill(csv_path, tmp_path / "alice.csv", "alice", all_supported),
        fill(csv_path, tmp_path / "bob.csv", "bob", all_supported),
    ]
    _, card = score(manifest_path, sheets)
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any("development draw" in b for b in card.eligibility_blockers)


def test_the_cli_samples_and_scores_a_k3_round_to_a_verdict(world, tmp_path, capsys):
    _twenty_sentence_world(world, logged_ingest=True, bound_themes=True)
    out = tmp_path / "cli" / "g2.csv"
    assert (
        make_review_sheets.main(
            [
                "sample-sentences",
                "--protocol",
                "k3-g2-v1",
                "--database",
                str(world.path),
                "--window-start",
                D1,
                "--window-end",
                D3,
                "--seed",
                "cli",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    sheets = [
        fill(out, tmp_path / "a.csv", "alice", all_supported),
        fill(out, tmp_path / "b.csv", "bob", all_supported),
    ]
    report = tmp_path / "cli" / "scorecard.json"
    code = make_review_sheets.main(
        ["score-sentences", "--round", str(out.with_name("g2.manifest.json"))]
        + [str(s) for s in sheets]
        + ["--report", str(report)]
    )
    assert "gate_result        PASS" in capsys.readouterr().out
    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["gate_result"] == "PASS"
    assert payload["protocol"] == {
        "id": "k3-g2-v1",
        "ratified": True,
        "fingerprint": K3.fingerprint(),
    }


# ----------------------------------------------------------------------
# Reviewer and adjudicator identity
# ----------------------------------------------------------------------


def test_reviewer_ids_differing_only_by_case_are_one_reviewer(k3_round, tmp_path):
    csv_path, manifest_path = k3_round
    sheets = [
        fill(csv_path, tmp_path / "a.csv", "Alice", all_supported),
        fill(csv_path, tmp_path / "b.csv", "alice", all_supported),
    ]
    with pytest.raises(review.ReviewSamplingError, match="case-insensitively"):
        score(manifest_path, sheets)


@pytest.mark.parametrize("adjudicator", ["alice", "bob", "ALICE", "Bob"])
def test_a_reviewer_cannot_adjudicate(k3_round, tmp_path, adjudicator):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv", [{"row_id": row_id, "adjudicator_id": adjudicator}]
    )
    with pytest.raises(review.ReviewSamplingError, match="third identity"):
        score(manifest_path, sheets, adjudication)


def test_adjudication_of_an_agreed_row_is_refused(k3_round, tmp_path):
    csv_path, manifest_path = k3_round
    disputed_row, sheets = disputed(k3_round, tmp_path)
    agreed_row = read_rows(csv_path)[1]["row_id"]
    adjudication = adjudicate(
        tmp_path / "adj.csv",
        [
            {"row_id": disputed_row},
            {
                "row_id": agreed_row,
                "final_verdict": "unsupported",
                "adjudication_notes": "ADDITION: overruled",
            },
        ],
    )
    with pytest.raises(review.ReviewSamplingError, match="not a disagreement"):
        score(manifest_path, sheets, adjudication)


def test_adjudication_of_a_half_blank_row_is_refused(k3_round, tmp_path):
    csv_path, manifest_path = k3_round
    sheets = two_reviewers(
        k3_round, tmp_path, all_supported, lambda i: "" if i == 0 else "supported"
    )
    row_id = read_rows(csv_path)[0]["row_id"]
    adjudication = adjudicate(tmp_path / "adj.csv", [{"row_id": row_id}])
    with pytest.raises(review.ReviewSamplingError, match="not a disagreement"):
        score(manifest_path, sheets, adjudication)


# ----------------------------------------------------------------------
# Reason codes and notes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "NUMBER: Evidence says $90B, sentence says $100B.",
        'ADDITION: "third recall this year" is not in cited evidence.',
        "UNVERIFIABLE",
        "CAUSAL:",
        "  MISCITATION: [2] supports nothing  ",
        "SYNTHESIS: joins [1] and [2]\nby a relation neither states",
    ],
)
def test_an_unsupported_verdict_with_a_reason_code_is_accepted(
    k3_round, tmp_path, note
):
    _, manifest_path = k3_round
    verdicts = unsupported_first(1)
    sheets = two_reviewers(
        k3_round, tmp_path, verdicts, verdicts, notes=lambda i: note if i == 0 else ""
    )
    _, card = score(manifest_path, sheets)
    assert card.gate_result is GateResult.PASS


@pytest.mark.parametrize(
    "note",
    [
        "",
        "number: lower case",
        "NUMBERS: not a code",
        "PARTIAL: not a v1 code",
        "NUMBER - wrong separator",
        "NUMBER, ENTITY: two codes",
        "see NUMBER",
        "Evidence says $90B",
    ],
)
def test_an_unsupported_verdict_without_a_valid_reason_code_is_refused(
    k3_round, tmp_path, note
):
    _, manifest_path = k3_round
    verdicts = unsupported_first(1)
    sheets = two_reviewers(
        k3_round, tmp_path, verdicts, verdicts, notes=lambda i: note if i == 0 else ""
    )
    with pytest.raises(review.ReviewSamplingError, match="reason code"):
        score(manifest_path, sheets)


def test_a_supported_verdict_needs_no_notes(k3_round, tmp_path):
    _, manifest_path = k3_round
    sheets = two_reviewers(
        k3_round, tmp_path, all_supported, all_supported, notes=lambda i: ""
    )
    _, card = score(manifest_path, sheets)
    assert card.gate_result is GateResult.PASS


def test_an_adjudicated_verdict_needs_notes(k3_round, tmp_path):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv", [{"row_id": row_id, "adjudication_notes": ""}]
    )
    with pytest.raises(review.ReviewSamplingError, match="adjudication_notes"):
        score(manifest_path, sheets, adjudication)


@pytest.mark.parametrize(
    "note, accepted", [("ADDITION: not in [1]", True), ("bob is right", False)]
)
def test_an_adjudicated_unsupported_verdict_needs_a_reason_code(
    k3_round, tmp_path, note, accepted
):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv",
        [
            {
                "row_id": row_id,
                "final_verdict": "unsupported",
                "adjudication_notes": note,
            }
        ],
    )
    if accepted:
        _, card = score(manifest_path, sheets, adjudication)
        assert card.positive_count == 19 and card.gate_result is GateResult.PASS
    else:
        with pytest.raises(review.ReviewSamplingError, match="reason code"):
            score(manifest_path, sheets, adjudication)


# ----------------------------------------------------------------------
# Timestamps
# ----------------------------------------------------------------------


VALID_STAMPS = [
    "2026-07-26",
    "2026-07-26T14:05",
    "2026-07-26T14:05Z",
    "2026-07-26T14:05:09Z",
    "2026-07-26T14:05:09.123456Z",
    "2026-07-26T14:05:09.123+05:30",
    "2026-07-26T14:05:09+01:30",
    "2026-07-26T14:05:09-05:00",
    "2026-07-26T14:05:09+23:59",
]
#: Offsets ``datetime.fromisoformat`` would normalise rather than refuse.
IMPOSSIBLE_OFFSETS = [
    "2026-07-26T14:05:09+01:99",
    "2026-07-26T14:05:09+00:60",
    "2026-07-26T14:05:09+24:00",
    "2026-07-26T14:05:09+99:00",
    "2026-07-26T14:05:09+05:60",
    "2026-07-26T14:05:09-24:00",
]
MALFORMED_STAMPS = IMPOSSIBLE_OFFSETS + [
    "26/07/2026",
    "2026-02-30",
    "2026-13-01",
    "2026-07-26 14:05",
    "July 26",
    "2026-07-26T25:00",
    "2026-07-26T14:60",
    "2026-07-26T14:05:60",
    "2026-7-26",
    "2026-07-26T14:05:09+0530",
    "2026-07-26T14:05:09+5:30",
    "2026-07-26T14:05:09z",
    "yesterday",
]


@pytest.mark.parametrize("stamp", VALID_STAMPS)
def test_iso_8601_reviewer_timestamps_are_accepted(k3_round, tmp_path, stamp):
    _, manifest_path = k3_round
    sheets = two_reviewers(
        k3_round, tmp_path, all_supported, all_supported, reviewed_at=stamp
    )
    _, card = score(manifest_path, sheets)
    assert card.gate_result is GateResult.PASS


@pytest.mark.parametrize("stamp", MALFORMED_STAMPS)
def test_malformed_reviewer_timestamps_are_refused(k3_round, tmp_path, stamp):
    _, manifest_path = k3_round
    sheets = two_reviewers(
        k3_round, tmp_path, all_supported, all_supported, reviewed_at=stamp
    )
    with pytest.raises(review.ReviewSamplingError, match="reviewed_at.*ISO-8601"):
        score(manifest_path, sheets)


@pytest.mark.parametrize("stamp", VALID_STAMPS)
def test_iso_8601_adjudication_timestamps_are_accepted(k3_round, tmp_path, stamp):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv", [{"row_id": row_id, "adjudicated_at": stamp}]
    )
    _, card = score(manifest_path, sheets, adjudication)
    assert card.gate_result is GateResult.PASS


@pytest.mark.parametrize("stamp", MALFORMED_STAMPS + ["27.07.2026"])
def test_malformed_adjudication_timestamps_are_refused(k3_round, tmp_path, stamp):
    _, manifest_path = k3_round
    row_id, sheets = disputed(k3_round, tmp_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv", [{"row_id": row_id, "adjudicated_at": stamp}]
    )
    with pytest.raises(review.ReviewSamplingError, match="adjudicated_at.*ISO-8601"):
        score(manifest_path, sheets, adjudication)


@pytest.mark.parametrize("stamp", IMPOSSIBLE_OFFSETS)
def test_impossible_offsets_are_refused_not_normalised(stamp):
    assert review.is_iso_8601(stamp) is False


def test_a_blank_timestamp_is_still_allowed(k3_round, tmp_path):
    _, manifest_path = k3_round
    sheets = two_reviewers(
        k3_round, tmp_path, all_supported, all_supported, reviewed_at=""
    )
    _, card = score(manifest_path, sheets)
    assert card.gate_result is GateResult.PASS


# ----------------------------------------------------------------------
# The coverage-framing exemption
# ----------------------------------------------------------------------


def test_the_exempt_framings_are_exactly_the_prompts():
    assert K3.exempt_framings == (
        "Coverage today is dominated by ",
        "The most-covered storyline is ",
    )
    prompt = summarization.SYSTEM_PROMPT
    for framing in K3.exempt_framings:
        lowered = framing[0].lower() + framing[1:].rstrip()
        assert f'"{lowered}..."' in prompt


@pytest.mark.parametrize(
    "sentence, framing",
    [
        (
            "Coverage today is dominated by Nvidia's quarterly revenue [1].",
            "Coverage today is dominated by ",
        ),
        (
            "The most-covered storyline is Apple's $90 billion buyback [1].",
            "The most-covered storyline is ",
        ),
        ("Coverage today is dominated by X", "Coverage today is dominated by "),
        ("The most-covered storyline is X", "The most-covered storyline is "),
        (
            "Coverage today is dominated by X  and\tY\nacross lines  [1].",
            "Coverage today is dominated by ",
        ),
    ],
)
def test_an_exact_framing_exempts_only_its_own_words(sentence, framing):
    found, remainder = review.split_exempt_framing(sentence, K3)
    assert found == framing
    assert remainder == sentence[len(framing) :]
    assert found + remainder == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "coverage today is dominated by Nvidia's revenue [1].",
        "Coverage is dominated by Nvidia's revenue [1].",
        "Coverage today was dominated by Nvidia's revenue [1].",
        "Coverage today is largely dominated by Nvidia's revenue [1].",
        "Coverage today is dominated by: Nvidia's revenue [1].",
        "Coverage  today is dominated by Nvidia's revenue [1].",
        "Today, coverage is dominated by Nvidia's revenue [1].",
        "Most coverage today focuses on Nvidia's revenue [1].",
        "The most covered storyline is Nvidia's revenue [1].",
        "The most-covered story is Nvidia's revenue [1].",
        "The most-covered storylines are Nvidia's revenue [1].",
        "Nvidia's revenue: the most-covered storyline is the beat [1].",
        "Several outlets report Nvidia's revenue [1].",
        "Coverage today is dominated by ",
        "Coverage today is dominated by   ",
        "The most-covered storyline is",
        "The most-covered storyline is ",
        # The boundary: exactly one space, then the remainder.
        "Coverage today is dominated by  X",
        "Coverage today is dominated by \tX",
        "Coverage today is dominated by \nX",
        "Coverage today is dominated by \rX",
        "Coverage today is dominated by \r\nX",
        "Coverage today is dominated by    X",
        "Coverage today is dominated by \u00a0X",
        "Coverage today is dominated by\tX",
        "Coverage today is dominated by\nX",
        "Coverage today is dominated byX",
        "Coverage today is dominated by \t\n ",
        "The most-covered storyline is  X",
        "The most-covered storyline is \tX",
        "The most-covered storyline is \nX",
        "The most-covered storyline is \rX",
        "The most-covered storyline is \n",
        "The most-covered storylineis X",
        "THE MOST-COVERED STORYLINE IS X",
        "The Most-Covered Storyline Is X",
        "the most-covered storyline is X",
        "The most covered storyline is X",
        "The most-covered storyline was X",
        "The most-covered storyline is: X",
        "X. Coverage today is dominated by Y",
        " Coverage today is dominated by X",
    ],
)
def test_near_miss_wording_gets_no_exemption(sentence):
    assert review.split_exempt_framing(sentence, K3) == ("", sentence)


def test_no_other_protocol_exempts_anything():
    sentence = "Coverage today is dominated by Nvidia's revenue [1]."
    for protocol in (g2.PROVISIONAL_G2_PROTOCOL, review.PROVISIONAL_PROTOCOL):
        assert review.split_exempt_framing(sentence, protocol) == ("", sentence)


# ----------------------------------------------------------------------
# Compatibility: nothing but k3-g2-v1 is held to its rules
# ----------------------------------------------------------------------


def test_the_provisional_protocol_keeps_its_old_reading_rules(sampled, tmp_path):
    """Case-variant ids, free-text times and bare ``unsupported`` still read."""

    _, csv_path, manifest_path = sampled
    sheets = [
        fill(
            csv_path,
            tmp_path / "a.csv",
            "Alice",
            unsupported_first(1),
            notes=lambda i: "",
            reviewed_at="last tuesday",
        ),
        fill(csv_path, tmp_path / "b.csv", "alice", all_supported, notes=lambda i: ""),
    ]
    rows = read_rows(csv_path)
    adjudication = adjudicate(
        tmp_path / "adj.csv",
        [
            {
                "row_id": rows[0]["row_id"],
                "adjudicator_id": "ALICE",
                "adjudicated_at": "soon",
                "adjudication_notes": "",
            },
            {"row_id": rows[1]["row_id"], "final_verdict": "unsupported"},
        ],
    )
    result, card = score(manifest_path, sheets, adjudication)
    assert result.adjudication_state is State.RESOLVED
    assert card.protocol_ratified is False
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    # The agreed row's adjudication is ignored, exactly as before.
    assert card.positive_count == card.sentence_count
