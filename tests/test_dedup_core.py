"""Behavioural tests for the M2 deduplication core (issue #64).

The core is precision-first: it may miss duplicates, but it must never
merge two records a reader would call different.  These tests exercise
externally observable behaviour through :func:`nlp.dedup.deduplicate`
rather than helpers, and are deliberately sized to stay auditable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
import os
import subprocess
import sys
import textwrap

import pytest

from nlp.dedup.config import POLICY_FINGERPRINTS
from nlp.dedup.selection import cluster_fingerprint_for
from nlp.dedup import (
    DedupCapacityError,
    DedupConfig,
    DedupConfigError,
    DedupInputError,
    MatchReason,
    RawItem,
    deduplicate,
)

UTC = timezone.utc
BASE = datetime(2026, 3, 2, 13, 0, tzinfo=UTC)
UNIVERSE = ["TSLA", "NVDA", "AMD", "AAPL", "META"]


def config(**overrides: object) -> DedupConfig:
    settings: dict[str, object] = {"supported_tickers": UNIVERSE}
    settings.update(overrides)
    return DedupConfig(**settings)  # type: ignore[arg-type]


def at(hours: float) -> datetime:
    return BASE + timedelta(hours=hours)


def item(item_id: str, **overrides: object) -> RawItem:
    """Return a raw item with sensible Phase 0 defaults."""

    fields: dict[str, object] = {
        "ticker": "NVDA",
        "title": "Nvidia reports record data centre revenue",
        "description": "The chipmaker reported quarterly revenue above estimates.",
        "url": f"https://reuters.com/tech/{item_id}",
        "source": "Reuters",
        "published_at": BASE,
    }
    fields.update(overrides)
    return RawItem(item_id=item_id, **fields)  # type: ignore[arg-type]


def run(items, **overrides):
    return deduplicate(items, config=config(**overrides))


def members(result) -> list[tuple[str, ...]]:
    return [cluster.member_ids for cluster in result.clusters]


def signature(result) -> str:
    """A total, comparable description of a run."""

    return json.dumps(
        {
            "clusters": [
                {
                    "fingerprint": cluster.cluster_fingerprint,
                    "ticker": cluster.ticker,
                    "canonical": cluster.canonical_item_id,
                    "members": list(cluster.member_ids),
                    "reasons": [
                        (member.item_id, member.match_reason.value)
                        for member in cluster.members
                    ],
                    "outlet_count": cluster.outlet_count,
                    "content_hash": cluster.content_hash,
                }
                for cluster in result.clusters
            ],
            "quarantined": list(result.quarantined_item_ids),
            "vetoes": list(result.stats.veto_counts),
        },
        sort_keys=True,
    )


@pytest.fixture
def syndicated() -> list[RawItem]:
    """A Reuters original, its copies, and two unrelated stories."""

    return [
        item(
            "reuters-1",
            url="https://reuters.com/technology/nvidia-record-revenue-2026-03-02",
            published_at=at(0),
            provider_item_id="rtrs-9001",
        ),
        item(
            "reuters-1-refetch",
            url=(
                "https://reuters.com/technology/nvidia-record-revenue-2026-03-02"
                "?utm_source=newsletter"
            ),
            published_at=at(0.5),
            provider_item_id="rtrs-9001",
        ),
        item(
            "yahoo-1",
            title="Nvidia reports record data centre revenue - Reuters",
            description=None,
            url="https://finance.yahoo.com/news/nvidia-record-revenue.html",
            source="Yahoo Finance",
            published_at=at(0.75),
        ),
        item(
            "cnbc-1",
            title="UPDATE 1-Nvidia reports record data centre revenue",
            url="https://cnbc.com/2026/03/02/nvidia-record-revenue.html",
            source="CNBC",
            published_at=at(1.25),
        ),
        item(
            "ft-1",
            title="Nvidia names a new chief financial officer",
            description="The company appointed a new finance chief.",
            url="https://ft.com/content/nvidia-cfo",
            source="Financial Times",
            published_at=at(3),
        ),
        item(
            "wsj-1",
            title="Nvidia names a new chief operating officer",
            description="The company appointed a new operations chief.",
            url="https://wsj.com/articles/nvidia-coo",
            source="The Wall Street Journal",
            published_at=at(3.5),
        ),
    ]


# --------------------------------------------------------------------------
# AC-3 syndication: one cluster, every link, an outlet count
# --------------------------------------------------------------------------


def test_syndicated_copies_collapse_into_one_cluster(syndicated):
    result = run(syndicated)
    assert len(result.clusters) == 3
    cluster = result.cluster_by_member["reuters-1"]
    assert cluster.member_ids == (
        "reuters-1",
        "reuters-1-refetch",
        "yahoo-1",
        "cnbc-1",
    )
    assert cluster.canonical_item_id == "reuters-1"
    assert cluster.published_at == at(0)
    assert cluster.outlet_count == 3
    assert cluster.is_syndicated


def test_every_member_keeps_its_link_and_join_reason(syndicated):
    cluster = run(syndicated).cluster_by_member["reuters-1"]
    assert all(member.url for member in cluster.members)
    assert {member.item_id: member.match_reason for member in cluster.members} == {
        "reuters-1": MatchReason.CANONICAL,
        "reuters-1-refetch": MatchReason.PROVIDER_ITEM,
        "yahoo-1": MatchReason.EXACT_TITLE,
        "cnbc-1": MatchReason.EXACT_CONTENT,
    }
    assert {member.outlet for member in cluster.members} == {
        "reuters",
        "yahoo finance",
        "cnbc",
    }


def test_a_role_change_keeps_two_clusters_apart(syndicated):
    result = run(syndicated)
    assert (
        result.cluster_by_member["ft-1"].cluster_fingerprint
        != result.cluster_by_member["wsj-1"].cluster_fingerprint
    )


def test_the_earliest_member_is_canonical_with_deterministic_tie_breaks():
    result = run(
        [
            item("z-item", source="CNBC", url="https://cnbc.com/z", published_at=at(0)),
            item("a-item", published_at=at(0)),
        ]
    )
    # Equal timestamps: outlet then item id decide, so "cnbc" wins.
    assert result.clusters[0].canonical_item_id == "z-item"


def test_stats_describe_the_run(syndicated):
    stats = run(syndicated).stats
    assert stats.input_count == 6
    assert stats.cluster_count == 3
    assert stats.merged_cluster_count == 1
    assert stats.duplicate_member_count == 3
    assert stats.syndicated_cluster_count == 1
    assert stats.reason_count(MatchReason.PROVIDER_ITEM) == 1
    assert stats.reason_count(MatchReason.EXACT_TITLE) == 1
    assert stats.reason_count(MatchReason.EXACT_CONTENT) == 1
    assert stats.provider_conflict_count == 0


# --------------------------------------------------------------------------
# The compatibility gate: explicit disagreement vetoes every signal
# --------------------------------------------------------------------------


def test_identical_titles_with_opposite_descriptions_never_merge():
    # The reference case: the headline normalizes identically, the bodies
    # say the opposite thing.
    result = run(
        [
            item(
                "a",
                title="Quarterly results",
                description="Profit rose sharply.",
                published_at=at(0),
            ),
            item(
                "b",
                title="Quarterly results",
                description="Profit fell sharply.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.veto_count("description") >= 1


def test_the_gate_vetoes_an_exact_title_edge():
    result = run(
        [
            item(
                "a",
                title="Quarterly results",
                description="Revenue rose 5%.",
                published_at=at(0),
            ),
            item(
                "b",
                title="Quarterly results",
                description="Revenue rose 8%.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.veto_count("protected_expression") >= 1


def test_the_gate_is_applied_to_exact_content_edges(monkeypatch):
    """The content signal is gated like every other circumstantial one.

    A pair with an identical title *and* description has nothing left to
    contradict, so the veto cannot be provoked with data.  What matters is
    that the tier goes through the same gate: forcing the gate to object
    stops the merge, proving there is no ungated path.
    """

    from nlp.dedup import detection

    pair = [
        item("a", published_at=at(0)),
        item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
    ]
    calls: list[int] = []
    real = detection.combine

    def recording(left, right):
        calls.append(1)
        return real(left, right)

    monkeypatch.setattr(detection, "combine", recording)
    merged = run(pair)
    assert merged.clusters[0].match_reasons == (MatchReason.EXACT_CONTENT,)
    assert calls, "the exact-content edge never reached the gate"

    monkeypatch.setattr(detection, "combine", lambda left, right: (None, "forced"))
    assert len(run(pair).clusters) == 2


def test_unparseable_numeric_notation_blocks_text_identity_entirely():
    result = run(
        [
            item(
                "a",
                title="Margin improved ½ point",
                description="Same body.",
                published_at=at(0),
            ),
            item(
                "b",
                title="Margin improved ½ point",
                description="Same body.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.uncertain_title_count == 2


def test_the_gate_vetoes_a_url_edge():
    url = "https://reuters.com/markets/live-blog"
    result = run(
        [
            item(
                "a",
                title="Markets live blog",
                description="Stocks opened higher.",
                url=url,
                published_at=at(0),
            ),
            item(
                "b",
                title="Markets live blog",
                description="Stocks closed lower.",
                url=url,
                published_at=at(6),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.veto_count("description") >= 1


def test_the_gate_vetoes_a_minhash_candidate_edge():
    # Titles are close enough to be proposed as candidates, and the gate
    # refuses them before verification can matter.
    result = run(
        [
            item(
                "a",
                title="Nvidia beats revenue estimates",
                description="Beat by a wide margin.",
                published_at=at(0),
            ),
            item(
                "b",
                title="Nvidia beats revenue estimate",
                description="Missed by a wide margin.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.candidate_pair_count >= 1


def test_a_url_edge_is_vetoed_by_text_presence():
    url = "https://reuters.com/technology/one-story"
    result = run(
        [
            item("a", url=url, published_at=at(0)),
            item("b", title=None, description=None, url=url, published_at=at(0.5)),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.veto_count("text_presence") >= 1


def test_a_negation_disagreement_vetoes():
    result = run(
        [
            item(
                "a",
                title="Regulator ruling",
                description="The deal was approved.",
                published_at=at(0),
            ),
            item(
                "b",
                title="Regulator ruling",
                description="The deal was not approved.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2
    assert result.stats.veto_count("negation") >= 1


def test_missing_information_is_not_a_contradiction(syndicated):
    # yahoo-1 carries the same headline and no description at all; it still
    # joins the cluster.
    cluster = run(syndicated).cluster_by_member["yahoo-1"]
    assert "reuters-1" in cluster.member_ids


def test_the_gate_never_blocks_authoritative_provider_identity():
    # One feed asserting "this is my same item" outranks a body rewrite;
    # a genuine payload disagreement is handled by quarantine instead.
    result = run(
        [
            item(
                "a",
                url="https://reuters.com/one",
                published_at=at(0),
                provider_item_id="rtrs-1",
            ),
            item(
                "b",
                url="https://reuters.com/one",
                published_at=at(0.5),
                provider_item_id="rtrs-1",
            ),
        ]
    )
    assert members(result) == [("a", "b")]
    assert result.clusters[0].match_reasons == (MatchReason.PROVIDER_ITEM,)


# --------------------------------------------------------------------------
# Provider-identity conflicts
# --------------------------------------------------------------------------


def _conflicting(**overrides) -> list[RawItem]:
    url = "https://reuters.com/technology/one-story"
    base: dict[str, object] = {
        "url": url,
        "published_at": at(0.25),
        "provider_item_id": "rtrs-1",
    }
    return [
        item("a", url=url, published_at=at(0), provider_item_id="rtrs-1"),
        item("b", **{**base, **overrides}),
    ]


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("title", {"title": "Nvidia recalls a product line"}),
        ("description", {"description": "A completely different standfirst."}),
        ("url", {"url": "https://reuters.com/somewhere-else"}),
        ("text_presence", {"title": None, "description": None}),
        ("published_at", {"published_at": at(9)}),
        ("ticker", {"ticker": "AMD"}),
    ],
)
def test_provider_conflicts_are_detected_per_field(field, overrides):
    result = run(_conflicting(**overrides))
    assert result.stats.provider_conflict_count == 1
    conflict = result.provider_conflicts[0]
    assert conflict.provider_namespace == "reuters"
    assert conflict.provider_item_id == "rtrs-1"
    assert conflict.item_ids == ("a", "b")
    assert field in conflict.fields


def test_a_conflicted_identity_blocks_every_other_signal():
    # Same URL, same title, same body, minutes apart: without the conflict
    # these would merge three different ways.
    url = "https://reuters.com/technology/one-story"
    result = run(
        [
            item("a", url=url, published_at=at(0), provider_item_id="rtrs-1"),
            item(
                "b",
                url=url,
                published_at=at(0.25),
                provider_item_id="rtrs-1",
                title="Nvidia recalls a product line",
            ),
            item("c", url=url, published_at=at(0.5), provider_item_id="rtrs-1"),
        ]
    )
    assert result.quarantined_item_ids == ("a", "b", "c")
    assert members(result) == [("a",), ("b",), ("c",)]
    assert result.stats.reason_counts == ()


def test_quarantined_items_are_still_emitted():
    result = run(_conflicting(title="Nvidia recalls a product line"))
    assert len(result.clusters) == 2
    assert {cluster.canonical_item_id for cluster in result.clusters} == {"a", "b"}


def test_a_small_timestamp_disagreement_is_not_a_conflict():
    result = run(_conflicting(published_at=at(0.75)))
    assert result.stats.provider_conflict_count == 0
    assert members(result) == [("a", "b")]


# --------------------------------------------------------------------------
# Meaning-bearing differences never merge
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("The Who announce a new tour", "Who announce a new tour"),
        ("Acme Inc names a new chief", "Acme names a new chief"),
        ("Apple names a new CFO", "Apple names a new COO"),
        ("Broadcom raises full-year guidance", "Broadcom cuts full-year guidance"),
        ("The regulator approved the merger", "The regulator rejected the merger"),
        ("Meta posts a quarterly profit", "Meta posts a quarterly loss"),
        ("Nvidia acquires a chip startup", "Nvidia considers acquiring a startup"),
        ("Tesla recalls 5 million cars", "Tesla recalls 5 billion cars"),
        ("Tesla recalls five million cars", "Tesla recalls six million cars"),
        ("Margin fell -5% in the quarter", "Margin fell 5% in the quarter"),
        ("Margin fell ~5% in the quarter", "Margin fell 5% in the quarter"),
        ("Growth of 5-10% is expected", "Growth of 5 10% is expected"),
        ("A deal worth ₹5 crore", "A deal worth $5 crore"),
        ("AMD gains 5% today", "AMD gains $5 today"),
        ("Nvidia Q1 revenue rises 2%", "Nvidia Q2 revenue rises 1%"),
        ("Shares up 5% to $10", "Shares up 10% to $5"),
        ("Apple guides for 2025 growth", "Apple guides for 2026 growth"),
        ("Fed lifts rates 50 bps", "Fed lifts rates 25 bps"),
        ("Tesla recalls 10,000 cars", "Tesla recalls 100,000 cars"),
        ("Nvidia stock jumps on AI demand", "Nvidia shares rally on AI demand"),
        ("Отчёт Nvidia о рекордной выручке", "Отчёт Nvidia о падении выручки"),
        ("エヌビディアが決算を発表", "エヌビディアが人事を発表"),
    ],
)
def test_meaning_bearing_pairs_never_merge(left, right):
    result = run(
        [
            item("a", title=left, description="Details follow.", published_at=at(0)),
            item(
                "b",
                title=right,
                description="Details follow.",
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(1),
            ),
        ]
    )
    assert len(result.clusters) == 2


@pytest.mark.parametrize(
    "variant",
    [
        "UPDATE 2-Nvidia reports record data centre revenue",
        "Nvidia reports record data centre revenue - Reuters",
        "NVIDIA REPORTS RECORD DATA CENTRE REVENUE",
        "Nvidia reports “record” data centre revenue!",
    ],
)
def test_trivial_formatting_variants_merge(variant):
    result = run(
        [
            item("a", published_at=at(0)),
            item(
                "b",
                title=variant,
                source="CNBC",
                url="https://cnbc.com/b",
                published_at=at(2),
            ),
        ]
    )
    assert members(result) == [("a", "b")]


# --------------------------------------------------------------------------
# Windows, spans, and timestamps
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("gap_hours", "expected"), [(71.9, 1), (72.0, 1), (72.1, 2)])
def test_exact_titles_respect_the_seventy_two_hour_window(gap_hours, expected):
    result = run(
        [
            item("a", published_at=at(0)),
            item(
                "b", source="CNBC", url="https://cnbc.com/b", published_at=at(gap_hours)
            ),
        ]
    )
    assert len(result.clusters) == expected


def test_cluster_span_bounds_transitive_chains():
    result = run(
        [
            item("a", published_at=at(0)),
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(40)),
            item(
                "c",
                source="Financial Times",
                url="https://ft.com/c",
                published_at=at(80),
            ),
        ]
    )
    assert members(result) == [("a", "b"), ("c",)]


def test_recurring_headlines_stay_distinct_across_quarters():
    title = "Tesla reports third-quarter vehicle deliveries"
    result = run(
        [
            item("a", ticker="TSLA", title=title, published_at=at(0)),
            item(
                "b",
                ticker="TSLA",
                title=title,
                url="https://reuters.com/b",
                published_at=at(24 * 92),
            ),
        ]
    )
    assert len(result.clusters) == 2


def test_undated_records_never_merge_on_text():
    result = run(
        [
            item("a", published_at=None),
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=None),
        ]
    )
    assert len(result.clusters) == 2


def test_dst_shifts_are_honoured_as_real_offsets():
    zoneinfo = pytest.importorskip("zoneinfo")
    eastern = zoneinfo.ZoneInfo("America/New_York")
    before = datetime(2026, 10, 30, 12, 0, tzinfo=eastern)
    after = datetime(2026, 11, 2, 12, 0, tzinfo=eastern)
    # Wall-clock arithmetic says 72 hours; the real gap is 73.
    assert after.astimezone(UTC) - before.astimezone(UTC) == timedelta(hours=73)
    result = run(
        [
            item("a", published_at=before),
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=after),
        ]
    )
    assert len(result.clusters) == 2


@pytest.mark.parametrize(
    "value", [datetime(2026, 1, 2, 14, 0), "2026-01-02T14:00:00", "2026-01-02"]
)
def test_naive_timestamps_are_rejected(value):
    with pytest.raises(DedupInputError, match="timezone offset"):
        run([item("a", published_at=value)])


@pytest.mark.parametrize(
    "value", [datetime(1970, 1, 1, tzinfo=UTC), datetime(2999, 1, 1, tzinfo=UTC)]
)
def test_implausible_timestamps_are_rejected(value):
    with pytest.raises(DedupInputError, match="plausible range"):
        run([item("a", published_at=value)])


# --------------------------------------------------------------------------
# Ticker contract
# --------------------------------------------------------------------------


def test_identical_headlines_never_merge_across_tickers():
    result = run(
        [
            item("a", ticker="NVDA", published_at=at(0)),
            item("b", ticker="AMD", url="https://reuters.com/b", published_at=at(1)),
        ]
    )
    assert len(result.clusters) == 2
    assert {cluster.ticker for cluster in result.clusters} == {"NVDA", "AMD"}


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("GOOG", "outside the supported universe"),
        (None, "ticker is required"),
        ("", "ticker is required"),
        ("   ", "ticker is required"),
        ("NVDA,AMD", "valid symbol"),
        ("NVDA AMD", "single symbol"),
        ("Tesla Inc", "single symbol"),
        ("1NVDA", "valid symbol"),
    ],
)
def test_unusable_tickers_are_rejected(value, match):
    with pytest.raises(DedupInputError, match=match):
        run([item("a", ticker=value, published_at=at(0))])


def test_the_supported_universe_is_required_and_immutable():
    with pytest.raises(TypeError):
        DedupConfig()  # type: ignore[call-arg]
    settings = config()
    assert settings.ticker_universe == frozenset(UNIVERSE)
    with pytest.raises(Exception):
        settings.supported_tickers = frozenset({"NVDA"})  # type: ignore[misc]


@pytest.mark.parametrize("value", ["NVDA", b"NVDA", [], [""], ["NVDA,AMD"], [1], 5])
def test_invalid_universes_are_rejected(value):
    with pytest.raises(DedupConfigError):
        DedupConfig(supported_tickers=value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Capacity: fail fast, never partial
# --------------------------------------------------------------------------


def _many(count: int, ticker: str = "NVDA") -> list[RawItem]:
    return [
        item(
            f"{ticker}-{index:03d}",
            ticker=ticker,
            title=f"{ticker} market note number {index}",
            url=f"https://reuters.com/{ticker}-{index}",
            published_at=at(index * 0.01),
        )
        for index in range(count)
    ]


def test_an_oversized_partition_fails_fast_with_scope():
    limit = 25
    with pytest.raises(DedupCapacityError) as caught:
        run(_many(limit + 1), max_partition_items=limit)
    error = caught.value
    assert error.ticker == "NVDA"
    assert error.item_count == limit + 1
    assert error.limit == limit
    assert "max_partition_items=25" in str(error)


def test_a_partition_at_the_limit_is_processed_completely():
    limit = 25
    result = run(_many(limit), max_partition_items=limit)
    assert result.stats.cluster_count == limit


def test_capacity_is_measured_per_ticker_not_per_batch():
    limit = 25
    corpus = _many(limit, "NVDA") + _many(limit, "AMD")
    result = run(corpus, max_partition_items=limit)
    assert result.stats.input_count == 2 * limit


def test_the_partition_limit_changes_the_configuration_fingerprint():
    assert (
        config(max_partition_items=25).fingerprint()
        != config(max_partition_items=26).fingerprint()
    )


# --------------------------------------------------------------------------
# Determinism, purity, and identity
# --------------------------------------------------------------------------


def test_output_is_invariant_under_every_input_permutation():
    core = [
        item("a", published_at=at(0)),
        item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
        item(
            "c",
            title="Nvidia names a new chief financial officer",
            description="A finance chief was appointed.",
            source="Financial Times",
            url="https://ft.com/c",
            published_at=at(2),
        ),
        item("d", ticker="AMD", url="https://reuters.com/d", published_at=at(3)),
        item(
            "e",
            source="The Wall Street Journal",
            url="https://wsj.com/e",
            published_at=at(100),
        ),
    ]
    expected = signature(run(core))
    for permutation in itertools.permutations(core):
        assert signature(run(list(permutation))) == expected


def test_an_unrelated_record_never_changes_whether_two_others_merge():
    left = item(
        "a",
        title="Nvidia reports record revenue - Apple",
        description="The chipmaker beat estimates.",
        published_at=at(0),
    )
    right = item(
        "b",
        title="Nvidia reports record revenue",
        description="The chipmaker beat estimates.",
        source="CNBC",
        url="https://cnbc.com/b",
        published_at=at(1),
    )
    unrelated = item(
        "c",
        ticker="TSLA",
        title="Tesla opens a delivery centre",
        description="Regional coverage.",
        source="Apple",
        url="https://example.com/c",
        published_at=at(2),
    )
    without = members(run([left, right]))
    with_extra = [
        cluster
        for cluster in members(run([left, right, unrelated]))
        if cluster != ("c",)
    ]
    assert with_extra == without


def test_mutating_the_callers_input_cannot_reach_the_result():
    items = [
        item("a", published_at=at(0)),
        item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
    ]
    result = run(items)
    before = signature(result)
    items.append(item("c", url="https://ft.com/c", published_at=at(2)))
    items.clear()
    assert signature(result) == before
    assert result.clusters[0].member_ids == ("a", "b")


def test_a_configuration_change_changes_every_content_hash(syndicated):
    default = run(syndicated)
    tightened = run(syndicated, near_exact_window_hours=1)
    assert default.config_fingerprint != tightened.config_fingerprint
    assert not {cluster.content_hash for cluster in default.clusters} & {
        cluster.content_hash for cluster in tightened.clusters
    }


def test_replaying_an_unchanged_run_reproduces_every_hash(syndicated):
    assert signature(run(syndicated)) == signature(run(list(reversed(syndicated))))


def test_fingerprints_are_stable_across_processes_and_hash_seeds():
    script = textwrap.dedent(
        """
        import json
        from datetime import datetime, timezone
        from nlp.dedup import DedupConfig, RawItem, deduplicate

        base = datetime(2026, 3, 2, 13, 0, tzinfo=timezone.utc)
        items = [
            RawItem(item_id="a", ticker="NVDA", title="Nvidia reports revenue",
                    description="Above estimates.", url="https://reuters.com/a",
                    source="Reuters", published_at=base),
            RawItem(item_id="b", ticker="NVDA",
                    title="UPDATE 1-Nvidia reports revenue",
                    description="Above estimates.", url="https://cnbc.com/b",
                    source="CNBC", published_at=base),
        ]
        result = deduplicate(items, config=DedupConfig(supported_tickers=["NVDA"]))
        print(json.dumps(
            [(c.cluster_fingerprint, c.content_hash) for c in result.clusters]
        ))
        """
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(outputs) == 1


def test_the_core_loads_no_model_and_touches_no_resource():
    """No embedding import, and no file, socket, or database access."""

    script = textwrap.dedent(
        """
        import sys
        from datetime import datetime, timezone
        from nlp.dedup import DedupConfig, RawItem, deduplicate

        base = datetime(2026, 3, 2, 13, 0, tzinfo=timezone.utc)
        config = DedupConfig(supported_tickers=["NVDA"])
        items = [
            RawItem(item_id=str(index), ticker="NVDA",
                    title="Nvidia reports record revenue",
                    description="Above estimates.",
                    url=f"https://reuters.com/{index}",
                    source="Reuters", published_at=base)
            for index in range(4)
        ]
        events = []
        sys.addaudithook(
            lambda name, args: events.append(name)
            if name == "open"
            or name.startswith(("socket.", "sqlite3.", "urllib.", "subprocess."))
            else None
        )
        deduplicate(items, config=config)
        banned = [
            name for name in sys.modules
            if name == "nlp.embeddings"
            or name.split(".")[0] in {"sentence_transformers", "torch",
                                      "transformers", "yaml", "sqlite3"}
        ]
        print(sorted(set(events)), banned)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[] []"


def test_cluster_fingerprints_are_order_independent_and_membership_sensitive():
    two = run(
        [
            item("a", published_at=at(0)),
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
        ]
    )
    reordered = run(
        [
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
            item("a", published_at=at(0)),
        ]
    )
    three = run(
        [
            item("a", published_at=at(0)),
            item("b", source="CNBC", url="https://cnbc.com/b", published_at=at(1)),
            item(
                "c",
                source="Financial Times",
                url="https://ft.com/c",
                published_at=at(2),
            ),
        ]
    )
    assert (
        two.clusters[0].cluster_fingerprint == reordered.clusters[0].cluster_fingerprint
    )
    assert two.clusters[0].cluster_fingerprint != three.clusters[0].cluster_fingerprint
    assert len(two.clusters[0].cluster_fingerprint) == 64


def test_item_ids_containing_separators_cannot_collide():
    left = run([item("a\x1fb", published_at=at(0))])
    right = run([item("a", published_at=at(0))])
    assert left.clusters[0].cluster_fingerprint != right.clusters[0].cluster_fingerprint


def test_no_durable_story_identifier_is_exposed(syndicated):
    cluster = run(syndicated).clusters[0]
    assert not hasattr(cluster, "story_id")
    assert not hasattr(cluster, "id")


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------


def test_empty_input_is_an_empty_result():
    result = run([])
    assert result.clusters == ()
    assert result.stats.input_count == 0
    assert result.provider_conflicts == ()


@pytest.mark.parametrize("value", ["items", b"items", 5, None, {"a": 1}])
def test_non_sequence_input_is_rejected(value):
    with pytest.raises(DedupInputError, match="sequence"):
        run(value)


def test_duplicate_item_ids_are_rejected():
    with pytest.raises(DedupInputError, match="duplicate item_id: a"):
        run([item("a", published_at=at(0)), item("a", published_at=at(1))])


def test_a_non_config_is_rejected():
    with pytest.raises(DedupInputError, match="DedupConfig"):
        deduplicate([], config={"url_window_hours": 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        RawItem(item_id="1", ticker="NVDA", title=42),
        RawItem(item_id="1", ticker="NVDA", description=[]),
        RawItem(item_id="1", ticker="NVDA", url=object()),
        RawItem(item_id="1", ticker="NVDA", source=b"reuters"),
        RawItem(item_id="1", ticker="NVDA", provider_item_id=7),
        RawItem(item_id="", ticker="NVDA"),
        {"item_id": "1"},
    ],
)
def test_malformed_records_are_rejected(bad):
    with pytest.raises(DedupInputError):
        run([bad])


def test_records_without_usable_text_survive_as_their_own_clusters():
    result = run(
        [
            RawItem(item_id="a", ticker="NVDA", published_at=at(0)),
            RawItem(item_id="b", ticker="NVDA", published_at=at(1)),
        ]
    )
    assert members(result) == [("a",), ("b",)]


def test_hostile_text_is_normalized_not_executed():
    result = run(
        [
            item(
                "a",
                title="<script>alert('x')</script> Nvidia &amp; AMD \x00 rally",
                description="‮RTL‬ control characters",
                published_at=at(0),
            )
        ]
    )
    cluster = result.clusters[0]
    assert "\x00" not in cluster.canonical_title
    assert "Nvidia & AMD" in cluster.canonical_title


# --------------------------------------------------------------------------
# Cluster-wide compatibility: a sparse record cannot bridge contradictions
# --------------------------------------------------------------------------


def _bridge(left_description, right_description, **shared):
    """A: known value, B: sparse, C: contradicting value — same title."""

    fields: dict[str, object] = {"title": "Quarterly results"}
    fields.update(shared)
    return [
        item("a", description=left_description, published_at=at(0), **fields),
        item(
            "b",
            description=None,
            source="CNBC",
            url="https://cnbc.com/b",
            published_at=at(1),
            **fields,
        ),
        item(
            "c",
            description=right_description,
            source="Financial Times",
            url="https://ft.com/c",
            published_at=at(2),
            **fields,
        ),
    ]


@pytest.mark.parametrize(
    ("case", "left", "right"),
    [
        ("opposite claims", "Profit rose sharply.", "Profit fell sharply."),
        ("numeric", "Revenue rose 5%.", "Revenue rose 8%."),
        ("direction", "Revenue rose 5%.", "Revenue fell 5%."),
        ("quarter", "Q1 revenue rose 5%.", "Q2 revenue rose 5%."),
        ("year", "Guidance for 2025.", "Guidance for 2026."),
        ("negation", "The deal was approved.", "The deal was not approved."),
    ],
)
def test_a_sparse_record_cannot_bridge_contradictory_records(case, left, right):
    result = run(_bridge(left, right))
    by_member = result.cluster_by_member
    assert (
        by_member["a"].cluster_fingerprint != by_member["c"].cluster_fingerprint
    ), case
    # B may join at most one side, never both.
    assert len(by_member["b"].member_ids) <= 2


def test_the_url_path_cannot_be_bridged_either():
    url = "https://reuters.com/markets/live-blog"
    result = run(
        [
            item(
                "a",
                title="Live blog",
                description="Stocks rose.",
                url=url,
                published_at=at(0),
            ),
            item(
                "b",
                title="Live blog",
                description=None,
                url=url,
                source="CNBC",
                published_at=at(1),
            ),
            item(
                "c",
                title="Live blog",
                description="Stocks fell.",
                url=url,
                source="Financial Times",
                published_at=at(2),
            ),
        ]
    )
    by_member = result.cluster_by_member
    assert by_member["a"].cluster_fingerprint != by_member["c"].cluster_fingerprint


def test_multiple_sparse_intermediates_still_cannot_bridge():
    records = [
        item(
            "a",
            title="Quarterly results",
            description="Profit rose sharply.",
            published_at=at(0),
        ),
        item(
            "b1",
            title="Quarterly results",
            description=None,
            source="CNBC",
            url="https://cnbc.com/b1",
            published_at=at(1),
        ),
        item(
            "b2",
            title="Quarterly results",
            description=None,
            source="Financial Times",
            url="https://ft.com/b2",
            published_at=at(2),
        ),
        item(
            "b3",
            title="Quarterly results",
            description=None,
            source="The Wall Street Journal",
            url="https://wsj.com/b3",
            published_at=at(3),
        ),
        item(
            "c",
            title="Quarterly results",
            description="Profit fell sharply.",
            source="Bloomberg",
            url="https://bloomberg.com/c",
            published_at=at(4),
        ),
    ]
    by_member = run(records).cluster_by_member
    assert by_member["a"].cluster_fingerprint != by_member["c"].cluster_fingerprint
    assert "c" not in by_member["a"].member_ids


def test_the_bridge_verdict_is_permutation_invariant():
    records = _bridge("Profit rose sharply.", "Profit fell sharply.")
    expected = signature(run(records))
    for permutation in itertools.permutations(records):
        assert signature(run(list(permutation))) == expected


def test_a_legitimate_sparse_bridge_still_merges_all_three():
    records = _bridge("Profit rose sharply.", "Profit rose sharply.")
    result = run(records)
    assert members(result) == [("a", "b", "c")]
    assert result.clusters[0].outlet_count == 3


def test_a_late_sparse_record_cannot_collapse_two_existing_clusters():
    # The contradictory pair first: two singleton clusters.
    contradictory = [
        item(
            "a",
            title="Quarterly results",
            description="Profit rose sharply.",
            published_at=at(0),
        ),
        item(
            "c",
            title="Quarterly results",
            description="Profit fell sharply.",
            source="Financial Times",
            url="https://ft.com/c",
            published_at=at(2),
        ),
    ]
    assert len(run(contradictory).clusters) == 2

    # Now the sparse record arrives and is processed with them.
    late = contradictory + [
        item(
            "b",
            title="Quarterly results",
            description=None,
            source="CNBC",
            url="https://cnbc.com/b",
            published_at=at(3),
        )
    ]
    result = run(late)
    by_member = result.cluster_by_member
    assert by_member["a"].cluster_fingerprint != by_member["c"].cluster_fingerprint
    assert len(result.clusters) == 2


# --------------------------------------------------------------------------
# Candidate generation must not depend on sort adjacency
# --------------------------------------------------------------------------


def test_a_compatible_pair_is_found_despite_an_incompatible_record_between():
    # A and C agree; B sorts between them in time and contradicts both.
    # Adjacent-only chaining would propose A-B and B-C, both vetoed, and
    # never consider A-C.
    records = [
        item(
            "a",
            title="Quarterly results",
            description="Profit rose sharply.",
            published_at=at(0),
        ),
        item(
            "b",
            title="Quarterly results",
            description="Profit fell sharply.",
            source="CNBC",
            url="https://cnbc.com/b",
            published_at=at(1),
        ),
        item(
            "c",
            title="Quarterly results",
            description="Profit rose sharply.",
            source="Financial Times",
            url="https://ft.com/c",
            published_at=at(2),
        ),
    ]
    result = run(records)
    assert members(result) == [("a", "c"), ("b",)]


def test_the_url_bucket_also_considers_non_adjacent_pairs():
    url = "https://reuters.com/markets/live-blog"
    records = [
        item(
            "a",
            title="Live blog",
            description="Stocks rose.",
            url=url,
            published_at=at(0),
        ),
        item(
            "b",
            title="Live blog",
            description="Stocks fell.",
            url=url,
            source="CNBC",
            published_at=at(1),
        ),
        item(
            "c",
            title="Live blog",
            description="Stocks rose.",
            url=url,
            source="Financial Times",
            published_at=at(2),
        ),
    ]
    result = run(records)
    assert members(result) == [("a", "c"), ("b",)]
    # The URL signal is the strongest applicable one, so it is what merged
    # the non-adjacent pair.
    assert MatchReason.CANONICAL_URL in result.clusters[0].match_reasons


def test_non_adjacent_recovery_is_permutation_invariant():
    records = [
        item(
            "a",
            title="Quarterly results",
            description="Profit rose sharply.",
            published_at=at(0),
        ),
        item(
            "b",
            title="Quarterly results",
            description="Profit fell sharply.",
            source="CNBC",
            url="https://cnbc.com/b",
            published_at=at(1),
        ),
        item(
            "c",
            title="Quarterly results",
            description="Profit rose sharply.",
            source="Financial Times",
            url="https://ft.com/c",
            published_at=at(2),
        ),
    ]
    expected = signature(run(records))
    for permutation in itertools.permutations(records):
        assert signature(run(list(permutation))) == expected


# --------------------------------------------------------------------------
# Provider identity: unified policy and conservative namespaces
# --------------------------------------------------------------------------


def test_uncertain_numeric_titles_under_one_provider_id_are_quarantined():
    # Neither title yields an ordinary key; that is not permission to merge.
    result = run(
        [
            item(
                "a",
                title="Profit ½ higher",
                description=None,
                url="https://reuters.com/a",
                published_at=at(0),
                provider_item_id="rtrs-1",
            ),
            item(
                "b",
                title="Revenue ⅓ lower",
                description=None,
                url="https://reuters.com/a",
                published_at=at(0.25),
                provider_item_id="rtrs-1",
            ),
        ]
    )
    assert result.stats.provider_conflict_count == 1
    assert "uncertain_numeric_notation" in result.provider_conflicts[0].fields
    assert result.quarantined_item_ids == ("a", "b")
    assert len(result.clusters) == 2


def test_legal_suffixes_do_not_collapse_distinct_providers():
    # "Acme Inc" and "Acme LLC" are different companies; sharing an item id
    # must not merge them, and must not be reported as a conflict either.
    result = run(
        [
            item(
                "a",
                source="Acme Inc",
                url="https://acme-inc.example/a",
                published_at=at(0),
                provider_item_id="1",
            ),
            item(
                "b",
                source="Acme LLC",
                title="A completely different story",
                description="Unrelated.",
                url="https://acme-llc.example/b",
                published_at=at(1),
                provider_item_id="1",
            ),
        ]
    )
    assert result.stats.provider_conflict_count == 0
    assert len(result.clusters) == 2
    assert result.stats.reason_count(MatchReason.PROVIDER_ITEM) == 0


def test_a_three_record_group_is_quarantined_when_only_one_pair_conflicts():
    records = [
        item(
            "a",
            url="https://reuters.com/one",
            published_at=at(0),
            provider_item_id="rtrs-1",
        ),
        item(
            "b",
            url="https://reuters.com/one",
            published_at=at(0.25),
            provider_item_id="rtrs-1",
        ),
        item(
            "c",
            url="https://reuters.com/one",
            published_at=at(0.5),
            provider_item_id="rtrs-1",
            description="A contradicting standfirst.",
        ),
    ]
    result = run(records)
    assert result.stats.provider_conflict_count == 1
    assert result.provider_conflicts[0].item_ids == ("a", "b", "c")
    assert result.quarantined_item_ids == ("a", "b", "c")
    assert members(result) == [("a",), ("b",), ("c",)]


def test_provider_conflict_detection_is_permutation_invariant():
    records = [
        item(
            "a",
            url="https://reuters.com/one",
            published_at=at(0),
            provider_item_id="rtrs-1",
        ),
        item(
            "b",
            url="https://reuters.com/one",
            published_at=at(0.25),
            provider_item_id="rtrs-1",
        ),
        item(
            "c",
            url="https://reuters.com/one",
            published_at=at(0.5),
            provider_item_id="rtrs-1",
            description="A contradicting standfirst.",
        ),
    ]
    expected = signature(run(records))
    for permutation in itertools.permutations(records):
        assert signature(run(list(permutation))) == expected


# --------------------------------------------------------------------------
# The configured URL window is enforced
# --------------------------------------------------------------------------

_REUSED_URL = "https://reuters.com/markets/quarterly-results"


def _untitled_pair(gap: timedelta) -> list[RawItem]:
    return [
        item("a", title=None, description=None, url=_REUSED_URL, published_at=at(0)),
        item(
            "b",
            title=None,
            description=None,
            url=_REUSED_URL,
            source="CNBC",
            published_at=BASE + gap,
        ),
    ]


def test_a_reused_url_does_not_merge_across_one_hundred_and_eighty_days():
    result = run(
        [
            item("a", url=_REUSED_URL, published_at=at(0)),
            item("b", url=_REUSED_URL, source="CNBC", published_at=at(24 * 180)),
        ]
    )
    assert len(result.clusters) == 2


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (timedelta(hours=71, minutes=59), 1),
        (timedelta(hours=72), 1),
        (timedelta(hours=72, microseconds=1), 2),
        (timedelta(hours=73), 2),
    ],
)
def test_the_url_window_boundary_is_exact(gap, expected):
    assert len(run(_untitled_pair(gap)).clusters) == expected


def test_undated_records_do_not_qualify_for_a_url_match():
    result = run(
        [
            item("a", url=_REUSED_URL, published_at=None),
            item("b", url=_REUSED_URL, source="CNBC", published_at=None),
        ]
    )
    assert len(result.clusters) == 2


def test_undated_records_at_one_url_still_merge_on_provider_identity():
    result = run(
        [
            item("a", url=_REUSED_URL, published_at=None, provider_item_id="rtrs-1"),
            item("b", url=_REUSED_URL, published_at=None, provider_item_id="rtrs-1"),
        ]
    )
    assert members(result) == [("a", "b")]
    assert result.clusters[0].match_reasons == (MatchReason.PROVIDER_ITEM,)


def test_url_plus_compatible_text_still_obeys_the_window():
    # Identical headline and body: agreement is total, and the window still
    # decides.
    result = run(
        [
            item("a", url=_REUSED_URL, published_at=at(0)),
            item("b", url=_REUSED_URL, source="CNBC", published_at=at(73)),
        ]
    )
    assert len(result.clusters) == 2


def test_a_mixed_provider_and_url_cluster_keeps_full_span_behaviour():
    # Provider identity merges a and b unconditionally; c shares a's URL but
    # would stretch the cluster past the URL window.
    result = run(
        [
            item("a", url=_REUSED_URL, published_at=at(0), provider_item_id="rtrs-1"),
            item("b", url=_REUSED_URL, published_at=at(0.5), provider_item_id="rtrs-1"),
            item("c", url=_REUSED_URL, source="CNBC", published_at=at(80)),
        ]
    )
    assert members(result) == [("a", "b"), ("c",)]


# --------------------------------------------------------------------------
# Ticker universes and policy fingerprinting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "universe",
    [["NVDA"], ["NVDA", "AMD"], ["BRK.B"], ["RDS-A"], ("nvda", "amd"), {"NVDA"}],
)
def test_valid_ticker_universes_are_accepted(universe):
    assert DedupConfig(supported_tickers=universe).ticker_universe


@pytest.mark.parametrize(
    "universe",
    [
        ["NVDA", "NVDA"],
        ["NVDA", "nvda"],
        ["NVDA", " NVDA "],
        ["$AAPL"],
        ["A@B"],
        ["A..B"],
        ["1NVDA"],
        [""],
        ["   "],
        ["TOOLONGSYMBOL"],
        ["NVDA,AMD"],
        ["NVDA AMD"],
    ],
)
def test_invalid_ticker_universes_are_rejected(universe):
    with pytest.raises(DedupConfigError):
        DedupConfig(supported_tickers=universe)


def test_a_configured_symbol_is_always_one_a_record_could_satisfy():
    from nlp.dedup.normalization import normalize_ticker

    for symbol in DedupConfig(supported_tickers=["BRK.B", "RDS-A"]).ticker_universe:
        assert normalize_ticker(symbol) == symbol


@pytest.mark.parametrize("policy", sorted(POLICY_FINGERPRINTS))
def test_every_static_policy_contributes_to_the_fingerprint(policy, monkeypatch):
    # Assembly is tested directly rather than by mutating the policy lists,
    # which would leak into other tests through module state.
    baseline = config().fingerprint()
    monkeypatch.setitem(POLICY_FINGERPRINTS, policy, lambda: f"changed-{policy}")
    assert config().fingerprint() != baseline


@pytest.mark.parametrize(
    "overrides",
    [
        {"content_window_hours": 80},
        {"exact_title_window_hours": 71},
        {"near_exact_window_hours": 35},
        {"url_window_hours": 71},
        {"provider_timestamp_tolerance_hours": 2},
        {"minhash_permutations": 64},
        {"minhash_shingle_size": 4},
        {"minhash_seed": "other"},
        {"candidate_min_similarity": 0.5},
        {"max_partition_items": 249},
        {"supported_tickers": ["NVDA"]},
    ],
)
def test_every_setting_contributes_to_the_fingerprint(overrides):
    assert config(**overrides).fingerprint() != config().fingerprint()


# --------------------------------------------------------------------------
# Internal helper hardening
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "member_ids"),
    [
        ("NVDA", []),
        ("NVDA", ["a", "a"]),
        ("NVDA", [""]),
        ("NVDA", ["  "]),
        ("NVDA", [None]),
        ("", ["a"]),
        ("   ", ["a"]),
    ],
)
def test_cluster_fingerprint_rejects_unusable_input(ticker, member_ids):
    with pytest.raises(DedupInputError):
        cluster_fingerprint_for(ticker, member_ids)


def test_cluster_fingerprint_is_stable_and_full_width():
    digest = cluster_fingerprint_for("NVDA", ["b", "a"])
    assert digest == cluster_fingerprint_for("NVDA", ["a", "b"])
    assert len(digest) == 64
