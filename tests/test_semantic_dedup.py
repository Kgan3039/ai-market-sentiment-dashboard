"""Tests for the M3 semantic deduplication stage (issue #70).

No test here loads a model or touches the network: the encoder is injected
everywhere, and the fake is a deterministic function of the text. Where a
test needs a specific cosine value it constructs the vectors directly, so
what is under test is the *predicate*, not the encoder's opinion.

The stage is precision-first. A missed rewrite costs a duplicate card; a
false merge attributes one event's coverage to another, so most of what
follows is about refusing to merge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from nlp.dedup import DedupConfig, RawItem, deduplicate
from nlp.semdedup import (
    SemanticDedupCapacityError,
    SemanticDedupConfig,
    SemanticDedupConfigError,
    SemanticDedupEncodingError,
    SemanticDedupInputError,
    SourceLink,
    StoryInput,
    merge_semantic_duplicates,
    stories_from_dedup,
    story_fingerprint_for,
    story_text,
)
from nlp.semdedup.evidence import (
    contradiction,
    numeric_signature,
    policy_fingerprint,
    roles,
    summarize,
    temporal_markers,
)

UTC = timezone.utc
BASE = datetime(2026, 3, 2, 13, 0, tzinfo=UTC)
UNIVERSE = ["TSLA", "NVDA", "AMD", "AAPL", "META"]
REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    """Deterministic encoder driven by an explicit text-to-vector table.

    Unlisted text gets a stable but unrelated vector, so a test that forgets
    to declare a pair's similarity gets a low score rather than an accidental
    merge.
    """

    model_name = "fake-encoder"
    model_revision = "v1"

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [self.vectors.get(text, self._fallback(text)) for text in texts]

    @staticmethod
    def _fallback(text: str) -> list[float]:
        seed = sum((index + 1) * ord(char) for index, char in enumerate(text))
        angle = (seed % 997) / 997 * math.pi / 2
        return [math.cos(angle), math.sin(angle), 0.0]


def unit(angle_degrees: float) -> list[float]:
    """A unit vector at a given angle; cosine is cos(difference)."""

    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians), 0.0]


def config(**overrides) -> SemanticDedupConfig:
    settings = {"supported_tickers": UNIVERSE}
    settings.update(overrides)
    return SemanticDedupConfig(**settings)


def story(key: str, title: str, **overrides) -> StoryInput:
    fields = {
        "ticker": "NVDA",
        "description": None,
        "published_at": BASE,
        "outlets": ("reuters",),
        "member_ids": (f"{key}-1",),
        "source_links": (
            SourceLink(item_id=f"{key}-1", outlet="reuters", url=f"https://x/{key}"),
        ),
    }
    fields.update(overrides)
    return StoryInput(story_key=key, title=title, **fields)


def run(stories, encoder=None, **overrides):
    return merge_semantic_duplicates(
        stories, config=config(**overrides), encoder=encoder or FakeEncoder()
    )


def grouped(result) -> list[tuple[str, ...]]:
    return [story.member_story_keys for story in result.stories]


def encoder_for(*pairs: tuple[str, float]) -> FakeEncoder:
    """Build an encoder placing each title at a given angle in degrees."""

    return FakeEncoder({title: unit(angle) for title, angle in pairs})


# --------------------------------------------------------------------------
# Same-story rewrites merge
# --------------------------------------------------------------------------


def test_two_rewrites_of_one_event_merge():
    encoder = encoder_for(
        ("Nvidia posts record quarterly sales", 0.0),
        ("Quarterly sales at the chipmaker hit an all-time high", 10.0),
    )
    result = run(
        [
            story("s1", "Nvidia posts record quarterly sales"),
            story(
                "s2",
                "Quarterly sales at the chipmaker hit an all-time high",
                published_at=BASE + timedelta(hours=2),
                outlets=("cnbc",),
            ),
        ],
        encoder,
    )

    assert grouped(result) == [("s1", "s2")]
    assert result.stats.merged_story_count == 1
    assert result.stats.collapsed_story_count == 1
    assert result.stories[0].merges[0].similarity == pytest.approx(
        math.cos(math.radians(10))
    )


def test_a_merged_story_keeps_every_member_and_source_link():
    encoder = encoder_for(("first headline", 0.0), ("a second telling", 5.0))
    result = run(
        [
            story(
                "s1",
                "first headline",
                member_ids=("a1", "a2"),
                outlets=("reuters",),
                source_links=(
                    SourceLink("a1", "reuters", "https://r/1"),
                    SourceLink("a2", "yahoo", "https://y/2"),
                ),
            ),
            story(
                "s2",
                "a second telling",
                member_ids=("b1",),
                outlets=("cnbc",),
                published_at=BASE + timedelta(hours=1),
                source_links=(SourceLink("b1", "cnbc", "https://c/1"),),
            ),
        ],
        encoder,
    )
    merged = result.stories[0]

    assert merged.member_ids == ("a1", "a2", "b1")
    assert merged.outlet_count == 3
    assert {link.item_id for link in merged.source_links} == {"a1", "a2", "b1"}
    assert merged.is_syndicated


def test_the_earliest_member_is_canonical():
    encoder = encoder_for(("later telling", 0.0), ("earlier telling", 5.0))
    result = run(
        [
            story("late", "later telling", published_at=BASE + timedelta(hours=3)),
            story("early", "earlier telling", published_at=BASE),
        ],
        encoder,
    )

    assert result.stories[0].canonical_story_key == "early"
    assert result.stories[0].canonical_title == "earlier telling"
    assert result.stories[0].member_story_keys == ("early", "late")


# --------------------------------------------------------------------------
# Hard negatives: the guards outrank the score
# --------------------------------------------------------------------------


def near_identical(left_title: str, right_title: str, **overrides):
    """Two titles the encoder considers essentially identical."""

    encoder = encoder_for((left_title, 0.0), (right_title, 0.5))
    return run(
        [
            story("s1", left_title, **overrides.pop("left", {})),
            story(
                "s2",
                right_title,
                published_at=BASE + timedelta(hours=1),
                **overrides.pop("right", {}),
            ),
        ],
        encoder,
        **overrides,
    )


@pytest.mark.parametrize(
    "left,right,reason",
    [
        (
            "Tesla recalls 12,000 vehicles over a wiper fault",
            "Tesla recalls 120,000 vehicles over a wiper fault",
            "numeric_disagreement",
        ),
        (
            "Apple ships 5 million units of the new headset",
            "Apple ships 5 billion units of the new headset",
            "numeric_disagreement",
        ),
        (
            "Tesla prices the Model 3 from $35,000 in the new market",
            "Tesla prices the Model 3 from €35,000 in the new market",
            "numeric_disagreement",
        ),
        (
            "AMD gross margin improves 50bps in the quarter",
            "AMD gross margin improves 50% in the quarter",
            "numeric_disagreement",
        ),
        (
            "AMD guides revenue growth of 5-10% for the year",
            "AMD guides revenue growth of 10-15% for the year",
            "numeric_disagreement",
        ),
        (
            "Nvidia margin moved -5% in the quarter",
            "Nvidia margin moved 5% in the quarter",
            "numeric_disagreement",
        ),
        (
            "Nvidia reports Q3 data centre revenue",
            "Nvidia reports Q4 data centre revenue",
            "temporal_disagreement",
        ),
        (
            "Apple reports fiscal 2025 services revenue",
            "Apple reports fiscal 2026 services revenue",
            "temporal_disagreement",
        ),
        (
            "Apple sets a launch date of 2026-09-12 for the next iPhone",
            "Apple sets a launch date of 2026-09-19 for the next iPhone",
            "temporal_disagreement",
        ),
        (
            "Apple names a new chief financial officer",
            "Apple names a new chief operating officer",
            "role_disagreement",
        ),
        (
            "AMD raises its full-year revenue guidance",
            "AMD cuts its full-year revenue guidance",
            "contrast_polarity",
        ),
        (
            "German regulator approves the Tesla plant expansion",
            "German regulator rejects the Tesla plant expansion",
            "contrast_polarity",
        ),
        (
            "Nvidia beats quarterly revenue estimates",
            "Nvidia misses quarterly revenue estimates",
            "contrast_polarity",
        ),
        (
            "Tesla posts a quarterly profit",
            "Tesla posts a quarterly loss",
            "contrast_polarity",
        ),
        (
            "Apple maintains its services revenue outlook",
            "Apple withdraws its services revenue outlook",
            "contrast_polarity",
        ),
        (
            "Nvidia expands packaging capacity for Blackwell",
            "The Nvidia packaging supplier expands capacity in Taiwan",
            "subject_shift",
        ),
        (
            "Apple opens its first store in Saudi Arabia",
            "Apple opens its first store in Vietnam",
            "same_frame_different_event",
        ),
        (
            "Nvidia appoints a new chair of the audit committee",
            "Nvidia appoints a new chair of the compensation committee",
            "same_frame_different_event",
        ),
        (
            "Meta rolls out teen account defaults on Instagram",
            "Meta rolls out teen account defaults on Facebook",
            "same_frame_different_event",
        ),
    ],
)
def test_a_guard_refuses_the_merge_however_close_the_vectors(left, right, reason):
    result = near_identical(left, right)

    assert grouped(result) == [("s1",), ("s2",)]
    assert result.stats.veto_count(reason) == 1
    assert result.rejected_pairs[0].reason == reason
    # The vectors really were near-identical: this is the guard's doing.
    assert result.rejected_pairs[0].similarity > 0.99


def test_a_contradiction_in_the_description_alone_refuses_the_merge():
    encoder = encoder_for(("Tesla updates its delivery guidance", 0.0))
    result = run(
        [
            story(
                "s1",
                "Tesla updates its delivery guidance",
                description="The company raised its delivery outlook for the year.",
            ),
            story(
                "s2",
                "Tesla updates its delivery guidance",
                description="The company cut its delivery outlook for the year.",
                published_at=BASE + timedelta(hours=1),
            ),
        ],
        encoder,
    )

    assert grouped(result) == [("s1",), ("s2",)]
    assert result.stats.veto_count("contrast_polarity") == 1


def test_missing_information_never_vetoes():
    """M2's asymmetry, preserved: silence is not disagreement."""

    encoder = encoder_for(
        ("Meta commits $10 billion to a Louisiana data centre", 0.0),
        ("Meta backs a new Louisiana computing campus", 10.0),
    )
    result = run(
        [
            story(
                "s1",
                "Meta commits $10 billion to a Louisiana data centre",
                ticker="META",
            ),
            story(
                "s2",
                "Meta backs a new Louisiana computing campus",
                ticker="META",
                published_at=BASE + timedelta(hours=2),
            ),
        ],
        encoder,
    )

    assert grouped(result) == [("s1", "s2")]


def test_agreeing_numbers_do_not_veto():
    encoder = encoder_for(
        ("Apple announces a $110 billion share buyback", 0.0),
        ("Apple authorises $110 billion of stock repurchases", 12.0),
    )
    result = run(
        [
            story("s1", "Apple announces a $110 billion share buyback", ticker="AAPL"),
            story(
                "s2",
                "Apple authorises $110 billion of stock repurchases",
                ticker="AAPL",
                published_at=BASE + timedelta(hours=1),
            ),
        ],
        encoder,
    )

    assert grouped(result) == [("s1", "s2")]


def test_the_same_role_named_two_ways_does_not_veto():
    assert roles("AMD appoints a new chief financial officer") == ("cfo",)
    assert roles("AMD hires a finance chief from a rival") == ("cfo",)
    assert (
        contradiction(
            summarize("AMD appoints a new chief financial officer"),
            summarize("AMD hires a finance chief from a rival"),
            frame_overlap=0.5,
        )
        != "role_disagreement"
    )


def test_a_strict_elaboration_is_not_a_frame_substitution():
    """Adding detail is not swapping the event."""

    assert (
        contradiction(
            summarize("Apple opens a store"),
            summarize("Apple opens a store in Riyadh"),
            frame_overlap=0.5,
        )
        is None
    )


# --------------------------------------------------------------------------
# Repeated but distinct events
# --------------------------------------------------------------------------


def test_the_same_headline_in_two_quarters_does_not_merge():
    result = near_identical(
        "Nvidia reports record data centre revenue",
        "Nvidia reports record data centre revenue",
        left={"description": "Revenue for the third quarter beat estimates."},
        right={"description": "Revenue for the fourth quarter beat estimates."},
    )

    assert grouped(result) == [("s1",), ("s2",)]
    assert result.stats.veto_count("temporal_disagreement") == 1


def test_a_recurring_headline_outside_the_window_never_becomes_a_candidate():
    encoder = encoder_for(("Nvidia declares a quarterly cash dividend", 0.0))
    result = run(
        [
            story("q1", "Nvidia declares a quarterly cash dividend"),
            story(
                "q2",
                "Nvidia declares a quarterly cash dividend",
                published_at=BASE + timedelta(days=90),
            ),
        ],
        encoder,
    )

    assert grouped(result) == [("q1",), ("q2",)]
    assert result.stats.candidate_pair_count == 0


# --------------------------------------------------------------------------
# Cluster semantics: no bridging
# --------------------------------------------------------------------------


def test_a_vague_story_cannot_bridge_two_contradictory_ones():
    """Prospective-cluster compatibility, not endpoint-only."""

    encoder = encoder_for(
        ("Chipmaker raises its full-year outlook", 0.0),
        ("Chipmaker updates its full-year outlook", 3.0),
        ("Chipmaker cuts its full-year outlook", 6.0),
    )
    result = run(
        [
            story("raised", "Chipmaker raises its full-year outlook"),
            story(
                "vague",
                "Chipmaker updates its full-year outlook",
                published_at=BASE + timedelta(hours=1),
            ),
            story(
                "cut",
                "Chipmaker cuts its full-year outlook",
                published_at=BASE + timedelta(hours=2),
            ),
        ],
        encoder,
    )
    members = grouped(result)

    assert all(len(group) <= 2 for group in members)
    assert not any({"raised", "cut"} <= set(group) for group in members)


def test_a_story_is_a_clique_so_chaining_cannot_stretch_it():
    """A and C never compared, so they never end up together."""

    encoder = encoder_for(
        ("alpha bravo charlie delta", 0.0),
        ("echo foxtrot golf hotel", 25.0),
        ("india juliett kilo lima", 50.0),
    )
    result = run(
        [
            story("a", "alpha bravo charlie delta"),
            story(
                "b", "echo foxtrot golf hotel", published_at=BASE + timedelta(hours=1)
            ),
            story(
                "c", "india juliett kilo lima", published_at=BASE + timedelta(hours=2)
            ),
        ],
        encoder,
        similarity_threshold=0.90,  # cos(25) = .906 yes, cos(50) = .643 no
    )

    assert not any({"a", "c"} <= set(group) for group in grouped(result))
    assert result.stats.veto_count("cluster_not_complete") >= 1


def test_merges_never_cross_a_ticker():
    encoder = encoder_for(("Chipmaker posts record revenue", 0.0))
    result = run(
        [
            story("nv", "Chipmaker posts record revenue", ticker="NVDA"),
            story("amd", "Chipmaker posts record revenue", ticker="AMD"),
        ],
        encoder,
    )

    assert grouped(result) == [("amd",), ("nv",)]
    assert result.stats.candidate_pair_count == 0


def test_an_undated_story_does_not_merge_by_default():
    encoder = encoder_for(("Nvidia posts record revenue", 0.0))
    result = run(
        [
            story("dated", "Nvidia posts record revenue"),
            story("undated", "Nvidia posts record revenue", published_at=None),
        ],
        encoder,
    )

    assert grouped(result) == [("dated",), ("undated",)]
    assert result.stats.candidate_pair_count == 0


def test_undated_merges_are_available_but_off_by_default():
    encoder = encoder_for(("Nvidia posts record revenue", 0.0))
    result = run(
        [
            story("dated", "Nvidia posts record revenue"),
            story("undated", "Nvidia posts record revenue", published_at=None),
        ],
        encoder,
        allow_undated_merges=True,
    )

    assert grouped(result) == [("dated", "undated")]


# --------------------------------------------------------------------------
# Threshold boundaries
# --------------------------------------------------------------------------


def _pair_at(angle: float):
    return [
        story("s1", "alpha bravo charlie"),
        story("s2", "delta echo foxtrot", published_at=BASE + timedelta(hours=1)),
    ], encoder_for(("alpha bravo charlie", 0.0), ("delta echo foxtrot", angle))


def test_the_threshold_is_a_closed_lower_bound():
    """A pair scoring exactly the threshold merges; one hair under does not.

    The exact score is measured rather than computed, because the encoder
    round-trips through float32 and a hand-derived cosine would be testing
    the arithmetic rather than the boundary.
    """

    stories, encoder = _pair_at(45.0)
    measured = run(stories, encoder, similarity_threshold=1.0).rejected_pairs[0]
    score = measured.similarity
    assert score is not None

    at_threshold = run(stories, encoder, similarity_threshold=score)
    just_above = run(stories, encoder, similarity_threshold=math.nextafter(score, 1.0))

    assert len(at_threshold.stories) == 1
    assert len(just_above.stories) == 2


@pytest.mark.parametrize("angle,expected_merge", [(0.0, True), (89.0, False)])
def test_the_threshold_separates_close_vectors_from_distant_ones(angle, expected_merge):
    stories, encoder = _pair_at(angle)
    result = run(stories, encoder, similarity_threshold=math.cos(math.radians(45)))

    assert (len(result.stories) == 1) is expected_merge


def test_a_pair_below_the_threshold_is_recorded_with_its_score():
    encoder = encoder_for(("alpha bravo charlie", 0.0), ("delta echo foxtrot", 60.0))
    result = run(
        [
            story("s1", "alpha bravo charlie"),
            story("s2", "delta echo foxtrot", published_at=BASE + timedelta(hours=1)),
        ],
        encoder,
        similarity_threshold=0.9,
    )
    rejection = result.rejected_pairs[0]

    assert rejection.reason == "below_threshold"
    assert rejection.similarity == pytest.approx(0.5, abs=1e-6)
    assert result.stats.above_threshold_count == 0


def test_the_committed_default_threshold_is_the_one_the_sweep_chose():
    from nlp.semdedup.config import DEFAULT_SIMILARITY_THRESHOLD

    sweep = json.loads(
        (
            REPO_ROOT / "nlp" / "eval" / "data" / "results" / "m3_threshold_sweep.json"
        ).read_text(encoding="utf-8")
    )["sweep"]
    chosen = next(
        point
        for point in sweep
        if point["threshold"] == pytest.approx(DEFAULT_SIMILARITY_THRESHOLD)
    )
    highest_false_merge = max(
        point["threshold"] for point in sweep if point["counts"]["false_positive"]
    )

    assert chosen["precision"] == 1.0
    assert chosen["recall"] >= 0.75
    # Chosen with margin over the last threshold that still merged something
    # it should not have, rather than parked on the knife edge.
    assert DEFAULT_SIMILARITY_THRESHOLD > highest_false_merge + 0.02


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def sample_stories():
    return [
        story("a", "Nvidia posts record quarterly sales"),
        story(
            "b",
            "Record quarterly sales reported by the chipmaker",
            published_at=BASE + timedelta(hours=1),
            outlets=("cnbc",),
        ),
        story(
            "c",
            "AMD opens a design centre in Dresden",
            ticker="AMD",
            published_at=BASE + timedelta(hours=2),
        ),
        story(
            "d",
            "Tesla recalls 12,000 Cybertrucks",
            ticker="TSLA",
            published_at=BASE + timedelta(hours=3),
        ),
    ]


def sample_encoder():
    return encoder_for(
        ("Nvidia posts record quarterly sales", 0.0),
        ("Record quarterly sales reported by the chipmaker", 10.0),
        ("AMD opens a design centre in Dresden", 80.0),
        ("Tesla recalls 12,000 Cybertrucks", 85.0),
    )


def fingerprints(result):
    return [story.story_fingerprint for story in result.stories]


def test_output_is_invariant_under_input_permutation():
    baseline = run(sample_stories(), sample_encoder())

    for permutation in itertools.islice(itertools.permutations(sample_stories()), 12):
        result = run(list(permutation), sample_encoder())
        assert fingerprints(result) == fingerprints(baseline)
        assert grouped(result) == grouped(baseline)
        assert [s.content_hash for s in result.stories] == [
            s.content_hash for s in baseline.stories
        ]


def test_the_same_run_twice_produces_identical_output():
    first = run(sample_stories(), sample_encoder())
    second = run(sample_stories(), sample_encoder())

    assert fingerprints(first) == fingerprints(second)
    assert first.config_fingerprint == second.config_fingerprint


def test_output_does_not_depend_on_the_hash_seed():
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from tests.test_semantic_dedup import sample_stories, sample_encoder, run;"
        "r = run(sample_stories(), sample_encoder());"
        "print([s.story_fingerprint for s in r.stories])"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout)

    assert len(outputs) == 1


def test_the_fingerprint_is_stable_under_member_order_and_changes_with_membership():
    first = story_fingerprint_for("NVDA", ["a", "b"])

    assert first == story_fingerprint_for("NVDA", ["b", "a"])
    assert first != story_fingerprint_for("NVDA", ["a", "b", "c"])
    assert first != story_fingerprint_for("AMD", ["a", "b"])


@pytest.mark.parametrize("member_keys", [[], ["a", "a"], ["a", ""], ["a", None]])
def test_the_fingerprint_helper_refuses_a_member_set_it_cannot_vouch_for(member_keys):
    with pytest.raises(SemanticDedupInputError):
        story_fingerprint_for("NVDA", member_keys)


def test_the_fingerprint_helper_refuses_a_blank_ticker():
    with pytest.raises(SemanticDedupInputError, match="non-blank ticker"):
        story_fingerprint_for("  ", ["a"])


# --------------------------------------------------------------------------
# Encoder contract
# --------------------------------------------------------------------------


def test_the_stage_encodes_in_one_batch():
    encoder = FakeEncoder()
    run(sample_stories(), encoder)

    assert len(encoder.calls) == 1
    assert len(encoder.calls[0]) == 4


def test_the_embedded_text_is_m1s_composition():
    assert story_text(story("s", "Title", description="Body")) == "Title\n\nBody"
    assert story_text(story("s", "Title")) == "Title"
    assert story_text(story("s", "  ")) is None


def test_a_story_with_no_text_stays_a_singleton_rather_than_failing():
    encoder = encoder_for(("Nvidia posts record revenue", 0.0))
    result = run(
        [
            story("text", "Nvidia posts record revenue"),
            story("silent", "   ", published_at=BASE + timedelta(hours=1)),
        ],
        encoder,
    )

    assert grouped(result) == [("text",), ("silent",)]
    assert result.stats.unencodable_story_count == 1
    assert result.stats.veto_count("no_encodable_text") == 1


def test_an_encoder_returning_the_wrong_count_is_an_error():
    class ShortEncoder(FakeEncoder):
        def embed_batch(self, texts):
            return super().embed_batch(texts)[:-1]

    with pytest.raises(SemanticDedupEncodingError, match="vectors for"):
        run(sample_stories(), ShortEncoder())


def test_an_encoder_returning_mixed_dimensions_is_an_error():
    class RaggedEncoder(FakeEncoder):
        def embed_batch(self, texts):
            return [
                [1.0, 0.0, 0.0] if index else [1.0, 0.0] for index in range(len(texts))
            ]

    with pytest.raises(SemanticDedupEncodingError, match="inconsistent vector"):
        run(sample_stories(), RaggedEncoder())


@pytest.mark.parametrize(
    "vector,message",
    [
        ([float("nan"), 1.0, 0.0], "non-finite"),
        ([0.0, 0.0, 0.0], "zero vector"),
        ([], "empty vector"),
        (["a", "b", "c"], "non-numeric"),
    ],
)
def test_an_invalid_vector_is_refused_rather_than_compared(vector, message):
    class BadEncoder(FakeEncoder):
        def embed_batch(self, texts):
            return [vector for _ in texts]

    with pytest.raises(SemanticDedupEncodingError, match=message):
        run(sample_stories(), BadEncoder())


def test_a_failing_encoder_propagates_rather_than_falling_back():
    class FailingEncoder(FakeEncoder):
        def embed_batch(self, texts):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        run(sample_stories(), FailingEncoder())


def test_an_object_that_is_not_an_encoder_is_refused():
    with pytest.raises(SemanticDedupInputError, match="must implement embed_batch"):
        merge_semantic_duplicates([], config=config(), encoder=object())


def test_the_encoder_identity_is_recorded_and_fingerprinted():
    result = run(sample_stories(), FakeEncoder())
    other = SemanticDedupConfig(supported_tickers=UNIVERSE).fingerprint(
        model_name="different", model_revision="v1"
    )

    assert result.model_name == "fake-encoder"
    assert result.model_revision == "v1"
    assert result.config_fingerprint != other


def test_no_model_is_loaded_by_importing_or_running_the_stage():
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from tests.test_semantic_dedup import sample_stories, sample_encoder, run;"
        "run(sample_stories(), sample_encoder());"
        "print('sentence_transformers' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"


# --------------------------------------------------------------------------
# Configuration and input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"similarity_threshold": 1.5}, "similarity_threshold"),
        ({"similarity_threshold": "high"}, "similarity_threshold"),
        ({"frame_overlap_threshold": -0.1}, "frame_overlap_threshold"),
        ({"window_hours": 0}, "greater than zero"),
        ({"window_hours": 400}, "must not exceed"),
        ({"max_partition_stories": 0}, "positive integer"),
        ({"allow_undated_merges": "yes"}, "must be a boolean"),
        ({"supported_tickers": []}, "must not be empty"),
        ({"supported_tickers": ["NVDA", "nvda"]}, "duplicate symbol"),
        ({"supported_tickers": ["not a ticker"]}, "invalid symbol"),
        ({"supported_tickers": "NVDA"}, "collection of symbols"),
    ],
)
def test_an_unusable_configuration_is_refused(overrides, message):
    with pytest.raises(SemanticDedupConfigError, match=message):
        config(**overrides)


def test_the_default_window_is_the_one_the_issue_specifies():
    assert config().window == timedelta(hours=36)


@pytest.mark.parametrize(
    "stories,message",
    [
        ([story("a", "x"), story("a", "y")], "duplicate story_key"),
        ([story("", "x")], "blank story_key"),
        ([story("a", "x", ticker="AMZN")], "outside the supported universe"),
        (
            [story("a", "x", published_at=datetime(2026, 3, 2, 13, 0))],
            "timezone offset",
        ),
        ([story("a", "x", member_ids=("m", "m"))], "repeats a member_id"),
        (["not a story"], "must be StoryInput instances"),
    ],
)
def test_invalid_input_is_refused(stories, message):
    with pytest.raises(SemanticDedupInputError, match=message):
        run(stories)


def test_an_oversized_partition_fails_before_any_output():
    stories = [
        story(
            f"s{index}",
            f"headline number {index}",
            published_at=BASE + timedelta(minutes=index),
        )
        for index in range(6)
    ]

    with pytest.raises(SemanticDedupCapacityError) as excinfo:
        run(stories, max_partition_stories=5)

    assert excinfo.value.ticker == "NVDA"
    assert excinfo.value.story_count == 6
    assert excinfo.value.limit == 5


def test_an_empty_run_is_valid_and_empty():
    result = run([])

    assert result.stories == ()
    assert result.stats.input_story_count == 0


def test_the_result_shares_no_state_with_the_caller_list():
    stories = sample_stories()
    result = run(stories, sample_encoder())
    stories.clear()

    assert len(result.stories) >= 1


# --------------------------------------------------------------------------
# M2 output integration
# --------------------------------------------------------------------------


def raw(item_id: str, title: str, **overrides) -> RawItem:
    fields = {
        "ticker": "NVDA",
        "description": None,
        "source": "Reuters",
        "url": f"https://reuters.com/{item_id}",
        "published_at": BASE,
    }
    fields.update(overrides)
    return RawItem(item_id=item_id, title=title, **fields)


def test_m2_clusters_project_onto_m3_input_without_losing_anything():
    items = [
        raw(
            "1",
            "Nvidia posts record quarterly sales",
            description="The chipmaker beat estimates.",
        ),
        raw(
            "2",
            "Nvidia posts record quarterly sales",
            description="The chipmaker beat estimates.",
            source="CNBC",
            url="https://cnbc.com/2",
            published_at=BASE + timedelta(minutes=20),
        ),
        raw(
            "3",
            "Record quarterly sales reported by the chipmaker",
            source="Yahoo Finance",
            url="https://finance.yahoo.com/3",
            published_at=BASE + timedelta(hours=1),
        ),
    ]
    exact = deduplicate(items, config=DedupConfig(supported_tickers=UNIVERSE))
    stories = stories_from_dedup(exact, items)

    assert len(stories) == 2
    syndicated = next(entry for entry in stories if len(entry.member_ids) == 2)
    assert syndicated.member_ids == ("1", "2")
    assert set(syndicated.outlets) == {"reuters", "cnbc"}
    assert syndicated.description == "The chipmaker beat estimates."
    assert {link.item_id for link in syndicated.source_links} == {"1", "2"}


def test_the_two_stages_compose_into_one_story_with_every_member():
    items = [
        raw("1", "Nvidia posts record quarterly sales"),
        raw(
            "2",
            "Nvidia posts record quarterly sales",
            source="CNBC",
            url="https://cnbc.com/2",
            published_at=BASE + timedelta(minutes=20),
        ),
        raw(
            "3",
            "Record quarterly sales reported by the chipmaker",
            source="Yahoo Finance",
            url="https://finance.yahoo.com/3",
            published_at=BASE + timedelta(hours=1),
        ),
    ]
    exact = deduplicate(items, config=DedupConfig(supported_tickers=UNIVERSE))
    encoder = encoder_for(
        ("Nvidia posts record quarterly sales", 0.0),
        ("Record quarterly sales reported by the chipmaker", 10.0),
    )
    result = merge_semantic_duplicates(
        stories_from_dedup(exact, items), config=config(), encoder=encoder
    )

    assert len(result.stories) == 1
    assert result.stories[0].member_ids == ("1", "2", "3")
    assert result.stories[0].outlet_count == 3


def test_m3_does_not_split_what_m2_merged():
    """M3 only ever collapses; it can never undo an exact-identity merge."""

    items = [
        raw("1", "Nvidia posts record quarterly sales"),
        raw(
            "2",
            "Nvidia posts record quarterly sales",
            source="CNBC",
            url="https://cnbc.com/2",
            published_at=BASE + timedelta(minutes=20),
        ),
    ]
    exact = deduplicate(items, config=DedupConfig(supported_tickers=UNIVERSE))
    result = merge_semantic_duplicates(
        stories_from_dedup(exact, items), config=config(), encoder=FakeEncoder()
    )

    assert len(result.stories) == 1
    assert result.stories[0].member_ids == ("1", "2")


# --------------------------------------------------------------------------
# Evidence helpers, directly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple ships 5 million units", ("5 million",)),
        ("Apple cuts 1,000 roles", ("1000",)),
        ("AMD margin up 5% to $10 per unit", ("5%", "$10")),
        ("AMD unveils the MI400 accelerator", ()),
        ("Tesla deploys 40 GWh of storage", ("40 gwh",)),
        ("Meta forecasts 60-65 billion dollars", ("60-65 billion",)),
    ],
)
def test_the_numeric_signature_binds_only_meaning_bearing_neighbours(text, expected):
    from nlp.dedup.structural import tokenize

    assert numeric_signature(tokenize(text)) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Nvidia reports Q3 revenue", ("q3",)),
        ("AMD issues first-quarter guidance", ("q1",)),
        ("Apple reports fiscal 2026 revenue", ("y2026",)),
        ("The dividend is payable in June", ("m06",)),
        # A full date stays one marker; it is not also read as a bare year.
        ("Launch date of 2026-09-12", ("d2026-09-12",)),
        ("Nvidia posts record revenue", ()),
    ],
)
def test_temporal_markers_are_normalized_and_sorted(text, expected):
    from nlp.dedup.structural import tokenize

    assert temporal_markers(tokenize(text)) == expected


def test_the_policy_fingerprint_covers_m2s_tokenizer():
    """A tokenizer change must invalidate M3's cached output too."""

    from nlp.dedup.structural import policy_fingerprint as structural

    assert len(policy_fingerprint()) == 64
    assert structural() != policy_fingerprint()


def test_the_config_fingerprint_moves_with_every_setting():
    baseline = config().fingerprint(model_name="m", model_revision=None)
    variants = [
        config(similarity_threshold=0.71),
        config(window_hours=24),
        config(frame_overlap_threshold=0.6),
        config(max_partition_stories=100),
        config(allow_undated_merges=True),
        config(supported_tickers=["NVDA"]),
    ]

    digests = {
        variant.fingerprint(model_name="m", model_revision=None) for variant in variants
    }
    assert baseline not in digests
    assert len(digests) == len(variants)


# --------------------------------------------------------------------------
# M4 evaluation integration
# --------------------------------------------------------------------------


def labelled_encoder():
    """Encode the labelled set without a model.

    The evaluator is what is under test here — that it runs M2 first, hands
    M2's output to M3, and attributes each merge to the right stage. Whether
    the real encoder likes a particular rewrite is measured separately, by
    the committed sweep.
    """

    return FakeEncoder()


def test_the_pipeline_predictor_attributes_each_merge_to_its_stage():
    from nlp.eval import default_pair_set
    from nlp.eval.dedup import config_for
    from nlp.eval.semantic import pipeline_predictor, semantic_config_for

    pair_set = default_pair_set()
    predict = pipeline_predictor(
        config_for(pair_set), semantic_config_for(pair_set), labelled_encoder()
    )

    exact = predict(pair_set.by_id("P001"))
    guarded = predict(pair_set.by_id("P120"))

    assert exact.merged and exact.stage == "m2"
    assert not guarded.merged
    assert "numeric_disagreement" in guarded.detail


def test_the_pipeline_never_loses_an_m2_merge():
    """Adding M3 can only ever raise recall, never lower it."""

    from nlp.eval import default_pair_set, evaluate_m2
    from nlp.eval.semantic import evaluate_pipeline

    pair_set = default_pair_set()
    exact = evaluate_m2(pair_set)
    combined = evaluate_pipeline(pair_set, labelled_encoder())

    assert set(exact.overall.confusion.true_positives) <= set(
        combined.overall.confusion.true_positives
    )


def test_the_caching_encoder_calls_through_once_per_distinct_text():
    from nlp.eval.semantic import CachingEncoder

    inner = FakeEncoder()
    cached = CachingEncoder(inner)

    first = cached.embed_batch(["alpha", "bravo", "alpha"])
    second = cached.embed_batch(["bravo", "charlie"])

    assert inner.calls == [["alpha", "bravo"], ["charlie"]]
    assert first[0] == first[2]
    assert second[0] == first[1]
    assert cached.model_name == "fake-encoder"


def test_the_committed_pipeline_result_matches_the_committed_sweep():
    """The headline numbers and the sweep row must tell the same story."""

    from nlp.semdedup.config import DEFAULT_SIMILARITY_THRESHOLD

    results = REPO_ROOT / "nlp" / "eval" / "data" / "results"
    pipeline = json.loads((results / "m2_m3_pipeline.json").read_text("utf-8"))
    sweep = json.loads((results / "m3_threshold_sweep.json").read_text("utf-8"))[
        "sweep"
    ]
    row = next(
        point
        for point in sweep
        if point["threshold"] == pytest.approx(DEFAULT_SIMILARITY_THRESHOLD)
    )

    assert pipeline["threshold"] == pytest.approx(DEFAULT_SIMILARITY_THRESHOLD)
    assert pipeline["overall"]["precision"] == pytest.approx(row["precision"])
    assert pipeline["overall"]["recall"] == pytest.approx(row["recall"])
    assert pipeline["overall"]["counts"]["false_positive"] == 0


def test_the_committed_pipeline_result_meets_ac3():
    results = REPO_ROOT / "nlp" / "eval" / "data" / "results"
    pipeline = json.loads((results / "m2_m3_pipeline.json").read_text("utf-8"))

    assert pipeline["overall"]["precision"] >= 0.85
    assert pipeline["overall"]["recall"] >= 0.75


def test_the_cli_can_score_and_sweep_the_pipeline(monkeypatch, capsys):
    from tools import eval_dedup

    monkeypatch.setattr(eval_dedup, "_shared_encoder", labelled_encoder)

    assert eval_dedup.main(["--stage", "m2+m3"]) == 0
    assert "predictor: m2+m3" in capsys.readouterr().out

    assert eval_dedup.main(["--stage", "m2+m3", "--sweep", "0.7", "0.9"]) == 0
    rendered = capsys.readouterr().out
    assert rendered.count("\n") == 3
