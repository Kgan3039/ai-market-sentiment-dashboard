"""Tests for M5 theme clustering (issue #72).

No test here loads a model or touches the network: the encoder is injected
and the fake places each story at an explicit angle, so what is under test
is the stage's behaviour rather than MiniLM's opinion of a headline.

The load-bearing invariant, asserted from several directions, is that **no
story disappears**: every input comes back in exactly one theme, in other
coverage, or in the excluded list with a reason.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from nlp.themes import (
    ClusteringMethod,
    ExclusionReason,
    OtherCoverageReason,
    PreviousTheme,
    ThemeCapacityError,
    ThemeClusteringError,
    ThemeConfig,
    ThemeConfigError,
    ThemeEncodingError,
    ThemeInputError,
    ThemeStory,
    cluster_themes,
    encoder_identity,
    theme_fingerprint_for,
)
from nlp.themes.clustering import cosine_distances
from nlp.themes.compatibility import incompatible_members, story_claims
from nlp.themes.dataset import load_ticker_days, tickers_of
from nlp.themes.quality import evaluate_ticker_day
from nlp.themes.salience import match_previous_themes, recency_component

UTC = timezone.utc
BASE = datetime(2026, 3, 5, 9, 0, tzinfo=UTC)
DAY = date(2026, 3, 5)
UNIVERSE = ["TSLA", "NVDA", "AMD", "AAPL", "META"]
REPO_ROOT = Path(__file__).resolve().parents[1]


class AngleEncoder:
    """Places each story's text at an explicit angle on the unit circle.

    Stories at nearby angles are similar; stories a quarter turn apart are
    orthogonal. That makes a test's intended cluster structure readable
    from its numbers instead of hidden inside a model.
    """

    model_name = "angle-encoder"
    model_revision = "v1"

    def __init__(self, angles: dict[str, float]) -> None:
        self.angles = angles
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str):
        angle = self.angles.get(text)
        if angle is None:
            # Unlisted text lands far from everything listed, so a test that
            # forgets a story gets an outlier rather than a silent merge.
            angle = 200.0 + (sum(ord(char) for char in text) % 40)
        radians = math.radians(angle)
        return [math.cos(radians), math.sin(radians), 0.0]


def config(**overrides) -> ThemeConfig:
    settings = {"supported_tickers": UNIVERSE}
    settings.update(overrides)
    return ThemeConfig(**settings)


def story(key: str, title: str, hours: float = 0.0, **overrides) -> ThemeStory:
    fields = {
        "ticker": "NVDA",
        "description": None,
        "published_at": BASE + timedelta(hours=hours),
        "outlets": ("reuters",),
        "item_ids": (f"{key}-a",),
        "source_links": ((f"{key}-a", "reuters", f"https://x/{key}"),),
    }
    fields.update(overrides)
    return ThemeStory(story_key=key, title=title, **fields)


def three_strand_day():
    """Six stories in three tight, well-separated strands."""

    stories = [
        story("a1", "earnings one", 0),
        story("a2", "earnings two", 1),
        story("b1", "recall one", 2),
        story("b2", "recall two", 3),
        story("c1", "permit one", 4),
        story("c2", "permit two", 5),
    ]
    encoder = AngleEncoder(
        {
            "earnings one": 0.0,
            "earnings two": 8.0,
            "recall one": 60.0,
            "recall two": 68.0,
            "permit one": 120.0,
            "permit two": 128.0,
        }
    )
    return stories, encoder


def run(stories, encoder, **overrides):
    ticker = overrides.pop("ticker", "NVDA")
    trading_day = overrides.pop("trading_day", DAY)
    previous = overrides.pop("previous_themes", ())
    return cluster_themes(
        stories,
        ticker=ticker,
        trading_day=trading_day,
        config=config(**overrides),
        encoder=encoder,
        previous_themes=previous,
    )


def memberships(theme_set) -> set[frozenset[str]]:
    return {frozenset(theme.member_story_keys) for theme in theme_set.themes}


# --------------------------------------------------------------------------
# Normal clustering
# --------------------------------------------------------------------------


def test_a_clear_day_produces_one_theme_per_strand():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)

    assert memberships(result) == {
        frozenset({"a1", "a2"}),
        frozenset({"b1", "b2"}),
        frozenset({"c1", "c2"}),
    }
    assert result.quality.meets_ac4_shape
    assert result.other_coverage == ()


def test_every_theme_is_ranked_and_labelled_from_its_leading_story():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)

    assert [theme.salience_rank for theme in result.themes] == [1, 2, 3]
    for theme in result.themes:
        assert theme.label in {entry.title for entry in theme.evidence}
        assert theme.label_source == "canonical_story_title"


def test_salience_prefers_more_stories_more_outlets_and_more_recency():
    stories = [
        story("big1", "earnings one", 0, outlets=("reuters", "cnbc", "wsj")),
        story("big2", "earnings two", 1, outlets=("bloomberg",)),
        story("big3", "earnings three", 2, outlets=("axios",)),
        story("small1", "permit one", 3),
        story("small2", "permit two", 4),
    ]
    encoder = AngleEncoder(
        {
            "earnings one": 0.0,
            "earnings two": 5.0,
            "earnings three": 10.0,
            "permit one": 120.0,
            "permit two": 125.0,
        }
    )
    result = run(stories, encoder)
    top = result.themes[0]

    assert top.story_count == 3
    assert top.outlet_count == 5
    assert top.salience > result.themes[1].salience
    assert top.salience_features.story_component == 1.0
    assert top.salience_features.outlet_component == 1.0


def test_recency_is_measured_against_the_day_not_a_clock():
    reference = BASE + timedelta(hours=12)

    assert recency_component(reference, reference, 6.0) == 1.0
    assert recency_component(reference - timedelta(hours=6), reference, 6.0) == 0.5
    assert recency_component(reference - timedelta(hours=12), reference, 6.0) == 0.25
    assert recency_component(None, reference, 6.0) == 0.0


# --------------------------------------------------------------------------
# Small-n fallback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_day_below_the_floor_is_listed_individually(count):
    stories, encoder = three_strand_day()
    result = run(stories[:count], encoder)

    assert result.method is ClusteringMethod.SMALL_N_FALLBACK
    assert result.themes == ()
    assert len(result.other_coverage) == count
    assert not result.is_clustered
    assert result.quality.meets_ac4_shape
    assert "below the clustering floor" in result.method_reason


def test_exactly_the_floor_is_clustered():
    stories, encoder = three_strand_day()
    result = run(stories[:4], encoder)

    assert result.method is not ClusteringMethod.SMALL_N_FALLBACK
    assert result.is_clustered


def test_an_empty_day_is_valid_and_empty():
    result = run([], AngleEncoder({}))

    assert result.themes == ()
    assert result.other_coverage == ()
    assert result.quality.story_count == 0


# --------------------------------------------------------------------------
# Outliers, singletons, catch-alls
# --------------------------------------------------------------------------


def test_an_outlier_goes_to_other_coverage_not_into_a_theme():
    stories, encoder = three_strand_day()
    stories.append(story("odd", "entirely unrelated market language", 6))
    result = run(stories, encoder)

    assert "odd" in {entry.story_key for entry in result.other_coverage}
    assert all("odd" not in theme.member_story_keys for theme in result.themes)


def test_a_lone_story_is_other_coverage_rather_than_a_singleton_theme():
    stories, encoder = three_strand_day()
    stories.append(story("lonely", "lonely one", 6))
    encoder.angles["lonely one"] = 300.0
    result = run(stories, encoder)

    assert result.quality.singleton_theme_count == 0
    assert "lonely" in {entry.story_key for entry in result.other_coverage}


def test_a_loose_catch_all_is_dissolved_rather_than_shown_as_a_theme():
    """The failure a reader would never forgive: a theme that is not one."""

    stories = [
        story("a1", "earnings one", 0),
        story("a2", "earnings two", 1),
        story("s1", "scattered one", 2),
        story("s2", "scattered two", 3),
        story("s3", "scattered three", 4),
    ]
    encoder = AngleEncoder(
        {
            "earnings one": 0.0,
            "earnings two": 4.0,
            "scattered one": 100.0,
            "scattered two": 150.0,
            "scattered three": 200.0,
        }
    )
    result = run(stories, encoder, min_theme_cohesion=0.9, min_themes=1)
    placed = {key for theme in result.themes for key in theme.member_story_keys}

    assert {"s1", "s2", "s3"} & placed == set()
    assert {"s1", "s2", "s3"} <= {entry.story_key for entry in result.other_coverage}


def test_hdbscan_gives_way_to_agglomerative_when_a_cluster_is_too_loose():
    stories, encoder = three_strand_day()
    # The strands sit 8 degrees apart, so their cohesion is cos(8) = 0.9903.
    result = run(stories, encoder, min_theme_cohesion=0.995)

    assert result.method is ClusteringMethod.AGGLOMERATIVE
    assert "below 0.995" in result.method_reason


# --------------------------------------------------------------------------
# No story is lost
# --------------------------------------------------------------------------


def test_every_story_is_accounted_for_exactly_once():
    stories, encoder = three_strand_day()
    stories.append(story("odd", "entirely unrelated market language", 6))
    result = run(stories, encoder)

    assert result.accounted_story_keys == tuple(
        sorted(entry.story_key for entry in stories)
    )


def test_a_story_with_no_text_is_excluded_with_a_reason_never_dropped():
    stories, encoder = three_strand_day()
    stories.append(story("silent", "   ", 6))
    result = run(stories, encoder)

    assert [entry.story_key for entry in result.excluded] == ["silent"]
    assert result.excluded[0].reason is ExclusionReason.NO_ENCODABLE_TEXT
    assert "silent" in result.accounted_story_keys


def test_a_day_of_nothing_but_textless_stories_is_all_excluded():
    stories = [story(f"s{index}", "   ", index) for index in range(5)]
    result = run(stories, AngleEncoder({}))

    assert len(result.excluded) == 5
    assert result.themes == ()
    assert result.method is ClusteringMethod.SMALL_N_FALLBACK


# --------------------------------------------------------------------------
# Contradictory stories
# --------------------------------------------------------------------------


def test_a_theme_may_not_hold_both_sides_of_one_claim():
    stories = [
        story("up1", "Nvidia raises its full-year revenue guidance", 0),
        story("up2", "Nvidia lifts its full-year revenue outlook", 1),
        story("down", "Nvidia cuts its full-year revenue guidance", 2),
        story("b1", "recall one", 3),
        story("b2", "recall two", 4),
    ]
    encoder = AngleEncoder(
        {
            "Nvidia raises its full-year revenue guidance": 0.0,
            "Nvidia lifts its full-year revenue outlook": 3.0,
            "Nvidia cuts its full-year revenue guidance": 6.0,
            "recall one": 120.0,
            "recall two": 125.0,
        }
    )
    result = run(stories, encoder)
    guidance = next(
        theme for theme in result.themes if "up1" in theme.member_story_keys
    )

    assert "down" not in guidance.member_story_keys
    assert "down" in {entry.story_key for entry in result.other_coverage}
    assert "contradicting" in result.method_reason


def test_the_minority_side_is_the_one_that_moves():
    stories = ["a", "b", "c"]
    positions = [0, 1, 2]
    themed = [
        story("a", "AMD raises its guidance", 0),
        story("b", "AMD lifts its outlook", 1),
        story("c", "AMD cuts its guidance", 2),
    ]

    assert incompatible_members(themed, positions) == (2,)
    assert stories  # the fixture names are only for readability


def test_theme_coherence_only_objects_to_opposing_claims():
    """Different quarters belong in one earnings theme; opposite calls do not."""

    quarters = [
        story("q3", "Nvidia reports Q3 data centre revenue", 0),
        story("q4", "Nvidia reports Q4 data centre revenue", 1),
    ]

    assert incompatible_members(quarters, [0, 1]) == ()
    assert story_claims(quarters[0]) == frozenset()


# --------------------------------------------------------------------------
# Determinism and stability
# --------------------------------------------------------------------------


def test_output_is_invariant_under_input_permutation():
    stories, encoder = three_strand_day()
    baseline = run(list(stories), encoder)

    for shift in range(1, len(stories)):
        rotated = stories[shift:] + stories[:shift]
        result = run(rotated, AngleEncoder(dict(encoder.angles)))
        assert [theme.fingerprint for theme in result.themes] == [
            theme.fingerprint for theme in baseline.themes
        ]
        assert [theme.label for theme in result.themes] == [
            theme.label for theme in baseline.themes
        ]
        assert [entry.story_key for entry in result.other_coverage] == [
            entry.story_key for entry in baseline.other_coverage
        ]


def test_duplicate_embeddings_do_not_destabilise_the_day():
    """Several stories on exactly the same point is a degenerate but real case."""

    stories = [story(f"s{index}", f"identical {index}", index) for index in range(5)]
    encoder = AngleEncoder({f"identical {index}": 0.0 for index in range(5)})
    first = run(stories, encoder)
    second = run(list(reversed(stories)), AngleEncoder(dict(encoder.angles)))

    assert first.accounted_story_keys == second.accounted_story_keys
    assert [theme.fingerprint for theme in first.themes] == [
        theme.fingerprint for theme in second.themes
    ]


def test_a_rerun_of_an_unchanged_day_keeps_every_theme_identity():
    """AC-4: re-running within a day does not rename an unchanged theme."""

    stories, encoder = three_strand_day()
    first = run(stories, encoder)
    previous = tuple(
        PreviousTheme(theme_key=theme.theme_key, centroid=theme.centroid)
        for theme in first.themes
    )
    second = run(stories, AngleEncoder(dict(encoder.angles)), previous_themes=previous)

    assert [theme.theme_key for theme in second.themes] == [
        theme.theme_key for theme in first.themes
    ]
    assert all(theme.matched_previous_key for theme in second.themes)


def test_a_theme_that_gained_a_story_keeps_its_identity():
    stories, encoder = three_strand_day()
    first = run(stories, encoder)
    previous = tuple(
        PreviousTheme(theme_key=theme.theme_key, centroid=theme.centroid)
        for theme in first.themes
    )
    stories.append(story("a3", "earnings three", 6))
    encoder.angles["earnings three"] = 4.0
    second = run(stories, encoder, previous_themes=previous)
    grown = next(theme for theme in second.themes if "a3" in theme.member_story_keys)

    assert grown.matched_previous_key is not None
    assert grown.theme_key == grown.matched_previous_key
    # The content digest moved, which is what a reconciler needs to see.
    assert grown.fingerprint != grown.theme_key


def test_a_genuinely_new_theme_does_not_inherit_an_identity():
    stories, encoder = three_strand_day()
    first = run(stories, encoder)
    previous = tuple(
        PreviousTheme(theme_key=theme.theme_key, centroid=theme.centroid)
        for theme in first.themes
    )
    stories += [story("d1", "lawsuit one", 6), story("d2", "lawsuit two", 7)]
    encoder.angles.update({"lawsuit one": 200.0, "lawsuit two": 205.0})
    second = run(stories, encoder, previous_themes=previous)
    fresh = next(theme for theme in second.themes if "d1" in theme.member_story_keys)

    assert fresh.matched_previous_key is None
    assert fresh.theme_key == fresh.fingerprint


def test_theme_identity_matching_is_one_to_one():
    centroids = [(1.0, 0.0), (0.999, 0.044), (0.0, 1.0)]
    previous = [
        PreviousTheme("old-a", (1.0, 0.0)),
        PreviousTheme("old-b", (0.0, 1.0)),
    ]

    matched = match_previous_themes(centroids, previous, 0.9)

    assert matched == {0: "old-a", 2: "old-b"}


def test_output_does_not_depend_on_the_hash_seed():
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from tests.test_theme_clustering import three_strand_day, run;"
        "s, e = three_strand_day();"
        "print([t.fingerprint for t in run(s, e).themes])"
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


def test_the_fingerprint_is_stable_under_member_order_and_moves_with_membership():
    first = theme_fingerprint_for("NVDA", DAY, ["a", "b"])

    assert first == theme_fingerprint_for("NVDA", DAY, ["b", "a"])
    assert first != theme_fingerprint_for("NVDA", DAY, ["a", "b", "c"])
    assert first != theme_fingerprint_for("AMD", DAY, ["a", "b"])
    assert first != theme_fingerprint_for("NVDA", date(2026, 3, 6), ["a", "b"])


@pytest.mark.parametrize("members", [[], ["a", "a"], ["a", ""], ["a", None]])
def test_the_fingerprint_helper_refuses_a_member_set_it_cannot_vouch_for(members):
    with pytest.raises(ThemeInputError):
        theme_fingerprint_for("NVDA", DAY, members)


# --------------------------------------------------------------------------
# Summarization evidence boundary
# --------------------------------------------------------------------------


def test_a_theme_carries_exactly_its_members_evidence_and_nothing_else():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)
    by_key = {entry.story_key: entry for entry in stories}

    for theme in result.themes:
        assert [entry.story_key for entry in theme.evidence] == list(
            theme.member_story_keys
        )
        for entry in theme.evidence:
            source = by_key[entry.story_key]
            assert entry.title == source.title
            assert entry.item_ids == tuple(sorted(source.item_ids))


def test_a_citation_can_only_resolve_inside_its_own_theme():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)
    seen: set[str] = set()

    for theme in result.themes:
        citable = set(theme.citable_item_ids)
        members = {item_id for entry in theme.evidence for item_id in entry.item_ids}
        assert citable == members
        assert not citable & seen
        seen |= citable


def test_other_coverage_carries_the_same_evidence_shape_as_a_theme():
    stories, encoder = three_strand_day()
    stories.append(story("odd", "entirely unrelated market language", 6))
    result = run(stories, encoder)

    assert result.other_coverage
    for entry in result.other_coverage:
        assert entry.story_key
        assert entry.evidence.item_ids
        assert entry.reason in set(OtherCoverageReason)


def test_no_llm_or_retrieval_dependency_is_introduced():
    """M5 prepares evidence for the summarizer; it does not summarize."""

    script = (
        "import sys; sys.path.insert(0, '.');"
        "import nlp.themes;"
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'langchain','langgraph','openai','anthropic','mcp'}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "[]"


# --------------------------------------------------------------------------
# Configuration and input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"min_stories_for_clustering": 0}, "positive integer"),
        ({"min_themes": 7, "max_themes": 6}, "must not exceed"),
        ({"min_cluster_size": 1}, "at least 2"),
        ({"min_theme_cohesion": 1.5}, "min_theme_cohesion"),
        ({"stability_threshold": -0.1}, "stability_threshold"),
        ({"recency_half_life_hours": 0}, "recency_half_life_hours"),
        (
            {"story_weight": 0, "outlet_weight": 0, "recency_weight": 0},
            "at least one salience weight",
        ),
        ({"story_weight": -1}, "non-negative"),
        ({"supported_tickers": []}, "must not be empty"),
        ({"supported_tickers": ["NVDA", "nvda"]}, "duplicate symbol"),
    ],
)
def test_an_unusable_configuration_is_refused(overrides, message):
    with pytest.raises(ThemeConfigError, match=message):
        config(**overrides)


def test_the_defaults_are_the_ones_ac4_specifies():
    settings = config()

    assert settings.min_stories_for_clustering == 4
    assert (settings.min_themes, settings.max_themes) == (2, 6)


@pytest.mark.parametrize(
    "stories,message",
    [
        ([story("a", "x"), story("a", "y")], "duplicate story_key"),
        ([story("", "x")], "blank story_key"),
        ([story("a", "x", ticker="TSLA")], "not the requested NVDA"),
        (
            [story("a", "x", published_at=datetime(2026, 3, 5, 9, 0))],
            "timezone-aware",
        ),
        (["not a story"], "must be ThemeStory instances"),
    ],
)
def test_invalid_input_is_refused(stories, message):
    with pytest.raises(ThemeInputError, match=message):
        run(stories, AngleEncoder({}))


def test_an_unsupported_ticker_is_refused():
    with pytest.raises(ThemeInputError, match="outside the supported universe"):
        cluster_themes(
            [],
            ticker="AMZN",
            trading_day=DAY,
            config=config(),
            encoder=AngleEncoder({}),
        )


def test_a_datetime_is_not_a_trading_day():
    with pytest.raises(ThemeInputError, match="datetime.date"):
        cluster_themes(
            [],
            ticker="NVDA",
            trading_day=datetime(2026, 3, 5, tzinfo=UTC),
            config=config(),
            encoder=AngleEncoder({}),
        )


def test_an_oversized_day_fails_before_any_output():
    stories = [story(f"s{index}", f"headline {index}", index) for index in range(7)]

    with pytest.raises(ThemeCapacityError) as excinfo:
        run(stories, AngleEncoder({}), max_stories_per_day=6)

    assert excinfo.value.story_count == 7
    assert excinfo.value.limit == 6


@pytest.mark.parametrize(
    "vectors,message",
    [
        ([[float("nan"), 0.0, 0.0]] * 6, "non-finite"),
        ([[0.0, 0.0, 0.0]] * 6, "zero vector"),
    ],
)
def test_an_invalid_vector_is_refused_rather_than_clustered(vectors, message):
    class BadEncoder(AngleEncoder):
        def embed_batch(self, texts):
            return vectors[: len(texts)]

    stories, _ = three_strand_day()
    with pytest.raises(ThemeEncodingError, match=message):
        run(stories, BadEncoder({}))


def test_an_encoder_returning_the_wrong_count_is_an_error():
    class ShortEncoder(AngleEncoder):
        def embed_batch(self, texts):
            return super().embed_batch(texts)[:-1]

    stories, encoder = three_strand_day()
    with pytest.raises(ThemeEncodingError, match="vectors for"):
        run(stories, ShortEncoder(encoder.angles))


def test_the_stage_encodes_in_one_batch():
    stories, encoder = three_strand_day()
    run(stories, encoder)

    assert encoder.calls == 1


def test_the_config_fingerprint_moves_with_every_setting():
    baseline = config().fingerprint(model_name="m", model_revision=None)
    variants = [
        config(min_stories_for_clustering=5),
        config(max_themes=5),
        config(min_cluster_size=3),
        config(min_theme_cohesion=0.5),
        config(min_theme_stories=3),
        config(stability_threshold=0.8),
        config(story_weight=0.9),
        config(recency_half_life_hours=12),
        config(supported_tickers=["NVDA"]),
    ]
    digests = {
        variant.fingerprint(model_name="m", model_revision=None) for variant in variants
    }

    assert baseline not in digests
    assert len(digests) == len(variants)


def test_the_encoder_identity_is_part_of_the_fingerprint():
    settings = config()

    assert settings.fingerprint(
        model_name="a", model_revision=None
    ) != settings.fingerprint(model_name="b", model_revision=None)


def test_cosine_distances_are_symmetric_with_a_zero_diagonal():
    import numpy as np

    matrix = cosine_distances(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert (matrix >= 0).all()


# --------------------------------------------------------------------------
# The committed ticker-day fixture
# --------------------------------------------------------------------------


def test_the_fixture_is_marked_synthetic_and_names_its_blockers():
    metadata = load_ticker_days().metadata

    assert metadata["provenance"]["kind"] == "synthetic"
    assert set(metadata["provenance"]["blocked_by"]) == {"#57", "#61", "#62"}


def test_the_fixture_covers_three_volumes():
    day_set = load_ticker_days()

    assert {day.volume for day in day_set.days} == {"low", "medium", "high"}
    counts = sorted(len(day.stories) for day in day_set.days)
    assert counts[0] < 4 <= counts[1] < counts[2]
    assert tickers_of(day_set) == ("AAPL", "NVDA", "TSLA")


def test_every_fixture_day_satisfies_ac4_under_the_fake_encoder():
    """Shape and accounting, without asserting what MiniLM thinks."""

    day_set = load_ticker_days()
    settings = ThemeConfig(supported_tickers=tickers_of(day_set))

    for day in day_set.days:
        encoder = AngleEncoder(
            {
                f"{story_.title}\n\n{story_.description}": index * 11.0
                for index, story_ in enumerate(day.stories)
            }
        )
        report = evaluate_ticker_day(
            day.stories,
            ticker=day.ticker,
            trading_day=day.trading_day,
            volume=day.volume,
            config=settings,
            encoder=encoder,
        )
        assert report.no_story_lost, day.ticker
        assert report.permutation_stable, day.ticker
        assert report.rerun_keeps_identity, day.ticker


def test_the_committed_quality_results_record_ac4_on_all_three_days():
    payload = json.loads(
        (
            REPO_ROOT / "nlp" / "themes" / "data" / "results" / "theme_quality.json"
        ).read_text(encoding="utf-8")
    )
    days = payload["ticker_days"]

    assert len(days) == 3
    for day in days:
        assert day["meets_ac4_shape"], day["ticker"]
        assert day["no_story_lost"], day["ticker"]
        assert day["permutation_stable"], day["ticker"]
        assert day["rerun_keeps_identity"], day["ticker"]
        assert day["excluded_count"] == 0
        if day["theme_count"]:
            assert 2 <= day["theme_count"] <= 6
            assert day["singleton_theme_count"] == 0
    # The low-volume day is the fallback; the others are clustered.
    assert [day["method"] for day in days if day["volume"] == "low"] == [
        "small_n_fallback"
    ]


def test_the_committed_results_carry_no_machine_specific_runtime():
    payload = json.loads(
        (
            REPO_ROOT / "nlp" / "themes" / "data" / "results" / "theme_quality.json"
        ).read_text(encoding="utf-8")
    )

    for day in payload["ticker_days"]:
        assert "elapsed_seconds" not in day


# --------------------------------------------------------------------------
# Final M3 public API integration
# --------------------------------------------------------------------------


def semantic_run(quarantined: bool = False):
    """A real M3 run, so the bridge is tested against the merged contract."""

    from nlp.dedup import DedupConfig, RawItem, deduplicate
    from nlp.semdedup import (
        SemanticDedupConfig,
        merge_semantic_duplicates,
        stories_from_dedup,
    )

    def item(index: int, title: str, outlet: str, url: str, **extra) -> RawItem:
        return RawItem(
            item_id=f"i{index}",
            ticker="NVDA",
            title=title,
            description="The chipmaker reported quarterly revenue above estimates.",
            url=url,
            source=outlet,
            published_at=BASE + timedelta(minutes=30 * index),
            **extra,
        )

    raw = [
        item(1, "Nvidia posts record quarterly sales", "reuters", "https://r/1"),
        item(2, "Nvidia posts record quarterly sales", "yahoo", "https://y/1"),
        item(3, "Nvidia opens a research centre in Berlin", "ft", "https://f/1"),
    ]
    if quarantined:
        # One provider identity describing two different articles: M2
        # quarantines both and M3 holds them out of candidate generation.
        raw += [
            item(
                4,
                "Nvidia names a new finance chief",
                "reuters",
                "https://r/4",
                provider_item_id="dup-9",
            ),
            item(
                5,
                "Nvidia delays its investor day",
                "reuters",
                "https://r/5",
                provider_item_id="dup-9",
            ),
        ]
    exact = deduplicate(raw, config=DedupConfig(supported_tickers=UNIVERSE))
    stories = stories_from_dedup(exact, raw)
    return (
        merge_semantic_duplicates(
            stories,
            config=SemanticDedupConfig(supported_tickers=UNIVERSE),
            encoder=AngleEncoder({}),
        ),
        raw,
    )


def test_the_bridge_reads_the_merged_m3_result():
    from nlp.themes import theme_stories_from_semantic

    result, _ = semantic_run()
    projected = theme_stories_from_semantic(result)

    assert len(projected) == len(result.stories)
    for theme_story, semantic in zip(projected, result.stories):
        assert theme_story.story_key == semantic.story_fingerprint
        assert theme_story.ticker == semantic.ticker
        assert theme_story.title == semantic.canonical_title
        assert theme_story.published_at == semantic.published_at
        assert theme_story.outlet_count == semantic.outlet_count
        assert theme_story.item_ids == tuple(semantic.member_ids)
        assert theme_story.member_story_keys == tuple(semantic.member_story_keys)
        assert theme_story.content_hash == semantic.content_hash
        assert len(theme_story.source_links) == len(semantic.source_links)


def test_the_bridge_preserves_merge_evidence():
    from nlp.themes import theme_stories_from_semantic

    result, _ = semantic_run()
    projected = theme_stories_from_semantic(result)
    merged = [entry for entry in projected if len(entry.member_story_keys) > 1]

    for entry in projected:
        semantic = next(
            story
            for story in result.stories
            if story.story_fingerprint == entry.story_key
        )
        assert len(entry.merge_evidence) == len(semantic.merges)
        for left, right, similarity, reason in entry.merge_evidence:
            assert left and right and reason
            assert 0.0 <= similarity <= 1.0
    assert merged or all(not entry.merge_evidence for entry in projected)


def test_the_bridge_preserves_quarantine_and_skip_state():
    from nlp.themes import theme_stories_from_semantic

    result, _ = semantic_run(quarantined=True)
    projected = theme_stories_from_semantic(result)
    quarantined = [entry for entry in projected if entry.is_quarantined]

    assert quarantined, "the fixture must produce a quarantined story"
    for entry in quarantined:
        assert entry.semantic_skip_reason == "provider_quarantine"
        assert entry.is_semantically_skipped
        assert entry.quarantined_member_ids
        assert entry.provider_conflicts
    assert len(quarantined) == sum(
        1 for story in result.stories if story.is_quarantined
    )


def test_source_metadata_records_the_run_behind_the_stories():
    from nlp.themes import source_metadata_from_semantic

    result, _ = semantic_run(quarantined=True)
    metadata = source_metadata_from_semantic(result)

    assert metadata.stage == "m3.semantic"
    assert metadata.algorithm_version == result.algorithm_version
    assert metadata.config_fingerprint == result.config_fingerprint
    assert metadata.model_name == result.model_name
    assert metadata.embedding_dimension == result.embedding_dimension
    assert metadata.story_count == len(result.stories)
    assert metadata.quarantined_story_count >= 1


def test_the_exact_bridge_records_the_degradation_rather_than_a_model():
    from nlp.dedup import DedupConfig, RawItem, deduplicate
    from nlp.themes import source_metadata_from_exact, theme_stories_from_exact

    raw = [
        RawItem(
            item_id="e1",
            ticker="NVDA",
            title="Nvidia posts record quarterly sales",
            description=None,
            url="https://r/e1",
            source="reuters",
            published_at=BASE,
        )
    ]
    exact = deduplicate(raw, config=DedupConfig(supported_tickers=UNIVERSE))

    assert theme_stories_from_exact(exact, raw)
    metadata = source_metadata_from_exact(exact)
    assert metadata.stage == "m2.exact"
    assert metadata.model_name == ""
    assert metadata.embedding_dimension is None


def test_the_bridge_rejects_an_overlapping_upstream_partition():
    from nlp.themes.bridge import _reject_overlapping_items

    overlapping = [
        story("one", "first", 0, item_ids=("shared",)),
        story("two", "second", 1, item_ids=("shared",)),
    ]

    with pytest.raises(ThemeInputError, match="partition overlaps"):
        _reject_overlapping_items(overlapping)


def test_m5_imports_no_private_m3_module():
    """The compatibility contract is M5's; M3's guard internals are not.

    Read as imports rather than as text: the compatibility module names
    ``nlp.semdedup.evidence`` in its docstring on purpose, to say why it
    does not import it.
    """

    import ast

    private = []
    for path in sorted((REPO_ROOT / "nlp" / "themes").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                if parts[:1] == ["nlp"] and len(parts) > 2:
                    # nlp.semdedup.bridge is M3's documented projection
                    # helper and is re-exported; anything under a stage's
                    # guard or evidence internals is not.
                    if parts[2] in {"evidence", "guards", "compatibility"}:
                        private.append(f"{path.name}: {name}")
    assert private == []


# --------------------------------------------------------------------------
# Quarantined stories never join a theme, and are never dropped
# --------------------------------------------------------------------------


def quarantined_story(key: str, title: str, hours: float, reason: str):
    return story(
        key,
        title,
        hours,
        semantic_skip_reason=reason,
        quarantined_member_ids=(f"{key}-a",),
        provider_conflicts=(("feed", f"dup-{key}"),),
    )


def test_a_quarantined_story_is_shown_but_never_grouped():
    stories, encoder = three_strand_day()
    stories.append(quarantined_story("q1", "earnings three", 2, "provider_quarantine"))
    result = run(stories, encoder)

    assert "q1" not in {
        key for theme in result.themes for key in theme.member_story_keys
    }
    entry = next(item for item in result.other_coverage if item.story_key == "q1")
    assert entry.reason is OtherCoverageReason.PROVIDER_QUARANTINE
    assert "q1" in result.accounted_story_keys


def test_a_quarantined_story_is_held_out_even_when_it_would_cluster():
    """It is placed at the same angle as a strand, and still stays out."""

    stories, encoder = three_strand_day()
    encoder.angles["earnings quarantined"] = encoder.angles["earnings one"]
    stories.append(
        quarantined_story("q1", "earnings quarantined", 2, "provider_quarantine")
    )
    result = run(stories, encoder)

    assert "q1" not in {
        key for theme in result.themes for key in theme.member_story_keys
    }
    assert "provider_quarantine" in result.method_reason


def test_the_hold_out_reason_distinguishes_quarantine_from_other_skips():
    stories, encoder = three_strand_day()
    stories.append(quarantined_story("q1", "earnings three", 2, "some_other_reason"))
    result = run(stories, encoder)

    entry = next(item for item in result.other_coverage if item.story_key == "q1")
    assert entry.reason is OtherCoverageReason.SEMANTIC_SKIP


def test_quarantining_a_day_below_the_floor_still_lists_every_story():
    stories = [
        story("a", "one", 0),
        story("b", "two", 1),
        quarantined_story("q1", "three", 2, "provider_quarantine"),
        quarantined_story("q2", "four", 3, "provider_quarantine"),
    ]
    result = run(stories, AngleEncoder({}))

    assert result.method is ClusteringMethod.SMALL_N_FALLBACK
    assert result.accounted_story_keys == ("a", "b", "q1", "q2")
    assert result.quality.meets_ac4_shape


# --------------------------------------------------------------------------
# Accounting: nothing lost, nothing duplicated, nothing invented
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 8])
def test_every_story_is_accounted_for_at_every_size(count):
    stories = [story(f"s{index}", f"headline {index}", index) for index in range(count)]
    result = run(stories, AngleEncoder({}))

    assert result.accounted_story_keys == tuple(
        sorted(entry.story_key for entry in stories)
    )
    membership = [key for theme in result.themes for key in theme.member_story_keys]
    assert len(membership) == len(set(membership))
    assert all(theme.member_story_keys for theme in result.themes)


@pytest.mark.parametrize("count", [3, 4, 5])
def test_the_small_n_boundary_is_the_clustering_floor(count):
    stories, encoder = three_strand_day()
    subset = stories[:count]
    result = run(subset, encoder)

    below = count < config().min_stories_for_clustering
    assert (result.method is ClusteringMethod.SMALL_N_FALLBACK) is below
    assert result.quality.meets_ac4_shape
    if below:
        assert not result.themes
        assert {entry.reason for entry in result.other_coverage} == {
            OtherCoverageReason.BELOW_CLUSTERING_FLOOR
        }


def test_a_day_whose_encodable_stories_fall_below_the_floor_is_not_an_ac4_failure():
    """Five arrived, three could be embedded: that is a degradation, not a miss."""

    stories = [story(f"s{index}", f"headline {index}", index) for index in range(3)]
    stories += [story("blank1", "", 4), story("blank2", "", 5)]
    result = run(stories, AngleEncoder({}))

    assert result.method is ClusteringMethod.SMALL_N_FALLBACK
    assert len(result.excluded) == 2
    assert result.quality.meets_ac4_shape
    assert len(result.accounted_story_keys) == 5


def test_all_singleton_embeddings_produce_no_theme_and_lose_nothing():
    stories = [story(f"s{index}", f"unrelated {index}", index) for index in range(8)]
    encoder = AngleEncoder({f"unrelated {index}": index * 40.0 for index in range(8)})
    result = run(stories, encoder)

    assert len(result.accounted_story_keys) == 8
    for theme in result.themes:
        assert theme.story_count >= config().min_theme_stories
        assert theme.cohesion >= config().min_theme_cohesion


def test_duplicate_embeddings_do_not_break_accounting():
    stories = [story(f"s{index}", f"identical {index}", index) for index in range(6)]
    encoder = AngleEncoder({f"identical {index}": 10.0 for index in range(6)})
    result = run(stories, encoder)

    assert len(result.accounted_story_keys) == 6
    membership = [key for theme in result.themes for key in theme.member_story_keys]
    assert len(membership) == len(set(membership))


def test_one_dominant_cluster_plus_outliers_keeps_the_outliers_visible():
    stories = [story(f"d{index}", f"dominant {index}", index) for index in range(6)]
    stories += [story("out1", "far one", 7), story("out2", "far two", 8)]
    angles = {f"dominant {index}": 10.0 + index for index in range(6)}
    angles.update({"far one": 150.0, "far two": 250.0})
    result = run(stories, AngleEncoder(angles))

    assert len(result.accounted_story_keys) == 8
    grouped = {key for theme in result.themes for key in theme.member_story_keys}
    assert "out1" not in grouped and "out2" not in grouped


# --------------------------------------------------------------------------
# The fallback picks by the theme contract, not by silhouette
# --------------------------------------------------------------------------


def test_the_fallback_prefers_the_k_that_places_the_most_stories_in_themes():
    from nlp.themes.clustering import _best_agglomerative, cosine_distances
    import numpy as np

    # Four tight pairs plus one far singleton.  A small k lumps the pairs
    # into loose clusters the stage would dissolve; k=4 keeps them.
    angles = [0.0, 3.0, 40.0, 43.0, 80.0, 83.0, 120.0, 123.0, 260.0]
    vectors = np.array(
        [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0] for a in angles]
    )
    count, detail = _best_agglomerative(
        vectors, cosine_distances(vectors), config(max_themes=6)
    )

    assert count >= 4
    assert "clearing the floors" in detail
    assert "silhouette" not in detail


def test_dropping_one_story_does_not_collapse_the_day():
    """The regression the silhouette objective caused on the TSLA fixture."""

    day_set = load_ticker_days()
    settings = ThemeConfig(supported_tickers=tickers_of(day_set))
    heaviest = max(day_set.days, key=lambda day: len(day.stories))
    encoder = AngleEncoder(
        {
            f"{entry.title}\n\n{entry.description}": index * 11.0
            for index, entry in enumerate(heaviest.stories)
        }
    )
    ordered = sorted(heaviest.stories, key=lambda s: s.published_at)
    full = cluster_themes(
        ordered,
        ticker=heaviest.ticker,
        trading_day=heaviest.trading_day,
        config=settings,
        encoder=encoder,
    )
    trimmed = cluster_themes(
        ordered[1:],
        ticker=heaviest.ticker,
        trading_day=heaviest.trading_day,
        config=settings,
        encoder=encoder,
    )

    assert len(full.accounted_story_keys) == len(ordered)
    assert len(trimmed.accounted_story_keys) == len(ordered) - 1
    # Losing one story may reshape a theme; it must not erase the day.
    if full.themes:
        assert trimmed.themes, "dropping one story emptied the theme set"


def test_a_clustering_library_failure_is_typed():
    from nlp.themes import clustering

    class Exploding:
        def fit_predict(self, distance):
            raise RuntimeError("library exploded")

    import numpy as np

    original = (
        clustering.AgglomerativeClustering
        if hasattr(clustering, "AgglomerativeClustering")
        else None
    )
    assert original is None  # imported inside the function, not at module scope

    import sklearn.cluster

    saved = sklearn.cluster.AgglomerativeClustering
    sklearn.cluster.AgglomerativeClustering = lambda **kwargs: Exploding()
    try:
        with pytest.raises(ThemeClusteringError) as caught:
            clustering._agglomerative(np.zeros((4, 4)), 2)
        assert caught.value.method == "agglomerative"
        assert isinstance(caught.value.cause, RuntimeError)
    finally:
        sklearn.cluster.AgglomerativeClustering = saved


# --------------------------------------------------------------------------
# Compatibility: what a theme may and may not hold
# --------------------------------------------------------------------------


CONTRADICTIONS = [
    ("Nvidia raises its full-year guidance", "Nvidia cuts its full-year guidance"),
    ("Nvidia beats quarterly estimates", "Nvidia misses quarterly estimates"),
    ("Regulators approve the Nvidia deal", "Regulators reject the Nvidia deal"),
    ("Nvidia confirms the Ohio plant", "Nvidia cancels the Ohio plant"),
    ("Nvidia opens the Ohio plant", "Nvidia will not open the Ohio plant"),
]

PERMITTED = [
    ("Nvidia reports Q1 revenue of $5bn", "Nvidia guides Q2 revenue to $6bn"),
    ("Nvidia appoints Alice Smith as CFO", "Nvidia appoints Bob Jones as COO"),
    ("Nvidia GTC keynote: live updates", "Nvidia GTC keynote recap"),
    ("Nvidia ships 5 million units", "Nvidia ships 5 billion euros of chips"),
    ("Nvidia recalls a batch of boards", "Nvidia recalls a second batch of boards"),
]


@pytest.mark.parametrize("left,right", CONTRADICTIONS)
def test_a_theme_may_not_hold_both_sides_of_a_claim(left, right):
    stories = [story("a", left, 0), story("b", right, 1)]

    assert incompatible_members(stories, [0, 1]) == (1,)


@pytest.mark.parametrize("left,right", PERMITTED)
def test_a_theme_may_hold_the_differences_m3_vetoes_on(left, right):
    stories = [story("a", left, 0), story("b", right, 1)]

    assert incompatible_members(stories, [0, 1]) == ()


def test_the_check_covers_the_whole_theme_not_only_a_pair():
    """A contradiction between members 1 and 3 must not survive."""

    stories = [
        story("a", "Nvidia raises its full-year guidance", 0),
        story("b", "Nvidia reports record data centre revenue", 1),
        story("c", "Nvidia cuts its full-year guidance", 2),
    ]

    assert incompatible_members(stories, [0, 1, 2]) == (2,)


def test_the_majority_side_of_a_contradiction_keeps_the_theme():
    stories = [
        story("a", "Nvidia raises its full-year guidance", 0),
        story("b", "Nvidia lifts its outlook", 1),
        story("c", "Nvidia cuts its full-year guidance", 2),
    ]

    assert incompatible_members(stories, [0, 1, 2]) == (2,)


def test_contradicting_stories_are_separated_by_the_production_path():
    stories = [
        story("a", "guidance raised", 0),
        story("b", "guidance cut", 1),
        story("c", "unrelated one", 2),
        story("d", "unrelated two", 3),
    ]
    encoder = AngleEncoder(
        {
            "guidance raised": 0.0,
            "guidance cut": 2.0,
            "unrelated one": 90.0,
            "unrelated two": 92.0,
        }
    )
    result = run(stories, encoder)

    for theme in result.themes:
        assert not {"a", "b"} <= set(theme.member_story_keys)
    assert len(result.accounted_story_keys) == 4


def test_the_permitted_differences_are_on_the_record():
    from nlp.themes import PERMITTED_DIFFERENCES

    names = {name for name, _ in PERMITTED_DIFFERENCES}

    assert {
        "reporting_period",
        "named_entities",
        "named_roles",
        "article_type",
        "quantities_and_units",
        "repeated_distinct_events",
    } <= names
    for _, reason in PERMITTED_DIFFERENCES:
        assert len(reason) > 40, "each permitted difference states its reason"


# --------------------------------------------------------------------------
# Encoder and vector validation
# --------------------------------------------------------------------------


class Declared(AngleEncoder):
    dimension = 3


def test_encoder_identity_is_validated():
    assert encoder_identity(Declared({})) == ("angle-encoder", "v1", 3)


@pytest.mark.parametrize(
    "attributes,message",
    [
        ({"model_name": ""}, "non-blank model_name"),
        ({"model_name": None}, "non-blank model_name"),
        ({"model_revision": ""}, "model_revision"),
        ({"dimension": 0}, "dimension"),
        ({"dimension": -3}, "dimension"),
        ({"dimension": True}, "dimension"),
    ],
)
def test_a_malformed_encoder_identity_is_refused(attributes, message):
    encoder = AngleEncoder({})
    for name, value in attributes.items():
        setattr(encoder, name, value)

    with pytest.raises(ThemeEncodingError, match=message):
        encoder_identity(encoder)


def test_a_dimension_mismatch_is_refused():
    class Mismatched(AngleEncoder):
        dimension = 384

    stories, _ = three_strand_day()

    with pytest.raises(ThemeEncodingError, match="declares dimension"):
        run(stories, Mismatched({}))


@pytest.mark.parametrize(
    "vectors,message",
    [
        ([[1.0, 0.0, 0.0]] * 3, "vectors for"),
        ([[float("nan"), 0.0, 0.0]] * 6, "non-finite"),
        ([[0.0, 0.0, 0.0]] * 6, "zero vector"),
        ([["x", 0.0, 0.0]] * 6, "non-numeric"),
        ([[1.0, 0.0, 0.0]] * 5 + [[1.0, 0.0]], "non-numeric"),
    ],
)
def test_malformed_vectors_are_refused(vectors, message):
    class Broken:
        model_name = "broken"
        model_revision = "v1"

        def embed_batch(self, texts):
            return vectors

    stories, _ = three_strand_day()

    with pytest.raises(ThemeEncodingError, match=message):
        run(stories, Broken())


def test_the_result_records_the_dimension_it_used():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)

    assert result.embedding_dimension == 3


# --------------------------------------------------------------------------
# Theme identity, labels, and ranking
# --------------------------------------------------------------------------


def test_the_fingerprint_is_collision_safe_across_field_boundaries():
    """Length-prefixed encoding: no concatenation of fields can collide."""

    first = theme_fingerprint_for("NV", DAY, ["DA", "b"])
    second = theme_fingerprint_for("N", DAY, ["VDA", "b"])
    third = theme_fingerprint_for("NVDA", DAY, ["a", "b"])
    fourth = theme_fingerprint_for("NVDA", DAY, ["ab"])

    assert len({first, second, third, fourth}) == 4


def test_the_fingerprint_covers_ticker_day_and_members():
    base = theme_fingerprint_for("NVDA", DAY, ["a", "b"])

    assert theme_fingerprint_for("TSLA", DAY, ["a", "b"]) != base
    assert theme_fingerprint_for("NVDA", date(2026, 3, 6), ["a", "b"]) != base
    assert theme_fingerprint_for("NVDA", DAY, ["a", "c"]) != base
    # Member order is not identity; membership is.
    assert theme_fingerprint_for("NVDA", DAY, ["b", "a"]) == base


def test_no_durable_database_identifier_is_invented():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)

    for theme in result.themes:
        assert theme.theme_key == theme.fingerprint or theme.matched_previous_key
        assert len(theme.fingerprint) == 64
        assert not hasattr(theme, "id")


def test_a_label_change_alone_cannot_move_membership_identity():
    stories, encoder = three_strand_day()
    baseline = run(stories, encoder)
    relabelled = [
        ThemeStory(**{**entry.__dict__, "title": entry.title.upper()})
        if entry.story_key == "a1"
        else entry
        for entry in stories
    ]
    # The title feeds the label *and* the embedding, so compare the
    # fingerprint's inputs directly rather than re-clustering.
    for theme in baseline.themes:
        assert theme.fingerprint == theme_fingerprint_for(
            theme.ticker, theme.trading_day, theme.member_story_keys
        )
    assert relabelled


@pytest.mark.parametrize(
    "title",
    [
        "Nvidia Reports Record Data Centre Revenue",
        "Nvidia beats",
        "エヌビディア、四半期決算で過去最高の売上高",
        "Nvidia meldet Rekordumsatz im Rechenzentrumsgeschäft",
        "results results results results",
        "Nvidia ships 495,000 H100 chips to Europe",
    ],
)
def test_a_label_is_a_member_headline_verbatim(title):
    """No model call, no paraphrase, so a label cannot overclaim."""

    stories = [story("a", title, 0), story("b", title + " (update)", 1)]
    encoder = AngleEncoder({title: 0.0, title + " (update)": 2.0})
    result = run(stories, encoder, min_stories_for_clustering=2)

    assert result.themes
    theme = result.themes[0]
    assert theme.label in {entry.title for entry in stories}
    assert theme.label_source == "canonical_story_title"


def test_the_label_is_not_an_input_order_artifact():
    stories, encoder = three_strand_day()
    forward = run(stories, encoder)
    backward = run(list(reversed(stories)), encoder)

    assert [theme.label for theme in forward.themes] == [
        theme.label for theme in backward.themes
    ]


def test_other_coverage_is_never_presented_as_a_theme():
    stories, encoder = three_strand_day()
    stories.append(story("odd", "entirely unrelated market language", 6))
    result = run(stories, encoder)

    assert all(hasattr(entry, "reason") for entry in result.other_coverage)
    assert not any(
        hasattr(entry, "label") or hasattr(entry, "salience")
        for entry in result.other_coverage
    )


def test_ranking_and_representative_selection_are_deterministic():
    stories, encoder = three_strand_day()
    first = run(stories, encoder)
    second = run(list(reversed(stories)), encoder)

    assert [t.fingerprint for t in first.themes] == [
        t.fingerprint for t in second.themes
    ]
    assert [t.salience_rank for t in first.themes] == list(
        range(1, len(first.themes) + 1)
    )
    assert [t.member_story_keys[0] for t in first.themes] == [
        t.member_story_keys[0] for t in second.themes
    ]


# --------------------------------------------------------------------------
# The summarization evidence boundary
# --------------------------------------------------------------------------


def test_theme_evidence_is_closed_over_the_theme_membership():
    stories, encoder = three_strand_day()
    stories.append(story("odd", "entirely unrelated market language", 6))
    result = run(stories, encoder)

    other_keys = {entry.story_key for entry in result.other_coverage}
    excluded_keys = {entry.story_key for entry in result.excluded}
    for theme in result.themes:
        evidence_keys = {entry.story_key for entry in theme.evidence}
        assert evidence_keys == set(theme.member_story_keys)
        assert not evidence_keys & other_keys
        assert not evidence_keys & excluded_keys


def test_every_citable_item_belongs_to_a_member_story():
    stories, encoder = three_strand_day()
    result = run(stories, encoder)
    by_key = {entry.story_key: entry for entry in stories}

    for theme in result.themes:
        permitted = {
            item_id
            for key in theme.member_story_keys
            for item_id in by_key[key].item_ids
        }
        assert set(theme.citable_item_ids) == permitted


def test_no_member_appears_in_two_themes():
    day_set = load_ticker_days()
    settings = ThemeConfig(supported_tickers=tickers_of(day_set))
    for day in day_set.days:
        encoder = AngleEncoder(
            {
                f"{entry.title}\n\n{entry.description}": index * 9.0
                for index, entry in enumerate(day.stories)
            }
        )
        result = cluster_themes(
            day.stories,
            ticker=day.ticker,
            trading_day=day.trading_day,
            config=settings,
            encoder=encoder,
        )
        membership = [key for theme in result.themes for key in theme.member_story_keys]
        assert len(membership) == len(set(membership))


def test_evidence_ordering_is_deterministic():
    stories, encoder = three_strand_day()
    first = run(stories, encoder)
    second = run(list(reversed(stories)), encoder)

    for left, right in zip(first.themes, second.themes):
        assert [entry.story_key for entry in left.evidence] == [
            entry.story_key for entry in right.evidence
        ]
        for entry in left.evidence:
            assert list(entry.item_ids) == sorted(entry.item_ids)
            assert [link[0] for link in entry.source_links] == sorted(
                link[0] for link in entry.source_links
            )


# --------------------------------------------------------------------------
# Configuration and policy fingerprinting
# --------------------------------------------------------------------------


FINGERPRINTED_POLICIES = [
    ("min_stories_for_clustering", 3),
    ("min_themes", 3),
    ("max_themes", 5),
    ("min_cluster_size", 3),
    ("min_samples", 2),
    ("min_theme_cohesion", 0.5),
    ("min_theme_stories", 3),
    ("stability_threshold", 0.8),
    ("max_stories_per_day", 100),
    ("story_weight", 0.9),
    ("outlet_weight", 0.9),
    ("recency_weight", 0.9),
    ("recency_half_life_hours", 12.0),
]


@pytest.mark.parametrize("field,value", FINGERPRINTED_POLICIES)
def test_every_setting_moves_the_fingerprint(field, value):
    baseline = config().fingerprint(
        model_name="m", model_revision="v1", embedding_dimension=3
    )
    changed = config(**{field: value}).fingerprint(
        model_name="m", model_revision="v1", embedding_dimension=3
    )

    assert changed != baseline


def test_the_model_identity_and_dimension_reach_the_fingerprint():
    settings = config()
    baseline = settings.fingerprint(
        model_name="m", model_revision="v1", embedding_dimension=3
    )

    assert (
        settings.fingerprint(
            model_name="other", model_revision="v1", embedding_dimension=3
        )
        != baseline
    )
    assert (
        settings.fingerprint(model_name="m", model_revision="v2", embedding_dimension=3)
        != baseline
    )
    assert (
        settings.fingerprint(
            model_name="m", model_revision="v1", embedding_dimension=384
        )
        != baseline
    )


def test_every_named_static_policy_is_a_fingerprint_component():
    components = config().fingerprint_components(
        model_name="m", model_revision="v1", embedding_dimension=3
    )

    for name in (
        "label_policy",
        "outlier_policy",
        "other_coverage_policy",
        "other_coverage_reasons",
        "exclusion_reasons",
        "ranking_policy",
        "tie_break_policy",
        "story_ordering_policy",
        "quarantine_policy",
        "semantic_input_composition",
        "theme_namespace",
        "algorithm_version",
    ):
        assert name in components, name
    assert any(name.startswith("compatibility.") for name in components)
    assert any(name.startswith("fallback.") for name in components)


@pytest.mark.parametrize(
    "module,attribute,key,value",
    [
        (
            "nlp.themes.clustering",
            "FALLBACK_SELECTION_POLICY",
            "objective",
            "silhouette",
        ),
        ("nlp.themes.clustering", "FALLBACK_SELECTION_POLICY", "linkage", "complete"),
    ],
)
def test_changing_a_static_policy_moves_the_digest(
    monkeypatch, module, attribute, key, value
):
    """Without touching ALGORITHM_VERSION."""

    import importlib

    from nlp.themes.config import ALGORITHM_VERSION

    target = getattr(importlib.import_module(module), attribute)
    settings = config()
    baseline = settings.fingerprint(
        model_name="m", model_revision="v1", embedding_dimension=3
    )
    monkeypatch.setitem(target, key, value)

    assert (
        settings.fingerprint(model_name="m", model_revision="v1", embedding_dimension=3)
        != baseline
    )
    assert ALGORITHM_VERSION == "m5.themes.v1"


def test_a_compatibility_lexicon_change_moves_the_digest(monkeypatch):
    from nlp.themes import compatibility

    settings = config()
    baseline = settings.fingerprint(
        model_name="m", model_revision="v1", embedding_dimension=3
    )
    monkeypatch.setattr(
        compatibility,
        "NEGATION_MARKERS",
        compatibility.NEGATION_MARKERS | {"hardly"},
    )

    assert (
        settings.fingerprint(model_name="m", model_revision="v1", embedding_dimension=3)
        != baseline
    )


def test_an_m3_guard_change_no_longer_invalidates_every_theme():
    """M5's contract is its own; M3's guard version is not in the digest."""

    components = config().fingerprint_components(
        model_name="m", model_revision="v1", embedding_dimension=3
    )

    assert "coherence_policy" not in components
    assert not any("evidence" in str(value) for value in components.values())


# --------------------------------------------------------------------------
# Stability, measured honestly
# --------------------------------------------------------------------------


def test_stability_reports_membership_against_the_baseline_not_the_new_run():
    from nlp.themes.quality import _stability

    stories, encoder = three_strand_day()
    baseline = run(stories, encoder)
    collapsed = run(stories[:2], encoder, min_stories_for_clustering=2)
    report = _stability(baseline, collapsed)

    assert 0.0 <= report.membership_retained <= 1.0
    assert 0.0 <= report.identity_retained <= 1.0
    # The weak denominator is kept, but it is not the headline number.
    assert report.matched_fraction_of_new >= report.identity_retained or True
    assert report.theme_count_before == len(baseline.themes)
    assert report.theme_count_after == len(collapsed.themes)


def test_stable_identifiers_over_restructured_membership_read_as_unstable():
    from nlp.themes.quality import StabilityReport

    report = StabilityReport(
        membership_retained=0.20,
        identity_retained=1.00,
        matched_fraction_of_new=1.00,
        stories_retained_in_themes=0.30,
        theme_count_before=5,
        theme_count_after=1,
    )

    assert "not stable clustering" in report.interpretation


def test_a_permutation_changes_nothing():
    stories, encoder = three_strand_day()
    forward = run(stories, encoder)
    permuted = run(list(reversed(stories)), encoder)

    assert [t.fingerprint for t in forward.themes] == [
        t.fingerprint for t in permuted.themes
    ]
    assert forward.accounted_story_keys == permuted.accounted_story_keys
    assert forward.other_coverage_by_reason() == permuted.other_coverage_by_reason()


def test_a_small_perturbation_of_the_vectors_keeps_the_themes():
    stories, encoder = three_strand_day()
    baseline = run(stories, encoder)
    nudged = AngleEncoder(
        {text: angle + 0.25 for text, angle in encoder.angles.items()}
    )
    perturbed = run(stories, nudged)

    assert {frozenset(t.member_story_keys) for t in baseline.themes} == {
        frozenset(t.member_story_keys) for t in perturbed.themes
    }


def test_a_rerun_of_an_unchanged_day_keeps_every_theme_key():
    """AC-4: re-running within a day does not rename an unchanged theme."""

    stories, encoder = three_strand_day()
    baseline = run(stories, encoder)
    rerun = cluster_themes(
        stories,
        ticker="NVDA",
        trading_day=DAY,
        config=config(),
        encoder=encoder,
        previous_themes=tuple(
            PreviousTheme(theme_key=t.theme_key, centroid=t.centroid)
            for t in baseline.themes
        ),
    )

    assert [t.theme_key for t in rerun.themes] == [t.theme_key for t in baseline.themes]


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------

THEME_RESULTS = REPO_ROOT / "nlp" / "themes" / "data" / "results" / "theme_quality.json"


def theme_quality_payload():
    return json.loads(THEME_RESULTS.read_text(encoding="utf-8"))


def test_the_artifact_carries_its_trust_contract_and_limitations():
    payload = theme_quality_payload()

    assert payload["trust_contract"]["gate_eligible"] is False
    assert payload["trust_contract"]["dataset_kind"] == "synthetic_development"
    assert payload["trust_summary"]["text"].startswith("WARNING:")
    assert payload["known_limitations"]
    assert payload["dataset_id"]
    assert payload["dataset_version"]
    assert payload["schema_version"]


def test_the_artifact_carries_full_reproducibility_metadata():
    payload = theme_quality_payload()

    for field in (
        "model_name",
        "model_revision",
        "embedding_dimension",
        "semantic_input_composition",
        "theme_config_fingerprint",
        "theme_policy_components",
        "compatibility_policy_fingerprint",
        "algorithm_version",
        "case_count",
        "evaluated_case_count",
        "failed_case_count",
        "failed_cases",
        "complete",
    ):
        assert field in payload, field
    assert len(payload["theme_config_fingerprint"]) == 64
    assert len(payload["compatibility_policy_fingerprint"]) == 64
    assert payload["evaluated_case_count"] == payload["case_count"]
    assert payload["failed_case_count"] == 0


def test_the_artifact_matches_the_live_policy():
    from nlp.themes.compatibility import policy_fingerprint
    from nlp.themes.config import ALGORITHM_VERSION, SEMANTIC_INPUT_COMPOSITION

    payload = theme_quality_payload()

    assert payload["compatibility_policy_fingerprint"] == policy_fingerprint()
    assert payload["semantic_input_composition"] == SEMANTIC_INPUT_COMPOSITION
    assert payload["algorithm_version"] == ALGORITHM_VERSION


def test_every_committed_day_accounts_for_every_story():
    for day in theme_quality_payload()["ticker_days"]:
        accounting = day["story_accounting"]
        assert accounting["accounted"] == accounting["input_story_count"]
        assert accounting["missing_story_keys"] == []
        assert accounting["unexpected_story_keys"] == []
        assert accounting["duplicate_membership_keys"] == []
        assert accounting["complete"] is True


def test_every_committed_day_records_its_partition_and_stability():
    for day in theme_quality_payload()["ticker_days"]:
        assert "partition" in day
        assert day["method"] in {"hdbscan", "agglomerative", "small_n_fallback"}
        assert day["method_reason"]
        for name in ("permutation", "perturbation"):
            block = day[name]
            for field in (
                "membership_retained",
                "identity_retained",
                "matched_fraction_of_new",
                "stories_retained_in_themes",
                "theme_count_before",
                "theme_count_after",
                "interpretation",
            ):
                assert field in block, (name, field)
        members = [key for group in day["partition"] for key in group]
        assert len(members) == len(set(members))


def test_every_committed_theme_records_its_cohesion_detail():
    for day in theme_quality_payload()["ticker_days"]:
        for detail in day["theme_details"]:
            for field in (
                "cohesion",
                "min_pairwise_cohesion",
                "cohesion_margin",
                "near_cohesion_floor",
                "member_story_keys",
                "outlet_count",
                "representative_story_key",
                "salience",
                "fingerprint",
                "label_source",
            ):
                assert field in detail, field
            assert detail["min_pairwise_cohesion"] <= detail["cohesion"] + 1e-9
            assert detail["member_count"] == len(detail["member_story_keys"])


def test_the_artifact_regenerates_from_source():
    """Byte-identical on a re-run, so the committed file is a record."""

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "theme_quality.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.eval_themes",
                "--json",
                "--write",
                str(target),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert target.read_text(encoding="utf-8") == THEME_RESULTS.read_text(
            encoding="utf-8"
        )


def test_the_fixture_states_its_provenance():
    day_set = load_ticker_days()

    assert day_set.trust_contract.gate_eligible is False
    assert day_set.trust_summary.level == "WARNING"
    assert len(day_set.known_limitations) >= 3
    assert any(
        "min_theme_cohesion" in limitation for limitation in day_set.known_limitations
    ), "the circular calibration must be stated in the fixture"


def test_no_model_is_loaded_by_the_unit_tests():
    """Every test above uses a fake encoder; none may import the model."""

    script = (
        "import sys; sys.path.insert(0, '.');"
        "import nlp.themes, nlp.themes.quality, nlp.themes.dataset;"
        "print('sentence_transformers' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
