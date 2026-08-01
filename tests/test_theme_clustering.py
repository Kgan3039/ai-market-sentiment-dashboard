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
    PreviousTheme,
    ThemeCapacityError,
    ThemeConfig,
    ThemeConfigError,
    ThemeEncodingError,
    ThemeInputError,
    ThemeStory,
    cluster_themes,
    theme_fingerprint_for,
)
from nlp.themes.clustering import cosine_distances
from nlp.themes.coherence import conflicting_members, story_polarities
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

    assert conflicting_members(themed, positions) == (2,)
    assert stories  # the fixture names are only for readability


def test_theme_coherence_only_objects_to_opposing_claims():
    """Different quarters belong in one earnings theme; opposite calls do not."""

    quarters = [
        story("q3", "Nvidia reports Q3 data centre revenue", 0),
        story("q4", "Nvidia reports Q4 data centre revenue", 1),
    ]

    assert conflicting_members(quarters, [0, 1]) == ()
    assert story_polarities(quarters[0]) == frozenset()


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
        assert entry.item_ids


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
