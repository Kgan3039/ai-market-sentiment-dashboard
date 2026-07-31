"""Signal-level tests for the M2 core: URL identity and MinHash (issue #64).

These two are the parts a behavioural test cannot fully pin down: URL
identity is a large space of near-miss spellings, and MinHash is the
stage 2 the issue requires, whose role has to be stated precisely rather
than assumed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import os
import subprocess
import sys
import textwrap

import pytest

from nlp.dedup import DedupConfig, RawItem, deduplicate
from nlp.dedup.detection import minhash_candidates, verify_candidate
from nlp.dedup.minhash import (
    DEFAULT_PERMUTATIONS,
    DEFAULT_SEED,
    DEFAULT_SHINGLE_SIZE,
    estimate_similarity,
    exact_similarity,
    permutation_coefficients,
    shingles,
    signature,
)
from nlp.dedup.service import _normalize_items
from nlp.dedup.urls import clean_url, url_identity_key

UTC = timezone.utc
BASE = datetime(2026, 3, 2, 13, 0, tzinfo=UTC)
CONFIG = DedupConfig(supported_tickers=["NVDA"])


def sign(text: str) -> tuple[int, ...]:
    return signature(
        shingles(text, DEFAULT_SHINGLE_SIZE),
        permutations=DEFAULT_PERMUTATIONS,
        seed=DEFAULT_SEED,
    )


# --------------------------------------------------------------------------
# URL identity is conservative by construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ("https://www.reuters.com/a", "https://reuters.com/a", "www is not an alias"),
        ("https://m.reuters.com/a", "https://reuters.com/a", "mobile host"),
        ("https://amp.cnn.com/a", "https://cnn.com/a", "amp host"),
        ("https://cnn.com/a/amp", "https://cnn.com/a", "amp path suffix"),
        ("https://cnn.com//a", "https://cnn.com/a", "duplicate slashes"),
        ("https://cnn.com/a/", "https://cnn.com/a", "trailing slash"),
        ("http://cnn.com/a", "https://cnn.com/a", "scheme is part of identity"),
        ("https://cnn.com/a?ref=home", "https://cnn.com/a", "ref can be meaningful"),
        ("https://cnn.com/a?page=2", "https://cnn.com/a?page=3", "pagination"),
        (
            "https://news.google.com/rss/articles/AB?url=https://cnn.com/a",
            "https://cnn.com/a",
            "redirect wrappers are never unwrapped",
        ),
        ("https://cnn.com/s?q=a+b", "https://cnn.com/s?q=a%20b", "plus is not space"),
        ("https://cnn.com/a?t=1&t=2", "https://cnn.com/a?t=2&t=1", "repeated order"),
        ("https://cnn.com/a?x=1&y=2", "https://cnn.com/a?y=2&x=1", "query order"),
        ("https://cnn.com/a%2Fb", "https://cnn.com/a/b", "percent-encoding"),
    ],
)
def test_potentially_distinct_urls_never_share_an_identity(left, right, reason):
    assert url_identity_key(left) != url_identity_key(right), reason


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://reuters.com/a?utm_source=x&utm_medium=y", "https://reuters.com/a"),
        ("https://reuters.com/a?fbclid=X", "https://reuters.com/a"),
        ("https://reuters.com/a?gclid=X", "https://reuters.com/a"),
        ("https://reuters.com/a#section", "https://reuters.com/a"),
        ("https://REUTERS.com/a", "https://reuters.com/a"),
        ("https://reuters.com:443/a", "https://reuters.com/a"),
        ("https://reuters.com./a", "https://reuters.com/a"),
        ("https://münchen.example/a", "https://xn--mnchen-3ya.example/a"),
        ("https://MÜNCHEN.example/a", "https://xn--mnchen-3ya.example/a"),
    ],
)
def test_safe_url_equivalences_share_an_identity(left, right):
    key = url_identity_key(left)
    assert key is not None and key == url_identity_key(right)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "not a url",
        "mailto:desk@reuters.com",
        "javascript:alert(1)",
        "ftp://reuters.com/a",
        "//reuters.com/a",
        "reuters.com/a",
        "https:///a",
        "https://localhost/a",
        "https://[2001:db8::1]/a",
        "https://reuters..com/a",
        "https://reuters.com:99999/a",
        "https://reuters.com:0/a",
    ],
)
def test_unusable_urls_have_no_identity(value):
    assert url_identity_key(value) is None


def test_credentials_block_identity_but_not_display():
    url = "https://user:secret@reuters.com/a"
    assert url_identity_key(url) is None
    assert clean_url(url) == "https://reuters.com/a"


def test_a_repeated_url_still_merges_compatible_syndicated_copies():
    url = "https://reuters.com/technology/one-story"
    result = deduplicate(
        [
            RawItem(
                "a",
                "NVDA",
                "Nvidia reports revenue",
                "Beat estimates.",
                url,
                None,
                "Reuters",
                BASE,
            ),
            RawItem(
                "b",
                "NVDA",
                None,
                None,
                f"{url}?utm_source=rss",
                None,
                "CNBC",
                BASE + timedelta(hours=1),
            ),
        ],
        config=CONFIG,
    )
    # One record is text-free, so the gate keeps them apart even here.
    assert len(result.clusters) == 2

    compatible = deduplicate(
        [
            RawItem(
                "a",
                "NVDA",
                "Nvidia reports revenue",
                "Beat estimates.",
                url,
                None,
                "Reuters",
                BASE,
            ),
            RawItem(
                "b",
                "NVDA",
                "Nvidia reports revenue",
                None,
                f"{url}?utm_source=rss",
                None,
                "CNBC",
                BASE + timedelta(hours=1),
            ),
        ],
        config=CONFIG,
    )
    assert [c.member_ids for c in compatible.clusters] == [("a", "b")]


# --------------------------------------------------------------------------
# MinHash: required by issue #64, deterministic, and honestly scoped
# --------------------------------------------------------------------------


def test_shingles_and_signatures_are_deterministic():
    assert shingles("abcdef", 5) == frozenset({"abcde", "bcdef"})
    assert shingles("amd", 5) == frozenset({"amd"})
    assert shingles("", 5) == frozenset()
    assert sign("nvidia reports revenue") == sign("nvidia reports revenue")
    assert len(sign("nvidia reports revenue")) == DEFAULT_PERMUTATIONS
    assert signature(frozenset(), permutations=8, seed=DEFAULT_SEED) is None


def test_permutation_coefficients_are_seeded_not_randomized():
    first = permutation_coefficients(16, "seed-a")
    assert first == permutation_coefficients(16, "seed-a")
    assert first != permutation_coefficients(16, "seed-b")
    assert all(multiplier > 0 for multiplier, _ in first)


def test_no_python_hash_is_used_anywhere_in_the_core():
    source = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            r"\bhash(",
            "--",
            "nlp/dedup",
        ],
        capture_output=True,
        text=True,
    )
    assert source.stdout == "", f"randomized hash() found:\n{source.stdout}"


def test_signatures_are_stable_across_processes_and_hash_seeds():
    script = textwrap.dedent(
        """
        from nlp.dedup.minhash import shingles, signature
        print(signature(shingles("nvidia reports record revenue", 5),
                        permutations=32, seed="m2.minhash.v1"))
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


def test_the_estimate_tracks_the_exact_jaccard():
    left, right = "nvidia reports record revenue", "nvidia reports recent revenue"
    exact = exact_similarity(
        shingles(left, DEFAULT_SHINGLE_SIZE), shingles(right, DEFAULT_SHINGLE_SIZE)
    )
    assert abs(estimate_similarity(sign(left), sign(right)) - exact) < 0.15
    assert estimate_similarity(None, sign(left)) == 0.0
    assert estimate_similarity((1, 2, 3), (1, 2)) == 0.0


def _batch(pairs):
    items = []
    for index, (left, right) in enumerate(pairs):
        items.append(
            RawItem(
                f"{index}-a",
                "NVDA",
                left,
                "Body.",
                f"https://reuters.com/{index}a",
                None,
                "Reuters",
                BASE,
            )
        )
        items.append(
            RawItem(
                f"{index}-b",
                "NVDA",
                right,
                "Body.",
                f"https://cnbc.com/{index}b",
                None,
                "CNBC",
                BASE + timedelta(hours=1),
            )
        )
    return items


def test_minhash_runs_and_reports_its_work():
    result = deduplicate(
        _batch(
            [
                (
                    "Nvidia reports record revenue",
                    "UPDATE 2-Nvidia reports record revenue",
                ),
                ("Nvidia opens a research site", "NVIDIA OPENS A RESEARCH SITE"),
            ]
        ),
        config=CONFIG,
    )
    assert result.stats.candidate_pair_count >= 2
    assert result.stats.verified_candidate_pair_count >= 2
    assert result.stats.cluster_count == 2


def test_verification_is_conservative_and_reads_only_the_two_records():
    items = _batch(
        [("Nvidia beats revenue estimates", "Nvidia beats revenue estimate")]
    )
    normalized = _normalize_items(items, CONFIG)
    candidates = minhash_candidates(normalized, [0, 1], CONFIG)
    assert candidates == [(0, 1)], "the pair must be proposed"
    assert (
        estimate_similarity(
            sign(normalized[0].title_key or ""), sign(normalized[1].title_key or "")
        )
        > 0.8
    )
    assert not verify_candidate(normalized[0], normalized[1])
    assert len(deduplicate(items, config=CONFIG).clusters) == 2


def test_minhash_adds_no_unique_merge_today_and_nothing_claims_otherwise():
    """Stage 2 currently confirms what stage 1 already finds.

    Verification requires identical normalized titles, which is exactly the
    exact-title signal, so every verified candidate is a pair stage 1 also
    proposes.  The stage is retained because issue #64 specifies it and
    because it is where M4's labelled data will widen matching — not
    because it contributes merges now.
    """

    pairs = [
        ("Nvidia reports record revenue", "UPDATE 2-Nvidia reports record revenue"),
        ("Nvidia opens a research site", "Nvidia opens a research site - Reuters"),
        ("Nvidia names a new CFO", "Nvidia names a new COO"),
    ]
    items = _batch(pairs)
    normalized = _normalize_items(items, CONFIG)
    ordered = list(range(len(normalized)))
    verified = {
        (left, right)
        for left, right in minhash_candidates(normalized, ordered, CONFIG)
        if verify_candidate(normalized[left], normalized[right])
    }
    exact_title = {
        (left, right)
        for left, right in itertools.combinations(ordered, 2)
        if normalized[left].title_key is not None
        and normalized[left].title_key == normalized[right].title_key
    }
    assert verified == exact_title
    # And the reason is therefore never the one recorded on a member.
    result = deduplicate(items, config=CONFIG)
    recorded = {
        member.match_reason.value
        for cluster in result.clusters
        for member in cluster.members
    }
    assert "near_exact_title" not in recorded


def test_candidates_are_deterministic_and_window_bounded():
    items = _batch([("Nvidia reports record revenue", "Nvidia reports record revenue")])
    normalized = _normalize_items(items, CONFIG)
    first = minhash_candidates(normalized, [0, 1], CONFIG)
    assert first == minhash_candidates(normalized, [0, 1], CONFIG)
    assert first == [(0, 1)]

    distant = [
        RawItem(
            "a",
            "NVDA",
            "Nvidia reports record revenue",
            "Body.",
            "https://reuters.com/a",
            None,
            "Reuters",
            BASE,
        ),
        RawItem(
            "b",
            "NVDA",
            "Nvidia reports record revenue",
            "Body.",
            "https://cnbc.com/b",
            None,
            "CNBC",
            BASE + timedelta(hours=40),
        ),
    ]
    assert minhash_candidates(_normalize_items(distant, CONFIG), [0, 1], CONFIG) == []


def test_undated_and_unparseable_titles_are_never_candidates():
    items = [
        RawItem(
            "a",
            "NVDA",
            "Nvidia reports record revenue",
            "Body.",
            "https://reuters.com/a",
            None,
            "Reuters",
            None,
        ),
        RawItem(
            "b",
            "NVDA",
            "Nvidia reports record revenue",
            "Body.",
            "https://cnbc.com/b",
            None,
            "CNBC",
            BASE,
        ),
        RawItem(
            "c",
            "NVDA",
            "Profit ½ higher",
            "Body.",
            "https://ft.com/c",
            None,
            "Financial Times",
            BASE,
        ),
    ]
    normalized = _normalize_items(items, CONFIG)
    assert minhash_candidates(normalized, [0, 1, 2], CONFIG) == []


def test_no_lsh_banding_is_implemented():
    # The issue specifies MinHash, not LSH; adding banding would be scope
    # the authoritative sources do not ask for.
    import nlp.dedup.minhash as module

    assert not [name for name in dir(module) if "band" in name.lower()]
