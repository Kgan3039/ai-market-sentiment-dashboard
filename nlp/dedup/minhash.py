"""MinHash over normalized title shingles (issue #64, stage 2).

The issue and ``docs/PHASE_0_SPEC.md`` both specify "MinHash over title
shingles" for near-exact detection.  This module implements it, and only
it: signatures are used to **generate candidate pairs**, never to decide a
merge.  Every candidate is afterwards verified by exact structural
comparison in :mod:`nlp.dedup.detection`.  A high MinHash similarity is
therefore necessary but never sufficient.

MinHash is a probabilistic estimator, so candidate generation is not a
proof of anything: nothing here claims equivalence with exhaustive search.
``tests/test_dedup_core_signals.py`` states the stage's actual reach — with
today's strict verification it confirms exactly the pairs the exact-title
signal already finds — instead of leaving it to be discovered.

Determinism: every hash is ``blake2b`` seeded from an explicit string.
Python's randomized :func:`hash` is never used, so signatures are
byte-identical across processes, machines, and runs.

The specification asks for MinHash but does not ask for LSH banding, so no
banding is implemented.  Phase 0 handles five tickers over bounded time
windows, and candidate pairs are only formed inside one ticker partition
and one near-exact time window, which keeps the pairwise signature
comparison small.  The quadratic cost is bounded by
:attr:`nlp.dedup.config.DedupConfig.max_partition_items`, which fails a run
fast rather than letting it degrade quietly.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib

#: Largest Mersenne prime below 2**64: the standard MinHash modulus.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1

#: Documented defaults.  All three participate in the configuration
#: fingerprint through :func:`policy_fingerprint`.
DEFAULT_PERMUTATIONS = 128
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_SEED = "m2.minhash.v1"
#: Estimated-Jaccard floor for proposing a candidate pair.  Deliberately
#: permissive: it is a work-saving prefilter, not a decision threshold.
DEFAULT_CANDIDATE_SIMILARITY = 0.4


def shingles(normalized_title: str, size: int) -> frozenset[str]:
    """Return the character n-gram shingle set of a normalized title.

    Shingling runs on the *normalized* title — after wire prefixes and
    publisher attribution are stripped — so trivially decorated copies
    produce identical shingle sets and are always proposed as candidates.
    A title shorter than one shingle degrades to a single whole-title
    shingle rather than an empty set.
    """

    if not normalized_title:
        return frozenset()
    if len(normalized_title) <= size:
        return frozenset({normalized_title})
    return frozenset(
        normalized_title[index : index + size]
        for index in range(len(normalized_title) - size + 1)
    )


def stable_hash64(value: str) -> int:
    """Return a stable 64-bit hash of ``value`` (never process-seeded)."""

    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
    )


@lru_cache(maxsize=8)
def permutation_coefficients(
    permutations: int, seed: str
) -> tuple[tuple[int, int], ...]:
    """Return the ``(a, b)`` coefficients of a seeded universal hash family."""

    coefficients: list[tuple[int, int]] = []
    for index in range(permutations):
        digest = hashlib.blake2b(
            f"{seed}|{index}".encode("utf-8"), digest_size=16
        ).digest()
        multiplier = int.from_bytes(digest[:8], "big") % (_MERSENNE_PRIME - 1) + 1
        offset = int.from_bytes(digest[8:], "big") % _MERSENNE_PRIME
        coefficients.append((multiplier, offset))
    return tuple(coefficients)


def signature(
    shingle_set: frozenset[str], *, permutations: int, seed: str
) -> tuple[int, ...] | None:
    """Return the MinHash signature of a shingle set, or ``None`` if empty."""

    if not shingle_set:
        return None
    hashed = [stable_hash64(shingle) for shingle in sorted(shingle_set)]
    return tuple(
        min(
            ((multiplier * value + offset) % _MERSENNE_PRIME) & _MAX_HASH
            for value in hashed
        )
        for multiplier, offset in permutation_coefficients(permutations, seed)
    )


def estimate_similarity(
    left: tuple[int, ...] | None, right: tuple[int, ...] | None
) -> float:
    """Return the MinHash estimate of the Jaccard similarity of two sets."""

    if not left or not right or len(left) != len(right):
        return 0.0
    agreements = sum(1 for a, b in zip(left, right) if a == b)
    return agreements / len(left)


def exact_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the exact Jaccard similarity, for measuring estimator error."""

    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@lru_cache(maxsize=1)
def policy_fingerprint() -> str:
    """Return a digest of this module's static MinHash policy."""

    payload = "|".join(
        (
            "m2.minhash.v3",
            str(DEFAULT_PERMUTATIONS),
            str(DEFAULT_SHINGLE_SIZE),
            DEFAULT_SEED,
            str(_MERSENNE_PRIME),
            str(_MAX_HASH),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
