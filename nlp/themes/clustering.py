"""The clustering step: HDBSCAN, with the fallback issue #72 allows.

Issue #72 specifies "HDBSCAN over story embeddings (agglomerative fallback
if unstable)".  "Unstable" is given a concrete, checkable meaning here
rather than left to judgement: HDBSCAN is unstable at this ``n`` when the
number of clusters it finds falls outside AC-4's 2-6 band, which at Phase 0
volumes it frequently does — it will happily call a nine-story day one
cluster, or all noise.

Both algorithms run on a **precomputed cosine-distance matrix**.  Vectors
are L2-normalized, so distance is ``1 - dot``, and using a precomputed
matrix means neither library's metric support can quietly change what
"close" means between versions.

Everything is deterministic.  Neither algorithm draws random numbers, the
input is sorted before it arrives, and cluster labels are renumbered by
first appearance so the numbering is a function of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ThemeConfig
from .errors import ThemeClusteringError

#: Label for stories HDBSCAN considers noise; they become "Other coverage".
NOISE = -1

#: How much finer than AC-4's theme cap the fallback may cut the
#: dendrogram.  A larger k is not a larger theme count: the surplus
#: clusters fail the quality floors and dissolve into other coverage.
FALLBACK_CANDIDATE_CAP_FACTOR = 2

#: How the fallback picks its cluster count, and when the fallback engages.
#: Stated here so both reach the configuration fingerprint: changing either
#: changes the themes produced from identical input.
FALLBACK_SELECTION_POLICY: dict[str, str] = {
    "trigger": "cluster_count_outside_band_or_a_cluster_below_cohesion_floor",
    "objective": "max_stories_in_themes_clearing_size_and_cohesion_floors",
    "tie_break": "higher_mean_cohesion, fewer_themes, smaller_k",
    "linkage": "average",
    "metric": "precomputed_cosine_distance",
    "label_numbering": "renumbered_by_first_appearance",
    "candidate_band": "min_themes..min(n-1, max_themes*cap_factor)",
    "candidate_cap_factor": str(FALLBACK_CANDIDATE_CAP_FACTOR),
    "band_applies_to": "surviving_theme_count_not_dendrogram_cut",
    "objective_order": (
        "1_mandatory_floors, 2_most_coherent_themes_in_band, "
        "3_max_min_pairwise, 4_max_mean_cohesion, 5_max_coverage, 6_min_k"
    ),
    "coverage_rank": "5_of_6_never_outranks_coherence",
    "no_valid_partition": "no_separable_structure_no_theme_invented",
    "subset_extraction": "shed_least_central_until_both_floors_or_give_up",
}


def fallback_selection_components() -> dict[str, str]:
    """The fallback policy, sorted, for the configuration fingerprint."""

    return dict(sorted(FALLBACK_SELECTION_POLICY.items()))


@dataclass(frozen=True)
class ClusterAssignment:
    """One clustering outcome: a label per story, plus how it was reached."""

    #: ``NOISE`` or a cluster index, one per input story, in input order.
    labels: tuple[int, ...]
    #: ``"hdbscan"`` or ``"agglomerative"``.
    method: str
    reason: str

    @property
    def cluster_count(self) -> int:
        return len({label for label in self.labels if label != NOISE})


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Return the symmetric cosine-distance matrix of unit-ish vectors.

    Rows are L2-normalized first, so a caller that hands over unnormalized
    vectors gets cosine distance rather than something between cosine and
    Euclidean.  The diagonal is forced to exactly zero and the matrix is
    symmetrized, because both libraries reject a matrix that is off by a
    float ulp.
    """

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1.0, norms)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = 1.0 - similarity
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, 0.0)


def _renumber(labels: Sequence[int]) -> tuple[int, ...]:
    """Renumber clusters by first appearance, keeping noise as ``NOISE``."""

    mapping: dict[int, int] = {}
    result: list[int] = []
    for label in labels:
        if label == NOISE:
            result.append(NOISE)
            continue
        if label not in mapping:
            mapping[label] = len(mapping)
        result.append(mapping[label])
    return tuple(result)


def _hdbscan(distance: np.ndarray, config: ThemeConfig) -> tuple[int, ...]:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=min(config.min_cluster_size, len(distance)),
        min_samples=min(config.min_samples, len(distance)),
        metric="precomputed",
        allow_single_cluster=False,
    )
    try:
        labels = model.fit_predict(distance)
    except Exception as exc:  # the library's failures are not M5's contract
        raise ThemeClusteringError("hdbscan", exc) from exc
    return _renumber(labels.tolist())


def _agglomerative(distance: np.ndarray, clusters: int) -> tuple[int, ...]:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=clusters, metric="precomputed", linkage="average"
    )
    try:
        labels = model.fit_predict(distance)
    except Exception as exc:
        raise ThemeClusteringError("agglomerative", exc) from exc
    return _renumber(labels.tolist())


def theme_quality_holds(
    vectors: np.ndarray, members: Sequence[int], config: ThemeConfig
) -> bool:
    """True when a candidate group clears every mandatory theme floor.

    Both floors, not one.  A mean hides its worst link, and the worst link
    is what a reader notices: the six-story TSLA candidate held a mean of
    0.4205 over a pair at 0.2676.
    """

    return (
        len(members) >= config.min_theme_stories
        and mean_pairwise_similarity(vectors, members) >= config.min_theme_cohesion
        and min_pairwise_similarity(vectors, members)
        >= config.min_theme_pairwise_cohesion
    )


def coherent_subset(
    vectors: np.ndarray, members: Sequence[int], config: ThemeConfig
) -> tuple[int, ...]:
    """Shed the least-central members until both floors hold, or give up.

    Deterministic: at each step the member with the lowest mean similarity
    to the rest leaves, ties broken by position, and the loop stops the
    moment the group qualifies.  Returns ``()`` when no subset of at least
    ``min_theme_stories`` qualifies, in which case the caller dissolves the
    whole cluster rather than shipping the least bad part of it.
    """

    remaining = sorted(members)
    while len(remaining) >= config.min_theme_stories:
        if theme_quality_holds(vectors, remaining, config):
            return tuple(remaining)
        subset = vectors[remaining]
        similarity = 1.0 - cosine_distances(subset)
        np.fill_diagonal(similarity, 0.0)
        centrality = similarity.sum(axis=1)
        weakest = min(
            range(len(remaining)), key=lambda index: (centrality[index], index)
        )
        remaining.pop(weakest)
    return ()


def theme_rank_key(
    vectors: np.ndarray, members: Sequence[int]
) -> tuple[float, float, int, int]:
    """Order candidate themes best-first, deterministically."""

    return (
        -min_pairwise_similarity(vectors, members),
        -mean_pairwise_similarity(vectors, members),
        -len(members),
        min(members),
    )


def surviving_themes(
    vectors: np.ndarray, labels: Sequence[int], config: ThemeConfig
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return ``(themes to ship, positions dissolved to other coverage)``.

    One function, two callers: the fallback scores candidates with it and
    the service assembles the winner with it, so a candidate can never be
    chosen on a shape the service would not actually produce.

    Each cluster first sheds its least-central members until both floors
    hold.  If more clusters survive than AC-4's cap allows, the **weakest
    surplus is dissolved** rather than the whole candidate being discarded:
    a finer cut that separates two genuine strands is worth having even
    when it produces a seventh group, and listing that group plainly under
    other coverage is better than refusing the split that separated the
    first six.
    """

    groups: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        if label != NOISE:
            groups.setdefault(label, []).append(position)
    dissolved: list[int] = []
    kept: list[tuple[int, ...]] = []
    for label in sorted(groups):
        members = groups[label]
        subset = coherent_subset(vectors, members, config)
        kept.append(subset) if subset else None
        dissolved.extend(
            position for position in members if position not in set(subset)
        )
    kept.sort(key=lambda members: theme_rank_key(vectors, members))
    if len(kept) > config.max_themes:
        for surplus in kept[config.max_themes :]:
            dissolved.extend(surplus)
        kept = kept[: config.max_themes]
    return tuple(kept), tuple(sorted(dissolved))


def _candidate_quality(
    vectors: np.ndarray, labels: Sequence[int], config: ThemeConfig
) -> tuple[int, float, float, int]:
    """Score one candidate clustering, quality first.

    Judged on the themes it would actually ship, after subset extraction
    and the surplus rule.
    """

    kept, _ = surviving_themes(vectors, labels, config)
    if not kept or len(kept) < config.min_themes:
        return 0, 0.0, 0.0, 0
    weakest = min(min_pairwise_similarity(vectors, subset) for subset in kept)
    mean = sum(mean_pairwise_similarity(vectors, subset) for subset in kept) / len(kept)
    covered = sum(len(subset) for subset in kept)
    return len(kept), weakest, mean, covered


def _best_agglomerative(
    vectors: np.ndarray, distance: np.ndarray, config: ThemeConfig
) -> tuple[int, str]:
    """Choose a cluster count in the allowed band, **quality before coverage**.

    Not by silhouette: that is a geometric statistic that knows nothing
    about the floors the stage enforces, and on the committed days its
    maximum and the stage's contract point in opposite directions.

    Nor by coverage first, which was the previous objective and was still
    wrong in the same direction.  Maximizing stories-in-themes rewards a
    candidate for sweeping loosely-related stories into a broad theme, and
    that is precisely the failure "Other coverage" exists to absorb.  A
    reader is better served by three themes they recognise and five stories
    listed plainly than by five themes one of which is a grab-bag.

    So the order is:

    1. every candidate theme must clear both mandatory floors, after
       deterministic subset extraction; candidates producing no theme inside
       AC-4's band are discarded outright;
    2. most coherent themes inside the band;
    3. highest *minimum* pairwise cohesion - the weakest link across the
       whole day, so one bad pair cannot be averaged away;
    4. highest mean cohesion;
    5. most stories covered - coverage breaks ties between clusterings that
       are already equally coherent, and never outranks coherence;
    6. smallest k.
    """

    # AC-4's band bounds the *themes shipped*, not the cut of the
    # dendrogram.  Capping k at max_themes conflated the two and forced
    # unrelated strands together: with k <= 6 the eighteen-story day could
    # not separate quarterly deliveries from grid storage, because a sixth
    # cut had to hold both.  A finer cut is allowed; the clusters it
    # produces that fail the floors dissolve into other coverage, and
    # _candidate_quality still discards any candidate whose surviving theme
    # count falls outside the band.
    upper = min(len(distance) - 1, config.max_themes * FALLBACK_CANDIDATE_CAP_FACTOR)
    lower = min(config.min_themes, upper)
    best: tuple[int, float, float, int, int] | None = None
    for count in range(lower, upper + 1):
        labels = _agglomerative(distance, count)
        if len(set(labels)) < 2:
            continue
        themes, weakest, mean, covered = _candidate_quality(vectors, labels, config)
        if not themes:
            continue
        candidate = (-themes, -weakest, -mean, -covered, count)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return 0, (
            f"no k in {lower}-{upper} produced {config.min_themes}-"
            f"{config.max_themes} themes clearing the mean floor "
            f"{config.min_theme_cohesion} and the pairwise floor "
            f"{config.min_theme_pairwise_cohesion}"
        )
    themes, weakest, mean, covered, count = (
        -best[0],
        -best[1],
        -best[2],
        -best[3],
        best[4],
    )
    return count, (
        f"k={count} yields {themes} theme(s) clearing both floors, weakest "
        f"pair {weakest:.4f}, mean cohesion {mean:.4f}, {covered} stories covered"
    )


def _weakest_cohesion(vectors: np.ndarray, labels: Sequence[int]) -> float | None:
    """Return the loosest cluster's mean pairwise cosine, ignoring noise."""

    groups: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        if label != NOISE:
            groups.setdefault(label, []).append(position)
    if not groups:
        return None
    return min(
        mean_pairwise_similarity(vectors, members) for members in groups.values()
    )


#: Method name for a day whose vectors carry no separable structure, and
#: for one where no partition clears the quality contract.
NO_STRUCTURE = "no_separable_structure"


def is_degenerate(vectors: np.ndarray, config: ThemeConfig) -> bool:
    """True when every story sits at the same point in the space.

    Four identical vectors are not four stories the stage failed to
    separate; they are one story repeated, and any 2-6 split of them is an
    arbitrary line drawn to satisfy a shape requirement.
    """

    if len(vectors) < 2:
        return False
    return bool(np.max(cosine_distances(vectors)) <= config.degenerate_geometry_epsilon)


def assign_clusters(vectors: np.ndarray, config: ThemeConfig) -> ClusterAssignment:
    """Cluster a ticker-day's story vectors into AC-4's 2-6 theme band.

    HDBSCAN is treated as **unstable at this n** — and the agglomerative
    fallback issue #72 allows is used instead — in either of two cases:

    * it finds a number of clusters outside AC-4's band, which at Phase 0
      volumes it frequently does; or
    * one of its clusters is looser than ``min_theme_cohesion``.  That
      second case is the one that matters in practice: HDBSCAN under-splits
      a busy day into a single grab-bag holding the factory story, the
      permit story, and the investor-day notice, which is exactly the
      "giant catch-all cluster" a reader must never be shown as a theme.

    The caller guarantees at least ``min_stories_for_clustering`` vectors.
    """

    if is_degenerate(vectors, config):
        return ClusterAssignment(
            labels=tuple(NOISE for _ in range(len(vectors))),
            method=NO_STRUCTURE,
            reason=(
                f"every pair of the {len(vectors)} stories is identical to "
                "within the degenerate-geometry epsilon; there is no "
                "structure to cluster, so no theme is invented"
            ),
        )
    distance = cosine_distances(vectors)
    labels = _hdbscan(distance, config)
    found = len({label for label in labels if label != NOISE})
    weakest = _weakest_cohesion(vectors, labels)
    if not config.min_themes <= found <= config.max_themes:
        objection = (
            f"hdbscan found {found} clusters, outside "
            f"{config.min_themes}-{config.max_themes}"
        )
    elif weakest is not None and weakest < config.min_theme_cohesion:
        objection = (
            f"hdbscan's loosest cluster has cohesion {weakest:.4f}, below "
            f"{config.min_theme_cohesion}"
        )
    else:
        return ClusterAssignment(
            labels=labels,
            method="hdbscan",
            reason=(
                f"hdbscan found {found} clusters inside the allowed band, "
                f"loosest cohesion {weakest:.4f}"
                if weakest is not None
                else f"hdbscan found {found} clusters inside the allowed band"
            ),
        )
    count, detail = _best_agglomerative(vectors, distance, config)
    if not count:
        # No partition anywhere in the band ships a theme a reader could
        # recognise.  Saying so is the honest outcome; forcing a split to
        # satisfy AC-4's count would ship exactly the theme AC-4's other
        # half - "no story is dropped" - was written to protect.
        return ClusterAssignment(
            labels=tuple(NOISE for _ in range(len(vectors))),
            method=NO_STRUCTURE,
            reason=f"{objection}; {detail}",
        )
    return ClusterAssignment(
        labels=_agglomerative(distance, count),
        method="agglomerative",
        reason=f"{objection}; {detail}",
    )


def mean_pairwise_similarity(vectors: np.ndarray, members: Sequence[int]) -> float:
    """Return the mean cosine between distinct members, 1.0 for a single one."""

    if len(members) < 2:
        return 1.0
    subset = vectors[list(members)]
    similarity = 1.0 - cosine_distances(subset)
    upper = similarity[np.triu_indices(len(members), k=1)]
    return float(np.mean(upper))


def min_pairwise_similarity(vectors: np.ndarray, members: Sequence[int]) -> float:
    """Return the lowest cosine between distinct members, 1.0 for a single one."""

    if len(members) < 2:
        return 1.0
    subset = vectors[list(members)]
    similarity = 1.0 - cosine_distances(subset)
    return float(np.min(similarity[np.triu_indices(len(members), k=1)]))


def centroid_of(vectors: np.ndarray, members: Sequence[int]) -> tuple[float, ...]:
    """Return the unit-length mean of some member vectors, rounded.

    Rounded to six places so a centroid written to storage and read back
    matches the one the next run computes, which is what theme-identity
    matching depends on.
    """

    from .config import CENTROID_PRECISION

    mean = np.mean(vectors[list(members)], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return tuple(0.0 for _ in mean)
    return tuple(round(float(value), CENTROID_PRECISION) for value in mean / norm)
