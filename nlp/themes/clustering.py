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
    "candidate_band": "min_themes..min(max_themes, n-1)",
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


def _qualifying_shape(
    vectors: np.ndarray, labels: Sequence[int], config: ThemeConfig
) -> tuple[int, float, int]:
    """Score one candidate clustering by the contract the stage will apply.

    Returns ``(stories in themes that will survive, their mean cohesion,
    theme count)``.  A cluster *qualifies* only if it clears both the size
    floor and the cohesion floor — the same two rules
    :func:`~nlp.themes.service.cluster_themes` applies afterwards.
    """

    groups: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        if label != NOISE:
            groups.setdefault(label, []).append(position)
    qualifying = [
        members
        for members in groups.values()
        if len(members) >= config.min_theme_stories
        and mean_pairwise_similarity(vectors, members) >= config.min_theme_cohesion
    ]
    if not qualifying:
        return 0, 0.0, 0
    covered = sum(len(members) for members in qualifying)
    cohesion = sum(
        mean_pairwise_similarity(vectors, members) for members in qualifying
    ) / len(qualifying)
    return covered, cohesion, len(qualifying)


def _best_agglomerative(
    vectors: np.ndarray, distance: np.ndarray, config: ThemeConfig
) -> tuple[int, str]:
    """Choose a cluster count in the allowed band by the theme contract.

    **Not by silhouette.**  Silhouette is a geometric shape statistic that
    knows nothing about the size and cohesion floors this stage enforces
    two steps later, and on the committed days the two objectives point in
    opposite directions: at n=17 the silhouette maximum sits at k=2, whose
    clusters the stage then dissolves, leaving one theme and fourteen
    stories in other coverage — while k=6 places sixteen of seventeen in
    themes that survive.  Choosing a clustering the stage is about to
    reject is not a defensible objective, and because the silhouette values
    involved differ in the third decimal, dropping a single story could
    flip the choice and collapse the day.

    So the objective is the outcome the stage actually wants: **the most
    stories placed in themes that will still be themes afterwards**, then
    the highest mean cohesion among them, then the fewest themes (broader
    over thinner), then the smallest k.  Every tie-break is a total order
    on data, so the choice is reproducible.
    """

    upper = min(config.max_themes, len(distance) - 1)
    lower = min(config.min_themes, upper)
    best: tuple[int, float, int, int] | None = None
    for count in range(lower, upper + 1):
        labels = _agglomerative(distance, count)
        if len(set(labels)) < 2:
            continue
        covered, cohesion, themes = _qualifying_shape(vectors, labels, config)
        # Maximize coverage, then cohesion; minimize theme count, then k.
        candidate = (-covered, -cohesion, themes, count)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return lower, f"no candidate k produced two clusters; used k={lower}"
    covered, cohesion, themes, count = -best[0], -best[1], best[2], best[3]
    if not covered:
        return count, (
            f"no k in {lower}-{upper} produced a theme clearing the size and "
            f"cohesion floors; used k={count}"
        )
    return count, (
        f"k={count} places {covered} stories in {themes} theme(s) clearing "
        f"the floors, mean cohesion {cohesion:.4f}"
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

    mean = np.mean(vectors[list(members)], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return tuple(0.0 for _ in mean)
    return tuple(round(float(value), 6) for value in mean / norm)
