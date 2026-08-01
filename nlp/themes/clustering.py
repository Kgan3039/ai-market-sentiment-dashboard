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

#: Label for stories HDBSCAN considers noise; they become "Other coverage".
NOISE = -1


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
    return _renumber(model.fit_predict(distance).tolist())


def _agglomerative(distance: np.ndarray, clusters: int) -> tuple[int, ...]:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=clusters, metric="precomputed", linkage="average"
    )
    return _renumber(model.fit_predict(distance).tolist())


def _best_agglomerative(distance: np.ndarray, config: ThemeConfig) -> tuple[int, str]:
    """Choose a cluster count in the allowed band by silhouette score.

    Silhouette is deterministic and needs no ground truth, which is what
    makes it usable here; it is a shape statistic, not a claim that the
    clustering is *correct*.  Ties go to the smaller count, so the stage
    prefers fewer, broader themes over more, thinner ones.
    """

    from sklearn.metrics import silhouette_score

    upper = min(config.max_themes, len(distance) - 1)
    lower = min(config.min_themes, upper)
    best_count, best_score = lower, None
    for count in range(lower, upper + 1):
        labels = _agglomerative(distance, count)
        if len({label for label in labels}) < 2:
            continue
        score = float(silhouette_score(distance, list(labels), metric="precomputed"))
        if best_score is None or score > best_score:
            best_count, best_score = count, score
    detail = "n/a" if best_score is None else f"{best_score:.4f}"
    return best_count, f"best silhouette {detail} at k={best_count}"


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
    count, detail = _best_agglomerative(distance, config)
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
