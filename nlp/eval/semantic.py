"""Scoring the M2 + M3 pipeline against the labeled sets.

The predictors run the stages in the order the pipeline runs them: M2
first, then M3 over M2's canonical output.  That is the number AC-3 is
about — a reader sees one deduplicated feed, not two stages — and it is
also the only honest way to measure M3, whose input is by definition
whatever M2 did not already collapse.

As in :mod:`nlp.eval.dedup`, nothing here reimplements a stage.  The
encoder is injected so a sweep can share one cache and a test can supply a
fake.

**Two measurements, and they are not interchangeable.**
:func:`evaluate_pipeline_isolated_pairs` invokes the stages on two records
at a time and answers "does a two-item call merge this pair"; it inherits
:data:`~nlp.eval.metrics.ISOLATED_PAIR_LIMITATION` verbatim, because M3
shares every batch-dependent behaviour M2 has — its compatibility summary
is combined across a whole prospective story and its merges are cliques,
so a third record can change what happens to two others.
:func:`evaluate_pipeline_clusters` runs a whole group in one call and is
where that behaviour is actually observable.
"""

from __future__ import annotations

from typing import Any, Sequence

from nlp.dedup import DedupConfig, deduplicate
from nlp.semdedup import (
    SemanticDedupConfig,
    StoryEncoder,
    merge_semantic_duplicates,
    stories_from_dedup,
)

from .clusters import (
    ClusterCase,
    ClusterCaseSet,
    ClusterEvaluationReport,
    ClusterPredictor,
    Partition,
    evaluate_clusters,
    to_raw_items as cluster_raw_items,
)
from .dataset import LabeledPair, PairSet
from .dedup import config_for, to_raw_items
from .metrics import (
    EvaluationReport,
    PairPrediction,
    PairPredictor,
    evaluate_isolated_pairs,
)


class CachingEncoder:
    """Memoize an encoder by text so a threshold sweep encodes once.

    A sweep re-scores the same stories at every threshold; without this the
    model would run once per point over identical input and the sweep would
    measure the encoder's throughput rather than the predicate's behaviour.
    The cache is keyed on the exact text handed to the encoder, so it can
    never return a vector for different input.
    """

    def __init__(self, encoder: StoryEncoder) -> None:
        self._encoder = encoder
        self._cache: dict[str, Any] = {}
        self.model_name = getattr(encoder, "model_name", "")
        self.model_revision = getattr(encoder, "model_revision", None)

    def embed_batch(self, texts: Sequence[str]) -> list[Any]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            for text, vector in zip(missing, self._encoder.embed_batch(missing)):
                self._cache[text] = vector
        return [self._cache[text] for text in texts]


def semantic_config_for(
    pair_set: PairSet | ClusterCaseSet,
    threshold: float | None = None,
    **overrides: Any,
) -> SemanticDedupConfig:
    """Build the M3 configuration the set declares.

    The ticker universe comes from the dataset manifest rather than a
    constant, so a set covering different symbols cannot be scored under a
    universe that silently rejects half of it.
    """

    settings: dict[str, Any] = {
        "supported_tickers": tuple(pair_set.metadata["tickers"])
    }
    if threshold is not None:
        settings["similarity_threshold"] = threshold
    settings.update(overrides)
    return SemanticDedupConfig(**settings)


def pipeline_isolated_pair_predictor(
    exact_config: DedupConfig,
    semantic_config: SemanticDedupConfig,
    encoder: StoryEncoder,
) -> PairPredictor:
    """Return a predictor that runs M2 and then M3 over one pair.

    Two records in, one two-item invocation of each stage out.  See the
    module docstring for what a two-item invocation cannot tell you.
    """

    def predict(pair: LabeledPair) -> PairPrediction:
        raw = list(to_raw_items(pair))
        exact = deduplicate(raw, config=exact_config)
        if any(len(cluster.member_ids) == 2 for cluster in exact.clusters):
            return PairPrediction(
                merged=True,
                stage="m2",
                candidate=True,
                detail="merged by m2 (exact identity)",
            )
        stories = stories_from_dedup(exact, raw)
        result = merge_semantic_duplicates(
            stories, config=semantic_config, encoder=encoder
        )
        merged_story = next(
            (story for story in result.stories if story.member_count == 2), None
        )
        if merged_story is not None:
            similarity = (
                merged_story.merges[0].similarity if merged_story.merges else None
            )
            return PairPrediction(
                merged=True,
                stage="m3",
                score=similarity,
                candidate=True,
                detail=(
                    f"merged by m3 (semantic similarity {similarity:.4f})"
                    if similarity is not None
                    else "merged by m3"
                ),
            )
        rejection = result.rejected_pairs[0] if result.rejected_pairs else None
        if rejection is None:
            return PairPrediction(
                merged=False,
                candidate=False,
                detail="not merged: no candidate (outside ticker or window)",
            )
        return PairPrediction(
            merged=False,
            score=rejection.similarity,
            candidate=True,
            detail=(
                f"not merged: {rejection.reason}"
                + (
                    f" (similarity {rejection.similarity:.4f})"
                    if rejection.similarity is not None
                    else ""
                )
            ),
        )

    return predict


#: Historical name for the same factory.
pipeline_predictor = pipeline_isolated_pair_predictor


def evaluate_pipeline_isolated_pairs(
    pair_set: PairSet,
    encoder: StoryEncoder,
    *,
    threshold: float | None = None,
    name: str = "m2+m3",
) -> EvaluationReport:
    """Score the M2 + M3 pipeline two records at a time."""

    semantic = semantic_config_for(pair_set, threshold)
    return evaluate_isolated_pairs(
        pair_set,
        pipeline_isolated_pair_predictor(config_for(pair_set), semantic, encoder),
        name=name,
        threshold=semantic.similarity_threshold,
    )


#: Historical name.  The report it returns is scoped ``isolated_pairs``.
evaluate_pipeline = evaluate_pipeline_isolated_pairs


def pipeline_cluster_predictor(
    exact_config: DedupConfig,
    semantic_config: SemanticDedupConfig,
    encoder: StoryEncoder,
) -> ClusterPredictor:
    """Return a predictor that runs M2 then M3 over a whole case at once.

    This is where M3's batch behaviour is observable: its evidence summary
    is combined across a whole prospective story and a merge is only
    accepted when every pair inside it survives, so a third record can
    change the outcome for two others exactly as it can in M2.
    """

    def predict(case: ClusterCase) -> Partition:
        raw = cluster_raw_items(case)
        exact = deduplicate(raw, config=exact_config)
        stories = stories_from_dedup(exact, raw)
        result = merge_semantic_duplicates(
            stories, config=semantic_config, encoder=encoder
        )
        return tuple(frozenset(story.member_ids) for story in result.stories)

    return predict


def evaluate_pipeline_clusters(
    case_set: ClusterCaseSet,
    encoder: StoryEncoder,
    *,
    threshold: float | None = None,
    target: str = "expected_partition",
    name: str = "m2+m3",
) -> ClusterEvaluationReport:
    """Score the M2 + M3 pipeline on whole batches.

    The default target is ground truth rather than ``exact_stage_partition``:
    the exact-stage expectation records what M2 *alone* should do, and the
    point of running M3 is to close the gap between that and what a reader
    would produce.
    """

    from nlp.dedup import DedupConfig as _DedupConfig

    exact_config = _DedupConfig(supported_tickers=tuple(case_set.metadata["tickers"]))
    semantic = semantic_config_for(case_set, threshold)
    return evaluate_clusters(
        case_set,
        pipeline_cluster_predictor(exact_config, semantic, encoder),
        name=name,
        target=target,
    )
