"""Scoring the M2 + M3 pipeline against the labeled set.

The predictor runs the stages in the order the pipeline runs them: M2
first, then M3 over M2's canonical output. That is the number AC-3 is
about — a reader sees one deduplicated feed, not two stages — and it is
also the only honest way to measure M3, whose input is by definition
whatever M2 did not already collapse.

As in :mod:`nlp.eval.dedup`, nothing here reimplements a stage. The encoder
is injected so a sweep can share one cache and a test can supply a fake.
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

from .dataset import LabeledPair, PairSet
from .dedup import config_for, to_raw_items
from .metrics import EvaluationReport, PairPrediction, PairPredictor, evaluate


class CachingEncoder:
    """Memoize an encoder by text so a threshold sweep encodes once.

    A sweep re-scores the same stories at every threshold; without this the
    model would run eleven times over identical input and the sweep would
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
    pair_set: PairSet, threshold: float | None = None, **overrides: Any
) -> SemanticDedupConfig:
    """Build the M3 configuration the set declares."""

    settings: dict[str, Any] = {
        "supported_tickers": tuple(pair_set.metadata["tickers"])
    }
    if threshold is not None:
        settings["similarity_threshold"] = threshold
    settings.update(overrides)
    return SemanticDedupConfig(**settings)


def pipeline_predictor(
    exact_config: DedupConfig,
    semantic_config: SemanticDedupConfig,
    encoder: StoryEncoder,
) -> PairPredictor:
    """Return a predictor that runs M2 and then M3 over one pair."""

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
                    "merged by m3 (semantic similarity " f"{similarity:.4f})"
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


def evaluate_pipeline(
    pair_set: PairSet,
    encoder: StoryEncoder,
    *,
    threshold: float | None = None,
    name: str = "m2+m3",
) -> EvaluationReport:
    """Score the M2 + M3 pipeline against the labeled set."""

    semantic = semantic_config_for(pair_set, threshold)
    return evaluate(
        pair_set,
        pipeline_predictor(config_for(pair_set), semantic, encoder),
        name=name,
        threshold=semantic.similarity_threshold,
    )
