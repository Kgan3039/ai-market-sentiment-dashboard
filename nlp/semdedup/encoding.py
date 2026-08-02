"""The embedding boundary: one protocol, one composition rule, no model.

M3 never loads a model.  It takes an encoder, which
:class:`nlp.embeddings.EmbeddingService` already satisfies, so unit tests
inject a deterministic fake and no test in this package touches the network
or the sentence-transformers cache.

The text that gets embedded is composed by M1's
:func:`~nlp.embeddings.compose_embedding_text` — the same composition the
rest of Phase 0 uses, so a stored vector is reusable across stages.  Note
the deliberate asymmetry with M2, which owns its content rules outright:
M2's identity keys must not move when the encoder's input composition
changes, whereas M3's whole job *is* to ask the encoder a question, so it
must ask it the same way everyone else does.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence, runtime_checkable

from nlp.embeddings import EmbeddingError, compose_embedding_text

from .errors import SemanticDedupEncodingError
from .models import StoryInput


@runtime_checkable
class StoryEncoder(Protocol):
    """What M3 needs from an embedding provider.

    :class:`nlp.embeddings.EmbeddingService` implements this as-is.
    """

    model_name: str
    model_revision: str | None

    def embed_batch(self, texts: Sequence[str]) -> list[Any]:
        """Return one vector per text, in the order supplied."""


def story_text(story: StoryInput) -> str | None:
    """Return the exact text M3 embeds, or ``None`` when there is none.

    A story with no usable text cannot be compared to anything.  That is a
    singleton, not an error: dropping it would lose coverage and guessing
    would invent a merge.
    """

    try:
        return compose_embedding_text(story.title, story.description)
    except EmbeddingError:
        return None


def validate_model_metadata(encoder: StoryEncoder) -> tuple[str, str | None]:
    """Return the encoder's declared identity, or refuse to record nothing.

    A stored vector is only reusable if the run that produced it is
    identifiable.  A blank model name makes the fingerprint meaningless and
    a cached result unfalsifiable, so it is rejected rather than recorded
    as an empty string.
    """

    name = getattr(encoder, "model_name", None)
    if not isinstance(name, str) or not name.strip():
        raise SemanticDedupEncodingError(
            "encoder must declare a non-blank model_name; a vector whose "
            "producer is unnamed cannot be invalidated when the model moves"
        )
    revision = getattr(encoder, "model_revision", None)
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise SemanticDedupEncodingError(
            "encoder model_revision must be a non-blank string or None"
        )
    return name.strip(), revision.strip() if isinstance(revision, str) else None


def validate_dimension(value: object) -> int | None:
    """Validate a declared embedding dimension, or ``None`` when unknown."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SemanticDedupEncodingError(
            f"declared embedding dimension must be a positive integer, got {value!r}"
        )
    return value


def encode_stories(
    stories: Sequence[StoryInput], encoder: StoryEncoder
) -> tuple[list[Any | None], int]:
    """Embed every story that has text, preserving input order.

    Returns the per-story vectors (``None`` where there was no text) and the
    number of stories that could not be encoded.  Everything encodable goes
    through the encoder in **one** call, so a batching provider sees the
    batch it was built for.
    """

    indices = [index for index, story in enumerate(stories) if story_text(story)]
    texts = [story_text(stories[index]) or "" for index in indices]
    vectors: list[Any | None] = [None] * len(stories)
    if not texts:
        return vectors, len(stories)
    encoded = list(encoder.embed_batch(texts))
    if len(encoded) != len(texts):
        raise SemanticDedupEncodingError(
            f"encoder returned {len(encoded)} vectors for {len(texts)} stories; "
            "the counts must match exactly, in order"
        )
    declared = validate_dimension(getattr(encoder, "dimension", None))
    dimension: int | None = declared
    for index, vector in zip(indices, encoded):
        size = _validated_size(vector, stories[index].story_key)
        if dimension is None:
            dimension = size
        elif size != dimension:
            raise SemanticDedupEncodingError(
                "encoder returned a vector of dimension "
                f"{size} where {dimension} was "
                + ("declared" if declared is not None else "seen first")
                + f" (story {stories[index].story_key!r})"
            )
        vectors[index] = vector
    return vectors, len(stories) - len(indices)


def _validated_size(vector: Any, story_key: str) -> int:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise SemanticDedupEncodingError(
            f"encoder returned a non-numeric vector for story {story_key!r}"
        ) from exc
    if not values:
        raise SemanticDedupEncodingError(
            f"encoder returned an empty vector for story {story_key!r}"
        )
    if not all(math.isfinite(value) for value in values):
        raise SemanticDedupEncodingError(
            f"encoder returned a non-finite vector for story {story_key!r}"
        )
    if not any(values):
        raise SemanticDedupEncodingError(
            f"encoder returned a zero vector for story {story_key!r}; "
            "cosine similarity is undefined for it"
        )
    return len(values)
