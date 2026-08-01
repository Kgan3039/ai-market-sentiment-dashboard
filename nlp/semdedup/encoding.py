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
    encoded = encoder.embed_batch(texts)
    if len(encoded) != len(texts):
        raise SemanticDedupEncodingError(
            f"encoder returned {len(encoded)} vectors for {len(texts)} stories"
        )
    dimension: int | None = None
    for index, vector in zip(indices, encoded):
        size = _validated_size(vector, stories[index].story_key)
        if dimension is None:
            dimension = size
        elif size != dimension:
            raise SemanticDedupEncodingError(
                "encoder returned inconsistent vector dimensions "
                f"({dimension} then {size})"
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
