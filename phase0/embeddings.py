"""Validation for the durable side of M1's embedding cache.

``nlp.embeddings`` owns encoding, normalization, and the versioned BLOB
format; this module owns nothing but the checks a *store* has to make
before it writes or after it reads.  The blob itself is validated by
``nlp.embeddings.deserialize_vector`` so there is exactly one definition of
"this vector is intact".
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from nlp.embeddings import (
    EMBEDDING_DTYPE,
    EmbeddingStorageError,
    PersistedEmbedding,
    deserialize_vector,
)

from .errors import Phase0Error, Phase0ValidationError


class EmbeddingPersistenceError(Phase0Error, EmbeddingStorageError):
    """Stored embedding data is corrupt, incompatible, or unwritable.

    Subclasses M1's :class:`~nlp.embeddings.EmbeddingStorageError` so
    ``EmbeddingService.embed_targets`` sees the failure it already handles,
    and :class:`~phase0.errors.Phase0Error` so the persistence layer's own
    callers can catch it with everything else.
    """


SOURCE_KINDS = frozenset({"raw_item", "story", "theme"})

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def normalize_source_id(value: Any) -> str:
    """Deterministic source identity: stripped, non-empty text."""

    if value is None:
        raise Phase0ValidationError("embedding source_id is required")
    if isinstance(value, bool):
        raise Phase0ValidationError("embedding source_id must not be a boolean")
    normalized = str(value).strip()
    if not normalized:
        raise Phase0ValidationError("embedding source_id must be non-empty")
    return normalized


def normalize_source_kind(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized not in SOURCE_KINDS:
        raise Phase0ValidationError(f"unsupported embedding source_kind: {value!r}")
    return normalized


def _required_text(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise Phase0ValidationError(f"embedding {field} is required")
    return normalized


def validate_embedding(embedding: Any) -> dict[str, Any]:
    """Validate one embedding and return its column values.

    Raises :class:`EmbeddingPersistenceError` when the vector BLOB does not
    match its own metadata, so a corrupt or incompatible vector fails
    explicitly instead of being written and read back as noise.
    """

    if not isinstance(embedding, PersistedEmbedding):
        raise Phase0ValidationError(
            "embedding must be an nlp.embeddings.PersistedEmbedding"
        )
    source_kind = normalize_source_kind(embedding.source_kind)
    source_id = normalize_source_id(embedding.source_id)
    model_name = _required_text(embedding.model_name, "model_name")
    if embedding.model_revision is None:
        model_revision: str | None = None
    else:
        model_revision = _required_text(embedding.model_revision, "model_revision")
    if isinstance(embedding.dimension, bool) or not isinstance(
        embedding.dimension, int
    ):
        raise Phase0ValidationError("embedding dimension must be an integer")
    if embedding.dimension <= 0:
        raise Phase0ValidationError("embedding dimension must be positive")
    dtype = str(embedding.dtype or "").strip()
    if dtype != EMBEDDING_DTYPE:
        raise EmbeddingPersistenceError(f"unsupported embedding dtype: {dtype!r}")
    fingerprint = str(embedding.input_fingerprint or "").strip().lower()
    if not _FINGERPRINT_PATTERN.match(fingerprint):
        raise Phase0ValidationError(
            "embedding input_fingerprint must be a SHA-256 hex digest"
        )
    if not isinstance(embedding.vector_blob, (bytes, bytearray)):
        raise EmbeddingPersistenceError("embedding vector_blob must be bytes")
    blob = bytes(embedding.vector_blob)
    try:
        deserialize_vector(
            blob, expected_dimension=embedding.dimension, expected_dtype=dtype
        )
    except EmbeddingStorageError as exc:
        raise EmbeddingPersistenceError(str(exc)) from exc
    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "model_name": model_name,
        "model_revision": model_revision,
        "dimension": int(embedding.dimension),
        "dtype": dtype,
        "input_fingerprint": fingerprint,
        "vector_blob": blob,
    }


def embedding_from_row(row: Mapping[str, Any]) -> PersistedEmbedding:
    """Rebuild M1's record from a stored row, validating the vector."""

    dimension = int(row["dimension"])
    dtype = str(row["dtype"])
    blob = row["vector_blob"]
    if not isinstance(blob, (bytes, bytearray)):
        raise EmbeddingPersistenceError("stored embedding vector is not a BLOB")
    try:
        deserialize_vector(
            bytes(blob), expected_dimension=dimension, expected_dtype=dtype
        )
    except EmbeddingStorageError as exc:
        raise EmbeddingPersistenceError(str(exc)) from exc
    return PersistedEmbedding(
        source_kind=str(row["source_kind"]),  # type: ignore[arg-type]
        source_id=str(row["source_id"]),
        model_name=str(row["model_name"]),
        model_revision=(
            None if row["model_revision"] is None else str(row["model_revision"])
        ),
        input_fingerprint=str(row["input_fingerprint"]),
        dimension=dimension,
        dtype=dtype,
        vector_blob=bytes(blob),
    )


__all__ = [
    "EmbeddingPersistenceError",
    "SOURCE_KINDS",
    "embedding_from_row",
    "normalize_source_id",
    "normalize_source_kind",
    "validate_embedding",
]
