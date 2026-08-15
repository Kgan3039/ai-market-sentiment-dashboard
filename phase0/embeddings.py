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
from .scalars import require_safe_identifier_scalar


class EmbeddingPersistenceError(Phase0Error, EmbeddingStorageError):
    """Stored embedding data is corrupt, incompatible, or unwritable.

    Subclasses M1's :class:`~nlp.embeddings.EmbeddingStorageError` so
    ``EmbeddingService.embed_targets`` sees the failure it already handles,
    and :class:`~phase0.errors.Phase0Error` so the persistence layer's own
    callers can catch it with everything else.
    """


SOURCE_KINDS = frozenset({"raw_item", "story", "theme"})

#: The table each source kind's durable id comes from, for error messages
#: that name the column a caller should have passed.
SOURCE_TABLES = {
    "raw_item": "raw_items",
    "story": "stories",
    "theme": "themes",
}

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: A durable row id as the schema writes it — ``CAST(id AS TEXT)`` of an
#: AUTOINCREMENT key, so a positive integer with no padding, sign, point,
#: or exponent.  Deliberately exact rather than ``str.isdigit()``: ``'01'``
#: and ``'1.0'`` are ids SQLite's integer affinity would match but the
#: ownership triggers' text comparison would not, and an identity that two
#: layers read differently is not an identity.
_DURABLE_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")


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


def require_durable_source_id(source_kind: str, value: Any) -> str:
    """The single-vector cache's identity: a durable row id, nothing else.

    ``embeddings`` is globally unique on ``(source_kind, source_id)``,
    while ``stories.cluster_fingerprint``, ``themes.fingerprint``, and
    ``themes.theme_key`` are unique only *within* one
    ticker/trading-day/pipeline-version partition.  A fingerprint is
    therefore not a cache key: two partitions may legitimately produce the
    same one, and then one partition's vector is what the other's text
    reads back.

    Nor is "unambiguous right now" enough to make one safe.  The single
    vector API has no run and no partition, so the identity it is handed
    is all it knows; and an identity that names one story today names two
    the moment another ticker-day clusters to the same fingerprint, with
    no write to this table in between to notice.  Only a globally unique
    identity is stable under that, and the durable row id is the one the
    schema guarantees.

    This is the repository's own rule, applied where it had not been:
    fingerprints are change-detection handles stored *beside* the durable
    ids, never instead of them.  The partition-aware batch
    (``persist_embeddings``) still accepts the wider identity set the
    ownership triggers define, because it has a run to check them against.
    """

    kind = normalize_source_kind(source_kind)
    normalized = normalize_source_id(value)
    if not _DURABLE_ID_PATTERN.match(normalized):
        raise Phase0ValidationError(
            f"embedding source_id {normalized!r} is not a durable {kind} id; "
            f"the single-vector cache is keyed globally, so it takes "
            f"{SOURCE_TABLES[kind]}.id and not a partition-scoped fingerprint "
            f"or key (the run-scoped persist_embeddings batch takes those)"
        )
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
    # The source id and the model identity together key the cache, so both
    # follow policy B: a credential-bearing value is refused, not redacted,
    # because a rewritten key silently returns the wrong vector forever.
    source_id = require_safe_identifier_scalar(
        normalize_source_id(embedding.source_id), "embedding source_id"
    )
    model_name = require_safe_identifier_scalar(
        _required_text(embedding.model_name, "model_name"), "embedding model_name"
    )
    if embedding.model_revision is None:
        model_revision: str | None = None
    else:
        model_revision = require_safe_identifier_scalar(
            _required_text(embedding.model_revision, "model_revision"),
            "embedding model_revision",
        )
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
    "require_durable_source_id",
    "validate_embedding",
]
