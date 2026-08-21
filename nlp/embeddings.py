"""Deterministic local embeddings shared by Phase 0 NLP stages.

The module deliberately knows nothing about SQLite.  Persistence is expressed
through :class:`EmbeddingRepository`, which the Phase 0 repository from issue
#57 must implement (or adapt) once it lands on ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import threading
from typing import (
    Any,
    Callable,
    Literal,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_VECTOR_DIMENSION = 384
MODEL_CACHE_ENV = "TICKER_NARRATIVES_MODEL_CACHE"
EMBEDDING_DTYPE = "float32"
_BLOB_MAGIC = b"TNEMB001"
_BLOB_HEADER = struct.Struct(">8sIB")
_FLOAT32_CODE = 1
_PARAGRAPH_BREAK_PATTERN = re.compile(r"\n(?:[ \t]*\n)+")
_SHARED_ENCODERS: dict[tuple[str, str | None, str | None], Any] = {}
_SHARED_ENCODERS_LOCK = threading.Lock()

FloatVector = NDArray[np.float32]
SourceKind = Literal["raw_item", "story", "theme"]
EncoderFactory = Callable[[str, str | None, str | None], Any]


class EmbeddingError(RuntimeError):
    """Base error for embedding failures."""


class EmbeddingInputError(ValueError, EmbeddingError):
    """Input text or vector data is invalid."""


class EmbeddingModelLoadError(EmbeddingError):
    """The local sentence-transformers model could not be loaded."""


class EmbeddingEncodingError(EmbeddingError):
    """The model failed to encode valid input."""


class EmbeddingStorageError(EmbeddingError):
    """Persisted embedding data is corrupt or incompatible."""


@dataclass(frozen=True)
class EmbeddingTarget:
    """A raw item, story, or theme whose text should be embedded."""

    source_kind: SourceKind
    source_id: str | int
    title: str | None = None
    description: str | None = None

    def normalized_source_id(self) -> str:
        value = str(self.source_id).strip()
        if not value:
            raise EmbeddingInputError("source_id must be non-empty")
        if self.source_kind not in {"raw_item", "story", "theme"}:
            raise EmbeddingInputError(f"unsupported source_kind: {self.source_kind}")
        return value


@dataclass(frozen=True)
class PersistedEmbedding:
    """Repository-neutral representation of one stored embedding."""

    source_kind: SourceKind
    source_id: str
    model_name: str
    model_revision: str | None
    input_fingerprint: str
    dimension: int
    dtype: str
    vector_blob: bytes


@runtime_checkable
class EmbeddingRepository(Protocol):
    """Public persistence boundary, implemented by issue #57's repository.

    :class:`phase0.repository.Phase0Repository` satisfies this protocol.
    Note that it accepts only a *durable row id* as ``source_id``; see
    :func:`phase0.embeddings.require_durable_source_id`.  Under the I5
    decision record (decision A) the Phase 0 pipeline does not use this
    cache for story or theme vectors, because a story has no durable id
    until after the stage that needs its vector has run.
    """

    def get_embedding(
        self, source_kind: SourceKind, source_id: str
    ) -> PersistedEmbedding | None:
        """Return the current embedding for a source, or ``None``."""

    def upsert_embedding(self, embedding: PersistedEmbedding) -> None:
        """Atomically insert or replace one source's embedding metadata/blob."""


def _normalize_component(value: str | None, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise EmbeddingInputError(f"{field} must be a string or None")
    return " ".join(value.split())


def canonicalize_model_input(text: str) -> str:
    """Normalize text while preserving canonical paragraph separators.

    Runs of two or more newlines become exactly ``"\\n\\n"``. Whitespace
    inside each paragraph is collapsed to a single space.
    """

    if not isinstance(text, str):
        raise EmbeddingInputError("text must be a string")
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in _PARAGRAPH_BREAK_PATTERN.split(normalized_newlines)
    ]
    canonical = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    if not canonical:
        raise EmbeddingInputError("text must contain usable content")
    return canonical


def compose_embedding_text(title: str | None, description: str | None) -> str:
    """Build deterministic model input from normalized title and description.

    Components have all surrounding/repeated whitespace collapsed.  When both
    are present they are joined by exactly two newline characters.  No labels
    or placeholder text are fabricated.
    """

    normalized_title = _normalize_component(title, "title")
    normalized_description = _normalize_component(description, "description")
    parts = [part for part in (normalized_title, normalized_description) if part]
    if not parts:
        raise EmbeddingInputError("title or description must contain usable text")
    return canonicalize_model_input("\n\n".join(parts))


def input_fingerprint(text: str) -> str:
    """Return a stable SHA-256 fingerprint for normalized model input."""

    canonical = canonicalize_model_input(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_vector(
    vector: Any, *, expected_dimension: int | None = None
) -> FloatVector:
    try:
        array = np.asarray(vector)
    except (TypeError, ValueError) as exc:
        raise EmbeddingInputError(
            "vector must be a one-dimensional numeric value"
        ) from exc
    if array.ndim != 1 or array.size == 0:
        raise EmbeddingInputError("vector must be non-empty and one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise EmbeddingInputError("vector must contain numeric values")
    try:
        converted = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingInputError("vector cannot be represented as float32") from exc
    if not np.all(np.isfinite(converted)):
        raise EmbeddingInputError("vector must contain only finite values")
    if expected_dimension is not None and converted.size != expected_dimension:
        raise EmbeddingInputError(
            f"vector dimension {converted.size} does not match "
            f"expected dimension {expected_dimension}"
        )
    return np.ascontiguousarray(converted)


def _l2_normalize(vector: Any) -> FloatVector:
    validated = _validated_vector(vector)
    norm = float(np.linalg.norm(validated.astype(np.float64)))
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbeddingInputError("embedding vector must have non-zero finite norm")
    normalized = validated / np.float32(norm)
    if not np.all(np.isfinite(normalized)):
        raise EmbeddingInputError("normalized embedding contains invalid values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    """Return finite cosine similarity in ``[-1, 1]`` for valid vectors."""

    first = _validated_vector(vector_a)
    second = _validated_vector(vector_b, expected_dimension=first.size)
    first_norm = float(np.linalg.norm(first.astype(np.float64)))
    second_norm = float(np.linalg.norm(second.astype(np.float64)))
    if first_norm == 0.0 or second_norm == 0.0:
        raise EmbeddingInputError("cosine similarity is undefined for zero vectors")
    similarity = float(
        np.dot(first.astype(np.float64), second.astype(np.float64))
        / (first_norm * second_norm)
    )
    if not math.isfinite(similarity):
        raise EmbeddingInputError("cosine similarity is not finite")
    return max(-1.0, min(1.0, similarity))


def serialize_vector(vector: Any) -> bytes:
    """Serialize a finite float32 vector using the versioned Phase 0 BLOB."""

    validated = _validated_vector(vector)
    little_endian = validated.astype("<f4", copy=False)
    header = _BLOB_HEADER.pack(_BLOB_MAGIC, little_endian.size, _FLOAT32_CODE)
    return header + little_endian.tobytes(order="C")


def deserialize_vector(
    blob: bytes,
    *,
    expected_dimension: int,
    expected_dtype: str = EMBEDDING_DTYPE,
) -> FloatVector:
    """Deserialize and strictly validate the versioned Phase 0 vector BLOB."""

    if not isinstance(blob, bytes):
        raise EmbeddingStorageError("embedding BLOB must be bytes")
    if expected_dimension <= 0:
        raise EmbeddingStorageError("expected vector dimension must be positive")
    if expected_dtype != EMBEDDING_DTYPE:
        raise EmbeddingStorageError(
            f"unsupported embedding dtype metadata: {expected_dtype}"
        )
    if len(blob) < _BLOB_HEADER.size:
        raise EmbeddingStorageError("embedding BLOB is truncated")
    magic, dimension, dtype_code = _BLOB_HEADER.unpack_from(blob)
    if magic != _BLOB_MAGIC:
        raise EmbeddingStorageError("embedding BLOB has invalid format marker")
    if dtype_code != _FLOAT32_CODE:
        raise EmbeddingStorageError("embedding BLOB has unsupported numeric dtype")
    if dimension != expected_dimension:
        raise EmbeddingStorageError(
            f"stored vector dimension {dimension} does not match metadata "
            f"dimension {expected_dimension}"
        )
    expected_size = _BLOB_HEADER.size + dimension * np.dtype("<f4").itemsize
    if len(blob) != expected_size:
        raise EmbeddingStorageError("embedding BLOB byte length is invalid")
    vector = np.frombuffer(blob, dtype="<f4", offset=_BLOB_HEADER.size).copy()
    try:
        return _validated_vector(vector, expected_dimension=dimension)
    except EmbeddingInputError as exc:
        raise EmbeddingStorageError("embedding BLOB contains invalid values") from exc


def _default_encoder_factory(
    model_name: str,
    cache_folder: str | None,
    model_revision: str | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, ModuleNotFoundError) as exc:
        raise EmbeddingModelLoadError(
            "sentence-transformers is required for local embeddings"
        ) from exc
    try:
        model_kwargs: dict[str, Any] = {"cache_folder": cache_folder}
        if model_revision is not None:
            model_kwargs["revision"] = model_revision
        return SentenceTransformer(model_name, **model_kwargs)
    except Exception as exc:
        raise EmbeddingModelLoadError(
            f"failed to load local embedding model {model_name!r}"
        ) from exc


class EmbeddingService:
    """Lazy, batch-oriented local embedding service."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = None,
        cache_location: str | Path | None = None,
        batch_size: int = 32,
        expected_dimension: int | None = None,
        encoder_factory: EncoderFactory | None = None,
    ) -> None:
        normalized_model = str(model_name).strip()
        if not normalized_model:
            raise EmbeddingInputError("model_name must be non-empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise EmbeddingInputError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise EmbeddingInputError("batch_size must be a positive integer")
        if expected_dimension is None:
            configured_dimension: int | None = (
                DEFAULT_VECTOR_DIMENSION
                if normalized_model == DEFAULT_MODEL_NAME
                else None
            )
        elif (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension <= 0
        ):
            raise EmbeddingInputError(
                "expected_dimension must be a positive integer or None"
            )
        else:
            configured_dimension = expected_dimension
        configured_cache = cache_location or os.getenv(MODEL_CACHE_ENV)
        self.model_name = normalized_model
        if model_revision is None:
            configured_revision: str | None = (
                DEFAULT_MODEL_REVISION
                if normalized_model == DEFAULT_MODEL_NAME
                else None
            )
        else:
            configured_revision = str(model_revision).strip()
        self.model_revision = configured_revision
        if model_revision is not None and not configured_revision:
            raise EmbeddingInputError("model_revision must be non-empty when provided")
        self.cache_location = (
            str(Path(configured_cache).expanduser()) if configured_cache else None
        )
        self.batch_size = batch_size
        self._uses_default_factory = encoder_factory is None
        self._encoder_factory = encoder_factory or _default_encoder_factory
        self._encoder: Any | None = None
        self._load_lock = threading.Lock()
        self._dimension = configured_dimension

    @property
    def dimension(self) -> int | None:
        """Configured or model-observed vector dimension."""

        return self._dimension

    def _get_encoder(self) -> Any:
        if self._uses_default_factory:
            cache_key = (
                self.model_name,
                self.cache_location,
                self.model_revision,
            )
            with _SHARED_ENCODERS_LOCK:
                encoder = _SHARED_ENCODERS.get(cache_key)
                if encoder is None:
                    encoder = self._load_encoder()
                    _SHARED_ENCODERS[cache_key] = encoder
                self._encoder = encoder
                return encoder
        if self._encoder is None:
            with self._load_lock:
                if self._encoder is None:
                    self._encoder = self._load_encoder()
        return self._encoder

    def _load_encoder(self) -> Any:
        try:
            return self._encoder_factory(
                self.model_name,
                self.cache_location,
                self.model_revision,
            )
        except EmbeddingModelLoadError:
            raise
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"failed to load local embedding model {self.model_name!r}"
            ) from exc

    def embed_text(self, text: str) -> FloatVector:
        """Embed one non-empty string as an L2-normalized float32 vector."""

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[FloatVector]:
        """Embed a batch in one model call while preserving input order."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingInputError("texts must be a sequence of strings")
        if not texts:
            return []
        canonical_texts = []
        for index, text in enumerate(texts):
            try:
                canonical_texts.append(canonicalize_model_input(text))
            except EmbeddingInputError as exc:
                raise EmbeddingInputError(f"texts[{index}] is invalid: {exc}") from exc
        try:
            encoder = self._get_encoder()
        except EmbeddingModelLoadError:
            raise
        try:
            encoded = encoder.encode(
                canonical_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingEncodingError(
                f"failed to encode {len(canonical_texts)} text item(s)"
            ) from exc
        try:
            matrix = np.asarray(encoded)
        except (TypeError, ValueError) as exc:
            raise EmbeddingEncodingError("encoder returned non-numeric output") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(canonical_texts):
            raise EmbeddingEncodingError(
                "encoder output count does not match input count"
            )
        vectors: list[FloatVector] = []
        for row in matrix:
            try:
                vector = _l2_normalize(row)
            except EmbeddingInputError as exc:
                raise EmbeddingEncodingError(
                    "encoder returned an invalid embedding vector"
                ) from exc
            if self._dimension is None:
                self._dimension = int(vector.size)
            elif vector.size != self._dimension:
                raise EmbeddingEncodingError(
                    "encoder returned inconsistent vector dimensions"
                )
            vectors.append(vector)
        return vectors

    def embed_record(self, title: str | None, description: str | None) -> FloatVector:
        """Compose and embed one title/description record."""

        return self.embed_text(compose_embedding_text(title, description))

    def embed_records(
        self, records: Sequence[tuple[str | None, str | None]]
    ) -> list[FloatVector]:
        """Compose and batch-embed title/description records."""

        texts = [
            compose_embedding_text(title, description) for title, description in records
        ]
        return self.embed_batch(texts)

    def embed_targets(
        self,
        targets: Sequence[EmbeddingTarget],
        repository: EmbeddingRepository,
    ) -> list[FloatVector]:
        """Return cached-compatible vectors or batch-encode and persist misses."""

        if not isinstance(repository, EmbeddingRepository):
            raise TypeError("repository does not implement EmbeddingRepository")
        normalized: list[tuple[EmbeddingTarget, str, str, str]] = []
        seen_identities: set[tuple[SourceKind, str]] = set()
        for target in targets:
            source_id = target.normalized_source_id()
            identity = (target.source_kind, source_id)
            if identity in seen_identities:
                raise EmbeddingInputError(
                    "duplicate embedding target identity: "
                    f"{target.source_kind}:{source_id}"
                )
            seen_identities.add(identity)
            text = compose_embedding_text(target.title, target.description)
            normalized.append((target, source_id, text, input_fingerprint(text)))

        results: list[FloatVector | None] = []
        misses: list[int] = []
        for index, (target, source_id, _, fingerprint) in enumerate(normalized):
            stored = repository.get_embedding(target.source_kind, source_id)
            if stored is not None and not isinstance(stored, PersistedEmbedding):
                raise EmbeddingStorageError(
                    "repository returned invalid embedding metadata"
                )
            if (
                stored is not None
                and stored.model_name == self.model_name
                and stored.model_revision == self.model_revision
                and stored.input_fingerprint == fingerprint
            ):
                if stored.source_kind != target.source_kind:
                    raise EmbeddingStorageError("stored source kind does not match")
                if stored.source_id != source_id:
                    raise EmbeddingStorageError(
                        "stored source identifier does not match"
                    )
                if self._dimension is None:
                    raise EmbeddingStorageError(
                        "expected_dimension is required to load a cached "
                        "custom-model embedding"
                    )
                vector = deserialize_vector(
                    stored.vector_blob,
                    expected_dimension=stored.dimension,
                    expected_dtype=stored.dtype,
                )
                if vector.size != self._dimension:
                    raise EmbeddingStorageError(
                        "stored vector dimension is incompatible with this model"
                    )
                results.append(vector)
            else:
                results.append(None)
                misses.append(index)

        if misses:
            vectors = self.embed_batch([normalized[index][2] for index in misses])
            for index, vector in zip(misses, vectors):
                target, source_id, _, fingerprint = normalized[index]
                stored = PersistedEmbedding(
                    source_kind=target.source_kind,
                    source_id=source_id,
                    model_name=self.model_name,
                    model_revision=self.model_revision,
                    input_fingerprint=fingerprint,
                    dimension=int(vector.size),
                    dtype=EMBEDDING_DTYPE,
                    vector_blob=serialize_vector(vector),
                )
                repository.upsert_embedding(stored)
                results[index] = vector
        if any(result is None for result in results):
            raise EmbeddingStorageError("embedding result assembly is incomplete")
        return [cast(FloatVector, result) for result in results]


_DEFAULT_SERVICE: EmbeddingService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_default_service() -> EmbeddingService:
    """Return the process-wide lazy default embedding service."""

    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = EmbeddingService()
    return _DEFAULT_SERVICE


def embed_text(text: str) -> FloatVector:
    """Embed one text using the process-wide default local model."""

    return get_default_service().embed_text(text)


def embed_batch(texts: Sequence[str]) -> list[FloatVector]:
    """Embed texts using the process-wide default local model."""

    return get_default_service().embed_batch(texts)
