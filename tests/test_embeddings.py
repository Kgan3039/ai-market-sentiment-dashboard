from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import nlp.embeddings as embeddings_module
from nlp.embeddings import (
    DEFAULT_MODEL_REVISION,
    EmbeddingEncodingError,
    EmbeddingInputError,
    EmbeddingModelLoadError,
    EmbeddingRepository,
    EmbeddingService,
    EmbeddingStorageError,
    EmbeddingTarget,
    PersistedEmbedding,
    canonicalize_model_input,
    compose_embedding_text,
    cosine_similarity,
    deserialize_vector,
    serialize_vector,
)


class FakeEncoder:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            seed = sum(text.encode("utf-8"))
            rows.append(
                [
                    float((seed + offset * 17) % 101 + 1)
                    for offset in range(self.dimension)
                ]
            )
        return np.asarray(rows, dtype=np.float32)


class ContractRepository:
    """In-memory implementation of the future #81 adapter contract."""

    def __init__(
        self, state: dict[tuple[str, str], PersistedEmbedding] | None = None
    ) -> None:
        self.state = state if state is not None else {}
        self.read_count = 0
        self.write_count = 0

    def get_embedding(
        self, source_kind: str, source_id: str
    ) -> PersistedEmbedding | None:
        self.read_count += 1
        return self.state.get((source_kind, source_id))

    def upsert_embedding(self, embedding: PersistedEmbedding) -> None:
        self.state[(embedding.source_kind, embedding.source_id)] = embedding
        self.write_count += 1


def service_with_fake(
    encoder: FakeEncoder | None = None,
    *,
    model_name: str = "fake/model-v1",
    model_revision: str | None = None,
) -> tuple[EmbeddingService, FakeEncoder, list[tuple[str, str | None]]]:
    fake = encoder or FakeEncoder()
    loads: list[tuple[str, str | None]] = []

    def factory(name: str, cache: str | None, revision: str | None) -> FakeEncoder:
        assert revision == model_revision
        loads.append((name, cache))
        return fake

    return (
        EmbeddingService(
            model_name=model_name,
            model_revision=model_revision,
            encoder_factory=factory,
            batch_size=8,
            expected_dimension=fake.dimension,
        ),
        fake,
        loads,
    )


def test_title_and_description_composition_is_deterministic() -> None:
    assert (
        compose_embedding_text("  NVIDIA   unveils GPU ", " New\naccelerator  ")
        == "NVIDIA unveils GPU\n\nNew accelerator"
    )


def test_canonical_model_input_preserves_paragraph_separator() -> None:
    assert (
        canonicalize_model_input("  Title \r\n \r\n Description  ")
        == "Title\n\nDescription"
    )


@pytest.mark.parametrize(
    ("title", "description", "expected"),
    [
        ("  title only  ", None, "title only"),
        (None, " description only ", "description only"),
        ("title", "", "title"),
        ("", "description", "description"),
    ],
)
def test_title_only_and_description_only(
    title: str | None, description: str | None, expected: str
) -> None:
    assert compose_embedding_text(title, description) == expected


@pytest.mark.parametrize(
    ("title", "description"), [(None, None), ("", ""), (" \n ", "\t")]
)
def test_empty_record_is_rejected(title: str | None, description: str | None) -> None:
    with pytest.raises(EmbeddingInputError):
        compose_embedding_text(title, description)


def test_single_embedding_is_float32_normalized() -> None:
    service, encoder, _ = service_with_fake()
    vector = service.embed_text("one")
    assert vector.dtype == np.float32
    assert vector.shape == (3,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert encoder.calls == [["one"]]


def test_batch_embedding_preserves_order_and_uses_one_model_call() -> None:
    service, encoder, _ = service_with_fake()
    vectors = service.embed_batch(["first", "second", "third"])
    singles = []
    for text in ["first", "second", "third"]:
        comparison, _, _ = service_with_fake()
        singles.append(comparison.embed_text(text))
    assert len(vectors) == 3
    assert all(vector.shape == (3,) for vector in vectors)
    assert all(
        np.allclose(actual, expected) for actual, expected in zip(vectors, singles)
    )
    assert encoder.calls == [["first", "second", "third"]]


def test_record_path_encodes_exact_composed_input() -> None:
    service, encoder, _ = service_with_fake()
    service.embed_record("  Title  ", " Description ")
    assert encoder.calls == [["Title\n\nDescription"]]


def test_direct_text_and_batch_paths_preserve_canonical_separator() -> None:
    service, encoder, _ = service_with_fake()
    service.embed_text("  Title \n\n Description  ")
    service.embed_batch([" First \n\n Description ", " Second "])
    assert encoder.calls == [
        ["Title\n\nDescription"],
        ["First\n\nDescription", "Second"],
    ]


def test_record_batch_preserves_composition_and_order() -> None:
    service, encoder, _ = service_with_fake()
    service.embed_records(
        [
            (" First title ", " First description "),
            (" Second title ", None),
            (None, " Third description "),
        ]
    )
    assert encoder.calls == [
        [
            "First title\n\nFirst description",
            "Second title",
            "Third description",
        ]
    ]


def test_target_fingerprint_hashes_exact_encoder_input() -> None:
    repository = ContractRepository()
    service, encoder, _ = service_with_fake()
    target = EmbeddingTarget("story", "story-1", " Title ", " Description ")
    service.embed_targets([target], repository)
    encoded_input = encoder.calls[0][0]
    stored = repository.get_embedding("story", "story-1")
    assert stored is not None
    assert encoded_input == compose_embedding_text(target.title, target.description)
    assert (
        stored.input_fingerprint
        == hashlib.sha256(encoded_input.encode("utf-8")).hexdigest()
    )


def test_model_is_lazy_loaded_once_and_reused() -> None:
    service, _, loads = service_with_fake()
    assert loads == []
    service.embed_text("first")
    service.embed_text("second")
    assert loads == [("fake/model-v1", None)]


def test_default_model_instance_is_shared_within_process(monkeypatch: Any) -> None:
    fake = FakeEncoder()
    loads = 0

    def factory(_: str, __: str | None, ___: str | None) -> FakeEncoder:
        nonlocal loads
        loads += 1
        return fake

    monkeypatch.setattr(embeddings_module, "_default_encoder_factory", factory)
    first = EmbeddingService(
        model_name="fake/shared-model",
        model_revision="shared-revision",
        expected_dimension=fake.dimension,
    )
    second = EmbeddingService(
        model_name="fake/shared-model",
        model_revision="shared-revision",
        expected_dimension=fake.dimension,
    )
    assert loads == 0
    first.embed_text("first")
    second.embed_text("second")
    assert loads == 1


def test_configurable_cache_location_is_passed_to_loader(tmp_path: Any) -> None:
    fake = FakeEncoder()
    loads: list[tuple[str, str | None]] = []

    def factory(name: str, cache: str | None, revision: str | None) -> FakeEncoder:
        assert revision == DEFAULT_MODEL_REVISION
        loads.append((name, cache))
        return fake

    service = EmbeddingService(
        cache_location=tmp_path / "models",
        expected_dimension=fake.dimension,
        encoder_factory=factory,
    )
    service.embed_text("text")
    assert loads == [
        ("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "models"))
    ]


def test_encoder_output_count_and_dimension_are_validated() -> None:
    class WrongCountEncoder(FakeEncoder):
        def encode(self, texts: list[str], **_: Any) -> np.ndarray:
            return np.ones((len(texts) + 1, 3), dtype=np.float32)

    service, _, _ = service_with_fake(WrongCountEncoder())
    with pytest.raises(EmbeddingEncodingError, match="count"):
        service.embed_batch(["one", "two"])


def test_cosine_identity_and_orthogonality() -> None:
    assert cosine_similarity([1, 2], [1, 2]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([], []),
        ([0, 0], [1, 0]),
        ([1, 2], [1]),
        ([math.nan, 1], [1, 1]),
        ([math.inf, 1], [1, 1]),
        ([1 + 2j, 2], [1, 2]),
        (["not-a-number"], [1]),
    ],
)
def test_cosine_rejects_invalid_vectors(first: Any, second: Any) -> None:
    with pytest.raises(EmbeddingInputError):
        cosine_similarity(first, second)


def test_vector_blob_round_trip() -> None:
    original = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    restored = deserialize_vector(serialize_vector(original), expected_dimension=3)
    assert restored.dtype == np.float32
    assert np.allclose(restored, original, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"not-an-embedding",
        serialize_vector([1, 2])[:-1],
        serialize_vector([1, 2]) + b"x",
    ],
)
def test_corrupt_blob_is_rejected(blob: bytes) -> None:
    with pytest.raises(EmbeddingStorageError):
        deserialize_vector(blob, expected_dimension=2)


def test_nonfinite_blob_payload_is_rejected() -> None:
    blob = bytearray(serialize_vector([1.0]))
    blob[-4:] = np.asarray([math.nan], dtype="<f4").tobytes()
    with pytest.raises(EmbeddingStorageError, match="invalid values"):
        deserialize_vector(bytes(blob), expected_dimension=1)


def test_dimension_and_dtype_metadata_mismatch_are_rejected() -> None:
    blob = serialize_vector([1, 2, 3])
    with pytest.raises(EmbeddingStorageError, match="dimension"):
        deserialize_vector(blob, expected_dimension=2)
    with pytest.raises(EmbeddingStorageError, match="dtype"):
        deserialize_vector(blob, expected_dimension=3, expected_dtype="float64")


def test_repository_adapter_recreation_uses_shared_backing_state() -> None:
    state: dict[tuple[str, str], PersistedEmbedding] = {}
    first_repository = ContractRepository(state)
    service, encoder, _ = service_with_fake()
    target = EmbeddingTarget("raw_item", 42, "Title", "Description")

    first = service.embed_targets([target], first_repository)[0]
    reconnected_repository = ContractRepository(state)
    second_service, second_encoder, _ = service_with_fake()
    second = second_service.embed_targets([target], reconnected_repository)[0]

    assert isinstance(first_repository, EmbeddingRepository)
    assert np.allclose(first, second)
    assert len(encoder.calls) == 1
    assert second_encoder.calls == []
    assert reconnected_repository.write_count == 0


def test_unchanged_embedding_is_idempotent() -> None:
    repository = ContractRepository()
    service, encoder, _ = service_with_fake()
    target = EmbeddingTarget("story", "story-1", "Title", "Description")
    first = service.embed_targets([target], repository)[0]
    second = service.embed_targets([target], repository)[0]
    assert np.allclose(first, second)
    assert len(encoder.calls) == 1
    assert repository.write_count == 1


def test_changed_input_invalidates_stored_embedding() -> None:
    repository = ContractRepository()
    service, encoder, _ = service_with_fake()
    original = EmbeddingTarget("story", "story-1", "Original", None)
    changed = EmbeddingTarget("story", "story-1", "Changed", None)
    service.embed_targets([original], repository)
    before = repository.get_embedding("story", "story-1")
    service.embed_targets([changed], repository)
    after = repository.get_embedding("story", "story-1")
    assert before is not None and after is not None
    assert before.input_fingerprint != after.input_fingerprint
    assert len(encoder.calls) == 2
    assert repository.write_count == 2


def test_changed_model_invalidates_stored_embedding() -> None:
    repository = ContractRepository()
    target = EmbeddingTarget("theme", "theme-1", "Title", None)
    first_service, _, _ = service_with_fake(model_name="fake/model-v1")
    second_service, second_encoder, _ = service_with_fake(model_name="fake/model-v2")
    first_service.embed_targets([target], repository)
    second_service.embed_targets([target], repository)
    stored = repository.get_embedding("theme", "theme-1")
    assert stored is not None
    assert stored.model_name == "fake/model-v2"
    assert len(second_encoder.calls) == 1
    assert repository.write_count == 2


def test_changed_model_revision_invalidates_stored_embedding() -> None:
    repository = ContractRepository()
    target = EmbeddingTarget("story", "story-1", "Title", None)
    first_service, _, _ = service_with_fake(model_revision="revision-1")
    second_service, second_encoder, _ = service_with_fake(model_revision="revision-2")
    first_service.embed_targets([target], repository)
    second_service.embed_targets([target], repository)
    stored = repository.get_embedding("story", "story-1")
    assert stored is not None
    assert stored.model_revision == "revision-2"
    assert len(second_encoder.calls) == 1
    assert repository.write_count == 2


@pytest.mark.parametrize(
    "targets",
    [
        [
            EmbeddingTarget("story", "same", "Title", "Description"),
            EmbeddingTarget("story", "same", "Title", "Description"),
        ],
        [
            EmbeddingTarget("story", "same", "First", None),
            EmbeddingTarget("story", "same", "Second", None),
        ],
    ],
)
def test_duplicate_target_identity_is_rejected_before_side_effects(
    targets: list[EmbeddingTarget],
) -> None:
    repository = ContractRepository()
    service, encoder, _ = service_with_fake()
    with pytest.raises(
        EmbeddingInputError,
        match="duplicate embedding target identity: story:same",
    ):
        service.embed_targets(targets, repository)
    assert encoder.calls == []
    assert repository.read_count == 0
    assert repository.write_count == 0


def test_distinct_target_identities_preserve_order() -> None:
    repository = ContractRepository()
    service, encoder, _ = service_with_fake()
    targets = [
        EmbeddingTarget("story", "story-2", "Second", None),
        EmbeddingTarget("raw_item", "raw-1", "First", None),
        EmbeddingTarget("theme", "theme-3", "Third", None),
    ]
    vectors = service.embed_targets(targets, repository)
    assert encoder.calls == [["Second", "First", "Third"]]
    assert len(vectors) == len(targets)
    assert [
        repository.get_embedding(target.source_kind, str(target.source_id)) is not None
        for target in targets
    ] == [True, True, True]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: replace(value, vector_blob=b"corrupt"),
        lambda value: replace(value, dimension=value.dimension + 1),
        lambda value: replace(value, dtype="float64"),
    ],
)
def test_incompatible_cached_embedding_is_rejected(mutation: Any) -> None:
    repository = ContractRepository()
    service, _, _ = service_with_fake()
    target = EmbeddingTarget("story", "story-1", "Title", None)
    service.embed_targets([target], repository)
    key = ("story", "story-1")
    repository.state[key] = mutation(repository.state[key])
    with pytest.raises(EmbeddingStorageError):
        service.embed_targets([target], repository)


def test_custom_model_cache_requires_expected_dimension() -> None:
    repository = ContractRepository()
    first_service, _, _ = service_with_fake()
    target = EmbeddingTarget("story", "story-1", "Title", None)
    first_service.embed_targets([target], repository)
    service = EmbeddingService(
        model_name="fake/model-v1",
        model_revision=None,
        expected_dimension=None,
        encoder_factory=lambda *_: FakeEncoder(),
    )
    with pytest.raises(EmbeddingStorageError, match="expected_dimension"):
        service.embed_targets([target], repository)


def test_repository_rejects_invalid_stored_metadata_type() -> None:
    class InvalidRepository(ContractRepository):
        def get_embedding(self, source_kind: str, source_id: str) -> Any:
            return {"vector_blob": b"not-a-record"}

    service, _, _ = service_with_fake()
    target = EmbeddingTarget("story", "story-1", "Title", None)
    with pytest.raises(EmbeddingStorageError, match="metadata"):
        service.embed_targets([target], InvalidRepository())


def test_model_load_failure_is_explicit() -> None:
    def failing_factory(_: str, __: str | None, ___: str | None) -> Any:
        raise OSError("model unavailable")

    service = EmbeddingService(
        model_revision=DEFAULT_MODEL_REVISION,
        encoder_factory=failing_factory,
    )
    with pytest.raises(EmbeddingModelLoadError, match="failed to load"):
        service.embed_text("valid input")


def test_missing_sentence_transformers_package_is_explicit(
    monkeypatch: Any,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    service = EmbeddingService(
        model_name="fake/missing-package-model",
        model_revision=None,
        expected_dimension=3,
    )
    with pytest.raises(
        EmbeddingModelLoadError, match="sentence-transformers is required"
    ):
        service.embed_text("valid input")


def test_encoding_failure_is_explicit() -> None:
    class FailingEncoder:
        def encode(self, _: list[str], **__: Any) -> np.ndarray:
            raise RuntimeError("encode failed")

    service = EmbeddingService(encoder_factory=lambda *_: FailingEncoder())
    with pytest.raises(EmbeddingEncodingError, match="failed to encode"):
        service.embed_text("valid input")


def test_module_import_does_not_import_sentence_transformers() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import nlp.embeddings; "
                "print('sentence_transformers' in sys.modules)"
            ),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
