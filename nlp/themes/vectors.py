"""The committed story vectors, and the encoder that replays them.

Evaluating M5 needs vectors, and computing them needs a model — which means
a model load, a warm cache, and a machine whose float arithmetic matches
whoever committed the last artifact.  None of those belong on the default
test route, and the third one quietly makes a "byte-identical regeneration"
claim untrue.

So the vectors are computed once, rounded to a documented precision, and
committed.  The evaluation replays them through :class:`FixtureEncoder`,
which is a real encoder as far as the stage is concerned — it declares a
model name, revision and dimension, and refuses text it has no vector for
rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import VECTOR_PRECISION
from .errors import ThemeEncodingError, ThemeInputError

SUPPORTED_VECTOR_SCHEMA = "phase0.theme_vectors.v1"

#: Every M5 JSON asset carries its own trust manifest, so none of them is
#: interpretable without knowing what it is worth.  A vector file looks like
#: neutral numbers; it is the evidence three ticker-days of themes rest on.
TRUST_MANIFEST_POLICY = (
    "every_m5_json_asset_carries_trust_contract_shared_summary_and_stage_"
    "summary; a vector or fixture file is not interpretable without it"
)
DEFAULT_VECTOR_PATH = Path(__file__).resolve().parent / "data" / "story_vectors.json"


@dataclass(frozen=True)
class StoryVectors:
    """One committed vector per fixture story, with its provenance."""

    dataset_id: str
    model_name: str
    model_revision: str | None
    dimension: int
    vector_precision: int
    composition: str
    vectors: Mapping[str, tuple[float, ...]]


def load_story_vectors(path: str | Path = DEFAULT_VECTOR_PATH) -> StoryVectors:
    """Load and validate the committed vectors, or refuse them."""

    location = Path(path).resolve()
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ThemeInputError(f"{location}: vector fixture not found") from exc
    except json.JSONDecodeError as exc:
        raise ThemeInputError(f"{location}: not valid JSON: {exc}") from exc
    if payload.get("schema_version") != SUPPORTED_VECTOR_SCHEMA:
        raise ThemeInputError(
            f"{location}: unsupported vector schema_version "
            f"{payload.get('schema_version')!r}"
        )
    unknown = sorted(
        set(payload)
        - {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "issue",
            "trust_contract",
            "trust_summary",
            "stage_specific_trust_summary",
            "model_name",
            "model_revision",
            "dimension",
            "vector_precision",
            "composition",
            "note",
            "vectors",
        }
    )
    if unknown:
        raise ThemeInputError(f"{location}: unknown vector-fixture field(s) {unknown}")
    raw = payload.get("vectors")
    if not isinstance(raw, dict) or not raw:
        raise ThemeInputError(f"{location}: holds no vectors")
    dimension = int(payload["dimension"])
    vectors: dict[str, tuple[float, ...]] = {}
    for key, values in raw.items():
        if not isinstance(values, list) or len(values) != dimension:
            raise ThemeInputError(
                f"{location}: story {key!r} has {len(values)} values, "
                f"not the declared {dimension}"
            )
        vectors[key] = tuple(float(value) for value in values)
    for field in ("trust_contract", "trust_summary", "stage_specific_trust_summary"):
        if field not in payload:
            raise ThemeInputError(
                f"{location}: vector fixture has no {field}; an M5 asset must "
                "state what it is worth"
            )
    if payload["trust_contract"].get("gate_eligible") is not False:
        raise ThemeInputError(
            f"{location}: vector fixture claims gate eligibility; these "
            "vectors come from an authored development fixture"
        )
    return StoryVectors(
        dataset_id=str(payload.get("dataset_id", "")),
        model_name=str(payload["model_name"]),
        model_revision=payload.get("model_revision"),
        dimension=dimension,
        vector_precision=int(payload.get("vector_precision", VECTOR_PRECISION)),
        composition=str(payload.get("composition", "")),
        vectors=vectors,
    )


class FixtureEncoder:
    """Replay committed vectors, keyed by the exact text the stage composes.

    Keyed on text rather than on story key so the stage's own composition
    is exercised: a change to what M5 embeds shows up here as a missing
    vector instead of passing silently on a stale one.
    """

    source = "committed_vectors"

    def __init__(self, store: StoryVectors, *, texts: Mapping[str, str] | None = None):
        self.store = store
        self.model_name = store.model_name
        self.model_revision = store.model_revision
        self.dimension = store.dimension
        self._by_text: dict[str, tuple[float, ...]] = {}
        if texts:
            for key, text in texts.items():
                self._by_text[text] = store.vectors[key]

    def bind(self, texts: Mapping[str, str]) -> None:
        """Register the text each committed story key was encoded from."""

        for key, text in texts.items():
            if key in self.store.vectors:
                self._by_text[text] = self.store.vectors[key]

    def embed_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        missing = [text for text in texts if text not in self._by_text]
        if missing:
            raise ThemeEncodingError(
                f"{len(missing)} text(s) have no committed vector; the fixture "
                "and the committed vectors have drifted apart. Refresh with "
                "tools.eval_themes --real-model --write-vectors"
            )
        return [self._by_text[text] for text in texts]


def write_story_vectors(day_set: Any, encoder: Any, path: str | Path) -> Path:
    """Recompute the committed vectors from a real encoder."""

    from nlp.embeddings import compose_embedding_text

    vectors: dict[str, list[float]] = {}
    dimension = 0
    for day in day_set.days:
        texts = [
            compose_embedding_text(story.title, story.description)
            for story in day.stories
        ]
        for story, vector in zip(day.stories, encoder.embed_batch(texts)):
            rounded = [round(float(value), VECTOR_PRECISION) for value in vector]
            dimension = len(rounded)
            vectors[story.story_key] = rounded
    payload = {
        "schema_version": SUPPORTED_VECTOR_SCHEMA,
        "dataset_id": day_set.dataset_id,
        "model_name": encoder.model_name,
        "model_revision": encoder.model_revision,
        "dimension": dimension,
        "vector_precision": VECTOR_PRECISION,
        "composition": "m1.compose_embedding_text(title, description)",
        "vectors": {key: vectors[key] for key in sorted(vectors)},
    }
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return location
