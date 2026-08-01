"""The labeled deduplication pair set: schema, loading, and validation.

A pair set is two files that travel together: a manifest naming the schema
version, the controlled vocabularies, and the provenance of the labels, and
a JSONL file holding one pair per line.  Splitting them keeps the manifest
readable and the 150-row body diff-friendly.

Loading is strict on purpose.  An evaluation set is only worth something if
a reviewer can trust that every row was checked: an unknown category, a
duplicated pair id, a naive timestamp, or a label that contradicts its
expected stage is a defect in the *dataset*, and this module refuses to
hand it to a scorer rather than quietly measuring against corrupt ground
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

#: Bumped when the pair schema changes shape.  A manifest declaring any
#: other version is rejected rather than best-effort parsed.
SUPPORTED_SCHEMA_VERSION = "phase0.dedup_eval.v1"

#: The committed set shipped with the repository.
DEFAULT_META_PATH = Path(__file__).resolve().parent / "data" / "dedup_pairs.meta.json"

_REQUIRED_META_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "pairs_file",
        "tickers",
        "labels",
        "expected_stages",
        "confidences",
        "categories",
        "provenance",
        "labeling",
    }
)

_REQUIRED_PAIR_KEYS = frozenset(
    {
        "pair_id",
        "ticker",
        "label",
        "category",
        "expected_stage",
        "rationale",
        "confidence",
        "item_a",
        "item_b",
    }
)

_ITEM_KEYS = frozenset(
    {
        "item_id",
        "ticker",
        "title",
        "description",
        "url",
        "canonical_url",
        "source",
        "published_at",
        "provider_item_id",
    }
)

#: A duplicate pair must be somebody's responsibility; a pair that must
#: never merge must be nobody's.  Enforcing the correspondence stops a
#: mislabelled row from silently inflating one stage's recall.
_STAGE_BY_LABEL = {
    "duplicate": frozenset({"m2", "m3"}),
    "distinct": frozenset({"none"}),
    "ambiguous": frozenset({"none"}),
}


class EvalDatasetError(ValueError):
    """The evaluation dataset is malformed and must not be scored against."""


@dataclass(frozen=True)
class LabeledItem:
    """One side of a labeled pair, projected from the ``raw_items`` columns.

    ``ticker`` is resolved at load time: a row that does not carry its own
    inherits the pair's.  Every optional field is ``None`` when the feed did
    not supply it, which is different from an empty string and is treated
    that way by every stage.
    """

    item_id: str
    ticker: str
    title: str | None
    description: str | None
    url: str | None
    canonical_url: str | None
    source: str | None
    published_at: str | None
    provider_item_id: str | None


@dataclass(frozen=True)
class LabeledPair:
    """One labeled candidate pair."""

    pair_id: str
    ticker: str
    #: ``duplicate`` | ``distinct`` | ``ambiguous``.
    label: str
    category: str
    #: ``m2`` | ``m3`` | ``none``; design intent, not a prediction.
    expected_stage: str
    rationale: str
    confidence: str
    item_a: LabeledItem
    item_b: LabeledItem

    @property
    def is_positive(self) -> bool:
        """True when the two records describe one event."""

        return self.label == "duplicate"

    @property
    def is_scored(self) -> bool:
        """True when this pair counts toward the headline metrics.

        ``ambiguous`` pairs do not: a metric computed against a label the
        labelling author already flagged as arguable measures the coin flip,
        not the stage.  They are reported separately instead.
        """

        return self.label in {"duplicate", "distinct"}


@dataclass(frozen=True)
class PairSet:
    """A validated labeled set plus the manifest it was loaded from."""

    dataset_id: str
    schema_version: str
    metadata: Mapping[str, Any]
    #: Ordered by ``pair_id``; the file is required to be sorted, so this
    #: order is the file order and is stable across processes.
    pairs: tuple[LabeledPair, ...]
    source_path: Path

    def __iter__(self) -> Iterator[LabeledPair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def scored(self) -> tuple[LabeledPair, ...]:
        """Pairs counting toward precision and recall."""

        return tuple(pair for pair in self.pairs if pair.is_scored)

    @property
    def ambiguous(self) -> tuple[LabeledPair, ...]:
        """Pairs deliberately excluded from the headline metrics."""

        return tuple(pair for pair in self.pairs if pair.label == "ambiguous")

    def by_id(self, pair_id: str) -> LabeledPair:
        """Return one pair, or raise :class:`KeyError`."""

        for pair in self.pairs:
            if pair.pair_id == pair_id:
                return pair
        raise KeyError(pair_id)

    def composition(self) -> dict[str, dict[str, int]]:
        """Return deterministic counts by label, category, stage, ticker."""

        return {
            "label": _tally(pair.label for pair in self.pairs),
            "category": _tally(pair.category for pair in self.pairs),
            "expected_stage": _tally(pair.expected_stage for pair in self.pairs),
            "ticker": _tally(pair.ticker for pair in self.pairs),
            "confidence": _tally(pair.confidence for pair in self.pairs),
        }


def _tally(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _require_str(value: Any, field: str, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalDatasetError(f"{where}: {field} must be a non-blank string")
    return value


def _optional_str(value: Any, field: str, *, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvalDatasetError(
            f"{where}: {field} must be a non-blank string or null; "
            "omit the key rather than supplying an empty value"
        )
    return value


def _require_vocabulary(
    value: Any, field: str, allowed: Sequence[str], *, where: str
) -> str:
    text = _require_str(value, field, where=where)
    if text not in allowed:
        raise EvalDatasetError(
            f"{where}: {field}={text!r} is not one of {sorted(allowed)}"
        )
    return text


def _parse_item(
    payload: Any,
    *,
    pair_id: str,
    side: str,
    default_ticker: str,
    tickers: Sequence[str],
) -> LabeledItem:
    where = f"{pair_id}.{side}"
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"{where}: must be an object")
    unknown = sorted(set(payload) - _ITEM_KEYS)
    if unknown:
        raise EvalDatasetError(f"{where}: unknown field(s) {unknown}")
    ticker = default_ticker
    if "ticker" in payload:
        ticker = _require_vocabulary(payload["ticker"], "ticker", tickers, where=where)
    published_at = _optional_str(
        payload.get("published_at"), "published_at", where=where
    )
    if published_at is not None:
        _validate_timestamp(published_at, where=where)
    return LabeledItem(
        item_id=_require_str(payload.get("item_id"), "item_id", where=where),
        ticker=ticker,
        title=_optional_str(payload.get("title"), "title", where=where),
        description=_optional_str(
            payload.get("description"), "description", where=where
        ),
        url=_optional_str(payload.get("url"), "url", where=where),
        canonical_url=_optional_str(
            payload.get("canonical_url"), "canonical_url", where=where
        ),
        source=_optional_str(payload.get("source"), "source", where=where),
        published_at=published_at,
        provider_item_id=_optional_str(
            payload.get("provider_item_id"), "provider_item_id", where=where
        ),
    )


def _validate_timestamp(value: str, *, where: str) -> None:
    """Reject anything the dedup core would reject, at load time.

    A naive timestamp in a fixture is a dataset bug that would otherwise
    surface as a stage exception halfway through a scoring run.
    """

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EvalDatasetError(
            f"{where}: published_at is not an ISO-8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise EvalDatasetError(
            f"{where}: published_at must carry a timezone offset: {value!r}"
        )


def _parse_pair(payload: Any, metadata: Mapping[str, Any], *, line: int) -> LabeledPair:
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"line {line}: each row must be a JSON object")
    missing = sorted(_REQUIRED_PAIR_KEYS - set(payload))
    if missing:
        raise EvalDatasetError(f"line {line}: missing field(s) {missing}")
    unknown = sorted(set(payload) - _REQUIRED_PAIR_KEYS)
    if unknown:
        raise EvalDatasetError(f"line {line}: unknown field(s) {unknown}")
    pair_id = _require_str(payload["pair_id"], "pair_id", where=f"line {line}")
    ticker = _require_vocabulary(
        payload["ticker"], "ticker", metadata["tickers"], where=pair_id
    )
    label = _require_vocabulary(
        payload["label"], "label", metadata["labels"], where=pair_id
    )
    expected_stage = _require_vocabulary(
        payload["expected_stage"],
        "expected_stage",
        metadata["expected_stages"],
        where=pair_id,
    )
    permitted = _STAGE_BY_LABEL.get(label)
    if permitted is not None and expected_stage not in permitted:
        raise EvalDatasetError(
            f"{pair_id}: label={label!r} is incompatible with "
            f"expected_stage={expected_stage!r}"
        )
    pair = LabeledPair(
        pair_id=pair_id,
        ticker=ticker,
        label=label,
        category=_require_vocabulary(
            payload["category"], "category", metadata["categories"], where=pair_id
        ),
        expected_stage=expected_stage,
        rationale=_require_str(payload["rationale"], "rationale", where=pair_id),
        confidence=_require_vocabulary(
            payload["confidence"], "confidence", metadata["confidences"], where=pair_id
        ),
        item_a=_parse_item(
            payload["item_a"],
            pair_id=pair_id,
            side="item_a",
            default_ticker=ticker,
            tickers=metadata["tickers"],
        ),
        item_b=_parse_item(
            payload["item_b"],
            pair_id=pair_id,
            side="item_b",
            default_ticker=ticker,
            tickers=metadata["tickers"],
        ),
    )
    if pair.item_a.item_id == pair.item_b.item_id:
        raise EvalDatasetError(f"{pair_id}: both sides share item_id")
    return pair


def _validate_metadata(metadata: Any, *, path: Path) -> Mapping[str, Any]:
    if not isinstance(metadata, dict):
        raise EvalDatasetError(f"{path}: manifest must be a JSON object")
    missing = sorted(_REQUIRED_META_KEYS - set(metadata))
    if missing:
        raise EvalDatasetError(f"{path}: manifest is missing {missing}")
    version = metadata["schema_version"]
    if version != SUPPORTED_SCHEMA_VERSION:
        raise EvalDatasetError(
            f"{path}: unsupported schema_version {version!r}; "
            f"this build reads {SUPPORTED_SCHEMA_VERSION!r}"
        )
    for field in ("tickers", "labels", "expected_stages", "confidences", "categories"):
        values = metadata[field]
        if not isinstance(values, list) or not values:
            raise EvalDatasetError(f"{path}: manifest {field} must be a non-empty list")
        if len(set(values)) != len(values):
            raise EvalDatasetError(f"{path}: manifest {field} repeats a value")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise EvalDatasetError(
                f"{path}: manifest {field} must hold non-blank strings"
            )
    return metadata


def load_pair_set(meta_path: str | Path = DEFAULT_META_PATH) -> PairSet:
    """Load and fully validate a labeled pair set.

    Raises :class:`EvalDatasetError` for any schema violation, unknown
    vocabulary value, repeated pair or item identifier, naive timestamp,
    label/stage contradiction, or unsorted row order.  Nothing is returned
    partially validated.
    """

    path = Path(meta_path).resolve()
    try:
        metadata = _validate_metadata(
            json.loads(path.read_text(encoding="utf-8")), path=path
        )
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"{path}: manifest not found") from exc
    except json.JSONDecodeError as exc:
        raise EvalDatasetError(f"{path}: manifest is not valid JSON: {exc}") from exc

    pairs_path = path.parent / str(metadata["pairs_file"])
    try:
        body = pairs_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"{pairs_path}: pairs file not found") from exc

    pairs: list[LabeledPair] = []
    seen_pair_ids: set[str] = set()
    seen_item_ids: set[str] = set()
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(
                f"{pairs_path} line {line_number}: not valid JSON: {exc}"
            ) from exc
        pair = _parse_pair(payload, metadata, line=line_number)
        if pair.pair_id in seen_pair_ids:
            raise EvalDatasetError(f"duplicate pair_id: {pair.pair_id}")
        seen_pair_ids.add(pair.pair_id)
        for item in (pair.item_a, pair.item_b):
            if item.item_id in seen_item_ids:
                raise EvalDatasetError(f"duplicate item_id: {item.item_id}")
            seen_item_ids.add(item.item_id)
        pairs.append(pair)

    if not pairs:
        raise EvalDatasetError(f"{pairs_path}: holds no pairs")
    identifiers = [pair.pair_id for pair in pairs]
    if identifiers != sorted(identifiers):
        raise EvalDatasetError(
            f"{pairs_path}: rows must be sorted by pair_id so the file order, "
            "the evaluation order, and every diff are the same order"
        )
    return PairSet(
        dataset_id=str(metadata["dataset_id"]),
        schema_version=str(metadata["schema_version"]),
        metadata=metadata,
        pairs=tuple(pairs),
        source_path=path,
    )


def default_pair_set() -> PairSet:
    """Load the committed Phase 0 labeled set."""

    return load_pair_set(DEFAULT_META_PATH)
