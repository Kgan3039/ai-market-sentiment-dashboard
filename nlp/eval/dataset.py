"""The labeled deduplication pair set: schema, loading, and validation.

A pair set is two files that travel together: a manifest naming the schema
version, the controlled vocabularies, the trust contract, and the
provenance of the labels, and a JSONL file holding one pair per line.
Splitting them keeps the manifest readable and the 150-row body
diff-friendly.

Loading is strict on purpose.  An evaluation set is only worth something if
a reviewer can trust that every row was checked: an unknown category, a
duplicated pair id, a pair that repeats another pair's content in the other
order, a naive timestamp, or a label that contradicts its expected stage is
a defect in the *dataset*, and this module refuses to hand it to a scorer
rather than quietly measuring against corrupt ground truth.

Provenance is enforced the same way.  The manifest must declare a complete
:class:`~nlp.eval.trust.TrustContract`, and every loaded pair carries it, so
no consumer can render a metric without also being able to render where it
came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

from nlp.dedup.normalization import (
    MAX_PLAUSIBLE_PUBLISHED_AT,
    MIN_PLAUSIBLE_PUBLISHED_AT,
)

from .trust import (
    TrustContract,
    TrustContractError,
    parse_trust_contract,
    validate_labeling,
    validate_provenance,
)

#: Bumped when the pair schema changes shape.  A manifest declaring any
#: other version is rejected rather than best-effort parsed.
SUPPORTED_SCHEMA_VERSION = "phase0.dedup_eval.v2"

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
        "trust_contract",
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

#: Present on every item even when null, so "this feed supplied no canonical
#: URL" is a recorded fact rather than a missing key nobody noticed.
_REQUIRED_ITEM_KEYS = frozenset({"item_id", "canonical_url"})

#: A duplicate pair must be somebody's responsibility; a pair that must
#: never merge must be nobody's.  Enforcing the correspondence stops a
#: mislabelled row from silently inflating one stage's recall.
_STAGE_BY_LABEL = {
    "duplicate": frozenset({"m2", "m3"}),
    "distinct": frozenset({"none"}),
    "ambiguous": frozenset({"none"}),
}

_URL_SCHEMES = frozenset({"http", "https"})


class EvalDatasetError(ValueError):
    """The evaluation dataset is malformed and must not be scored against."""


@dataclass(frozen=True)
class LabeledItem:
    """One side of a labeled pair, projected from the ``raw_items`` columns.

    ``ticker`` is resolved at load time: a row that does not carry its own
    inherits the pair's.  Every optional field is ``None`` when the feed did
    not supply it, which is different from an empty string and is treated
    that way by every stage.  ``synthetic`` is stamped from the manifest, so
    a record cannot travel without its provenance.
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
    synthetic: bool = True

    def content_key(self) -> tuple[Any, ...]:
        """What makes this record the record it is, for duplicate detection."""

        return (
            self.ticker,
            self.title,
            self.description,
            self.url,
            self.canonical_url,
            self.source,
            self.published_at,
            self.provider_item_id,
        )


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
    synthetic: bool = True

    @property
    def is_positive(self) -> bool:
        """True when the two records describe one event."""

        return self.label == "duplicate"

    @property
    def is_scored(self) -> bool:
        """True when this pair counts toward the headline metrics.

        ``ambiguous`` pairs do not: the records do not contain enough to
        decide them, so scoring against the label measures the coin flip.
        They are reported separately instead.
        """

        return self.label in {"duplicate", "distinct"}

    def content_key(self) -> frozenset[tuple[Any, ...]]:
        """Order-insensitive identity of the pair's two records.

        A frozenset, so a pair that repeats another pair's content with the
        two sides swapped collides with it.  Two identical pairs under
        different ids double-count in every metric they appear in.
        """

        return frozenset({self.item_a.content_key(), self.item_b.content_key()})


@dataclass(frozen=True)
class PairSet:
    """A validated labeled set plus the manifest it was loaded from."""

    dataset_id: str
    schema_version: str
    trust: TrustContract
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


def validate_url(value: str | None, field: str, *, where: str) -> str | None:
    """Reject a link a fetcher could not have produced.

    This is a *shape* check, not a claim of authenticity: the manifest
    states that every URL here is synthetic and resolves to nothing. What
    it catches is a fixture link that could never have come out of an
    ingester at all, which would exercise the URL-identity rules on input
    production will never see.
    """

    if value is None:
        return None
    split = urlsplit(value)
    if split.scheme not in _URL_SCHEMES:
        raise EvalDatasetError(f"{where}: {field} must be an http(s) URL: {value!r}")
    if not split.netloc or "." not in split.netloc:
        raise EvalDatasetError(f"{where}: {field} has no usable host: {value!r}")
    if " " in value:
        raise EvalDatasetError(f"{where}: {field} contains whitespace: {value!r}")
    return value


def _validate_timestamp(value: str, *, where: str) -> None:
    """Reject anything the dedup core would reject, at load time.

    A naive or implausible timestamp in a fixture is a dataset bug that
    would otherwise surface as a stage exception halfway through a scoring
    run. The bounds are M2's own, imported rather than restated, so the two
    contracts cannot drift apart.
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
    utc = parsed.astimezone(MIN_PLAUSIBLE_PUBLISHED_AT.tzinfo)
    if not MIN_PLAUSIBLE_PUBLISHED_AT <= utc < MAX_PLAUSIBLE_PUBLISHED_AT:
        raise EvalDatasetError(
            f"{where}: published_at is outside the range the dedup core "
            f"accepts ({MIN_PLAUSIBLE_PUBLISHED_AT.date()} to "
            f"{MAX_PLAUSIBLE_PUBLISHED_AT.date()}): {value!r}"
        )


def _parse_item(
    payload: Any,
    *,
    pair_id: str,
    side: str,
    default_ticker: str,
    tickers: Sequence[str],
    synthetic: bool,
) -> LabeledItem:
    where = f"{pair_id}.{side}"
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"{where}: must be an object")
    unknown = sorted(set(payload) - _ITEM_KEYS)
    if unknown:
        raise EvalDatasetError(f"{where}: unknown field(s) {unknown}")
    missing = sorted(_REQUIRED_ITEM_KEYS - set(payload))
    if missing:
        raise EvalDatasetError(
            f"{where}: missing field(s) {missing}; canonical_url must be "
            "present even when null so 'no canonical URL' is recorded rather "
            "than assumed"
        )
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
        url=validate_url(
            _optional_str(payload.get("url"), "url", where=where), "url", where=where
        ),
        canonical_url=validate_url(
            _optional_str(payload.get("canonical_url"), "canonical_url", where=where),
            "canonical_url",
            where=where,
        ),
        source=_optional_str(payload.get("source"), "source", where=where),
        published_at=published_at,
        provider_item_id=_optional_str(
            payload.get("provider_item_id"), "provider_item_id", where=where
        ),
        synthetic=synthetic,
    )


def _parse_pair(
    payload: Any, metadata: Mapping[str, Any], *, line: int, synthetic: bool
) -> LabeledPair:
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
            synthetic=synthetic,
        ),
        item_b=_parse_item(
            payload["item_b"],
            pair_id=pair_id,
            side="item_b",
            default_ticker=ticker,
            tickers=metadata["tickers"],
            synthetic=synthetic,
        ),
        synthetic=synthetic,
    )
    if pair.item_a.item_id == pair.item_b.item_id:
        raise EvalDatasetError(f"{pair_id}: both sides share item_id")
    if label != "duplicate" and pair.item_a.content_key() == pair.item_b.content_key():
        # Two byte-identical records cannot be two different events. Within a
        # *duplicate* pair it is legitimate and common: a feed re-poll emits
        # the same row twice under two raw_items ids, which is precisely the
        # provider_repeat case.
        raise EvalDatasetError(
            f"{pair_id}: both sides carry identical content but the pair is "
            f"labelled {label!r}; identical records cannot be different events"
        )
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
    vocabulary value, repeated pair or item identifier, repeated pair
    *content* (in either order), naive or implausible timestamp,
    unparseable URL, missing ``canonical_url`` key, label/stage
    contradiction, or unsorted row order, and
    :class:`~nlp.eval.trust.TrustContractError` for a manifest that does not
    state its own provenance or contradicts it.  Nothing is returned
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

    trust = parse_trust_contract(metadata, where=str(path))
    validate_provenance(metadata, trust, where=str(path))
    validate_labeling(metadata, trust, where=str(path))

    pairs_path = path.parent / str(metadata["pairs_file"])
    try:
        body = pairs_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"{pairs_path}: pairs file not found") from exc

    pairs: list[LabeledPair] = []
    seen_pair_ids: set[str] = set()
    seen_item_ids: set[str] = set()
    seen_content: dict[frozenset[tuple[Any, ...]], str] = {}
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(
                f"{pairs_path} line {line_number}: not valid JSON: {exc}"
            ) from exc
        pair = _parse_pair(
            payload, metadata, line=line_number, synthetic=trust.is_synthetic
        )
        if pair.pair_id in seen_pair_ids:
            raise EvalDatasetError(f"duplicate pair_id: {pair.pair_id}")
        seen_pair_ids.add(pair.pair_id)
        for item in (pair.item_a, pair.item_b):
            if item.item_id in seen_item_ids:
                raise EvalDatasetError(f"duplicate item_id: {item.item_id}")
            seen_item_ids.add(item.item_id)
        key = pair.content_key()
        if key in seen_content:
            raise EvalDatasetError(
                f"{pair.pair_id} repeats the content of {seen_content[key]} "
                "(possibly with the two sides swapped); a repeated pair "
                "double-counts in every metric it appears in"
            )
        seen_content[key] = pair.pair_id
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
        trust=trust,
        metadata=metadata,
        pairs=tuple(pairs),
        source_path=path,
    )


def default_pair_set() -> PairSet:
    """Load the committed Phase 0 labeled set."""

    return load_pair_set(DEFAULT_META_PATH)


__all__ = [
    "DEFAULT_META_PATH",
    "SUPPORTED_SCHEMA_VERSION",
    "EvalDatasetError",
    "LabeledItem",
    "LabeledPair",
    "PairSet",
    "TrustContractError",
    "default_pair_set",
    "load_pair_set",
    "validate_url",
]
