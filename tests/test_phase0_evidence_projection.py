"""The I5 evidence read/projection boundary.

What these tests hold is the accounting, not the algorithms: that a
partition accounts for every row it claims, that the day accounts for every
row it holds, that the two are deliberately *not* the same arithmetic, and
that evidence which cannot be processed stays counted rather than
disappearing.  No stage runs here.
"""

from __future__ import annotations

import json

import pytest

from nlp.dedup.text import normalize_source
from phase0.errors import Phase0IntegrityError
from phase0.evidence import (
    EvidenceDay,
    EvidencePartition,
    project_raw_item,
    provider_item_id,
)
from phase0.publishers import (
    PUBLISHER_MAPPING,
    PublisherPolicyError,
    canonical_publisher,
    source_scheme,
)
from phase0.repository import Phase0Reader, Phase0Repository

DAY = "2026-08-20"


def migrated(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    return repository


def item(**updates):
    """One ordinary stored row, on DAY, before any association."""

    row = {
        "source": "yahoo:Barron's",
        "title": "NVDA headline",
        "url": "https://publisher.example/story",
        "canonical_url": "https://publisher.example/story",
        "external_id": "prov-9001",
        "published_at": f"{DAY}T12:00:00+00:00",
        "fetched_at": f"{DAY}T12:30:00+00:00",
        "raw_json": {},
    }
    row.update(updates)
    return row


def stored(repository, *items):
    return [result.item_id for result in repository.admin.insert_raw_items(items)]


def associate(repository, item_id, ticker, association_type):
    """Write one association row directly -- the tables, not a pipeline."""

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO raw_item_tickers "
            "(raw_item_id, ticker, association_type) VALUES (?, ?, ?)",
            (item_id, ticker, association_type),
        )


def record_match(repository, item_id, ticker, decision, evidence=("title",)):
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT INTO raw_item_match_evidence "
            "(raw_item_id, ticker, decision, evidence) VALUES (?, ?, ?, ?)",
            (item_id, ticker, decision, json.dumps(list(evidence))),
        )


def reader(repository):
    return Phase0Reader(repository.database_path)


def partition(repository, ticker, day=DAY):
    rows = reader(repository).evidence_partitions(trading_day=day, ticker=ticker)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Counting the population a partition claims
# ---------------------------------------------------------------------------


def test_one_item_holding_both_association_types_is_counted_once(tmp_path):
    """The primary key allows both rows; the partition is not twice as big."""

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item())
    associate(repository, item_id, "NVDA", "source")
    associate(repository, item_id, "NVDA", "relevance")

    counted = partition(repository, "NVDA")

    assert counted.associated_item_count == 1
    assert counted.eligible_item_count == 1
    assert len(reader(repository).partition_evidence("NVDA", DAY).items) == 1


def test_a_multi_ticker_item_joins_two_partitions_but_one_day(tmp_path):
    """Decision E, and the identity A1 says does not hold."""

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(title="NVDA and AMD"))
    associate(repository, item_id, "NVDA", "source")
    associate(repository, item_id, "AMD", "source")

    partitions = reader(repository).evidence_partitions(trading_day=DAY)
    (day,) = reader(repository).evidence_days(trading_day=DAY)

    assert [(row.ticker, row.associated_item_count) for row in partitions] == [
        ("AMD", 1),
        ("NVDA", 1),
    ]
    assert day.associated_any_ticker == 1
    # Stated as a fact so nobody "fixes" the read into agreeing.
    assert sum(row.associated_item_count for row in partitions) == 2
    assert sum(row.associated_item_count for row in partitions) != (
        day.associated_any_ticker
    )


@pytest.mark.parametrize("status", ["invalid", "ambiguous"])
def test_associated_non_valid_evidence_still_balances_the_partition(tmp_path, status):
    """Excluded-invalid means every status that is not 'valid'."""

    repository = migrated(tmp_path)
    good, bad = stored(
        repository,
        item(),
        item(
            canonical_url="urn:yahoo:nvda:deadbeef",
            url=None,
            title=None,
            ingest_status=status,
            validation_errors=["missing title"],
        ),
    )
    associate(repository, good, "NVDA", "source")
    associate(repository, bad, "NVDA", "source")

    counted = partition(repository, "NVDA")

    assert counted.associated_item_count == 2
    assert counted.eligible_item_count == 1
    assert counted.excluded_invalid == 1
    assert counted.excluded_unprojectable == 0
    assert counted.excluded_ambiguous == (1 if status == "ambiguous" else 0)


def test_an_ambiguous_row_is_a_subset_of_invalid_not_a_fourth_term(tmp_path):
    """The A1 invariant keeps exactly three terms; ambiguous is diagnostic."""

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(ingest_status="ambiguous"))
    associate(repository, item_id, "NVDA", "relevance")

    counted = partition(repository, "NVDA")

    assert (counted.excluded_invalid, counted.excluded_ambiguous) == (1, 1)
    assert (
        counted.eligible_item_count
        + counted.excluded_invalid
        + counted.excluded_unprojectable
        == counted.associated_item_count
    )


@pytest.mark.parametrize(
    "updates, day",
    [
        ({"published_at": "1901-05-05T12:00:00+00:00"}, "1901-05-05"),
        ({"source": "Example News"}, DAY),
        ({"source": "ftp:example.com"}, DAY),
        ({"source": "yahoo:Inc"}, DAY),
    ],
    ids=["implausible-timestamp", "no-scheme", "unknown-scheme", "not-a-fixed-point"],
)
def test_a_valid_but_unprojectable_row_is_measured_not_eligible(tmp_path, updates, day):
    """Unprojectable evidence is counted and inspectable, never a crash.

    The timestamp case is the one that reaches the dedup core's own
    refusal: the column accepts any datetime SQLite can parse, and M2
    rejects a year outside 1990-2100 as corrupt.
    """

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(**updates))
    associate(repository, item_id, "NVDA", "source")

    counted = partition(repository, "NVDA", day)
    evidence = reader(repository).partition_evidence("NVDA", day)

    assert counted.associated_item_count == 1
    assert counted.eligible_item_count == 0
    assert counted.excluded_invalid == 0
    assert counted.excluded_unprojectable == 1
    assert evidence.items == ()
    assert [row.outcome for row in evidence.excluded] == ["unprojectable"]
    assert evidence.excluded[0].ingest_status == "valid"
    assert evidence.excluded[0].detail


def test_the_partition_invariant_is_enforced_by_the_record_itself(tmp_path):
    """A miscount is an integrity error, not a plausible-looking row."""

    with pytest.raises(Phase0IntegrityError, match="account for its own population"):
        EvidencePartition(
            ticker="NVDA",
            trading_day=DAY,
            associated_item_count=3,
            eligible_item_count=1,
            excluded_invalid=1,
            excluded_unprojectable=0,
            excluded_ambiguous=0,
            latest_fetched_at=None,
        )


# ---------------------------------------------------------------------------
# Counting the day, ownership or not
# ---------------------------------------------------------------------------


def test_the_day_accounts_for_every_row_including_the_unowned(tmp_path):
    repository = migrated(tmp_path)
    owned, unowned = stored(
        repository,
        item(),
        item(canonical_url="https://publisher.example/b", url="https://p.example/b"),
    )
    associate(repository, owned, "NVDA", "source")

    (day,) = reader(repository).evidence_days(trading_day=DAY)

    assert day.total_item_count == 2
    assert day.associated_any_ticker == 1
    assert day.unassociated_item_count == 1
    assert (
        day.associated_any_ticker + day.unassociated_item_count == day.total_item_count
    )


def test_overlapping_diagnostic_signals_are_counted_independently(tmp_path):
    """One item can be ambiguous, have candidates, and have match evidence."""

    repository = migrated(tmp_path)
    (item_id,) = stored(
        repository,
        item(
            source="rss:example-news.com",
            ingest_status="ambiguous",
            candidate_tickers=[{"ticker": "NVDA", "reason": "ambiguous match"}],
        ),
    )
    record_match(repository, item_id, "NVDA", "matched")
    record_match(repository, item_id, "AMD", "matched")

    (day,) = reader(repository).evidence_days(trading_day=DAY)

    assert day.total_item_count == 1
    assert day.unassociated_item_count == 1
    assert day.unassociated_ambiguous == 1
    assert day.unassociated_with_candidates == 1
    assert day.unassociated_with_match_evidence == 1
    assert day.unassociated_without_evidence == 0
    # The signals overlap on one row; only the two-term invariant is real.
    assert (
        day.unassociated_ambiguous
        + day.unassociated_with_candidates
        + day.unassociated_with_match_evidence
        > day.unassociated_item_count
    )
    assert (
        day.associated_any_ticker + day.unassociated_item_count == day.total_item_count
    )


def test_the_day_invariant_is_enforced_by_the_record_itself():
    with pytest.raises(Phase0IntegrityError, match="account for its own population"):
        EvidenceDay(
            trading_day=DAY,
            total_item_count=3,
            associated_any_ticker=1,
            unassociated_item_count=1,
            unassociated_invalid=0,
            unassociated_ambiguous=0,
            unassociated_with_candidates=0,
            unassociated_with_match_evidence=0,
            unassociated_without_evidence=1,
            latest_fetched_at=None,
        )


# ---------------------------------------------------------------------------
# Row-level observability
# ---------------------------------------------------------------------------


def test_unassociated_items_carry_their_candidate_and_match_evidence(tmp_path):
    repository = migrated(tmp_path)
    (item_id,) = stored(
        repository,
        item(
            source="rss:example-news.com",
            candidate_tickers=[{"ticker": "NVDA", "reason": "ambiguous match"}],
        ),
    )
    record_match(repository, item_id, "NVDA", "matched", ["title mentions NVDA"])
    record_match(repository, item_id, "AMD", "excluded", ["ticker not present"])

    (row,) = reader(repository).unassociated_items(DAY)

    assert row.raw_item_id == item_id
    assert [candidate.ticker for candidate in row.candidates] == ["NVDA"]
    assert [(match.ticker, match.decision) for match in row.match_evidence] == [
        ("AMD", "excluded"),
        ("NVDA", "matched"),
    ]
    assert row.match_evidence[1].evidence == ("title mentions NVDA",)


def test_the_join_does_not_multiply_an_item_with_several_evidence_rows(tmp_path):
    """Two candidates and two match rows are four joined rows, one item."""

    repository = migrated(tmp_path)
    (item_id,) = stored(
        repository,
        item(
            source="rss:example-news.com",
            candidate_tickers=[
                {"ticker": "NVDA", "reason": "ambiguous match"},
                {"ticker": "AMD", "reason": "ambiguous match"},
            ],
        ),
    )
    record_match(repository, item_id, "NVDA", "matched")
    record_match(repository, item_id, "AMD", "matched")

    rows = reader(repository).unassociated_items(DAY)

    assert len(rows) == 1
    assert len(rows[0].candidates) == 2
    assert len(rows[0].match_evidence) == 2


def test_unassociated_items_bounds_items_not_joined_rows(tmp_path):
    repository = migrated(tmp_path)
    stored(
        repository,
        item(canonical_url="https://publisher.example/a"),
        item(canonical_url="https://publisher.example/b"),
        item(canonical_url="https://publisher.example/c"),
    )

    assert len(reader(repository).unassociated_items(DAY, limit=2)) == 2


def test_a_withheld_match_survives_an_association_to_another_ticker(tmp_path):
    """Keyed on (item, ticker): AMD's association cannot hide NVDA's match."""

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(source="rss:example-news.com"))
    associate(repository, item_id, "AMD", "relevance")
    record_match(repository, item_id, "AMD", "matched")
    record_match(repository, item_id, "NVDA", "matched")

    withheld = reader(repository).withheld_matches(DAY)

    assert [(row.raw_item_id, row.ticker) for row in withheld] == [(item_id, "NVDA")]
    # And it confers nothing: NVDA has no partition at all here.
    assert reader(repository).evidence_partitions(trading_day=DAY, ticker="NVDA") == []
    assert partition(repository, "AMD").eligible_item_count == 1


def test_an_accepted_association_is_not_a_withheld_match(tmp_path):
    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(source="rss:example-news.com"))
    associate(repository, item_id, "NVDA", "relevance")
    record_match(repository, item_id, "NVDA", "matched")

    assert reader(repository).withheld_matches(DAY) == []


def test_candidates_and_match_evidence_confer_no_ownership(tmp_path):
    """Observability only -- decision F, rules 2 and the note under it."""

    repository = migrated(tmp_path)
    (item_id,) = stored(
        repository,
        item(
            source="rss:example-news.com",
            candidate_tickers=[{"ticker": "NVDA", "reason": "ambiguous match"}],
        ),
    )
    record_match(repository, item_id, "NVDA", "matched")

    assert reader(repository).evidence_partitions(trading_day=DAY) == []
    assert reader(repository).partition_evidence("NVDA", DAY).items == ()


def test_the_primary_ticker_column_alone_does_not_own_evidence(tmp_path):
    """`raw_items.ticker` is the first claimant; the table is the authority."""

    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(ticker="NVDA"))
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM raw_item_tickers WHERE raw_item_id = ?", (item_id,)
        )

    (day,) = reader(repository).evidence_days(trading_day=DAY)

    assert reader(repository).evidence_partitions(trading_day=DAY) == []
    assert day.unassociated_item_count == 1
    assert [row.raw_item_id for row in reader(repository).unassociated_items(DAY)] == [
        item_id
    ]


# ---------------------------------------------------------------------------
# Projection: publisher identity and provider identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        ("yahoo:Barron's", "yahoo barrons"),
        ("rss:example-news.com", "rss examplenewscom"),
        ("rss:example-news.co", "rss examplenewsco"),
        ("yahoo:Yahoo Finance", "yahoo yahoofinance"),
        ("rss:www.example.com", "rss wwwexamplecom"),
        ("yahoo:", "yahoo"),
    ],
)
def test_the_fallback_emits_the_values_the_decision_record_names(source, expected):
    assert canonical_publisher(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "yahoo:Barron's",
        "rss:example-news.com",
        "rss:example-news.co",
        "yahoo:Yahoo Finance",
        "yahoo:Reuters",
        "rss:marketwatch.com",
        "yahoo:24/7 Wall St.",
        "yahoo:Investor's Business Daily",
        "yahoo:",
    ],
)
def test_every_emitted_outlet_is_a_fixed_point_of_the_real_normalize_source(source):
    """Established by calling it, never by predicting what it would do."""

    value = canonical_publisher(source)

    assert normalize_source(value) == value
    assert value


@pytest.mark.parametrize(
    "source",
    ["Example News", "ftp:example.com", "yahoo:Inc", "   ", ""],
    ids=["no-scheme", "unknown-scheme", "not-a-fixed-point", "blank", "empty"],
)
def test_a_source_the_policy_cannot_represent_is_refused_not_repaired(source):
    with pytest.raises(PublisherPolicyError):
        canonical_publisher(source)


def test_the_reviewed_mapping_ships_empty():
    """Decision B's stop condition: no observed Yahoo/RSS equivalence."""

    assert dict(PUBLISHER_MAPPING) == {}


def test_an_explicit_mapping_unifies_two_spellings(monkeypatch):
    """The mapping path, exercised without inventing a shipped entry."""

    monkeypatch.setattr(
        "phase0.publishers.PUBLISHER_MAPPING",
        {"yahoo:Reuters": "reuters", "rss:reuters.com": "reuters"},
    )

    assert canonical_publisher("yahoo:Reuters") == "reuters"
    assert canonical_publisher("rss:reuters.com") == "reuters"
    assert normalize_source("reuters") == "reuters"
    # The scheme still separates the id spaces the unified outlet merged.
    assert provider_item_id("yahoo:Reuters", "abc") == "yahoo:abc"
    assert provider_item_id("rss:reuters.com", "abc") == "rss:abc"


@pytest.mark.parametrize(
    "identifier",
    ["yahoo", "rss", "yahoo finance", "rss example"],
)
def test_a_mapped_id_may_not_occupy_the_unmapped_namespace(identifier):
    """`yahoo:Finance` emits `yahoo finance`; a mapping must not collide."""

    from phase0.publishers import _validate_mapping

    with pytest.raises(PublisherPolicyError, match="reserved"):
        _validate_mapping({"yahoo:Anything": identifier})


@pytest.mark.parametrize(
    "external_id", [None, "", "   ", "\t\n"], ids=["absent", "empty", "spaces", "tabs"]
)
def test_no_provider_id_means_none_never_a_fabricated_one(external_id):
    assert provider_item_id("yahoo:Barron's", external_id) is None


def test_one_bare_id_from_two_providers_stays_two_identities():
    """A2: the scheme qualifier is what keeps the id spaces apart."""

    assert provider_item_id("yahoo:Barron's", "abc") == "yahoo:abc"
    assert provider_item_id("rss:example-news.com", "abc") == "rss:abc"
    assert provider_item_id("yahoo:Barron's", "abc") != provider_item_id(
        "rss:example-news.com", "abc"
    )


def test_the_qualifier_comes_from_the_stored_scheme_not_the_outlet(monkeypatch):
    """Even when canonicalization has unified the two publishers."""

    monkeypatch.setattr(
        "phase0.publishers.PUBLISHER_MAPPING",
        {"yahoo:Reuters": "reuters", "rss:reuters.com": "reuters"},
    )
    yahoo = project_raw_item(
        {
            "id": 1,
            "source": "yahoo:Reuters",
            "title": "One story",
            "url": "https://reuters.example/a",
            "canonical_url": "https://reuters.example/a",
            "external_id": "abc",
            "published_at": f"{DAY}T12:00:00+00:00",
        },
        "NVDA",
    )
    rss = project_raw_item(
        {
            "id": 2,
            "source": "rss:reuters.com",
            "title": "One story",
            "url": "https://reuters.example/a",
            "canonical_url": "https://reuters.example/a",
            "external_id": "abc",
            "published_at": f"{DAY}T12:00:00+00:00",
        },
        "NVDA",
    )

    assert yahoo.source == rss.source == "reuters"
    assert (yahoo.provider_item_id, rss.provider_item_id) == ("yahoo:abc", "rss:abc")


def test_only_recognized_schemes_qualify_a_provider_id():
    for source in ("Example News", "ftp:example.com"):
        assert source_scheme(source) is None
        with pytest.raises(PublisherPolicyError):
            provider_item_id(source, "abc")


def test_the_projection_carries_the_row_across_unchanged(tmp_path):
    repository = migrated(tmp_path)
    (item_id,) = stored(repository, item(description="A description.", ticker="AMD"))
    associate(repository, item_id, "NVDA", "relevance")

    (projected,) = reader(repository).partition_evidence("NVDA", DAY).items

    assert projected.item_id == str(item_id)
    # The caller's authoritative association, not `raw_items.ticker`.
    assert projected.ticker == "NVDA"
    assert projected.title == "NVDA headline"
    assert projected.description == "A description."
    assert projected.url == "https://publisher.example/story"
    assert projected.canonical_url == "https://publisher.example/story"
    assert projected.published_at == f"{DAY}T12:00:00+00:00"
    assert projected.source == "yahoo barrons"
    assert projected.provider_item_id == "yahoo:prov-9001"


def test_evidence_takes_its_day_from_its_own_timestamps(tmp_path):
    """Decision C, unchanged: the UTC date of published_at or fetched_at."""

    repository = migrated(tmp_path)
    published, undated = stored(
        repository,
        item(published_at=f"{DAY}T23:59:00+00:00"),
        item(
            canonical_url="https://publisher.example/b",
            published_at=None,
            fetched_at="2026-08-21T00:30:00+00:00",
        ),
    )
    associate(repository, published, "NVDA", "source")
    associate(repository, undated, "NVDA", "source")

    days = [row.trading_day for row in reader(repository).evidence_partitions()]

    assert days == [DAY, "2026-08-21"]
