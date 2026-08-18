-- RSS ingestion (#62) needs three things the final I1 schema has nowhere to
-- put.  Each one is evidence rather than derived state, which is why none of
-- them fits an existing table:
--
-- 1. The exact bytes of a feed response.  `raw_items.raw_json` is per-entry
--    publisher evidence, not the per-response body, and `source_state.metadata`
--    is operational metadata that is *always* redacted -- storing a response
--    body there would corrupt the evidence it exists to preserve.
--
-- 2. Which feed an item arrived on.  `raw_items` is unique by
--    `(source, canonical_url)`, and RSS deliberately keys `source` by the
--    *publisher* host so the same story syndicated by two approved feeds keeps
--    one identity.  Without a join table that identity costs the knowledge of
--    where it came from, which is the provenance #62 is required to keep.
--
-- 3. Why a ticker was matched or excluded.  `raw_item_candidates` carries one
--    free-text `reason` per (item, ticker); a decision carries an *array* of
--    evidence (which alias, which field), and exclusions are not candidates.
--
-- The ticker domain is a literal here, not a lookup against
-- `supported_tickers`, for the reason migration 009 gives: a lookup is a
-- domain any ordinary write can widen.

CREATE TABLE IF NOT EXISTS feed_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_source TEXT NOT NULL CHECK (feed_source LIKE 'rss:%'),
    fetched_at TEXT NOT NULL CHECK (datetime(fetched_at) IS NOT NULL),
    response_url TEXT NOT NULL CHECK (length(trim(response_url)) > 0),
    content_type TEXT,
    content_encoding TEXT,
    body BLOB NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    UNIQUE(feed_source, sha256)
);

CREATE INDEX IF NOT EXISTS idx_feed_snapshots_source_fetched
    ON feed_snapshots(feed_source, fetched_at);

-- One row per (item, feed, entry): the same story on two feeds is one
-- `raw_items` row with two provenance rows, which is the whole point.
-- `ON DELETE RESTRICT` on the snapshot is deliberate: provenance may not
-- outlive the bytes it points at.
CREATE TABLE IF NOT EXISTS raw_item_feeds (
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    feed_source TEXT NOT NULL CHECK (feed_source LIKE 'rss:%'),
    external_id TEXT NOT NULL,
    snapshot_id INTEGER NOT NULL
        REFERENCES feed_snapshots(id) ON DELETE RESTRICT,
    entry_digest TEXT NOT NULL CHECK (length(entry_digest) = 64),
    PRIMARY KEY (raw_item_id, feed_source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_item_feeds_source
    ON raw_item_feeds(feed_source, raw_item_id);

CREATE INDEX IF NOT EXISTS idx_raw_item_feeds_snapshot
    ON raw_item_feeds(snapshot_id);

-- Derived state, unlike the two tables above: reclassification replaces these
-- rows and must leave the evidence they were derived from untouched.
CREATE TABLE IF NOT EXISTS raw_item_match_evidence (
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('matched', 'excluded')),
    evidence TEXT NOT NULL CHECK (
        json_valid(evidence) AND json_type(evidence) = 'array'
    ),
    PRIMARY KEY (raw_item_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_raw_item_match_evidence_ticker
    ON raw_item_match_evidence(ticker, decision);

-- The approved universe, as a literal, on both INSERT and UPDATE -- the
-- pairing migration 009 established so an UPDATE cannot walk a row into a
-- state its INSERT would have refused.
DROP TRIGGER IF EXISTS trg_raw_item_match_evidence_ticker_insert;
DROP TRIGGER IF EXISTS trg_raw_item_match_evidence_ticker_update;

CREATE TRIGGER trg_raw_item_match_evidence_ticker_insert
BEFORE INSERT ON raw_item_match_evidence
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_match_evidence_ticker_update
BEFORE UPDATE OF ticker ON raw_item_match_evidence
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;
