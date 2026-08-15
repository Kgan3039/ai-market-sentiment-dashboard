-- The approved Phase 0 ticker universe (spec section 2), made
-- authoritative in the database so no storage boundary can bypass it.

CREATE TABLE IF NOT EXISTS supported_tickers (
    ticker TEXT PRIMARY KEY CHECK (
        ticker = upper(trim(ticker))
        AND length(ticker) BETWEEN 1 AND 5
        AND ticker NOT GLOB '*[^A-Z]*'
    ),
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    position INTEGER NOT NULL UNIQUE CHECK (position > 0)
);

INSERT OR IGNORE INTO supported_tickers (ticker, display_name, position)
VALUES
    ('TSLA', 'Tesla', 1),
    ('NVDA', 'NVIDIA', 2),
    ('AMD', 'Advanced Micro Devices', 3),
    ('AAPL', 'Apple', 4),
    ('META', 'Meta Platforms', 5);

-- Existing rows are normalized before the constraints below start firing.
-- Raw evidence is never deleted: an unsupported symbol on a raw item is
-- cleared to NULL, which is exactly how the spec stores an item that
-- matches no ticker.  Derived rows carry no evidence and are removed.

UPDATE raw_items
SET ticker = upper(trim(ticker))
WHERE ticker IS NOT NULL
  AND ticker <> upper(trim(ticker));

UPDATE raw_items
SET ticker = NULL
WHERE ticker IS NOT NULL
  AND ticker NOT IN (SELECT ticker FROM supported_tickers);

DELETE FROM raw_item_tickers
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM raw_item_tickers
    GROUP BY raw_item_id, upper(trim(ticker)), association_type
);

UPDATE raw_item_tickers
SET ticker = upper(trim(ticker))
WHERE ticker <> upper(trim(ticker));

DELETE FROM raw_item_tickers
WHERE ticker NOT IN (SELECT ticker FROM supported_tickers);

DELETE FROM raw_item_candidates
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM raw_item_candidates
    GROUP BY raw_item_id, upper(trim(ticker))
);

UPDATE raw_item_candidates
SET ticker = upper(trim(ticker))
WHERE ticker <> upper(trim(ticker));

DELETE FROM raw_item_candidates
WHERE ticker NOT IN (SELECT ticker FROM supported_tickers);

UPDATE stories
SET ticker = upper(trim(ticker))
WHERE ticker <> upper(trim(ticker));

UPDATE themes
SET ticker = upper(trim(ticker))
WHERE ticker <> upper(trim(ticker));

UPDATE pipeline_stage_keys
SET ticker = upper(trim(ticker))
WHERE ticker <> upper(trim(ticker));

DELETE FROM theme_citations
WHERE theme_id IN (
    SELECT id FROM themes
    WHERE ticker NOT IN (SELECT ticker FROM supported_tickers)
)
   OR theme_id IN (
    SELECT theme_stories.theme_id
    FROM theme_stories
    JOIN stories ON stories.id = theme_stories.story_id
    WHERE stories.ticker NOT IN (SELECT ticker FROM supported_tickers)
);

DELETE FROM theme_stories
WHERE theme_id IN (
    SELECT id FROM themes
    WHERE ticker NOT IN (SELECT ticker FROM supported_tickers)
)
   OR story_id IN (
    SELECT id FROM stories
    WHERE ticker NOT IN (SELECT ticker FROM supported_tickers)
);

DELETE FROM themes
WHERE ticker NOT IN (SELECT ticker FROM supported_tickers);

DELETE FROM stories
WHERE ticker NOT IN (SELECT ticker FROM supported_tickers);

DELETE FROM pipeline_stage_keys
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM pipeline_stage_keys
    GROUP BY stage, ticker, trading_day, pipeline_version
);

DELETE FROM pipeline_stage_keys
WHERE ticker NOT IN (SELECT ticker FROM supported_tickers);

-- Direct SQL is held to the same universe as the repository API.

CREATE TRIGGER IF NOT EXISTS trg_raw_item_ticker_insert
BEFORE INSERT ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_ticker_update
BEFORE UPDATE OF ticker ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_association_insert
BEFORE INSERT ON raw_item_tickers
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_association_update
BEFORE UPDATE OF ticker ON raw_item_tickers
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_candidate_insert
BEFORE INSERT ON raw_item_candidates
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_candidate_update
BEFORE UPDATE OF ticker ON raw_item_candidates
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_story_ticker_insert
BEFORE INSERT ON stories
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_story_ticker_update
BEFORE UPDATE OF ticker ON stories
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_ticker_insert
BEFORE INSERT ON themes
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_ticker_update
BEFORE UPDATE OF ticker ON themes
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_stage_key_ticker_insert
BEFORE INSERT ON pipeline_stage_keys
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_stage_key_ticker_update
BEFORE UPDATE OF ticker ON pipeline_stage_keys
WHEN NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

-- The universe itself is not editable out from under stored rows.

CREATE TRIGGER IF NOT EXISTS trg_supported_ticker_delete
BEFORE DELETE ON supported_tickers
WHEN EXISTS (SELECT 1 FROM raw_items WHERE ticker = OLD.ticker)
  OR EXISTS (SELECT 1 FROM raw_item_tickers WHERE ticker = OLD.ticker)
  OR EXISTS (SELECT 1 FROM raw_item_candidates WHERE ticker = OLD.ticker)
  OR EXISTS (SELECT 1 FROM stories WHERE ticker = OLD.ticker)
  OR EXISTS (SELECT 1 FROM themes WHERE ticker = OLD.ticker)
  OR EXISTS (SELECT 1 FROM pipeline_stage_keys WHERE ticker = OLD.ticker)
BEGIN
    SELECT RAISE(ABORT, 'supported ticker is still referenced');
END;
