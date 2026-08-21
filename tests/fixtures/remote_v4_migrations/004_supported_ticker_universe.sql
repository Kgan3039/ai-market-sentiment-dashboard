UPDATE raw_items
SET ticker = NULL
WHERE ticker IS NOT NULL
  AND upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

UPDATE raw_items
SET ticker = upper(trim(ticker))
WHERE ticker IS NOT NULL;

DELETE FROM raw_item_tickers
WHERE upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

DELETE FROM raw_item_tickers
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM raw_item_tickers
    GROUP BY raw_item_id, upper(trim(ticker)), association_type
);

UPDATE raw_item_tickers SET ticker = upper(trim(ticker));

DELETE FROM raw_item_candidates
WHERE upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

DELETE FROM raw_item_candidates
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM raw_item_candidates
    GROUP BY raw_item_id, upper(trim(ticker))
);

UPDATE raw_item_candidates SET ticker = upper(trim(ticker));

DELETE FROM themes
WHERE id NOT IN (
    SELECT min(id)
    FROM themes
    GROUP BY
        upper(trim(ticker)),
        trading_day,
        content_hash,
        pipeline_version
);

DELETE FROM themes
WHERE upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

DELETE FROM theme_citations
WHERE NOT EXISTS (
    SELECT 1
    FROM theme_stories
    JOIN stories
      ON stories.id = theme_stories.story_id
    JOIN story_members
      ON story_members.story_id = stories.id
    WHERE theme_stories.theme_id = theme_citations.theme_id
      AND story_members.raw_item_id = theme_citations.raw_item_id
      AND upper(trim(stories.ticker))
          IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
);

DELETE FROM theme_stories
WHERE story_id IN (
    SELECT id
    FROM stories
    WHERE upper(trim(ticker))
        NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
);

DELETE FROM stories
WHERE upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

DELETE FROM pipeline_stage_keys
WHERE upper(trim(ticker)) NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META');

DELETE FROM pipeline_stage_keys
WHERE rowid NOT IN (
    SELECT min(rowid)
    FROM pipeline_stage_keys
    GROUP BY
        stage,
        upper(trim(ticker)),
        trading_day,
        pipeline_version
);

UPDATE stories SET ticker = upper(trim(ticker));
UPDATE themes SET ticker = upper(trim(ticker));
UPDATE pipeline_stage_keys SET ticker = upper(trim(ticker));

CREATE TRIGGER enforce_raw_item_ticker_insert
BEFORE INSERT ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_raw_item_ticker_update
BEFORE UPDATE OF ticker ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_raw_item_association_insert
BEFORE INSERT ON raw_item_tickers
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_raw_item_association_update
BEFORE UPDATE OF ticker ON raw_item_tickers
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_raw_item_candidate_insert
BEFORE INSERT ON raw_item_candidates
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_raw_item_candidate_update
BEFORE UPDATE OF ticker ON raw_item_candidates
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_story_ticker_insert
BEFORE INSERT ON stories
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_story_ticker_update
BEFORE UPDATE OF ticker ON stories
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_theme_ticker_insert
BEFORE INSERT ON themes
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_theme_ticker_update
BEFORE UPDATE OF ticker ON themes
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_stage_key_ticker_insert
BEFORE INSERT ON pipeline_stage_keys
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER enforce_stage_key_ticker_update
BEFORE UPDATE OF ticker ON pipeline_stage_keys
WHEN NEW.ticker NOT IN ('TSLA', 'NVDA', 'AMD', 'AAPL', 'META')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;
