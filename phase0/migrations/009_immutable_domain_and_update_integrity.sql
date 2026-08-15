-- Two defects migration 004 left open, closed additively.
--
-- 1. The approved universe was a *lookup table*.  Every ticker trigger
--    asked `NOT IN (SELECT ticker FROM supported_tickers)`, so a direct
--    `INSERT INTO supported_tickers VALUES ('GOOG', ...)` widened the
--    domain for every other table at once.  The universe is now a literal
--    in each trigger — the constraint no longer reads from a table that
--    ordinary writes can edit — and `supported_tickers` is sealed so it
--    stays a readable projection of that literal rather than its source.
--
-- 2. Several theme integrity triggers were INSERT-only, so an UPDATE could
--    walk a row into a state its INSERT would have been refused.  Each one
--    gains its UPDATE counterpart, and the ticker/day of a theme or theme
--    set can no longer be moved out from under the rows that hang off it.

-- ------------------------------------------------------------------
-- Clean up anything a widened universe already allowed in.
-- ------------------------------------------------------------------

UPDATE raw_items
SET ticker = NULL
WHERE ticker IS NOT NULL
  AND ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM raw_item_tickers
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM raw_item_candidates
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM theme_citations
WHERE theme_id IN (
    SELECT id FROM themes
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
)
   OR theme_id IN (
    SELECT theme_stories.theme_id
    FROM theme_stories
    JOIN stories ON stories.id = theme_stories.story_id
    WHERE stories.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
);

DELETE FROM theme_stories
WHERE theme_id IN (
    SELECT id FROM themes
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
)
   OR story_id IN (
    SELECT id FROM stories
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
);

DELETE FROM theme_other_coverage
WHERE theme_set_id IN (
    SELECT id FROM theme_sets
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
)
   OR story_id IN (
    SELECT id FROM stories
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
);

DELETE FROM theme_excluded_stories
WHERE theme_set_id IN (
    SELECT id FROM theme_sets
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
)
   OR story_id IN (
    SELECT id FROM stories
    WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
);

DELETE FROM themes
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM theme_sets
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM stories
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM run_log_stage_keys
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

DELETE FROM pipeline_stage_keys
WHERE ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

UPDATE run_log
SET ticker = NULL
WHERE ticker IS NOT NULL
  AND ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA');

-- The old delete guard was conditional on references; the new one is not,
-- so it has to go before the table is restored to the approved five.
DROP TRIGGER IF EXISTS trg_supported_ticker_delete;

-- Rebuild the table rather than reposition it in place.
--
-- `position` is UNIQUE and SQLite enforces that per row, so an upsert
-- that moves TSLA to position 1 while NVDA still holds position 1 aborts
-- the whole migration.  That is not an edge case: of the 120 orderings a
-- valid pre-009 database could legitimately be in, 119 collide, because
-- before 009 this table is not yet sealed and anything could have
-- reordered it.  No ordering of the writes avoids it in general — any
-- permutation with a cycle has some row that must pass through an
-- occupied position — so the fix is to leave no occupied positions to
-- collide with.
--
-- Emptying it first loses nothing: no table references
-- `supported_tickers`, and the five rows below are the whole approved
-- universe, so this converges to the canonical set and ordering from any
-- prior state — reordered, partial, duplicated, or carrying unsupported
-- rows the cleanup above already handles elsewhere.
DELETE FROM supported_tickers;

INSERT INTO supported_tickers (ticker, display_name, position)
VALUES
    ('TSLA', 'Tesla', 1),
    ('NVDA', 'NVIDIA', 2),
    ('AMD', 'Advanced Micro Devices', 3),
    ('AAPL', 'Apple', 4),
    ('META', 'Meta Platforms', 5);

-- ------------------------------------------------------------------
-- The universe is a constant, not a row set.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_raw_item_ticker_insert;
DROP TRIGGER IF EXISTS trg_raw_item_ticker_update;
DROP TRIGGER IF EXISTS trg_raw_item_association_insert;
DROP TRIGGER IF EXISTS trg_raw_item_association_update;
DROP TRIGGER IF EXISTS trg_raw_item_candidate_insert;
DROP TRIGGER IF EXISTS trg_raw_item_candidate_update;
DROP TRIGGER IF EXISTS trg_story_ticker_insert;
DROP TRIGGER IF EXISTS trg_story_ticker_update;
DROP TRIGGER IF EXISTS trg_theme_ticker_insert;
DROP TRIGGER IF EXISTS trg_theme_ticker_update;
DROP TRIGGER IF EXISTS trg_stage_key_ticker_insert;
DROP TRIGGER IF EXISTS trg_stage_key_ticker_update;
DROP TRIGGER IF EXISTS trg_run_log_ticker_insert;
DROP TRIGGER IF EXISTS trg_run_log_ticker_update;

CREATE TRIGGER trg_raw_item_ticker_insert
BEFORE INSERT ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_ticker_update
BEFORE UPDATE OF ticker ON raw_items
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_association_insert
BEFORE INSERT ON raw_item_tickers
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_association_update
BEFORE UPDATE OF ticker ON raw_item_tickers
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_candidate_insert
BEFORE INSERT ON raw_item_candidates
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_raw_item_candidate_update
BEFORE UPDATE OF ticker ON raw_item_candidates
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_story_ticker_insert
BEFORE INSERT ON stories
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_story_ticker_update
BEFORE UPDATE OF ticker ON stories
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_theme_ticker_insert
BEFORE INSERT ON themes
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_theme_ticker_update
BEFORE UPDATE OF ticker ON themes
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_stage_key_ticker_insert
BEFORE INSERT ON pipeline_stage_keys
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_stage_key_ticker_update
BEFORE UPDATE OF ticker ON pipeline_stage_keys
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

-- Ticker-bearing tables migration 004 missed entirely.

CREATE TRIGGER trg_theme_set_ticker_insert
BEFORE INSERT ON theme_sets
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_theme_set_ticker_update
BEFORE UPDATE OF ticker ON theme_sets
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_run_log_ticker_insert
BEFORE INSERT ON run_log
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_run_log_ticker_update
BEFORE UPDATE OF ticker ON run_log
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_run_log_stage_key_ticker_insert
BEFORE INSERT ON run_log_stage_keys
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER trg_run_log_stage_key_ticker_update
BEFORE UPDATE OF ticker ON run_log_stage_keys
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

-- ------------------------------------------------------------------
-- `supported_tickers` is sealed: it can be read, never written.
-- ------------------------------------------------------------------

CREATE TRIGGER trg_supported_tickers_immutable_insert
BEFORE INSERT ON supported_tickers
BEGIN
    SELECT RAISE(ABORT, 'the Phase 0 ticker universe is immutable');
END;

CREATE TRIGGER trg_supported_tickers_immutable_update
BEFORE UPDATE ON supported_tickers
BEGIN
    SELECT RAISE(ABORT, 'the Phase 0 ticker universe is immutable');
END;

CREATE TRIGGER trg_supported_tickers_immutable_delete
BEFORE DELETE ON supported_tickers
BEGIN
    SELECT RAISE(ABORT, 'the Phase 0 ticker universe is immutable');
END;

-- ------------------------------------------------------------------
-- M5: everything that was guarded on INSERT is now guarded on UPDATE.
-- ------------------------------------------------------------------

CREATE TRIGGER trg_other_coverage_not_in_theme_update
BEFORE UPDATE ON theme_other_coverage
WHEN EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN themes
      ON themes.ticker = theme_sets.ticker
     AND themes.trading_day = theme_sets.trading_day
     AND themes.pipeline_version = theme_sets.pipeline_version
    JOIN theme_stories ON theme_stories.theme_id = themes.id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_stories.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already a member of a theme');
END;

CREATE TRIGGER trg_theme_story_not_in_other_coverage_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_other_coverage
    JOIN theme_sets ON theme_sets.id = theme_other_coverage.theme_set_id
    JOIN themes ON themes.id = NEW.theme_id
    WHERE theme_other_coverage.story_id = NEW.story_id
      AND theme_sets.ticker = themes.ticker
      AND theme_sets.trading_day = themes.trading_day
      AND theme_sets.pipeline_version = themes.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already listed under other coverage');
END;

CREATE TRIGGER trg_theme_story_not_excluded_insert
BEFORE INSERT ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_excluded_stories
    JOIN theme_sets ON theme_sets.id = theme_excluded_stories.theme_set_id
    JOIN themes ON themes.id = NEW.theme_id
    WHERE theme_excluded_stories.story_id = NEW.story_id
      AND theme_sets.ticker = themes.ticker
      AND theme_sets.trading_day = themes.trading_day
      AND theme_sets.pipeline_version = themes.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already excluded from this theme set');
END;

CREATE TRIGGER trg_theme_story_not_excluded_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_excluded_stories
    JOIN theme_sets ON theme_sets.id = theme_excluded_stories.theme_set_id
    JOIN themes ON themes.id = NEW.theme_id
    WHERE theme_excluded_stories.story_id = NEW.story_id
      AND theme_sets.ticker = themes.ticker
      AND theme_sets.trading_day = themes.trading_day
      AND theme_sets.pipeline_version = themes.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already excluded from this theme set');
END;

CREATE TRIGGER trg_excluded_story_not_in_theme_update
BEFORE UPDATE ON theme_excluded_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN themes
      ON themes.ticker = theme_sets.ticker
     AND themes.trading_day = theme_sets.trading_day
     AND themes.pipeline_version = theme_sets.pipeline_version
    JOIN theme_stories ON theme_stories.theme_id = themes.id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_stories.story_id = NEW.story_id
)
   OR EXISTS (
    SELECT 1 FROM theme_other_coverage
    WHERE theme_other_coverage.theme_set_id = NEW.theme_set_id
      AND theme_other_coverage.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already accounted for in this theme set');
END;

CREATE TRIGGER trg_other_coverage_not_excluded_insert
BEFORE INSERT ON theme_other_coverage
WHEN EXISTS (
    SELECT 1 FROM theme_excluded_stories
    WHERE theme_excluded_stories.theme_set_id = NEW.theme_set_id
      AND theme_excluded_stories.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already accounted for in this theme set');
END;

CREATE TRIGGER trg_other_coverage_not_excluded_update
BEFORE UPDATE ON theme_other_coverage
WHEN EXISTS (
    SELECT 1 FROM theme_excluded_stories
    WHERE theme_excluded_stories.theme_set_id = NEW.theme_set_id
      AND theme_excluded_stories.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already accounted for in this theme set');
END;

CREATE TRIGGER trg_other_coverage_same_ticker_day_update
BEFORE UPDATE ON theme_other_coverage
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'other coverage must share the theme-set ticker/day');
END;

CREATE TRIGGER trg_excluded_story_same_ticker_day_update
BEFORE UPDATE ON theme_excluded_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'exclusion must share the theme-set ticker/day');
END;

-- A parent cannot be moved out from under the rows that hang off it: the
-- child-side triggers above all check against the parent's *current*
-- ticker/day, so relocating the parent would slip past every one of them.

CREATE TRIGGER trg_theme_partition_locked
BEFORE UPDATE OF ticker, trading_day, pipeline_version ON themes
WHEN (
    OLD.ticker <> NEW.ticker
    OR OLD.trading_day <> NEW.trading_day
    OR OLD.pipeline_version <> NEW.pipeline_version
)
AND (
    EXISTS (SELECT 1 FROM theme_stories WHERE theme_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_citations WHERE theme_id = OLD.id)
)
BEGIN
    SELECT RAISE(
        ABORT,
        'a populated theme cannot change ticker, day, or pipeline version'
    );
END;

CREATE TRIGGER trg_theme_set_partition_locked
BEFORE UPDATE OF ticker, trading_day, pipeline_version ON theme_sets
WHEN (
    OLD.ticker <> NEW.ticker
    OR OLD.trading_day <> NEW.trading_day
    OR OLD.pipeline_version <> NEW.pipeline_version
)
AND (
    EXISTS (SELECT 1 FROM theme_other_coverage WHERE theme_set_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_excluded_stories WHERE theme_set_id = OLD.id)
)
BEGIN
    SELECT RAISE(
        ABORT,
        'a populated theme set cannot change ticker, day, or pipeline version'
    );
END;

CREATE TRIGGER trg_story_partition_locked
BEFORE UPDATE OF ticker, trading_day ON stories
WHEN (OLD.ticker <> NEW.ticker OR OLD.trading_day <> NEW.trading_day)
AND (
    EXISTS (SELECT 1 FROM theme_stories WHERE story_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_other_coverage WHERE story_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_excluded_stories WHERE story_id = OLD.id)
)
BEGIN
    SELECT RAISE(
        ABORT,
        'a story referenced by a theme set cannot change ticker or day'
    );
END;
