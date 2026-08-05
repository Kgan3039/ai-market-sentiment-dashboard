-- Migration 009 closed the ticker and trading-day halves of M5 partition
-- integrity but left the pipeline_version half open: `theme_stories`,
-- `theme_other_coverage`, and `theme_excluded_stories` compared ticker and
-- day only, so a v1 theme happily accepted a v2 story, and a story could
-- change pipeline_version out from under the theme set citing it.
--
-- A story whose pipeline_version is NULL predates versioning (the
-- administrative `insert_story` path never set it) and is treated as
-- belonging to whichever partition claims it.  Two *stated* versions that
-- disagree are the defect, and that is what these triggers refuse.

-- ------------------------------------------------------------------
-- Remove relationships that already cross a version boundary.
-- ------------------------------------------------------------------

DELETE FROM theme_citations
WHERE theme_id IN (
    SELECT theme_stories.theme_id
    FROM theme_stories
    JOIN themes ON themes.id = theme_stories.theme_id
    JOIN stories ON stories.id = theme_stories.story_id
    WHERE stories.pipeline_version IS NOT NULL
      AND stories.pipeline_version <> themes.pipeline_version
);

DELETE FROM theme_stories
WHERE rowid IN (
    SELECT theme_stories.rowid
    FROM theme_stories
    JOIN themes ON themes.id = theme_stories.theme_id
    JOIN stories ON stories.id = theme_stories.story_id
    WHERE stories.pipeline_version IS NOT NULL
      AND stories.pipeline_version <> themes.pipeline_version
);

DELETE FROM theme_other_coverage
WHERE rowid IN (
    SELECT theme_other_coverage.rowid
    FROM theme_other_coverage
    JOIN theme_sets ON theme_sets.id = theme_other_coverage.theme_set_id
    JOIN stories ON stories.id = theme_other_coverage.story_id
    WHERE stories.pipeline_version IS NOT NULL
      AND stories.pipeline_version <> theme_sets.pipeline_version
);

DELETE FROM theme_excluded_stories
WHERE rowid IN (
    SELECT theme_excluded_stories.rowid
    FROM theme_excluded_stories
    JOIN theme_sets ON theme_sets.id = theme_excluded_stories.theme_set_id
    JOIN stories ON stories.id = theme_excluded_stories.story_id
    WHERE stories.pipeline_version IS NOT NULL
      AND stories.pipeline_version <> theme_sets.pipeline_version
);

-- ------------------------------------------------------------------
-- Theme membership: ticker, day, AND pipeline version.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_theme_story_same_ticker_day_insert;
DROP TRIGGER IF EXISTS trg_theme_story_same_ticker_day_update;

CREATE TRIGGER trg_theme_story_same_partition_insert
BEFORE INSERT ON theme_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM themes
    JOIN stories ON stories.id = NEW.story_id
    WHERE themes.id = NEW.theme_id
      AND themes.ticker = stories.ticker
      AND themes.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = themes.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme story must share the theme ticker, day, and pipeline version'
    );
END;

CREATE TRIGGER trg_theme_story_same_partition_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM themes
    JOIN stories ON stories.id = NEW.story_id
    WHERE themes.id = NEW.theme_id
      AND themes.ticker = stories.ticker
      AND themes.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = themes.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme story must share the theme ticker, day, and pipeline version'
    );
END;

-- ------------------------------------------------------------------
-- Other coverage and exclusions: same three fields.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_other_coverage_same_ticker_day;
DROP TRIGGER IF EXISTS trg_other_coverage_same_ticker_day_update;
DROP TRIGGER IF EXISTS trg_excluded_story_same_ticker_day;
DROP TRIGGER IF EXISTS trg_excluded_story_same_ticker_day_update;

CREATE TRIGGER trg_other_coverage_same_partition_insert
BEFORE INSERT ON theme_other_coverage
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = theme_sets.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'other coverage must share the theme-set ticker/day/version'
    );
END;

CREATE TRIGGER trg_other_coverage_same_partition_update
BEFORE UPDATE ON theme_other_coverage
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = theme_sets.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'other coverage must share the theme-set ticker/day/version'
    );
END;

CREATE TRIGGER trg_excluded_story_same_partition_insert
BEFORE INSERT ON theme_excluded_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = theme_sets.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'exclusion must share the theme-set ticker/day/version'
    );
END;

CREATE TRIGGER trg_excluded_story_same_partition_update
BEFORE UPDATE ON theme_excluded_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND (
          stories.pipeline_version IS NULL
          OR stories.pipeline_version = theme_sets.pipeline_version
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'exclusion must share the theme-set ticker/day/version'
    );
END;

-- ------------------------------------------------------------------
-- A referenced story cannot change partition, version included.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_story_partition_locked;

CREATE TRIGGER trg_story_partition_locked
BEFORE UPDATE OF ticker, trading_day, pipeline_version ON stories
WHEN (
    OLD.ticker <> NEW.ticker
    OR OLD.trading_day <> NEW.trading_day
    OR OLD.pipeline_version IS NOT NEW.pipeline_version
)
AND (
    EXISTS (SELECT 1 FROM theme_stories WHERE story_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_other_coverage WHERE story_id = OLD.id)
    OR EXISTS (SELECT 1 FROM theme_excluded_stories WHERE story_id = OLD.id)
)
BEGIN
    SELECT RAISE(
        ABORT,
        'a story referenced by a theme set cannot change ticker, day, '
        || 'or pipeline version'
    );
END;

-- ------------------------------------------------------------------
-- A populated theme set is anything a theme, membership, citation,
-- other-coverage row, or exclusion still points at.  009 only looked at
-- other coverage and exclusions, so a theme set carrying nothing but
-- themes could still be relocated.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_theme_set_partition_locked;

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
    OR EXISTS (
        SELECT 1 FROM themes
        WHERE themes.ticker = OLD.ticker
          AND themes.trading_day = OLD.trading_day
          AND themes.pipeline_version = OLD.pipeline_version
    )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'a populated theme set cannot change ticker, day, or pipeline version'
    );
END;

-- A populated theme was already locked by 009; it is restated here only so
-- the citation table counts as "populated" too.

DROP TRIGGER IF EXISTS trg_theme_partition_locked;

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
