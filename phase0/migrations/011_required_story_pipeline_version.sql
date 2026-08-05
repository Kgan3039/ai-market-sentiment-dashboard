-- Migration 010 tolerated `stories.pipeline_version IS NULL` as "legacy,
-- belongs to whichever partition claims it".  That was a wildcard, and a
-- wildcard is an escape: nothing stopped a *newly* inserted story from
-- leaving the column NULL and then joining a v1 theme and a v2 theme at
-- the same time.
--
-- The fix is to stop treating NULL as a category at all.  Existing NULL
-- rows are converted to an explicit version — inferred from the
-- relationships they already have, or the deterministic sentinel
-- `legacy-v0` when they have none — and from then on every story carries a
-- stated version that must match exactly.
--
-- A NULL story attached to two *different* versions is genuinely ambiguous:
-- there is no correct answer, and picking one would silently move
-- evidence between partitions.  The migration aborts instead, rolling back
-- whole, so the operator resolves it rather than discovering it later.

-- ------------------------------------------------------------------
-- Refuse to guess: abort on ambiguity before changing anything.
-- ------------------------------------------------------------------

CREATE TEMPORARY TABLE legacy_story_versions AS
SELECT
    stories.id AS story_id,
    COUNT(DISTINCT versions.pipeline_version) AS version_count,
    MIN(versions.pipeline_version) AS inferred_version
FROM stories
LEFT JOIN (
    SELECT theme_stories.story_id AS story_id, themes.pipeline_version
    FROM theme_stories
    JOIN themes ON themes.id = theme_stories.theme_id
    UNION
    SELECT theme_other_coverage.story_id, theme_sets.pipeline_version
    FROM theme_other_coverage
    JOIN theme_sets ON theme_sets.id = theme_other_coverage.theme_set_id
    UNION
    SELECT theme_excluded_stories.story_id, theme_sets.pipeline_version
    FROM theme_excluded_stories
    JOIN theme_sets ON theme_sets.id = theme_excluded_stories.theme_set_id
) AS versions ON versions.story_id = stories.id
WHERE stories.pipeline_version IS NULL
GROUP BY stories.id;

-- RAISE() is only available inside a trigger, so the check borrows one on
-- a scratch table: inserting the single sentinel row fires it, and the
-- trigger aborts the whole migration when any legacy story is ambiguous.
CREATE TEMPORARY TABLE legacy_version_guard (checked INTEGER PRIMARY KEY);

CREATE TEMPORARY TRIGGER trg_legacy_version_ambiguity
BEFORE INSERT ON legacy_version_guard
WHEN EXISTS (SELECT 1 FROM legacy_story_versions WHERE version_count > 1)
BEGIN
    SELECT RAISE(
        ABORT,
        'cannot infer pipeline_version: a legacy story is referenced by more '
        || 'than one pipeline version; resolve it before upgrading'
    );
END;

INSERT INTO legacy_version_guard (checked) VALUES (1);

-- ------------------------------------------------------------------
-- Convert every NULL to an explicit, deterministic version.
--
-- 010's partition lock refuses to let a referenced story change version,
-- which is right for callers and wrong for the one migration whose job is
-- to give those stories a version at all.  It comes off for the backfill
-- and goes straight back on below, inside the same transaction.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_story_partition_locked;

UPDATE stories
SET pipeline_version = COALESCE(
    (
        SELECT inferred_version
        FROM legacy_story_versions
        WHERE legacy_story_versions.story_id = stories.id
    ),
    'legacy-v0'
)
WHERE pipeline_version IS NULL;

DROP TRIGGER trg_legacy_version_ambiguity;
DROP TABLE legacy_version_guard;
DROP TABLE legacy_story_versions;

CREATE TRIGGER trg_story_partition_locked
BEFORE UPDATE OF ticker, trading_day, pipeline_version ON stories
WHEN (
    OLD.ticker <> NEW.ticker
    OR OLD.trading_day <> NEW.trading_day
    OR OLD.pipeline_version <> NEW.pipeline_version
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
-- SQLite cannot add NOT NULL to an existing column without rebuilding the
-- table, and a rebuild is not an additive upgrade.  Triggers enforce the
-- same thing, on INSERT and on UPDATE.
-- ------------------------------------------------------------------

CREATE TRIGGER trg_story_pipeline_version_required_insert
BEFORE INSERT ON stories
WHEN NEW.pipeline_version IS NULL OR trim(NEW.pipeline_version) = ''
BEGIN
    SELECT RAISE(ABORT, 'stories require an explicit pipeline_version');
END;

CREATE TRIGGER trg_story_pipeline_version_required_update
BEFORE UPDATE OF pipeline_version ON stories
WHEN NEW.pipeline_version IS NULL OR trim(NEW.pipeline_version) = ''
BEGIN
    SELECT RAISE(ABORT, 'stories require an explicit pipeline_version');
END;

-- ------------------------------------------------------------------
-- With no NULLs left, the relationship triggers compare exactly.  The
-- `IS NULL OR` escape hatch from 010 is gone.
-- ------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_theme_story_same_partition_insert;
DROP TRIGGER IF EXISTS trg_theme_story_same_partition_update;

CREATE TRIGGER trg_theme_story_same_partition_insert
BEFORE INSERT ON theme_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM themes
    JOIN stories ON stories.id = NEW.story_id
    WHERE themes.id = NEW.theme_id
      AND themes.ticker = stories.ticker
      AND themes.trading_day = stories.trading_day
      AND themes.pipeline_version = stories.pipeline_version
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
      AND themes.pipeline_version = stories.pipeline_version
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme story must share the theme ticker, day, and pipeline version'
    );
END;

DROP TRIGGER IF EXISTS trg_other_coverage_same_partition_insert;
DROP TRIGGER IF EXISTS trg_other_coverage_same_partition_update;

CREATE TRIGGER trg_other_coverage_same_partition_insert
BEFORE INSERT ON theme_other_coverage
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND theme_sets.pipeline_version = stories.pipeline_version
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
      AND theme_sets.pipeline_version = stories.pipeline_version
)
BEGIN
    SELECT RAISE(
        ABORT,
        'other coverage must share the theme-set ticker/day/version'
    );
END;

DROP TRIGGER IF EXISTS trg_excluded_story_same_partition_insert;
DROP TRIGGER IF EXISTS trg_excluded_story_same_partition_update;

CREATE TRIGGER trg_excluded_story_same_partition_insert
BEFORE INSERT ON theme_excluded_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
      AND theme_sets.pipeline_version = stories.pipeline_version
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
      AND theme_sets.pipeline_version = stories.pipeline_version
)
BEGIN
    SELECT RAISE(
        ABORT,
        'exclusion must share the theme-set ticker/day/version'
    );
END;
