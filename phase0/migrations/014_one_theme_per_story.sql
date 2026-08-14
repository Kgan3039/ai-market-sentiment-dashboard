-- One story, one accounting bucket — including "theme" against itself.
--
-- The M5 accounting rules are pairwise, and by 013 five of the six pairs
-- were written: coverage against themes, themes against coverage,
-- exclusions against themes, exclusions against coverage, and coverage
-- against exclusions, each on INSERT and on UPDATE.  The pair nobody
-- wrote is the one where both sides are the same table: a story in *two
-- themes* of one partition.
--
-- It is not blocked incidentally either.  A theme's citations must
-- reference raw items belonging to its member stories, and no two themes
-- in a partition may cite the same raw item — which stops the obvious
-- attempt, since two themes sharing a one-item story would have to share
-- its citation.  Give that story a second member and the collision goes
-- away: theme A cites the first item, theme B the second, both claim the
-- story, and every existing rule is satisfied.  The result is a day whose
-- theme cards double-count a story, and whose Other Coverage arithmetic
-- no longer adds up.
--
-- The invariant is stated per partition, matching the M5 triggers, and
-- that is also the whole of it: migration 011 made `stories.pipeline_version`
-- required and 010/011's partition triggers require a theme's member to
-- share the theme's ticker, trading day, *and* pipeline version exactly.
-- So a story belongs to exactly one partition and the only themes that
-- can ever claim it are in that partition — "at most one theme here" is
-- therefore also "at most one theme anywhere", and neither reading needs
-- the durable id to be treated as something it is not.
--
-- The `UPDATE` trigger excludes the row being updated by its own
-- (theme_id, story_id): `BEFORE UPDATE` still sees the old row, so a
-- membership merely moving from one theme to another would otherwise
-- collide with itself.

-- ------------------------------------------------------------------
-- Refuse to guess: abort on data that already violates the rule.
--
-- Same policy as 011's ambiguous legacy versions.  Choosing a winner
-- among two themes claiming a story is a content decision — which card
-- the story belongs on — and nothing here can make it.  A database
-- carrying such a pair stays at version 13, whole, until an operator
-- resolves it.
-- ------------------------------------------------------------------

CREATE TEMPORARY TABLE duplicate_theme_membership AS
SELECT theme_stories.story_id AS story_id
FROM theme_stories
JOIN themes ON themes.id = theme_stories.theme_id
GROUP BY
    theme_stories.story_id,
    themes.ticker,
    themes.trading_day,
    themes.pipeline_version
HAVING COUNT(*) > 1;

-- RAISE() only exists inside a trigger, so the check borrows one on a
-- scratch table: inserting the sentinel fires it, and the trigger aborts
-- the whole migration.
CREATE TEMPORARY TABLE theme_membership_guard (checked INTEGER PRIMARY KEY);

CREATE TEMPORARY TRIGGER trg_duplicate_theme_membership
BEFORE INSERT ON theme_membership_guard
WHEN EXISTS (SELECT 1 FROM duplicate_theme_membership)
BEGIN
    -- One literal, not a concatenation; see 010 for why.
    SELECT RAISE(
        ABORT,
        'a story already belongs to more than one theme in the same ticker, day, and pipeline version; resolve which theme owns it before upgrading'
    );
END;

INSERT INTO theme_membership_guard (checked) VALUES (1);

DROP TRIGGER trg_duplicate_theme_membership;
DROP TABLE theme_membership_guard;
DROP TABLE duplicate_theme_membership;

-- ------------------------------------------------------------------
-- And from here on, both write paths.
-- ------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS trg_theme_story_in_one_theme_insert
BEFORE INSERT ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_stories AS other
    JOIN themes AS other_theme ON other_theme.id = other.theme_id
    JOIN themes AS this_theme ON this_theme.id = NEW.theme_id
    WHERE other.story_id = NEW.story_id
      AND other.theme_id <> NEW.theme_id
      AND other_theme.ticker = this_theme.ticker
      AND other_theme.trading_day = this_theme.trading_day
      AND other_theme.pipeline_version = this_theme.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already a member of another theme');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_in_one_theme_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_stories AS other
    JOIN themes AS other_theme ON other_theme.id = other.theme_id
    JOIN themes AS this_theme ON this_theme.id = NEW.theme_id
    WHERE other.story_id = NEW.story_id
      AND other.theme_id <> NEW.theme_id
      AND NOT (
          other.theme_id = OLD.theme_id AND other.story_id = OLD.story_id
      )
      AND other_theme.ticker = this_theme.ticker
      AND other_theme.trading_day = this_theme.trading_day
      AND other_theme.pipeline_version = this_theme.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already a member of another theme');
END;
