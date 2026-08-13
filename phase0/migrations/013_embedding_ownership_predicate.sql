-- One ownership predicate, asked by every verb.
--
-- `trg_embedding_owner_insert` accepts an embedding whose `source_id` is a
-- durable row id *or* any handle that row answers to.  The cleanup
-- triggers were narrower on both sides of the question, and each gap
-- collects a vector somebody still owns:
--
--   * the handle branch asked only "does another row still carry this
--     *fingerprint*", so an alias colliding with some other live row's
--     durable id — or, for themes, with its other alias — looked
--     orphaned.  Story A with `cluster_fingerprint = '2'` dying took the
--     vector belonging to story B whose id is 2.
--
--   * the durable-id branch asked nothing at all.  Deleting story 1 took
--     the vector cached under `'1'` even when a live story's
--     `cluster_fingerprint` was the string `'1'`, which the insert
--     trigger would have accepted as ownership a moment earlier.
--
-- Themes are worse off than stories only because they have one more
-- accepted form: `fingerprint` and `theme_key` collide with each other as
-- readily as either collides with an id.  These are not exotic states.
-- Handles are unique only within a ticker/trading-day/pipeline-version,
-- so two partitions repeating one is ordinary, and a digest that happens
-- to be a short numeral is a coincidence nothing forbids.
--
-- So the rule is stated once, in the shape the insert trigger already
-- uses: a vector is collected only when *no* live row of its kind would
-- still be allowed to own that `source_id`, through any accepted form.
-- The `NOT EXISTS` is correlated to `embeddings.source_id` rather than to
-- one `OLD` column, which is what lets both branches share it — and is
-- why this reads as one question rather than three.
--
-- Migration 012 already asks exactly this on the rename path, so nothing
-- there changes; this brings deletion into line with it.  `raw_items` is
-- left alone deliberately: its ownership predicate has exactly one form,
-- the durable id, which is the one its cleanup already uses, and
-- AUTOINCREMENT means a deleted id is never handed out again.
--
-- Replacing a trigger is additive.  Migrations 007 and 012 keep their
-- bytes and their checksums; this file drops what 007 created and creates
-- its replacement, which is a schema change like any other and settles in
-- its own transaction.

DROP TRIGGER IF EXISTS trg_embedding_story_cleanup;

CREATE TRIGGER trg_embedding_story_cleanup
AFTER DELETE ON stories
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'story'
      AND source_id IN (CAST(OLD.id AS TEXT), OLD.cluster_fingerprint)
      AND NOT EXISTS (
          SELECT 1 FROM stories
          WHERE CAST(stories.id AS TEXT) = embeddings.source_id
             OR stories.cluster_fingerprint = embeddings.source_id
      );
END;

DROP TRIGGER IF EXISTS trg_embedding_theme_cleanup;

CREATE TRIGGER trg_embedding_theme_cleanup
AFTER DELETE ON themes
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'theme'
      AND source_id IN (
          CAST(OLD.id AS TEXT), OLD.fingerprint, OLD.theme_key
      )
      AND NOT EXISTS (
          SELECT 1 FROM themes
          WHERE CAST(themes.id AS TEXT) = embeddings.source_id
             OR themes.fingerprint = embeddings.source_id
             OR themes.theme_key = embeddings.source_id
      );
END;
