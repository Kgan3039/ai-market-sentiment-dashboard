-- Migration 007 gave the handle-keyed embeddings a survivor guard on the
-- *delete* path: a vector cached under `cluster_fingerprint`, `fingerprint`,
-- or `theme_key` outlives one owner's deletion exactly as long as some
-- other live row still carries that handle.
--
-- It said nothing about the owner that stays alive and renames its handle,
-- which reaches the same end state by a different verb.  `reconcile_themes`
-- does exactly this: it matches a theme by `fingerprint` and writes
-- `theme_key` as an owned column, so a re-run that assigns a new key leaves
-- the old key belonging to nobody while its vector stays in the cache.
-- Nothing reads that row afterwards — every read resolves a handle through
-- a live parent — and nothing ever removes it, because the only cleanup
-- there was fires on DELETE.  A handle later reused by an unrelated
-- partition would then find a vector encoding text it never saw.
--
-- The rule the delete path already states, applied to the other verb: a
-- handle-keyed embedding lives exactly while some live row still answers
-- to that handle.  Renaming is judged the same way deleting is — by asking
-- who is left, not by asking what this row used to be.
--
-- These are AFTER UPDATE triggers, so the row already carries its new
-- handle and the survivor check below naturally excludes it.  That is
-- also what makes rewriting a handle to the value it already had safe:
-- the row still answers to it, so there is nothing to collect.  The
-- `WHEN` clause is a short-circuit on top of that, not the thing keeping
-- a no-op harmless; it uses `IS NOT` rather than `<>` so a NULL on either
-- side compares rather than making the whole condition vanish.
--
-- The survivor check is the ownership predicate from
-- `trg_embedding_owner_insert`, not just the one column: an embedding is
-- orphaned only when *no* live row of that kind would still be allowed to
-- own it, in any of the identity forms migration 007 accepts.  That is
-- what keeps a durable-id vector out of this: `source_id` matching some
-- live row's id is an owner, so a fingerprint moving out from under a
-- string that happens to be a row id changes nothing.
--
-- Nothing here moves a vector to the new handle.  An embedding names the
-- text that produced it; a renamed owner is a re-encode the repository
-- performs explicitly, not a rename the schema performs silently.

CREATE TRIGGER IF NOT EXISTS trg_embedding_story_fingerprint_moved
AFTER UPDATE OF cluster_fingerprint ON stories
WHEN OLD.cluster_fingerprint IS NOT NEW.cluster_fingerprint
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'story'
      AND source_id = OLD.cluster_fingerprint
      AND NOT EXISTS (
          SELECT 1 FROM stories
          WHERE CAST(stories.id AS TEXT) = OLD.cluster_fingerprint
             OR stories.cluster_fingerprint = OLD.cluster_fingerprint
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_embedding_theme_fingerprint_moved
AFTER UPDATE OF fingerprint ON themes
WHEN OLD.fingerprint IS NOT NEW.fingerprint
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'theme'
      AND source_id = OLD.fingerprint
      AND NOT EXISTS (
          SELECT 1 FROM themes
          WHERE CAST(themes.id AS TEXT) = OLD.fingerprint
             OR themes.fingerprint = OLD.fingerprint
             OR themes.theme_key = OLD.fingerprint
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_embedding_theme_key_moved
AFTER UPDATE OF theme_key ON themes
WHEN OLD.theme_key IS NOT NEW.theme_key
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'theme'
      AND source_id = OLD.theme_key
      AND NOT EXISTS (
          SELECT 1 FROM themes
          WHERE CAST(themes.id AS TEXT) = OLD.theme_key
             OR themes.fingerprint = OLD.theme_key
             OR themes.theme_key = OLD.theme_key
      );
END;
