-- A4c: review provenance binding.
--
-- A4b could describe where a reviewed summary's evidence came from but not
-- verify it: nothing persisted linked a raw item to the logged run that
-- admitted it, a story to the run that built it, or a theme set to the story
-- generation it was clustered over.  This migration adds exactly those
-- links, as nullable columns on the rows they describe, and nothing else.
-- There is no provenance table: `run_log` is already the durable record of
-- a stage run, and every column below names a row in it.
--
-- **No backfill.**  Every existing row keeps NULL, and NULL means "origin
-- not established".  A historical row cannot be blessed later: raw-item
-- provenance is write-once at INSERT (below), and a derived row is bound
-- only by a logged reconciliation that recomputed its exact content.
--
-- **Provenance never outlives the content it attests.**  The triggers below
-- clear a story's or theme set's binding whenever anything the binding
-- speaks for is written *without* re-binding it in a separate statement.
-- The logged repository paths re-bind as their last step; the admin paths
-- and raw SQL do not, so after them the binding is gone rather than stale.
-- Raw-item provenance cannot be cleared, so the content it attests is
-- instead frozen once it is set.
--
-- **Only the repository may set a binding.**  Every trigger named
-- `*_authorized*` below refuses a write that sets a binding unless the
-- application function `phase0_provenance_write_authorized()` returns 1 on
-- the writing connection.  `phase0.repository` registers it on each of its
-- connections, switched off, and switches it on only for the statements a
-- logged stage run binds with.  Knowing a valid run id therefore
-- authorizes nothing: ordinary SQL cannot set, copy, or restore a binding.
-- A bare connection has no such function, so there every write that could
-- set one fails.  (Registering an impostor function is, like dropping these
-- triggers, a bypass of the file itself.)
--
-- **What this is not.**  A run id is a relational link, not a signature.
-- A writer with raw access to this file can forge `run_log` rows and every
-- binding; nothing here claims otherwise.  See docs/PHASE0_DATA_PIPELINE.md.

-- ------------------------------------------------------------------
-- Raw items: the logged ingestion run that inserted the row.
-- ------------------------------------------------------------------

ALTER TABLE raw_items ADD COLUMN ingest_run_id TEXT CHECK (
    ingest_run_id IS NULL
    OR (ingest_run_id = trim(ingest_run_id) AND length(ingest_run_id) > 0)
);

ALTER TABLE raw_items ADD COLUMN ingest_stage TEXT CHECK (
    ingest_stage IS NULL
    OR (ingest_stage = trim(ingest_stage) AND length(ingest_stage) > 0)
);

CREATE TRIGGER IF NOT EXISTS trg_raw_item_provenance_authorized
BEFORE INSERT ON raw_items
WHEN (NEW.ingest_run_id IS NOT NULL OR NEW.ingest_stage IS NOT NULL)
 AND NOT phase0_provenance_write_authorized()
BEGIN
    SELECT RAISE(ABORT, 'raw item ingest provenance is written only by a logged ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_raw_item_provenance_pair
BEFORE INSERT ON raw_items
WHEN (NEW.ingest_run_id IS NULL) <> (NEW.ingest_stage IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'raw item ingest provenance is both run and stage, or neither');
END;

-- Write-once: set at INSERT or never.  NULL -> value is refused as firmly as
-- value -> value, because a later re-fetch proves the URL still resolves,
-- not what the stored row was made from.
CREATE TRIGGER IF NOT EXISTS trg_raw_item_provenance_immutable
BEFORE UPDATE OF ingest_run_id, ingest_stage ON raw_items
WHEN OLD.ingest_run_id IS NOT NEW.ingest_run_id
  OR OLD.ingest_stage IS NOT NEW.ingest_stage
BEGIN
    SELECT RAISE(ABORT, 'raw item ingest provenance is immutable');
END;

-- The evidence an ingest binding attests is frozen with it.  `ticker` and
-- `ingest_status` stay writable: relevance classification owns them, and
-- neither is shown to a summarizer.
CREATE TRIGGER IF NOT EXISTS trg_raw_item_attested_content_immutable
BEFORE UPDATE OF
    source, title, description, url, canonical_url, external_id,
    published_at, fetched_at, raw_json
ON raw_items
WHEN OLD.ingest_run_id IS NOT NULL
 AND (
    OLD.source IS NOT NEW.source
    OR OLD.title IS NOT NEW.title
    OR OLD.description IS NOT NEW.description
    OR OLD.url IS NOT NEW.url
    OR OLD.canonical_url IS NOT NEW.canonical_url
    OR OLD.external_id IS NOT NEW.external_id
    OR OLD.published_at IS NOT NEW.published_at
    OR OLD.fetched_at IS NOT NEW.fetched_at
    OR OLD.raw_json IS NOT NEW.raw_json
 )
BEGIN
    SELECT RAISE(ABORT, 'evidence admitted by a logged ingestion run is immutable');
END;

-- ------------------------------------------------------------------
-- Stories: the logged story run that last wrote the row's content.
-- ------------------------------------------------------------------

ALTER TABLE stories ADD COLUMN build_run_id TEXT CHECK (
    build_run_id IS NULL
    OR (build_run_id = trim(build_run_id) AND length(build_run_id) > 0)
);

CREATE TRIGGER IF NOT EXISTS trg_story_provenance_authorized_insert
BEFORE INSERT ON stories
WHEN NEW.build_run_id IS NOT NULL
 AND NOT phase0_provenance_write_authorized()
BEGIN
    SELECT RAISE(ABORT, 'story provenance is written only by a logged story run');
END;

CREATE TRIGGER IF NOT EXISTS trg_story_provenance_authorized_update
BEFORE UPDATE OF build_run_id ON stories
WHEN NEW.build_run_id IS NOT NULL
 AND OLD.build_run_id IS NOT NEW.build_run_id
 AND NOT phase0_provenance_write_authorized()
BEGIN
    SELECT RAISE(ABORT, 'story provenance is written only by a logged story run');
END;

-- Any update that leaves the binding as it was clears it.  The logged path
-- writes content first and binds in a statement of its own, which changes
-- `build_run_id` and so does not match this condition.
CREATE TRIGGER IF NOT EXISTS trg_story_provenance_cleared
AFTER UPDATE ON stories
WHEN NEW.build_run_id IS NOT NULL
 AND OLD.build_run_id IS NEW.build_run_id
BEGIN
    UPDATE stories SET build_run_id = NULL WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_member_insert_clears_provenance
AFTER INSERT ON story_members
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = NEW.story_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_member_update_clears_provenance
AFTER UPDATE ON story_members
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id IN (OLD.story_id, NEW.story_id) AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_member_delete_clears_provenance
AFTER DELETE ON story_members
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = OLD.story_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_conflict_insert_clears_provenance
AFTER INSERT ON story_provider_conflicts
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = NEW.story_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_conflict_update_clears_provenance
AFTER UPDATE ON story_provider_conflicts
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id IN (OLD.story_id, NEW.story_id) AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_conflict_delete_clears_provenance
AFTER DELETE ON story_provider_conflicts
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = OLD.story_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_merge_insert_clears_provenance
AFTER INSERT ON story_semantic_merges
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = NEW.story_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_merge_update_clears_provenance
AFTER UPDATE ON story_semantic_merges
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id IN (OLD.story_id, NEW.story_id) AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_story_merge_delete_clears_provenance
AFTER DELETE ON story_semantic_merges
BEGIN
    UPDATE stories SET build_run_id = NULL
    WHERE id = OLD.story_id AND build_run_id IS NOT NULL;
END;

-- ------------------------------------------------------------------
-- Theme sets: the logged theme run that last wrote the partition's theme
-- output, and the story-generation signature it verified before writing.
-- ------------------------------------------------------------------

ALTER TABLE theme_sets ADD COLUMN build_run_id TEXT CHECK (
    build_run_id IS NULL
    OR (build_run_id = trim(build_run_id) AND length(build_run_id) > 0)
);

ALTER TABLE theme_sets ADD COLUMN build_story_signature TEXT CHECK (
    build_story_signature IS NULL
    OR (
        length(build_story_signature) = 64
        AND lower(build_story_signature) = build_story_signature
        AND build_story_signature NOT GLOB '*[^0-9a-f]*'
    )
);

-- Stored as an exact INTEGER or not at all: INTEGER affinity keeps a REAL
-- that is not a whole number (1.5) as REAL, and this refuses it rather than
-- let any reader round it into a version.
ALTER TABLE theme_sets ADD COLUMN build_story_signature_version INTEGER CHECK (
    build_story_signature_version IS NULL
    OR (
        typeof(build_story_signature_version) = 'integer'
        AND build_story_signature_version > 0
    )
);

CREATE TRIGGER IF NOT EXISTS trg_theme_set_binding_authorized_insert
BEFORE INSERT ON theme_sets
WHEN (
    NEW.build_run_id IS NOT NULL
    OR NEW.build_story_signature IS NOT NULL
    OR NEW.build_story_signature_version IS NOT NULL
)
 AND NOT phase0_provenance_write_authorized()
BEGIN
    SELECT RAISE(ABORT, 'a theme set build binding is written only by a logged theme run');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_set_binding_authorized_update
BEFORE UPDATE OF build_run_id, build_story_signature, build_story_signature_version
ON theme_sets
WHEN (
    NEW.build_run_id IS NOT NULL
    OR NEW.build_story_signature IS NOT NULL
    OR NEW.build_story_signature_version IS NOT NULL
)
 AND (
    OLD.build_run_id IS NOT NEW.build_run_id
    OR OLD.build_story_signature IS NOT NEW.build_story_signature
    OR OLD.build_story_signature_version IS NOT NEW.build_story_signature_version
 )
 AND NOT phase0_provenance_write_authorized()
BEGIN
    SELECT RAISE(ABORT, 'a theme set build binding is written only by a logged theme run');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_set_binding_complete_insert
BEFORE INSERT ON theme_sets
WHEN NOT (
    (
        NEW.build_run_id IS NULL
        AND NEW.build_story_signature IS NULL
        AND NEW.build_story_signature_version IS NULL
    )
    OR (
        NEW.build_run_id IS NOT NULL
        AND NEW.build_story_signature IS NOT NULL
        AND NEW.build_story_signature_version IS NOT NULL
    )
)
BEGIN
    SELECT RAISE(ABORT, 'a theme set build binding is all three fields, or none');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_set_binding_complete_update
BEFORE UPDATE OF build_run_id, build_story_signature, build_story_signature_version
ON theme_sets
WHEN NOT (
    (
        NEW.build_run_id IS NULL
        AND NEW.build_story_signature IS NULL
        AND NEW.build_story_signature_version IS NULL
    )
    OR (
        NEW.build_run_id IS NOT NULL
        AND NEW.build_story_signature IS NOT NULL
        AND NEW.build_story_signature_version IS NOT NULL
    )
)
BEGIN
    SELECT RAISE(ABORT, 'a theme set build binding is all three fields, or none');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_set_binding_cleared
AFTER UPDATE ON theme_sets
WHEN NEW.build_run_id IS NOT NULL
 AND OLD.build_run_id IS NEW.build_run_id
 AND OLD.build_story_signature IS NEW.build_story_signature
 AND OLD.build_story_signature_version IS NEW.build_story_signature_version
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_insert_clears_binding
AFTER INSERT ON themes
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE ticker = NEW.ticker AND trading_day = NEW.trading_day
      AND pipeline_version = NEW.pipeline_version
      AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_update_clears_binding
AFTER UPDATE ON themes
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE (
        (ticker = OLD.ticker AND trading_day = OLD.trading_day
         AND pipeline_version = OLD.pipeline_version)
        OR (ticker = NEW.ticker AND trading_day = NEW.trading_day
            AND pipeline_version = NEW.pipeline_version)
    )
      AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_delete_clears_binding
AFTER DELETE ON themes
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE ticker = OLD.ticker AND trading_day = OLD.trading_day
      AND pipeline_version = OLD.pipeline_version
      AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_insert_clears_binding
AFTER INSERT ON theme_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id = NEW.theme_id
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_update_clears_binding
AFTER UPDATE ON theme_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id IN (OLD.theme_id, NEW.theme_id)
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_delete_clears_binding
AFTER DELETE ON theme_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id = OLD.theme_id
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_insert_clears_binding
AFTER INSERT ON theme_citations
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id = NEW.theme_id
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_update_clears_binding
AFTER UPDATE ON theme_citations
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id IN (OLD.theme_id, NEW.theme_id)
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_delete_clears_binding
AFTER DELETE ON theme_citations
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE build_run_id IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM themes
        WHERE themes.id = OLD.theme_id
          AND themes.ticker = theme_sets.ticker
          AND themes.trading_day = theme_sets.trading_day
          AND themes.pipeline_version = theme_sets.pipeline_version
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_other_coverage_insert_clears_binding
AFTER INSERT ON theme_other_coverage
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id = NEW.theme_set_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_other_coverage_update_clears_binding
AFTER UPDATE ON theme_other_coverage
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id IN (OLD.theme_set_id, NEW.theme_set_id) AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_other_coverage_delete_clears_binding
AFTER DELETE ON theme_other_coverage
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id = OLD.theme_set_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_excluded_story_insert_clears_binding
AFTER INSERT ON theme_excluded_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id = NEW.theme_set_id AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_excluded_story_update_clears_binding
AFTER UPDATE ON theme_excluded_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id IN (OLD.theme_set_id, NEW.theme_set_id) AND build_run_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_excluded_story_delete_clears_binding
AFTER DELETE ON theme_excluded_stories
BEGIN
    UPDATE theme_sets
    SET build_run_id = NULL,
        build_story_signature = NULL,
        build_story_signature_version = NULL
    WHERE id = OLD.theme_set_id AND build_run_id IS NOT NULL;
END;
