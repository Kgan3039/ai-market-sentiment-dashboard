-- The durable adapter behind nlp.embeddings.EmbeddingRepository (M1).
--
-- One row per (source_kind, source_id): M1 asks for "the current embedding
-- for a source" and re-encodes whenever the model contract or the input
-- fingerprint moved, so a second row for the same source would only ever
-- be a stale cache nobody can reach.

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_kind TEXT NOT NULL CHECK (
        source_kind IN ('raw_item', 'story', 'theme')
    ),
    source_id TEXT NOT NULL CHECK (
        source_id = trim(source_id) AND length(source_id) > 0
    ),
    model_name TEXT NOT NULL CHECK (
        model_name = trim(model_name) AND length(model_name) > 0
    ),
    model_revision TEXT CHECK (
        model_revision IS NULL
        OR (
            model_revision = trim(model_revision)
            AND length(model_revision) > 0
        )
    ),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    dtype TEXT NOT NULL CHECK (dtype = 'float32'),
    input_fingerprint TEXT NOT NULL CHECK (
        length(input_fingerprint) = 64
        AND lower(input_fingerprint) = input_fingerprint
        AND input_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    vector_blob BLOB NOT NULL CHECK (
        typeof(vector_blob) = 'blob'
        AND length(vector_blob) = 13 + dimension * 4
        AND substr(vector_blob, 1, 8) = CAST('TNEMB001' AS BLOB)
    ),
    created_at TEXT NOT NULL CHECK (datetime(created_at) IS NOT NULL),
    updated_at TEXT NOT NULL CHECK (datetime(updated_at) IS NOT NULL),
    UNIQUE(source_kind, source_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON embeddings(model_name, model_revision, source_kind);

-- Ownership: an embedding names a row that exists.  ``source_id`` is text
-- because a story may be addressed by its durable id or by the upstream
-- fingerprint that produced it before an id existed.

CREATE TRIGGER IF NOT EXISTS trg_embedding_owner_insert
BEFORE INSERT ON embeddings
WHEN NOT (
    (
        NEW.source_kind = 'raw_item'
        AND EXISTS (
            SELECT 1 FROM raw_items
            WHERE CAST(raw_items.id AS TEXT) = NEW.source_id
        )
    )
    OR (
        NEW.source_kind = 'story'
        AND EXISTS (
            SELECT 1 FROM stories
            WHERE CAST(stories.id AS TEXT) = NEW.source_id
               OR stories.cluster_fingerprint = NEW.source_id
        )
    )
    OR (
        NEW.source_kind = 'theme'
        AND EXISTS (
            SELECT 1 FROM themes
            WHERE CAST(themes.id AS TEXT) = NEW.source_id
               OR themes.fingerprint = NEW.source_id
               OR themes.theme_key = NEW.source_id
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'embedding source does not exist');
END;

CREATE TRIGGER IF NOT EXISTS trg_embedding_owner_update
BEFORE UPDATE OF source_kind, source_id ON embeddings
WHEN NOT (
    (
        NEW.source_kind = 'raw_item'
        AND EXISTS (
            SELECT 1 FROM raw_items
            WHERE CAST(raw_items.id AS TEXT) = NEW.source_id
        )
    )
    OR (
        NEW.source_kind = 'story'
        AND EXISTS (
            SELECT 1 FROM stories
            WHERE CAST(stories.id AS TEXT) = NEW.source_id
               OR stories.cluster_fingerprint = NEW.source_id
        )
    )
    OR (
        NEW.source_kind = 'theme'
        AND EXISTS (
            SELECT 1 FROM themes
            WHERE CAST(themes.id AS TEXT) = NEW.source_id
               OR themes.fingerprint = NEW.source_id
               OR themes.theme_key = NEW.source_id
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'embedding source does not exist');
END;

-- Lifecycle: an embedding never outlives the row it describes — but it
-- also never dies while some *other* row still describes it.
--
-- `embeddings` is globally unique on (source_kind, source_id), while
-- `cluster_fingerprint`, `fingerprint`, and `theme_key` are unique only
-- within one ticker/trading-day/pipeline-version.  Two partitions may
-- therefore hold different rows bearing the same handle, and deleting
-- either one used to take the shared embedding with it, leaving the
-- surviving owner without its vector.
--
-- The durable id needs no such guard: it is globally unique, so nothing
-- else can be addressed by it.  Every handle-shaped identity is deleted
-- only once no live row still carries that handle.  These are AFTER
-- DELETE triggers, so the row being deleted is already gone and the
-- NOT EXISTS below asks exactly the right question: is anyone left?

CREATE TRIGGER IF NOT EXISTS trg_embedding_raw_item_cleanup
AFTER DELETE ON raw_items
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'raw_item'
      AND source_id = CAST(OLD.id AS TEXT);
END;

CREATE TRIGGER IF NOT EXISTS trg_embedding_story_cleanup
AFTER DELETE ON stories
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'story'
      AND (
        source_id = CAST(OLD.id AS TEXT)
        OR (
            source_id = OLD.cluster_fingerprint
            AND NOT EXISTS (
                SELECT 1 FROM stories
                WHERE cluster_fingerprint = OLD.cluster_fingerprint
            )
        )
      );
END;

CREATE TRIGGER IF NOT EXISTS trg_embedding_theme_cleanup
AFTER DELETE ON themes
BEGIN
    DELETE FROM embeddings
    WHERE source_kind = 'theme'
      AND (
        source_id = CAST(OLD.id AS TEXT)
        OR (
            source_id = OLD.fingerprint
            AND NOT EXISTS (
                SELECT 1 FROM themes WHERE fingerprint = OLD.fingerprint
            )
        )
        OR (
            source_id = OLD.theme_key
            AND NOT EXISTS (
                SELECT 1 FROM themes WHERE theme_key = OLD.theme_key
            )
        )
      );
END;
