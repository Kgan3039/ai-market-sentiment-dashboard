-- Durable storage for one M2/M3 reconciliation of a ticker/trading-day.
--
-- M2's ``cluster_fingerprint`` and M3's ``story_fingerprint`` are
-- deliberately *not* durable identifiers (see nlp/dedup/models.py and
-- nlp/semdedup/models.py).  They are stored beside ``stories.id`` as
-- change-detection handles so a reconciler can tell insert from update
-- from unchanged without ever mistaking one for a primary key.

ALTER TABLE stories ADD COLUMN cluster_fingerprint TEXT;
ALTER TABLE stories ADD COLUMN pipeline_version TEXT;
ALTER TABLE stories ADD COLUMN stage TEXT CHECK (
    stage IS NULL OR stage IN ('m2.exact', 'm3.semantic')
);
ALTER TABLE stories ADD COLUMN canonical_item_id INTEGER;
ALTER TABLE stories ADD COLUMN canonical_url TEXT;
ALTER TABLE stories ADD COLUMN source TEXT;
ALTER TABLE stories ADD COLUMN outlet TEXT;
ALTER TABLE stories ADD COLUMN published_at TEXT CHECK (
    published_at IS NULL OR datetime(published_at) IS NOT NULL
);
ALTER TABLE stories ADD COLUMN content_hash TEXT;
ALTER TABLE stories ADD COLUMN algorithm_version TEXT;
ALTER TABLE stories ADD COLUMN config_fingerprint TEXT;
ALTER TABLE stories ADD COLUMN model_name TEXT;
ALTER TABLE stories ADD COLUMN model_revision TEXT;
ALTER TABLE stories ADD COLUMN embedding_dimension INTEGER CHECK (
    embedding_dimension IS NULL OR embedding_dimension > 0
);
ALTER TABLE stories ADD COLUMN quarantined INTEGER NOT NULL DEFAULT 0
    CHECK (quarantined IN (0, 1));
ALTER TABLE stories ADD COLUMN semantic_skip_reason TEXT;
ALTER TABLE stories ADD COLUMN member_story_keys TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(member_story_keys));
ALTER TABLE stories ADD COLUMN invalidated_at TEXT CHECK (
    invalidated_at IS NULL OR datetime(invalidated_at) IS NOT NULL
);
ALTER TABLE stories ADD COLUMN updated_at TEXT CHECK (
    updated_at IS NULL OR datetime(updated_at) IS NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_stories_cluster_fingerprint
    ON stories(ticker, trading_day, pipeline_version, cluster_fingerprint)
    WHERE cluster_fingerprint IS NOT NULL AND pipeline_version IS NOT NULL;

ALTER TABLE story_members ADD COLUMN outlet TEXT;
ALTER TABLE story_members ADD COLUMN url TEXT;
ALTER TABLE story_members ADD COLUMN canonical_url TEXT;
ALTER TABLE story_members ADD COLUMN match_reason TEXT;
ALTER TABLE story_members ADD COLUMN quarantined INTEGER NOT NULL DEFAULT 0
    CHECK (quarantined IN (0, 1));

CREATE TABLE IF NOT EXISTS story_provider_conflicts (
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE CASCADE,
    provider_namespace TEXT NOT NULL CHECK (
        length(trim(provider_namespace)) > 0
    ),
    provider_item_id TEXT NOT NULL CHECK (
        length(trim(provider_item_id)) > 0
    ),
    item_ids TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(item_ids) AND json_type(item_ids) = 'array'
    ),
    fields TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(fields) AND json_type(fields) = 'array'
    ),
    PRIMARY KEY (story_id, provider_namespace, provider_item_id)
);

CREATE TABLE IF NOT EXISTS story_semantic_merges (
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE CASCADE,
    left_story_key TEXT NOT NULL CHECK (length(trim(left_story_key)) > 0),
    right_story_key TEXT NOT NULL CHECK (length(trim(right_story_key)) > 0),
    similarity REAL NOT NULL CHECK (similarity BETWEEN -1.0 AND 1.0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    PRIMARY KEY (story_id, left_story_key, right_story_key)
);

-- The canonical member is set after membership exists, so the database can
-- check it rather than trusting insert ordering.

CREATE TRIGGER IF NOT EXISTS trg_story_canonical_member_update
BEFORE UPDATE OF canonical_item_id ON stories
WHEN NEW.canonical_item_id IS NOT NULL
 AND NOT EXISTS (
    SELECT 1 FROM story_members
    WHERE story_members.story_id = NEW.id
      AND story_members.raw_item_id = NEW.canonical_item_id
 )
BEGIN
    SELECT RAISE(ABORT, 'canonical item must be a member of the story');
END;

CREATE TRIGGER IF NOT EXISTS trg_story_canonical_member_delete
BEFORE DELETE ON story_members
WHEN EXISTS (
    SELECT 1 FROM stories
    WHERE stories.id = OLD.story_id
      AND stories.canonical_item_id = OLD.raw_item_id
)
BEGIN
    SELECT RAISE(ABORT, 'canonical member cannot leave its story');
END;

-- A theme may only group stories from its own ticker and trading day.

CREATE TRIGGER IF NOT EXISTS trg_theme_story_same_ticker_day_insert
BEFORE INSERT ON theme_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM themes
    JOIN stories ON stories.id = NEW.story_id
    WHERE themes.id = NEW.theme_id
      AND themes.ticker = stories.ticker
      AND themes.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'theme story must share the theme ticker and day');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_same_ticker_day_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM themes
    JOIN stories ON stories.id = NEW.story_id
    WHERE themes.id = NEW.theme_id
      AND themes.ticker = stories.ticker
      AND themes.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'theme story must share the theme ticker and day');
END;
