ALTER TABLE pipeline_stage_keys
    ADD COLUMN lease_expires_at TEXT CHECK (
        lease_expires_at IS NULL OR datetime(lease_expires_at) IS NOT NULL
    );

UPDATE pipeline_stage_keys
SET lease_expires_at = updated_at
WHERE status = 'running' AND lease_expires_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_stage_keys_lease
    ON pipeline_stage_keys(status, lease_expires_at);

DELETE FROM theme_citations
WHERE NOT EXISTS (
    SELECT 1
    FROM theme_stories
    JOIN story_members
        ON story_members.story_id = theme_stories.story_id
    WHERE theme_stories.theme_id = theme_citations.theme_id
      AND story_members.raw_item_id = theme_citations.raw_item_id
);

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_member_insert
BEFORE INSERT ON theme_citations
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_stories
    JOIN story_members
        ON story_members.story_id = theme_stories.story_id
    WHERE theme_stories.theme_id = NEW.theme_id
      AND story_members.raw_item_id = NEW.raw_item_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme citation must belong to a member story'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_member_update
BEFORE UPDATE OF theme_id, raw_item_id ON theme_citations
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_stories
    JOIN story_members
        ON story_members.story_id = theme_stories.story_id
    WHERE theme_stories.theme_id = NEW.theme_id
      AND story_members.raw_item_id = NEW.raw_item_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme citation must belong to a member story'
    );
END;
