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

CREATE TRIGGER IF NOT EXISTS trg_story_member_citation_delete
BEFORE DELETE ON story_members
WHEN EXISTS (
    SELECT 1
    FROM theme_stories
    JOIN theme_citations
      ON theme_citations.theme_id = theme_stories.theme_id
     AND theme_citations.raw_item_id = OLD.raw_item_id
    WHERE theme_stories.story_id = OLD.story_id
      AND NOT EXISTS (
          SELECT 1
          FROM theme_stories AS alternate_theme_story
          JOIN story_members AS alternate_member
            ON alternate_member.story_id = alternate_theme_story.story_id
          WHERE alternate_theme_story.theme_id = theme_stories.theme_id
            AND alternate_member.raw_item_id = OLD.raw_item_id
            AND NOT (
                alternate_member.story_id = OLD.story_id
                AND alternate_member.raw_item_id = OLD.raw_item_id
            )
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'story member is required by a theme citation'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_story_member_citation_update
BEFORE UPDATE OF story_id, raw_item_id ON story_members
WHEN OLD.story_id <> NEW.story_id OR OLD.raw_item_id <> NEW.raw_item_id
BEGIN
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM theme_stories
            JOIN theme_citations
              ON theme_citations.theme_id = theme_stories.theme_id
             AND theme_citations.raw_item_id = OLD.raw_item_id
            WHERE theme_stories.story_id = OLD.story_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM theme_stories AS alternate_theme_story
                  JOIN story_members AS alternate_member
                    ON alternate_member.story_id =
                        alternate_theme_story.story_id
                  WHERE alternate_theme_story.theme_id =
                        theme_stories.theme_id
                    AND alternate_member.raw_item_id = OLD.raw_item_id
                    AND NOT (
                        alternate_member.story_id = OLD.story_id
                        AND alternate_member.raw_item_id = OLD.raw_item_id
                    )
              )
        )
        THEN RAISE(
            ABORT,
            'story member is required by a theme citation'
        )
    END;
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_citation_delete
BEFORE DELETE ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_citations
    JOIN story_members
      ON story_members.story_id = OLD.story_id
     AND story_members.raw_item_id = theme_citations.raw_item_id
    WHERE theme_citations.theme_id = OLD.theme_id
      AND NOT EXISTS (
          SELECT 1
          FROM theme_stories AS alternate_theme_story
          JOIN story_members AS alternate_member
            ON alternate_member.story_id = alternate_theme_story.story_id
          WHERE alternate_theme_story.theme_id = OLD.theme_id
            AND alternate_theme_story.story_id <> OLD.story_id
            AND alternate_member.raw_item_id =
                theme_citations.raw_item_id
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'theme story is required by a theme citation'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_citation_update
BEFORE UPDATE OF theme_id, story_id ON theme_stories
WHEN OLD.theme_id <> NEW.theme_id OR OLD.story_id <> NEW.story_id
BEGIN
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM theme_citations
            JOIN story_members
              ON story_members.story_id = OLD.story_id
             AND story_members.raw_item_id =
                 theme_citations.raw_item_id
            WHERE theme_citations.theme_id = OLD.theme_id
              AND NOT EXISTS (
                  SELECT 1
                  FROM theme_stories AS alternate_theme_story
                  JOIN story_members AS alternate_member
                    ON alternate_member.story_id =
                        alternate_theme_story.story_id
                  WHERE alternate_theme_story.theme_id = OLD.theme_id
                    AND alternate_theme_story.story_id <> OLD.story_id
                    AND alternate_member.raw_item_id =
                        theme_citations.raw_item_id
              )
        )
        THEN RAISE(
            ABORT,
            'theme story is required by a theme citation'
        )
    END;
END;
