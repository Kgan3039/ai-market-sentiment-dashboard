-- Durable storage for one M5 theme set of a ticker/trading-day.
--
-- ``themes.fingerprint`` is M5's non-durable membership digest and
-- ``themes.theme_key`` is M5's cross-run identity; both are stored beside
-- the durable ``themes.id`` and neither replaces it.

ALTER TABLE themes ADD COLUMN theme_key TEXT;
ALTER TABLE themes ADD COLUMN fingerprint TEXT;
ALTER TABLE themes ADD COLUMN label_source TEXT;
ALTER TABLE themes ADD COLUMN method TEXT CHECK (
    method IS NULL OR method IN (
        'hdbscan', 'agglomerative', 'small_n_fallback',
        'no_separable_structure'
    )
);
ALTER TABLE themes ADD COLUMN salience REAL;
ALTER TABLE themes ADD COLUMN cohesion REAL CHECK (
    cohesion IS NULL OR cohesion BETWEEN -1.0 AND 1.0
);
ALTER TABLE themes ADD COLUMN min_pairwise_cohesion REAL CHECK (
    min_pairwise_cohesion IS NULL
    OR min_pairwise_cohesion BETWEEN -1.0 AND 1.0
);
ALTER TABLE themes ADD COLUMN story_count INTEGER CHECK (
    story_count IS NULL OR story_count >= 0
);
ALTER TABLE themes ADD COLUMN outlet_count INTEGER CHECK (
    outlet_count IS NULL OR outlet_count >= 0
);
ALTER TABLE themes ADD COLUMN latest_published_at TEXT CHECK (
    latest_published_at IS NULL OR datetime(latest_published_at) IS NOT NULL
);
ALTER TABLE themes ADD COLUMN salience_story_component REAL CHECK (
    salience_story_component IS NULL
    OR salience_story_component BETWEEN 0.0 AND 1.0
);
ALTER TABLE themes ADD COLUMN salience_outlet_component REAL CHECK (
    salience_outlet_component IS NULL
    OR salience_outlet_component BETWEEN 0.0 AND 1.0
);
ALTER TABLE themes ADD COLUMN salience_recency_component REAL CHECK (
    salience_recency_component IS NULL
    OR salience_recency_component BETWEEN 0.0 AND 1.0
);
ALTER TABLE themes ADD COLUMN matched_previous_key TEXT;
ALTER TABLE themes ADD COLUMN algorithm_version TEXT;
ALTER TABLE themes ADD COLUMN config_fingerprint TEXT;
ALTER TABLE themes ADD COLUMN model_name TEXT;
ALTER TABLE themes ADD COLUMN model_revision TEXT;
ALTER TABLE themes ADD COLUMN embedding_dimension INTEGER CHECK (
    embedding_dimension IS NULL OR embedding_dimension > 0
);
ALTER TABLE themes ADD COLUMN updated_at TEXT CHECK (
    updated_at IS NULL OR datetime(updated_at) IS NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_themes_fingerprint
    ON themes(ticker, trading_day, pipeline_version, fingerprint)
    WHERE fingerprint IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_themes_theme_key
    ON themes(ticker, trading_day, pipeline_version, theme_key)
    WHERE theme_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS theme_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    pipeline_version TEXT NOT NULL CHECK (
        length(trim(pipeline_version)) > 0
    ),
    method TEXT NOT NULL CHECK (
        method IN (
            'hdbscan', 'agglomerative', 'small_n_fallback',
            'no_separable_structure'
        )
    ),
    method_reason TEXT NOT NULL DEFAULT '',
    quality TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(quality) AND json_type(quality) = 'object'
    ),
    source_metadata TEXT CHECK (
        source_metadata IS NULL
        OR (
            json_valid(source_metadata)
            AND json_type(source_metadata) = 'object'
        )
    ),
    trust_metadata TEXT NOT NULL DEFAULT '{}' CHECK (
        json_valid(trust_metadata) AND json_type(trust_metadata) = 'object'
    ),
    config_fingerprint TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    model_name TEXT,
    model_revision TEXT,
    embedding_dimension INTEGER CHECK (
        embedding_dimension IS NULL OR embedding_dimension > 0
    ),
    updated_at TEXT NOT NULL CHECK (datetime(updated_at) IS NOT NULL),
    UNIQUE(ticker, trading_day, pipeline_version)
);

CREATE TABLE IF NOT EXISTS theme_other_coverage (
    theme_set_id INTEGER NOT NULL
        REFERENCES theme_sets(id) ON DELETE CASCADE,
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE CASCADE,
    reason TEXT NOT NULL CHECK (
        reason IN (
            'below_clustering_floor', 'clustering_noise',
            'below_theme_size_floor', 'below_cohesion_floor',
            'narrative_mismatch', 'surplus_to_theme_cap',
            'theme_incompatible', 'provider_quarantine', 'semantic_skip',
            'degenerate_embedding_geometry', 'insufficient_theme_structure'
        )
    ),
    position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
    PRIMARY KEY (theme_set_id, story_id)
);

CREATE TABLE IF NOT EXISTS theme_excluded_stories (
    theme_set_id INTEGER NOT NULL
        REFERENCES theme_sets(id) ON DELETE CASCADE,
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE CASCADE,
    reason TEXT NOT NULL CHECK (reason IN ('no_encodable_text')),
    PRIMARY KEY (theme_set_id, story_id)
);

CREATE INDEX IF NOT EXISTS idx_theme_sets_day
    ON theme_sets(trading_day, ticker);

-- A raw item may be citable from exactly one theme within a ticker-day and
-- pipeline version.  M5 asserts this in memory
-- (validate_theme_set_invariants); here it is enforced against direct SQL.

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_single_theme_insert
BEFORE INSERT ON theme_citations
WHEN EXISTS (
    SELECT 1
    FROM theme_citations AS other
    JOIN themes AS other_theme ON other_theme.id = other.theme_id
    JOIN themes AS new_theme ON new_theme.id = NEW.theme_id
    WHERE other.raw_item_id = NEW.raw_item_id
      AND other.theme_id <> NEW.theme_id
      AND other_theme.ticker = new_theme.ticker
      AND other_theme.trading_day = new_theme.trading_day
      AND other_theme.pipeline_version = new_theme.pipeline_version
)
BEGIN
    SELECT RAISE(
        ABORT,
        'raw item is already citable from another theme in this ticker-day'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_citation_single_theme_update
BEFORE UPDATE OF theme_id, raw_item_id ON theme_citations
WHEN EXISTS (
    SELECT 1
    FROM theme_citations AS other
    JOIN themes AS other_theme ON other_theme.id = other.theme_id
    JOIN themes AS new_theme ON new_theme.id = NEW.theme_id
    WHERE other.raw_item_id = NEW.raw_item_id
      AND other.theme_id <> NEW.theme_id
      AND other_theme.ticker = new_theme.ticker
      AND other_theme.trading_day = new_theme.trading_day
      AND other_theme.pipeline_version = new_theme.pipeline_version
)
BEGIN
    SELECT RAISE(
        ABORT,
        'raw item is already citable from another theme in this ticker-day'
    );
END;

-- A story is in a theme, or in other coverage, or excluded — never two of
-- them for the same theme set.

CREATE TRIGGER IF NOT EXISTS trg_other_coverage_not_in_theme
BEFORE INSERT ON theme_other_coverage
WHEN EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN themes
      ON themes.ticker = theme_sets.ticker
     AND themes.trading_day = theme_sets.trading_day
     AND themes.pipeline_version = theme_sets.pipeline_version
    JOIN theme_stories ON theme_stories.theme_id = themes.id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_stories.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already a member of a theme');
END;

CREATE TRIGGER IF NOT EXISTS trg_theme_story_not_in_other_coverage
BEFORE INSERT ON theme_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_other_coverage
    JOIN theme_sets ON theme_sets.id = theme_other_coverage.theme_set_id
    JOIN themes ON themes.id = NEW.theme_id
    WHERE theme_other_coverage.story_id = NEW.story_id
      AND theme_sets.ticker = themes.ticker
      AND theme_sets.trading_day = themes.trading_day
      AND theme_sets.pipeline_version = themes.pipeline_version
)
BEGIN
    SELECT RAISE(ABORT, 'story is already listed under other coverage');
END;

CREATE TRIGGER IF NOT EXISTS trg_excluded_story_not_in_theme
BEFORE INSERT ON theme_excluded_stories
WHEN EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN themes
      ON themes.ticker = theme_sets.ticker
     AND themes.trading_day = theme_sets.trading_day
     AND themes.pipeline_version = theme_sets.pipeline_version
    JOIN theme_stories ON theme_stories.theme_id = themes.id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_stories.story_id = NEW.story_id
)
   OR EXISTS (
    SELECT 1 FROM theme_other_coverage
    WHERE theme_other_coverage.theme_set_id = NEW.theme_set_id
      AND theme_other_coverage.story_id = NEW.story_id
)
BEGIN
    SELECT RAISE(ABORT, 'story is already accounted for in this theme set');
END;

-- Other coverage and exclusions may only name stories from their own
-- ticker and trading day.

CREATE TRIGGER IF NOT EXISTS trg_other_coverage_same_ticker_day
BEFORE INSERT ON theme_other_coverage
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'other coverage must share the theme-set ticker/day');
END;

CREATE TRIGGER IF NOT EXISTS trg_excluded_story_same_ticker_day
BEFORE INSERT ON theme_excluded_stories
WHEN NOT EXISTS (
    SELECT 1
    FROM theme_sets
    JOIN stories ON stories.id = NEW.story_id
    WHERE theme_sets.id = NEW.theme_set_id
      AND theme_sets.ticker = stories.ticker
      AND theme_sets.trading_day = stories.trading_day
)
BEGIN
    SELECT RAISE(ABORT, 'exclusion must share the theme-set ticker/day');
END;
