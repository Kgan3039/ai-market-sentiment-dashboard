CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ticker TEXT,
    title TEXT,
    description TEXT,
    url TEXT,
    canonical_url TEXT NOT NULL,
    external_id TEXT,
    published_at TEXT CHECK (
        published_at IS NULL OR datetime(published_at) IS NOT NULL
    ),
    fetched_at TEXT NOT NULL CHECK (datetime(fetched_at) IS NOT NULL),
    ingest_status TEXT NOT NULL DEFAULT 'valid'
        CHECK (ingest_status IN ('valid', 'invalid', 'ambiguous')),
    validation_errors TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(validation_errors)),
    raw_json TEXT NOT NULL CHECK (json_valid(raw_json)),
    CHECK (
        ingest_status <> 'valid'
        OR (
            length(trim(COALESCE(title, ''))) > 0
            AND length(trim(COALESCE(url, ''))) > 0
        )
    ),
    UNIQUE(source, canonical_url)
);

CREATE INDEX IF NOT EXISTS idx_raw_items_ticker_published
    ON raw_items(ticker, published_at);

CREATE TABLE IF NOT EXISTS raw_item_tickers (
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    association_type TEXT NOT NULL
        CHECK (association_type IN ('source', 'relevance')),
    PRIMARY KEY (raw_item_id, ticker, association_type)
);

CREATE TABLE IF NOT EXISTS raw_item_candidates (
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (raw_item_id, ticker)
);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    canonical_title TEXT NOT NULL,
    embedding BLOB,
    outlet_count INTEGER NOT NULL DEFAULT 1 CHECK (outlet_count > 0),
    member_ids TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(member_ids))
);

CREATE INDEX IF NOT EXISTS idx_stories_ticker_day
    ON stories(ticker, trading_day);

CREATE TABLE IF NOT EXISTS story_members (
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE CASCADE,
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
    PRIMARY KEY (story_id, raw_item_id)
);

CREATE TABLE IF NOT EXISTS themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    label TEXT NOT NULL,
    summary TEXT,
    citations TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(citations)),
    salience_rank INTEGER NOT NULL CHECK (salience_rank > 0),
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'ready', 'degraded', 'failed')),
    centroid BLOB,
    content_hash TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    UNIQUE(ticker, trading_day, content_hash, pipeline_version)
);

CREATE INDEX IF NOT EXISTS idx_themes_ticker_day
    ON themes(ticker, trading_day);

CREATE TABLE IF NOT EXISTS theme_stories (
    theme_id INTEGER NOT NULL
        REFERENCES themes(id) ON DELETE CASCADE,
    story_id INTEGER NOT NULL
        REFERENCES stories(id) ON DELETE RESTRICT,
    PRIMARY KEY (theme_id, story_id)
);

CREATE TABLE IF NOT EXISTS theme_citations (
    theme_id INTEGER NOT NULL
        REFERENCES themes(id) ON DELETE CASCADE,
    raw_item_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE RESTRICT,
    PRIMARY KEY (theme_id, raw_item_id)
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    counts TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(counts)),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    errors TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(errors)),
    started_at TEXT NOT NULL CHECK (datetime(started_at) IS NOT NULL),
    completed_at TEXT NOT NULL CHECK (datetime(completed_at) IS NOT NULL),
    status TEXT NOT NULL
        CHECK (status IN ('success', 'degraded', 'failed')),
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    pipeline_version TEXT NOT NULL,
    UNIQUE(run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_run_log_stage_started
    ON run_log(stage, started_at DESC);

CREATE TABLE IF NOT EXISTS eval_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_type TEXT NOT NULL,
    item_a_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    item_b_id INTEGER NOT NULL
        REFERENCES raw_items(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    label TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL CHECK (datetime(created_at) IS NOT NULL),
    CHECK (item_a_id <> item_b_id)
);
