PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ticker TEXT,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE(source, canonical_url)
);

CREATE INDEX IF NOT EXISTS idx_raw_items_ticker_published
    ON raw_items(ticker, published_at);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL,
    canonical_title TEXT NOT NULL,
    embedding BLOB,
    outlet_count INTEGER NOT NULL DEFAULT 1,
    member_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_stories_ticker_day
    ON stories(ticker, trading_day);

CREATE TABLE IF NOT EXISTS themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL,
    label TEXT NOT NULL,
    summary TEXT,
    citations TEXT NOT NULL DEFAULT '[]',
    salience_rank INTEGER NOT NULL,
    status TEXT NOT NULL,
    centroid BLOB,
    content_hash TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    UNIQUE(ticker, trading_day, content_hash, pipeline_version)
);

CREATE INDEX IF NOT EXISTS idx_themes_ticker_day
    ON themes(ticker, trading_day);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    counts TEXT NOT NULL DEFAULT '{}',
    duration_ms INTEGER NOT NULL,
    errors TEXT NOT NULL DEFAULT '[]',
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    trading_day TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    UNIQUE(run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_run_log_stage_started
    ON run_log(stage, started_at DESC);

CREATE TABLE IF NOT EXISTS eval_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_type TEXT NOT NULL,
    item_a_id INTEGER,
    item_b_id INTEGER,
    reviewer TEXT,
    label TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

PRAGMA user_version = 1;
