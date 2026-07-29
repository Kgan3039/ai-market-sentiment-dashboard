CREATE TABLE source_state (
    source TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT,
    last_checked_at TEXT NOT NULL,
    last_success_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE pipeline_stage_keys (
    stage TEXT NOT NULL,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL,
    run_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(stage, ticker, trading_day, pipeline_version)
);
