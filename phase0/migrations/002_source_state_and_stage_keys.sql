CREATE TABLE IF NOT EXISTS source_state (
    source TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT,
    last_checked_at TEXT NOT NULL CHECK (datetime(last_checked_at) IS NOT NULL),
    last_success_at TEXT CHECK (
        last_success_at IS NULL OR datetime(last_success_at) IS NOT NULL
    ),
    metadata TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata))
);

CREATE TABLE IF NOT EXISTS pipeline_stage_keys (
    stage TEXT NOT NULL,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('running', 'success', 'degraded', 'failed')),
    run_id TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (datetime(updated_at) IS NOT NULL),
    PRIMARY KEY(stage, ticker, trading_day, pipeline_version)
);
