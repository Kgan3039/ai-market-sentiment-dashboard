-- Mandatory stage/run logging, retry and replay state, and the
-- database-enforced link between a run-log row and the stage key it ran.

ALTER TABLE run_log ADD COLUMN ticker TEXT;
ALTER TABLE run_log ADD COLUMN success_count INTEGER NOT NULL DEFAULT 0
    CHECK (success_count >= 0);
ALTER TABLE run_log ADD COLUMN partial_count INTEGER NOT NULL DEFAULT 0
    CHECK (partial_count >= 0);
ALTER TABLE run_log ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0
    CHECK (failure_count >= 0);
ALTER TABLE run_log ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1
    CHECK (attempt > 0);
ALTER TABLE run_log ADD COLUMN replay INTEGER NOT NULL DEFAULT 0
    CHECK (replay IN (0, 1));

CREATE INDEX IF NOT EXISTS idx_run_log_day_stage
    ON run_log(trading_day, stage);

CREATE TRIGGER IF NOT EXISTS trg_run_log_ticker_insert
BEFORE INSERT ON run_log
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_run_log_ticker_update
BEFORE UPDATE OF ticker ON run_log
WHEN NEW.ticker IS NOT NULL
 AND NEW.ticker NOT IN (SELECT ticker FROM supported_tickers)
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

ALTER TABLE pipeline_stage_keys ADD COLUMN attempts INTEGER NOT NULL
    DEFAULT 0 CHECK (attempts >= 0);
ALTER TABLE pipeline_stage_keys ADD COLUMN claimed_at TEXT CHECK (
    claimed_at IS NULL OR datetime(claimed_at) IS NOT NULL
);
ALTER TABLE pipeline_stage_keys ADD COLUMN completed_at TEXT CHECK (
    completed_at IS NULL OR datetime(completed_at) IS NOT NULL
);
ALTER TABLE pipeline_stage_keys ADD COLUMN recovered_count INTEGER NOT NULL
    DEFAULT 0 CHECK (recovered_count >= 0);
ALTER TABLE pipeline_stage_keys ADD COLUMN last_error TEXT;

UPDATE pipeline_stage_keys
SET attempts = 1, claimed_at = updated_at
WHERE attempts = 0;

CREATE TABLE IF NOT EXISTS run_log_stage_keys (
    run_log_id INTEGER NOT NULL
        REFERENCES run_log(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    PRIMARY KEY (run_log_id, stage, ticker, trading_day, pipeline_version),
    FOREIGN KEY (stage, ticker, trading_day, pipeline_version)
        REFERENCES pipeline_stage_keys(
            stage, ticker, trading_day, pipeline_version
        ) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_log_stage_keys_key
    ON run_log_stage_keys(stage, ticker, trading_day, pipeline_version);

ALTER TABLE source_state ADD COLUMN status TEXT NOT NULL DEFAULT 'unknown'
    CHECK (status IN ('success', 'partial', 'empty', 'failed', 'unknown'));
ALTER TABLE source_state ADD COLUMN consecutive_failures INTEGER NOT NULL
    DEFAULT 0 CHECK (consecutive_failures >= 0);
ALTER TABLE source_state ADD COLUMN last_error TEXT;
ALTER TABLE source_state ADD COLUMN retry_after TEXT CHECK (
    retry_after IS NULL OR datetime(retry_after) IS NOT NULL
);
