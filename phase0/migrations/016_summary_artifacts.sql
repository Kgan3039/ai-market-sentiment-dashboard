-- A3: durable storage for guarded summaries and the generations that
-- produced them.
--
-- The summary a reader sees is *not* `themes.summary`.  That column, with
-- `themes.status` and `themes.citations`, belongs to theme reconciliation,
-- which rewrites all three on every settlement (it writes `summary` as
-- NULL), so anything a summarizer put there would be erased on the next
-- replay -- and a replay would then report a change that nothing made.
-- A3 therefore owns five tables of its own and touches none of the theme
-- stage's.
--
-- Two design rules shape them:
--
-- 1. **No foreign key to `themes` or `stories`.**  Both are legitimately
--    deleted and recreated by ordinary reconciliation: any story change
--    deletes the whole theme set, and an obsolete story is hard-deleted.
--    A CASCADE would erase summary history every time a partition moved;
--    a RESTRICT would stop the story and theme stages from settling.  So
--    `theme_id` and `story_id` are persisted as *logical* identifiers of
--    the rows they named when the artifact was accepted.  Both are
--    AUTOINCREMENT ids, so a recreated theme or story can never collide
--    with an old artifact's key.
--
-- 2. **An accepted artifact is immutable.**  "Current" is not a stored
--    flag: it is derived, at read time, by rebuilding the frozen A2 input
--    from the live theme population and matching the artifact's exact
--    `(theme_id, input_fingerprint, policy_fingerprint)` key, then
--    re-running A2's pure validator.  The one mutation a row admits is the
--    audited transition `accepted -> invalidated`, made only when a
--    replacement generation found the stored artifact structurally
--    invalid and is about to take its key.  `status = 'accepted'` means
--    "an originally accepted generation, not yet explicitly invalidated";
--    lifecycle currentness additionally requires successful validation.
--
--    Immutability is enforced twice.  Triggers refuse every UPDATE of an
--    artifact and its children, and refuse INSERT or DELETE of a sentence
--    or citation once the artifact is *sealed* -- once a
--    `summary_generations` row with outcome 'accepted' names it, which the
--    write path inserts last, in the same transaction.  Cascade deletes
--    fire those child triggers too, so a sealed artifact cannot be deleted
--    from the parent either; cleanup goes through its generation rows
--    first, on purpose.  And `content_digest` is a SHA-256 over the whole
--    accepted content (see `phase0.repository.summary_artifact_digest`),
--    recomputed from the stored rows before any reuse, so damage the
--    triggers did not see -- a trailing sentence gone, a citation
--    reordered, an identity column forged -- is still refused.
--
-- The ticker domain is a literal here, not a lookup, for the reason
-- migration 009 gives.
--
-- This migration also adds one column to `run_log`, at the end of the
-- file: `last_mutation_id`, the identity of the logged mutation that
-- wrote the row.  It is shared infrastructure rather than an A3 table,
-- but it ships here because A3's crash semantics are what exposed the
-- need for it, and a schema version is a whole.

-- ------------------------------------------------------------------
-- The accepted artifact: what a reader may be shown.
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS summary_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    pipeline_version TEXT NOT NULL CHECK (
        length(trim(pipeline_version)) > 0
    ),
    theme_id INTEGER NOT NULL CHECK (theme_id > 0),
    theme_key TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL CHECK (length(input_fingerprint) = 64),
    policy_fingerprint TEXT NOT NULL CHECK (length(policy_fingerprint) = 64),
    citation_convention TEXT NOT NULL CHECK (
        length(trim(citation_convention)) > 0
    ),
    prompt_version TEXT NOT NULL CHECK (length(trim(prompt_version)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    label TEXT NOT NULL CHECK (length(trim(label)) > 0),
    guarantee TEXT NOT NULL CHECK (length(trim(guarantee)) > 0),
    -- SHA-256 over the canonical accepted content: identity, label,
    -- guarantee, every sentence (ordinal, text) and every citation
    -- (position, story_id).  Recomputed from the stored rows on read.
    content_digest TEXT NOT NULL CHECK (length(content_digest) = 64),
    status TEXT NOT NULL DEFAULT 'accepted'
        CHECK (status IN ('accepted', 'invalidated')),
    created_at TEXT NOT NULL CHECK (datetime(created_at) IS NOT NULL),
    invalidated_at TEXT CHECK (
        invalidated_at IS NULL OR datetime(invalidated_at) IS NOT NULL
    ),
    invalidated_reason TEXT,
    CHECK (
        (
            status = 'accepted'
            AND invalidated_at IS NULL
            AND invalidated_reason IS NULL
        )
        OR (
            status = 'invalidated'
            AND invalidated_at IS NOT NULL
            AND length(trim(COALESCE(invalidated_reason, ''))) > 0
        )
    )
);

-- One accepted artifact per exact generation key.  Two workers that both
-- generated for the same frozen input under the same policy cannot both
-- land; the second is recorded as a duplicate of the first.
CREATE UNIQUE INDEX IF NOT EXISTS idx_summary_artifacts_accepted_key
    ON summary_artifacts(theme_id, input_fingerprint, policy_fingerprint)
    WHERE status = 'accepted';

CREATE INDEX IF NOT EXISTS idx_summary_artifacts_partition
    ON summary_artifacts(ticker, trading_day, pipeline_version, theme_id);

CREATE TRIGGER IF NOT EXISTS trg_summary_artifact_ticker_insert
BEFORE INSERT ON summary_artifacts
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_artifact_ticker_update
BEFORE UPDATE OF ticker ON summary_artifacts
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

-- The only permitted update is accepted -> invalidated, and it may change
-- nothing but the three invalidation columns.
CREATE TRIGGER IF NOT EXISTS trg_summary_artifact_immutable
BEFORE UPDATE ON summary_artifacts
WHEN OLD.status <> 'accepted'
  OR NEW.status <> 'invalidated'
  OR OLD.id IS NOT NEW.id
  OR OLD.ticker IS NOT NEW.ticker
  OR OLD.trading_day IS NOT NEW.trading_day
  OR OLD.pipeline_version IS NOT NEW.pipeline_version
  OR OLD.theme_id IS NOT NEW.theme_id
  OR OLD.theme_key IS NOT NEW.theme_key
  OR OLD.input_fingerprint IS NOT NEW.input_fingerprint
  OR OLD.policy_fingerprint IS NOT NEW.policy_fingerprint
  OR OLD.citation_convention IS NOT NEW.citation_convention
  OR OLD.prompt_version IS NOT NEW.prompt_version
  OR OLD.model IS NOT NEW.model
  OR OLD.label IS NOT NEW.label
  OR OLD.guarantee IS NOT NEW.guarantee
  OR OLD.content_digest IS NOT NEW.content_digest
  OR OLD.created_at IS NOT NEW.created_at
BEGIN
    SELECT RAISE(
        ABORT,
        'an accepted summary artifact is immutable except for its invalidation'
    );
END;

-- ------------------------------------------------------------------
-- Ordered sentences, and each sentence's ordered story citations.
--
-- A citation is a persisted story id; its citation id is exactly
-- `story:<story_id>` under the artifact's convention and is not stored
-- twice.  These are *generated sentence selections*, not the theme's
-- evidence membership (`theme_citations`, which is raw-item grained and
-- owned by the theme stage).
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS summary_sentences (
    artifact_id INTEGER NOT NULL
        REFERENCES summary_artifacts(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    PRIMARY KEY (artifact_id, ordinal)
);

CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_immutable
BEFORE UPDATE ON summary_sentences
BEGIN
    SELECT RAISE(ABORT, 'summary sentences are immutable');
END;

-- Sealed: an accepted generation names the artifact.  From then on its
-- sentence set is fixed -- nothing added, nothing removed -- whether the
-- artifact is still accepted or has since been invalidated (history is
-- kept whole).  The write path inserts the sentences before the sealing
-- generation row, so its own inserts pass.
CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_sealed_insert
BEFORE INSERT ON summary_sentences
WHEN EXISTS (
    SELECT 1 FROM summary_generations
    WHERE summary_generations.artifact_id = NEW.artifact_id
      AND summary_generations.outcome = 'accepted'
)
BEGIN
    SELECT RAISE(ABORT, 'a sealed summary artifact cannot gain a sentence');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_sealed_delete
BEFORE DELETE ON summary_sentences
WHEN EXISTS (
    SELECT 1 FROM summary_generations
    WHERE summary_generations.artifact_id = OLD.artifact_id
      AND summary_generations.outcome = 'accepted'
)
BEGIN
    SELECT RAISE(ABORT, 'a sealed summary artifact cannot lose a sentence');
END;

CREATE TABLE IF NOT EXISTS summary_sentence_citations (
    artifact_id INTEGER NOT NULL,
    sentence_ordinal INTEGER NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    story_id INTEGER NOT NULL CHECK (story_id > 0),
    PRIMARY KEY (artifact_id, sentence_ordinal, position),
    UNIQUE (artifact_id, sentence_ordinal, story_id),
    FOREIGN KEY (artifact_id, sentence_ordinal)
        REFERENCES summary_sentences(artifact_id, ordinal) ON DELETE CASCADE
);

CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_citation_immutable
BEFORE UPDATE ON summary_sentence_citations
BEGIN
    SELECT RAISE(ABORT, 'summary sentence citations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_citation_sealed_insert
BEFORE INSERT ON summary_sentence_citations
WHEN EXISTS (
    SELECT 1 FROM summary_generations
    WHERE summary_generations.artifact_id = NEW.artifact_id
      AND summary_generations.outcome = 'accepted'
)
BEGIN
    SELECT RAISE(ABORT, 'a sealed summary artifact cannot gain a citation');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_sentence_citation_sealed_delete
BEFORE DELETE ON summary_sentence_citations
WHEN EXISTS (
    SELECT 1 FROM summary_generations
    WHERE summary_generations.artifact_id = OLD.artifact_id
      AND summary_generations.outcome = 'accepted'
)
BEGIN
    SELECT RAISE(ABORT, 'a sealed summary artifact cannot lose a citation');
END;

-- ------------------------------------------------------------------
-- Generation accounting: one row per lifecycle invocation that called
-- the provider, whatever came of it, and one row per attempt inside it.
--
-- `outcome`:
--   accepted             the artifact in `artifact_id` was inserted here
--   unavailable          A2 returned no summary; `reason` says why
--   discarded_stale      A2 accepted, but at write time the theme was
--                        gone, the population unhealthy, or the input
--                        fingerprint changed; `detail` names which
--   discarded_duplicate  A2 accepted, but an accepted artifact already
--                        held the key; `artifact_id` is that winner
--
-- Token counts are NULL when the provider reported none.  Unknown is not
-- zero, and these are dedicated columns rather than JSON keys so no
-- redaction rule keyed on the word "token" can ever touch them.
--
-- Derived, not stored: provider calls = attempt rows; the accepted
-- attempt = the attempt row with outcome 'accepted'; total latency = the
-- sum of attempt latencies.  No monetary cost: pricing is not a fact
-- this database can vouch for.
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS summary_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    ticker TEXT NOT NULL,
    trading_day TEXT NOT NULL CHECK (
        date(trading_day) IS NOT NULL AND date(trading_day) = trading_day
    ),
    pipeline_version TEXT NOT NULL CHECK (
        length(trim(pipeline_version)) > 0
    ),
    theme_id INTEGER NOT NULL CHECK (theme_id > 0),
    theme_key TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL CHECK (length(input_fingerprint) = 64),
    policy_fingerprint TEXT NOT NULL CHECK (length(policy_fingerprint) = 64),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    max_attempts INTEGER NOT NULL CHECK (max_attempts IN (1, 2)),
    outcome TEXT NOT NULL CHECK (
        outcome IN (
            'accepted', 'unavailable', 'discarded_stale', 'discarded_duplicate'
        )
    ),
    reason TEXT CHECK (
        reason IS NULL OR reason IN (
            'validation_exhausted', 'provider_unavailable',
            'provider_unconfigured'
        )
    ),
    detail TEXT,
    artifact_id INTEGER
        REFERENCES summary_artifacts(id) ON DELETE RESTRICT,
    completed_at TEXT NOT NULL CHECK (datetime(completed_at) IS NOT NULL),
    CHECK (
        (outcome = 'unavailable' AND reason IS NOT NULL)
        OR (outcome <> 'unavailable' AND reason IS NULL)
    ),
    CHECK (
        (
            outcome IN ('accepted', 'discarded_duplicate')
            AND artifact_id IS NOT NULL
        )
        OR (
            outcome IN ('unavailable', 'discarded_stale')
            AND artifact_id IS NULL
        )
    ),
    CHECK (
        outcome <> 'discarded_stale'
        OR length(trim(COALESCE(detail, ''))) > 0
    ),
    -- Persistence idempotency: a retry of the same result under the same
    -- run finds this row instead of writing a second one.
    UNIQUE (run_id, theme_id, input_fingerprint, policy_fingerprint)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_summary_generations_accepted_artifact
    ON summary_generations(artifact_id)
    WHERE outcome = 'accepted';

CREATE INDEX IF NOT EXISTS idx_summary_generations_key
    ON summary_generations(
        theme_id, input_fingerprint, policy_fingerprint, id
    );

CREATE INDEX IF NOT EXISTS idx_summary_generations_partition
    ON summary_generations(ticker, trading_day, pipeline_version, theme_id);

CREATE TRIGGER IF NOT EXISTS trg_summary_generation_ticker_insert
BEFORE INSERT ON summary_generations
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_generation_ticker_update
BEFORE UPDATE OF ticker ON summary_generations
WHEN NEW.ticker NOT IN ('AAPL', 'AMD', 'META', 'NVDA', 'TSLA')
BEGIN
    SELECT RAISE(ABORT, 'unsupported Phase 0 ticker');
END;

CREATE TRIGGER IF NOT EXISTS trg_summary_generation_immutable
BEFORE UPDATE ON summary_generations
BEGIN
    SELECT RAISE(ABORT, 'summary generation accounting is immutable');
END;

CREATE TABLE IF NOT EXISTS summary_generation_attempts (
    generation_id INTEGER NOT NULL
        REFERENCES summary_generations(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    outcome TEXT NOT NULL CHECK (
        outcome IN (
            'accepted', 'rejected', 'provider_error', 'provider_timeout',
            'provider_unconfigured'
        )
    ),
    failures TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(failures) AND json_type(failures) = 'array'
    ),
    latency_ms REAL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    prompt_tokens INTEGER CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0),
    candidate_tokens INTEGER CHECK (
        candidate_tokens IS NULL OR candidate_tokens >= 0
    ),
    total_tokens INTEGER CHECK (total_tokens IS NULL OR total_tokens >= 0),
    error TEXT,
    PRIMARY KEY (generation_id, attempt)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_summary_generation_attempts_accepted
    ON summary_generation_attempts(generation_id)
    WHERE outcome = 'accepted';

CREATE TRIGGER IF NOT EXISTS trg_summary_generation_attempt_immutable
BEFORE UPDATE ON summary_generation_attempts
BEGIN
    SELECT RAISE(ABORT, 'summary generation attempts are immutable');
END;

-- ------------------------------------------------------------------
-- run_log: which logged mutation wrote the row.
-- ------------------------------------------------------------------
--
-- `(run_id, stage)` names a *row*; it does not name the transaction that
-- last wrote it.  A logged mutation whose `commit()` raised has to find
-- out from the disk whether it became durable, and comparing the row's
-- outcome columns with what it intended is not proof: a second writer
-- holding the same run identity can commit a different mutation whose
-- accounting happens to look identical, and the first would then claim a
-- commit that was never its own.
--
-- So every logged mutation writes a fresh, random, opaque identifier of
-- itself into the row, in the same transaction as the data it describes.
-- The durable probe then asks "did *my* marker become durable?" rather
-- than "does the row look like what I meant?".  Settlement, recovery and
-- operator writes of the row supply no marker and leave the existing one
-- as it is (see `Phase0Repository._write_run_log`): the column always
-- reads as the identity of the most recent logged mutation whose
-- accounting was atomically written with the row.  It is NULL on rows
-- written before this migration and on rows no logged mutation has ever
-- written.  It is not a secret and is never supplied by a caller.
ALTER TABLE run_log ADD COLUMN last_mutation_id TEXT
    CHECK (last_mutation_id IS NULL OR length(last_mutation_id) = 32);
