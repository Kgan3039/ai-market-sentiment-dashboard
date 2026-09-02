# AI Market Sentiment Dashboard Engineering Contract

## Phase 0 Dependency Order

I1 Persistence
→ I2 Yahoo ingestion
→ I3 RSS ingestion
→ I4 Pipeline orchestration
→ real M4 evaluation
→ M3/M5 recalibration
→ downstream summarization / RAG

Do not treat a dependent stage as production-complete while its upstream contract is unstable.

## Approved Ticker Universe

- AAPL
- AMD
- META
- NVDA
- TSLA

Ticker, trading-day, and pipeline-version boundaries must be explicit and deterministic.

## Evidence Principles

- Raw publisher evidence is preserved according to the persistence contract.
- Operational metadata is credential-safe.
- Canonical stories retain provenance to source evidence.
- Themes must never silently lose stories.
- Citations resolve only to evidence belonging to the associated canonical story/theme.
- Quarantine and conflict state may not be silently discarded downstream.

## Persistence Principles

- Normal pipeline writes are run-scoped and logged.
- Replay and idempotency behavior are deterministic.
- Released migrations are immutable.
- New schema work uses additive migrations.
- Direct database bypasses are not normal pipeline APIs.

## Evaluation Principles

- Synthetic M4 fixtures are development-regression evidence.
- Synthetic, single-author, unadjudicated evaluation is not an acceptance gate.
- M3 thresholds remain provisional until real adjudicated evaluation.
- M5 metrics on authored ticker-days do not establish G1 or AC-4 production acceptance.

## Product AI Direction

ingestion
→ embeddings
→ exact dedup
→ semantic dedup
→ themes
→ grounded retrieval / RAG
→ citation-backed market explanations

## Engineering Automation Direction

GitHub
→ implementation agent
→ tests
→ independent reviewer
→ repair loop
→ human merge gate

LangGraph may orchestrate the stateful engineering workflow after the basic GitHub Action loop is proven reliable.

MCP may provide standardized GitHub, repository, CI, and test tooling.

LangChain may be used in the product RAG layer where it reduces retrieval, model, or prompt plumbing without replacing working M1–M5 architecture merely for framework adoption.
