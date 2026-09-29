# Repository Agent Instructions

Before reviewing or changing this repository, read:

- .agent/PROJECT_CONTRACT.md
- .agent/SAFETY.md

## Code Review Rules

When performing an independent review, also read:

- .agent/REVIEWER.md

Focus especially on:

- correctness and hostile failure modes;
- migration integrity and additive schema evolution;
- persistence, replay, and idempotency;
- ticker/day/pipeline-version partition boundaries;
- provenance and citation integrity;
- credential and security boundaries;
- dependency-contract compatibility;
- synthetic-vs-production trust claims.

Do not approve merely because tests pass.

Do not demand unrelated refactors.

Separate merge blockers from follow-up work.

Never treat synthetic or single-author development evaluation as production acceptance evidence.
