# Agent Automation Safety Policy

These rules override convenience and speed.

## Human Approval Required

Automation must stop before:

- force-pushing any branch;
- merging any PR;
- deleting branches;
- rewriting Git history;
- rewriting released or applied migrations;
- changing a PR base;
- closing milestone issues;
- destructive database operations;
- production secret or configuration changes;
- changing security boundaries;
- bypassing failed tests or quality gates.

## Git Safety

- Never commit directly to main.
- Verify current branch and HEAD before changes.
- Fetch before comparing remote history.
- An unexpected remote force-push is a hard stop.
- Force-with-lease requires explicit human approval and an exact expected remote SHA.
- Preserve backup refs before replacing divergent history.

## Database Safety

- Released migrations are immutable.
- Schema evolution is additive.
- Migration failure must fail closed.
- Replay and idempotency contracts must not be weakened.
- Raw evidence and operational metadata must follow their documented trust boundaries.

## Agent Loop Limits

Maximum automatic repair iterations: 3.

Stop sooner when:

- the same blocker appears twice;
- reviewer and implementer disagree about the governing specification;
- scope materially expands;
- dependency contracts are unstable;
- branch history changes;
- a destructive migration or persistence change becomes necessary.

## Trust

Synthetic, authored, or unadjudicated datasets are development-regression evidence only unless explicitly approved otherwise.

No automated agent may convert a development metric into a production-readiness claim.
