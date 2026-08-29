# Phase 0 Deployment Handoff

**Issue:** B4 / #76
**Status:** Not deployed. This is a readiness runbook, not evidence of a live
deployment.

## Current Blockers

1. The current B1 API deliberately reads committed fixtures. Its live-data
   adapter must resolve cited sentences to persisted evidence, map run and
   persistence health into the API degradation state, and exclude incomplete
   or failed outputs.
2. The merged pipeline must be demonstrated to persist API-eligible, completed
   narrative/theme output before the ticker page switches to live data. Raw
   items and intermediate story records, including degraded intermediate story
   output, do not satisfy this gate.
3. A deployment host, private URL, backup destination, and responsible
   operator have not been provided to this repository.

## Preconditions Before Host Work

1. Complete the B1 live-data readiness gate in
   `docs/phase0_api_contract.md` against the merged I1–I4 persistence and
   pipeline stack.
2. Verify the B2 page against persisted SQLite data, not fixtures.
3. Record B3 screenshots and Kartik’s copy sign-off.
4. Select the VM hostname, private access mechanism, backup destination, and
   environment-secret owner.

## Host Runbook

1. Deploy a pinned commit to `/opt/ticker-narratives` and create a virtual
   environment outside the repository checkout.
2. Set a persistent `PHASE0_DATABASE_PATH`, for example
   `/var/lib/ticker-narratives/phase0.sqlite3`; do not place the database in a
   temporary build directory.
3. Supply the LLM credential through the host environment or secret store.
   Never commit it to `.env` or the repository.
4. Build the frontend with `npm ci && npm run build`, then serve the built
   assets and FastAPI from the same private host behind nginx. The production
   frontend uses same-origin `/api/v1` requests by default; set
   `VITE_API_BASE_URL` only when the API is intentionally hosted elsewhere.
   nginx basic auth is acceptable at the private boundary.
5. Install the merged, reviewed I4 scheduler configuration. Verify that a
   scheduled run updates SQLite `run_log` and that `/api/v1/meta/status`
   reports the corresponding run metadata.
6. Run a nightly SQLite backup using SQLite’s backup mechanism to a separate
   persistent location. Restore it into a temporary database and verify API
   reads before declaring backup recovery complete.
7. Reboot the VM and verify nginx, FastAPI, the scheduler, and the backup job
   recover without manual intervention.

## Acceptance Evidence

- Private URL shared with the team.
- Frontend fixture and live-data screenshots attached to B4.
- First unattended pipeline run and `/meta/status` response recorded.
- Backup restore command and successful read verification recorded.
- Reboot verification recorded.

Do not start the soak window until every item above is evidenced on the issue.
