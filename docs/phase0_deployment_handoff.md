# Phase 0 Deployment Handoff

**Issue:** B4 / #76
**Status:** Not deployed. This is a readiness runbook, not evidence of a live
deployment.

## Current Blockers

1. I1–I4 are available only on their feature branches and are not merged into
   `main`.
2. The current B1 API deliberately reads committed fixtures. It needs a tested
   `SQLiteNarrativeRepository` before it can read the pipeline database.
3. The current I4 pipeline runs Yahoo and RSS fetch stages. Downstream
   deduplication, clustering, and summarization must be registered and writing
   themes before the ticker page can show live narrative data.
4. A deployment host, private URL, backup destination, and responsible
   operator have not been provided to this repository.

## Preconditions Before Host Work

1. Merge I1–I4 into `main` and complete the B1 live-data readiness gate in
   `docs/phase0_api_contract.md`.
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
5. Install the reviewed I4 cron template only after it lands on `main`. Verify
   that a scheduled run updates SQLite `run_log` and that `/api/v1/meta/status`
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
