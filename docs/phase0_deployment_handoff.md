# Phase 0 Deployment Handoff

Issue B4 remains blocked until Isaac's pipeline runner (I4), the B1 live-data
adapter, and the B2 page are all merged. This checklist keeps the deployment
work small once those prerequisites are ready.

1. Build the frontend with `cd frontend && npm run build`.
2. Run FastAPI behind nginx and serve the built frontend from the same private
   host. Keep the API under `/api/v1`; nginx basic auth is acceptable at the
   private URL boundary and is not product authentication.
3. Set the SQLite database path and Gemini credentials through `.env`; never
   commit the key.
4. Install Isaac's scheduled pipeline job and confirm its `run_log` updates
   appear through `GET /api/v1/meta/status`.
5. Configure a nightly copy of the SQLite database to a separate location.
6. Reboot the VM, confirm the services and cron job restart, then restore the
   backup into a temporary path and verify API reads.
7. Share the private URL only after the UI has fixture and live-API screenshots
   and the first unattended pipeline run succeeds.

This repository cannot complete the VM, nginx, cron, or backup operations on
its own because those require the deployment host and I4's pipeline service.
