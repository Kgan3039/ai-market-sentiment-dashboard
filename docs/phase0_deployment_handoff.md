# Phase 0 Deployment Handoff

**Issue:** B4 / #76
**Status:** Not deployed. This is a readiness runbook, not evidence of a live
deployment.

## Current Blockers

1. The B1 SQLite read source exists but is opt-in
   (`PHASE0_NARRATIVE_SOURCE=sqlite`; the default is still the fixture). Its
   behaviour and API-side requirements -- matching `GEMINI_MODEL` and
   `GEMINI_MAX_OUTPUT_TOKENS` with the scheduler, and `PYTHONPATH` including
   the project root -- are in `docs/phase0_api_contract.md`. B3 adds the
   writer-side WAL keeper and the preflight gate the switch requires; see
   "SQLite serving (B3)" below. Neither has been run on a target host yet.
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
   temporary build directory. Give the API the same value, but switch it to
   `PHASE0_NARRATIVE_SOURCE=sqlite` only through the B3 rollout below.
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

## SQLite serving (B3)

The API reads SQLite read-only (`mode=ro`, `query_only`, write-denying
authorizer). A WAL database cannot be read that way unless its `-wal` and
`-shm` coordination files already exist, and SQLite deletes both when the
last connection closes -- which, with the cron pipeline, is the end of every
run. B3 keeps them in place with a writer-side keeper and puts a hard,
read-only gate in front of the switch.

### What is implemented and tested in the repository

- `phase0/wal_keeper.py` (`python -m phase0.wal_keeper --database <abs path>`):
  holds one `mode=ro`, `query_only` connection with the write-denying
  authorizer. It refuses to start (exit `2`) if the path is unset, relative,
  a symlink, missing, not SQLite, not WAL (checked in the file header before
  any connection is made), or not the schema version the checkout's
  migrations produce (16). It never creates, migrates, re-journals or
  checkpoints the database, proves its connection refuses a write before it
  logs `keeper_ready`, and holds no transaction. Every heartbeat it compares
  the path's `(st_dev, st_ino)` with what it opened and checks that `-wal`
  and `-shm` still exist; on a mismatch it logs `keeper_stopped` and exits
  `3`. SIGTERM/SIGINT close the connection and exit `0`. Logs are one
  redacted JSON line per event on stderr (journald).
- `tools/phase0_api_preflight.py`: the enablement gate (exit `0` ready, `1`
  not ready, `2` bad arguments). It checks, in order and without opening
  SQLite until every filesystem check has passed: not root; an explicit,
  absolute, non-symlink `PHASE0_DATABASE_PATH` to an existing regular file;
  the directory is **not** writable; neither it, the database, `-wal` nor
  `-shm` is owned by the running identity; the database, `-wal` and `-shm`
  are readable, regular and **not** writable; the header says WAL. Then, read-only through
  `Phase0Reader`: schema version, `PRAGMA journal_mode`, a representative
  query, write refusal (`DELETE ... WHERE 0`, which cannot change a row), and
  that no file appeared in the directory. It **reports but does not fail on**
  availability (a completed run for the pipeline version, and per-ticker
  what `/tickers/{t}/themes` would serve, including how many themes have a
  current summary) and freshness (`data_as_of`, `is_stale`, the resolved
  summary policy fingerprint, model and output cap). It never runs
  `pipeline.py --status` or `--database-info` (both migrate), needs no
  `GEMINI_API_KEY`, makes no network request, and reports only whether the
  API environment carries a Gemini key.
- `deploy/phase0-wal-keeper.service`: the systemd unit (placeholders for the
  writer user and reader group; `Restart=always`, `RestartSec=10`, no start
  limit, `UMask=0027`, no network address families, no secrets).
- Cross-process tests (`tests/test_phase0_wal_keeper.py`,
  `tests/test_phase0_api_preflight.py`) run the keeper, short-lived writer
  processes and a long-lived API reader process against temporary databases:
  API before keeper fails closed and creates nothing; with the keeper up a
  read-only API reads, creates nothing, and sees later commits without a
  restart; keeper stop plus a writer run opens a `503` window that closes
  when the keeper returns; a restarted API reads; a SIGKILLed writer and
  keeper lose no committed data and `integrity_check` stays `ok`; a replaced
  or removed database makes the keeper exit `3`; the writer can run a full
  `wal_checkpoint(TRUNCATE)` while the keeper is up, so it pins no snapshot.
  The fixture source remains the default.

These tests run as one OS user and simulate the two identities with file
modes. They prove SQLite's behaviour under those modes on the machine that
ran them (developed on macOS); they do **not** prove Linux ownership and
group semantics between two real users. That is the host smoke checklist.

### Supported deployment contract

- One Linux VM; the database on a local POSIX filesystem (not NFS, SMB, EFS
  or any network filesystem -- WAL relies on shared memory through `-shm`).
- The API, the cron pipeline and the keeper all use the same absolute
  `PHASE0_DATABASE_PATH`, on the same filesystem.
- Two service identities: the **writer** (owns the crontab and runs the
  keeper) and the **API**. They share one **reader group**.
- The existing cron schedule stays the only application-data writer; the
  keeper is systemd-managed and read-only; the API is read-only.
- Enablement is gated by the preflight, run as the API user. There is no
  HTTP readiness endpoint; evidence is the preflight output plus the keeper's
  journal.

### Permission model

Placeholders: `<writer>`, `<api>`, `<readers>`; directory
`/var/lib/ticker-narratives`.

```
sudo groupadd --system <readers>
sudo useradd  --system --no-create-home --shell /usr/sbin/nologin <writer>
sudo useradd  --system --no-create-home --shell /usr/sbin/nologin <api>
sudo usermod  -aG <readers> <api>
# setgid (2xxx): every file created here gets group <readers>
sudo install -d -o <writer> -g <readers> -m 2750 /var/lib/ticker-narratives
```

| Path | Owner:group | Mode | Writer | API | World |
|---|---|---|---|---|---|
| `/var/lib/ticker-narratives/` | `<writer>:<readers>` | `2750` | create/remove/rename | traverse + list only | none |
| `phase0.sqlite3` | `<writer>:<readers>` | `0640` | read/write | read | none |
| `phase0.sqlite3-wal`, `-shm` | `<writer>:<readers>` | `0640` (copied from the database) | read/write | read | none |

- After the first pipeline run creates the database, run
  `sudo chmod 0640 /var/lib/ticker-narratives/phase0.sqlite3` once if the
  writer's umask left it `0644` (the `2750` directory already keeps the world
  out). SQLite gives `-wal` and `-shm` the database file's permission bits;
  the setgid directory gives them group `<readers>`.
- The API user must never own the directory or the files, and must never
  have write permission on them. Never `chmod 777`.
- The API's `mode=ro` connection opens `-shm` read-only; SQLite supports that
  while the files exist. This is exactly what the smoke checklist verifies
  with real users.

### Rollout (host steps; not yet executed)

1. Keep the API on the fixture (`PHASE0_NARRATIVE_SOURCE` unset or `fixture`).
2. Create `<writer>`, `<api>` and `<readers>` as above.
3. Create the database directory with the permissions above.
4. Install the existing cron schedule (`deploy/phase0-pipeline.cron`) in
   `<writer>`'s crontab with the same `PHASE0_DATABASE_PATH`.
5. Run the pipeline once as `<writer>` to create, migrate and populate the
   database; then `chmod 0640` it if needed.
6. Fill in the placeholders in `deploy/phase0-wal-keeper.service`, install
   it, and `systemctl enable --now phase0-wal-keeper`.
7. Confirm `journalctl -u phase0-wal-keeper` shows `keeper_ready` with
   `"schema_version": 16`, `"journal_mode": "wal"`, and that `-wal` and `-shm`
   exist with owner `<writer>`, group `<readers>`, mode `0640`.
8. Run the preflight **as the API user**, with the API's environment:
   `sudo -u <api> env PHASE0_DATABASE_PATH=... PHASE0_PIPELINE_VERSION=...
   GEMINI_MODEL=... GEMINI_MAX_OUTPUT_TOKENS=...
   PYTHONPATH=/opt/ticker-narratives:/opt/ticker-narratives/backend
   /opt/ticker-narratives/.venv/bin/python /opt/ticker-narratives/tools/phase0_api_preflight.py`
9. Require exit `0` and `"failed_checks": []`.
10. Read `availability` and `freshness` separately: a completed run exists,
    tickers show the expected themes and current summaries, `data_as_of` is
    recent, and the policy fingerprint is the one the scheduler uses.
11. Set for the API: `PHASE0_NARRATIVE_SOURCE=sqlite`,
    `PHASE0_DATABASE_PATH=<the same explicit path>`,
    `PHASE0_PIPELINE_VERSION=<the scheduler's value>`, plus matching
    `GEMINI_MODEL` / `GEMINI_MAX_OUTPUT_TOKENS`.
12. Do **not** give the API `GEMINI_API_KEY`.
13. Restart the API.
14. Smoke `/api/v1/meta/status`, `/api/v1/tickers` and
    `/api/v1/tickers/{ticker}/themes`.
15. Confirm the API's reads created no new entry in the database directory
    (`ls -la` before and after).
16. Let cron run again.
17. Confirm `/api/v1/meta/status` `data_as_of` advanced without an API
    restart.
18. Verify the frontend through nginx.

### Rollback

1. Set `PHASE0_NARRATIVE_SOURCE=fixture` (or remove it) for the API.
2. Restart the API; the narrative source is chosen once per process.
3. The keeper and cron may keep running.
4. Do not delete `-wal` or `-shm` by hand.
5. Do not modify or delete the database as part of rollback.
6. Stop the keeper only when retiring SQLite serving altogether; the next
   pipeline run then removes the coordination files itself.

Rollback touches only the API's configuration, so it cannot lose persisted
data.

### Restoring or replacing the database

The keeper holds an open file; a restore that replaces the path would leave
it holding the old inode. It detects this and exits `3` so systemd restarts
it on the new file, but the safe sequence is explicit:

1. Roll the API back to the fixture (above), or accept `503`s meanwhile.
2. `systemctl stop phase0-wal-keeper`, and make sure no pipeline run is in
   progress (it holds `/var/lock/phase0-pipeline.lock`).
3. Restore the database with SQLite's backup mechanism, owner `<writer>`,
   group `<readers>`, mode `0640`.
4. `systemctl start phase0-wal-keeper` and confirm `keeper_ready`.
5. Run the preflight as the API user.
6. Only then serve SQLite again.

### WAL growth

The keeper holds no transaction and never checkpoints. The pipeline's
connections keep SQLite's automatic checkpoints (every 1000 pages), and a
test shows a full `TRUNCATE` checkpoint succeeds while the keeper is up, so
the WAL is reused rather than growing without bound. While the keeper runs,
the `-wal` file is not deleted between runs and is not shrunk below its
high-water mark. Watch its size on the host; a `journal_size_limit` or a
periodic writer-side `TRUNCATE` checkpoint is a follow-up only if it grows
materially.

### Host smoke checklist (must pass before claiming production SQLite serving)

Record the output of each step on the B4 issue.

- [ ] `python -V` and `python -c "import sqlite3; print(sqlite3.sqlite_version)"`
      from the deployed virtualenv.
- [ ] `findmnt -T /var/lib/ticker-narratives` (or `stat -f`) shows a local
      filesystem (ext4/xfs/btrfs), not NFS/SMB/EFS/CIFS/FUSE network storage.
- [ ] `id <writer>`, `id <api>`: separate users; `<api>` is in `<readers>`.
- [ ] `stat` of the directory: `<writer>:<readers>`, `2750`.
- [ ] `stat` of the database, `-wal`, `-shm`: `<writer>:<readers>`, `0640`.
- [ ] `sudo -u <api> touch /var/lib/ticker-narratives/probe` fails with
      permission denied.
- [ ] `sudo -u <api> head -c 16 <file> | od -c` succeeds for each of the
      database, `-wal` and `-shm`.
- [ ] The keeper starts (`keeper_ready`) and restarts after
      `systemctl kill -s KILL phase0-wal-keeper`.
- [ ] With the keeper stopped and a pipeline run since, API narrative
      routes return `503` and the preflight exits `1`.
- [ ] After `systemctl start phase0-wal-keeper`, the same running API
      returns `200` again.
- [ ] A cron run commits while the API stays up, and `data_as_of`
      advances without an API restart.
- [ ] API reads create no directory entries (`ls -la` before/after).
- [ ] The API's environment has no `GEMINI_API_KEY`
      (`sudo cat /proc/<api pid>/environ | tr '\0' '\n' | grep -c GEMINI_API_KEY`
      is `0`).
- [ ] Reboot: the keeper, the API, cron and nginx come back; the preflight
      passes; the API serves `200`.
- [ ] Fixture rollback: switch back, restart the API, the page serves the
      fixture; the database and its files are unchanged.

### Not yet verified

- The actual target host, its OS, Python and SQLite versions.
- The actual Linux users, group, ownership and modes.
- The actual filesystem.
- Keeper and API behaviour across a real reboot.
- The production switch to `PHASE0_NARRATIVE_SOURCE=sqlite` itself.
