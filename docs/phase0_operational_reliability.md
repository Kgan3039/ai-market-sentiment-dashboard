# Phase 0 operational reliability evidence (#117)

This procedure reads the existing SQLite ledger. It makes no provider calls,
does not migrate the database, and writes no application data. Use an explicit
window and pipeline version so two operators can reproduce the same report.
The report does **not** declare G5, G6, or G7 passed from partial evidence.

## Run the local report

Run against a migrated fixture, the deployed database with read access, or a
consistent SQLite backup. A plain copy of a WAL-mode `.sqlite3` file can miss
recent committed rows unless its WAL is included.

```sh
python tools/phase0_operational_report.py \
  --database /path/to/phase0.sqlite3 \
  --start 2026-11-02T00:00:00Z \
  --end 2026-11-05T00:00:00Z \
  --pipeline-version phase0-v1 > operational-report.json
```

Both bounds require an explicit UTC offset; start is inclusive and end is
exclusive. A run is included by its start time, even if it completes after
the end bound. Keep the command, database snapshot identity, report JSON, cron
configuration, and relevant structured pipeline logs together as soak
evidence. The report uses the checked-in Eastern cron cadence, including
daylight saving transitions. Confirm the host actually honors
`CRON_TZ=America/New_York` before treating that cadence as its schedule
(`deploy/phase0-pipeline.cron` explains the check).

Each scheduled slot reports whether any live partition run was durable, its
Yahoo outcome for each of the five tickers, each RSS checkpoint known to the
database, and failed or degraded stages. `empty` is a completed but quiet
source check; it can be a clean candidate when every other known source and
stage is healthy. `failed` is a provider or processing failure. `partial`
and `degraded` remain separate. Replay rows are excluded
from scheduled-slot accounting. A Yahoo item-partition row without its
source checkpoint remains `missing`; it cannot prove the provider check
completed. The source registry comes from persisted
source state and run rows; a feed that has never left either is unknowable
from SQLite alone.
Compare that list with the approved feed configuration before scoring.
The `source_checkpoints_from_window_runs` field shows each source's latest
recorded and latest successful checkpoint from those included runs.

`raw_evidence` reports source and feed provenance, missing publication and
ingest-run timestamps, invalid evidence, and publish-to-**fetch** latency.
That latency is a diagnostic proxy. It is not publish-to-visible G6.

## What the gates still need

| Gate | Available now | Needed before a decision |
| --- | --- | --- |
| G5: ≥95% scheduled runs succeed; no >2h market-hours gap | Scheduled opportunities, durable run coverage, source and stage outcomes, gaps without durable rows | Correlate every slot with structured `invocation_started`, `component_completed`, `invocation_completed`, and `invocation_skipped` logs. A process may fail before its first partition row; a partition row also does not prove that the whole invocation finished. Retain explicit numerator, denominator, missed slots, and unknowns. |
| G6: p95 publish→visible ≤45 minutes during market hours | Publish→fetch timestamps and missing-date counts | A historical timestamp for when each eligible item became visible in the served theme/summary. Current rows and the latest API response cannot reconstruct past visibility after later reconciliation. |
| G7: ≤15% themes shipped without summary | Current theme and summary state can be inspected elsewhere | A historical record of themes actually served, with and without a current accepted summary, over the preregistered soak window. A present-day snapshot is not the shipped denominator. |

Do not turn `clean_candidates` into a G5 pass rate. It is a diagnostic count
of slots where all *known* sources report success or a quiet `empty` check and no stage row is
degraded or failed. Missing invocations, component failures that
left no row, retired or never-recorded feeds, and process completion require
the log correlation above. If logs or visibility history are missing, record
the gate as **unmeasured**, with the missing evidence named. Gate objects
carry `numerator`, `denominator`, and `unknowns` explicitly. G5 knows the
scheduled denominator but marks every result unknown without invocation
correlation. G6 and G7 leave counts null because the historical served
population itself is not recorded; null does not mean zero.

## Controlled local recovery evidence

The offline test suite exercises these conditions with fake providers and a
temporary SQLite database:

| Condition | Expected | Observed in offline tests |
| --- | --- | --- |
| Yahoo request times out | The ticker records a failed checkpoint; a later healthy request restores it. | Timeout and recovery assertions pass in `tests/test_phase0_yahoo.py::test_a_request_timeout_is_reported_and_checkpointed` and `test_the_next_fetch_recovers_once_the_hung_request_finally_returns`. |
| Yahoo returns a malformed item | The bad item stays as invalid evidence and later items continue. | Both assertions pass in `tests/test_phase0_yahoo.py::test_a_malformed_item_does_not_abort_later_items_and_is_durable`. |
| One RSS feed returns malformed XML | The other feed succeeds; a later healthy fetch restores the failed feed without deleting earlier runs or snapshots. | Both feeds, restored checkpoint, and retained history are asserted in `tests/test_phase0_ingestion.py::test_malformed_feed_does_not_prevent_later_feed_success`. |
| One ticker and one feed fail | The invocation is degraded while healthy sources retain their evidence; a later ordinary run restores failed checkpoints. | Degradation and recovery assertions pass in `tests/test_phase0_pipeline.py::test_partial_evidence_is_degraded_and_never_a_clean_success` and `test_source_outage_recovers_on_next_unattended_run`. |
| Process stops between Yahoo and RSS | Yahoo's committed rows survive; a new process completes; offline replay preserves raw evidence. | All three assertions pass in `tests/test_phase0_pipeline.py::test_restart_after_interruption_preserves_evidence_and_recovers`. |
| Replay runs | It makes no provider calls and does not count as a scheduled live opportunity. | `tests/test_phase0_pipeline.py::test_replay_touches_no_network_at_all` and `tests/test_phase0_operational_report.py` assert both properties. |
| Quiet source and incomplete checkpoint | A zero-headline check remains distinct from failure; an ingest-only row cannot claim a completed Yahoo check. | `tests/test_phase0_operational_report.py` asserts both classifications. |

The process-stop test interrupts the orchestrator at a controlled boundary.
It does not test power loss inside a SQLite transaction or a host reboot.
Host recovery and unattended operation remain deployment evidence under #76.

## Operator response

1. Check the scheduler's most recent expected slot and verify its timezone.
   Read the structured pipeline log for the matching invocation and its
   exit code: 0 clean or overlap-skipped, 1 degraded, 2 failed. An overlap
   skip is not a successful data refresh.
2. Run this report for the affected window. Identify missing durable slots,
   the ticker or RSS source that failed, its last successful checkpoint,
   and any missing publication or provenance timestamps. A zero-headline
   check is `empty`, not a provider outage.
3. Check whether another process holds the configured production lock,
   whether the database is readable, and whether a provider failure is
   source-local. Preserve the original log and database snapshot before
   intervention. Do not substitute a manually triggered run for a missed
   scheduled opportunity in the denominator.
4. Let the next scheduled invocation retry normally when it is safe.
   If an operator changes configuration, restarts a service, runs the
   pipeline manually, repairs data, or otherwise intervenes, record the
   exact time, actor, action, reason, and observed result in the soak log.
   Follow the preregistered window restart rule from #77; do not silently
   continue counting the interrupted window.
5. If the same source fails again, preserve the failure and recovery
   sequence. Escalate any unresolved provider-use question to #44 and any
   deployment fault to #76. Report G5/G6/G7 as missed or unmeasured with
   their numerator, denominator, and unknowns; do not move thresholds.

The target-host three-day unattended run and final gate measurements begin
only after #76 supplies a host and #44 establishes the provider-use boundary.
