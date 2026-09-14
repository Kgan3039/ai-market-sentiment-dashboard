"""``make_review_sheets.py``: sample and score Phase 0 G1 review sheets (A4a, #74).

    # draw 40 placements from what the themes stage persisted for a day
    python -m tools.make_review_sheets sample-assignments \\
        --database "$PHASE0_DATABASE_PATH" \\
        --day 2026-09-10 --seed phase0-g1-r1 --round-id r1 \\
        --out reviews/g1/r1.csv

    # a second, non-overlapping round toward section 8's >= 80
    python -m tools.make_review_sheets sample-assignments \\
        --database ... --day 2026-09-10 --seed phase0-g1-r2 --round-id r2 \\
        --exclude-manifest reviews/g1/r1.manifest.json --out reviews/g1/r2.csv

    # score the gate from the artifacts themselves: each --round names a
    # manifest and its one or two completed sheets; --adjudication pairs a
    # manifest with its adjudication sheet
    python -m tools.make_review_sheets score-assignments \\
        --round reviews/g1/r1.manifest.json reviews/g1/r1.alice.csv \\
                reviews/g1/r1.bob.csv \\
        --adjudication reviews/g1/r1.manifest.json reviews/g1/r1.adjudicated.csv \\
        --round reviews/g1/r2.manifest.json reviews/g1/r2.alice.csv \\
                reviews/g1/r2.bob.csv \\
        --report reviews/g1/scorecard.json

Sampling reads the persisted theme output of a Phase 0 database, as one
snapshot per partition, and never re-runs clustering.  ``--fixture`` is the
one development substitute: it clusters the committed M5 fixture offline and
the sample is classified synthetic, so nothing drawn from it can reach PASS.

Every sample writes a sidecar ``<name>.manifest.json`` beside the CSV.  The
manifest's captured snapshot is what a review is *of*: scoring holds the
completed sheets to it and never consults the database again.  Scoring
takes only manifests and sheets; a saved scorecard or round report is
output, never input.

Exit status: 0 PASS, 1 FAIL, 2 usage or input error, 3 INCOMPLETE or
NOT_ELIGIBLE.  The last two share a code because both mean "no gate
verdict is available"; the scorecard text says which, and a script that
needs to branch on eligibility is a script making a decision K4 owns.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from nlp.eval.review import (
    DEFAULT_ROUND_SIZE,
    UNRATIFIED_PROTOCOL,
    DevelopmentOverrides,
    GateResult,
    OperatorAttestation,
    ReviewSamplingError,
    build_manifest,
    load_fixture_population,
    load_persisted_population,
    read_manifest,
    render_scorecard,
    sample_assignments,
    score_gate,
    score_round,
    write_sample,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_NO_VERDICT = 3

EXIT_BY_RESULT = {
    GateResult.PASS: EXIT_PASS,
    GateResult.FAIL: EXIT_FAIL,
    GateResult.INCOMPLETE: EXIT_NO_VERDICT,
    GateResult.NOT_ELIGIBLE: EXIT_NO_VERDICT,
}


def _cmd_sample(args: argparse.Namespace) -> int:
    if args.fixture:
        if args.database or args.day or args.ticker or args.pipeline_version:
            raise ReviewSamplingError(
                "--fixture is a development mode and takes no database selection"
            )
        population = load_fixture_population(args.fixture_path, args.vectors)
    else:
        if not args.database or not args.day:
            raise ReviewSamplingError(
                "sampling needs --database and at least one --day (or --fixture)"
            )
        population = load_persisted_population(
            args.database,
            trading_days=args.day,
            tickers=args.ticker or None,
            pipeline_version=args.pipeline_version,
        )
    attestation = None
    if args.attestation or args.attested_by:
        if not (args.attestation and args.attested_by):
            raise ReviewSamplingError("--attestation and --attested-by go together")
        attestation = OperatorAttestation(
            attested_by=args.attested_by,
            statement=args.attestation,
            attested_at=datetime.now(timezone.utc).isoformat(),
        )
    priors = [read_manifest(path) for path in args.exclude_manifest or ()]
    sample = sample_assignments(
        population,
        seed=args.seed,
        size=args.size,
        round_id=args.round_id,
        prior_manifests=priors,
    )
    out = Path(args.out)
    manifest = build_manifest(
        sample,
        protocol_id=args.protocol,
        csv_name=out.name,
        operator_attestation=attestation,
    )
    csv_path, manifest_path = write_sample(sample, out, manifest=manifest)
    print(
        f"wrote {sample.actual_size}/{sample.requested_size} placements "
        f"(population {len(population.rows)}, {len(sample.excluded_row_ids)} "
        f"excluded from prior rounds) to {csv_path}"
    )
    print(f"manifest: {manifest_path}")
    print(
        f"origin: {manifest['origin']['status']}; protocol: {args.protocol}; "
        f"snapshot {manifest['snapshot']['sha256'][:12]}"
    )
    if attestation is not None:
        print(
            "NOTE: the operator attestation is recorded as audit metadata and does "
            "not change origin or eligibility",
            file=sys.stderr,
        )
    if sample.shortfall:
        print(
            f"NOTE: {sample.shortfall} fewer than requested; the population was "
            "exhausted and nothing was padded",
            file=sys.stderr,
        )
    for skipped in population.skipped:
        print(
            f"skipped {skipped.ticker} {skipped.trading_day}: {skipped.reason} "
            f"({skipped.detail})",
            file=sys.stderr,
        )
    return EXIT_PASS


def _cmd_score(args: argparse.Namespace) -> int:
    adjudications: dict[Path, Path] = {}
    for manifest_path, sheet_path in args.adjudication or ():
        key = Path(manifest_path).resolve()
        if key in adjudications:
            raise ReviewSamplingError(f"{manifest_path}: adjudication given twice")
        adjudications[key] = Path(sheet_path)
    rounds = []
    seen: set[Path] = set()
    for group in args.round:
        if not 2 <= len(group) <= 3:
            raise ReviewSamplingError(
                "--round takes a manifest and one or two completed sheets"
            )
        manifest_path = Path(group[0])
        key = manifest_path.resolve()
        if key in seen:
            raise ReviewSamplingError(f"{manifest_path}: round given twice")
        seen.add(key)
        manifest = read_manifest(manifest_path)
        rounds.append(
            score_round(manifest, group[1:], adjudicated=adjudications.pop(key, None))
        )
    if adjudications:
        raise ReviewSamplingError(
            "an --adjudication names a manifest that is not among the --round groups"
        )
    development = DevelopmentOverrides(
        threshold=args.development_threshold,
        required_unique_assignments=args.development_required_unique,
    )
    scorecard = score_gate(rounds, development=development)
    payload = scorecard.as_dict()
    payload["round_reports"] = [r.as_dict() for r in rounds]
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_scorecard(scorecard))
    return EXIT_BY_RESULT[scorecard.gate_result]


def _finite_unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc
    try:
        DevelopmentOverrides(threshold=value)
    except ReviewSamplingError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    try:
        DevelopmentOverrides(required_unique_assignments=value)
    except ReviewSamplingError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_review_sheets",
        description="Sample and score Phase 0 G1 review sheets (issue #74 / A4a).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser(
        "sample-assignments",
        help="draw story->theme placements from persisted theme output (gate G1)",
    )
    sample.add_argument("--database", type=Path, help="Phase 0 SQLite database")
    sample.add_argument("--day", action="append", help="trading day, repeatable")
    sample.add_argument(
        "--ticker", action="append", help="restrict to a ticker, repeatable"
    )
    sample.add_argument(
        "--pipeline-version", help="required when the day spans several"
    )
    sample.add_argument(
        "--fixture",
        action="store_true",
        help="development mode: the committed M5 fixture",
    )
    sample.add_argument("--fixture-path", type=Path)
    sample.add_argument("--vectors", type=Path)
    sample.add_argument(
        "--seed", required=True, help="the draw is a pure function of this"
    )
    sample.add_argument("--size", type=_positive_int, default=DEFAULT_ROUND_SIZE)
    sample.add_argument("--round-id", default="round-1")
    sample.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        help="a prior round's manifest; its rows are not drawn again",
    )
    sample.add_argument("--protocol", default=UNRATIFIED_PROTOCOL)
    sample.add_argument(
        "--attested-by", help="who is recording an attestation (audit metadata only)"
    )
    sample.add_argument(
        "--attestation",
        help="what the operator checked, in their words (audit metadata only)",
    )
    sample.add_argument("--out", type=Path, required=True)

    score = sub.add_parser(
        "score-assignments", help="compute gate G1 from manifests and completed sheets"
    )
    score.add_argument(
        "--round",
        action="append",
        nargs="+",
        required=True,
        metavar="PATH",
        help="a manifest followed by its one or two completed sheets; repeatable",
    )
    score.add_argument(
        "--adjudication",
        action="append",
        nargs=2,
        metavar=("MANIFEST", "SHEET"),
        help="a manifest and its adjudication sheet; repeatable",
    )
    score.add_argument(
        "--development-threshold",
        type=_finite_unit_float,
        help="development only: forces a NOT_ELIGIBLE evaluation",
    )
    score.add_argument(
        "--development-required-unique",
        type=_positive_int,
        help="development only: forces a NOT_ELIGIBLE evaluation",
    )
    score.add_argument("--report", type=Path, help="write the scorecard JSON here")
    score.add_argument("--json", action="store_true")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_PASS
    try:
        if args.command == "sample-assignments":
            return _cmd_sample(args)
        return _cmd_score(args)
    except ReviewSamplingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # a CLI reports a domain error; it does not trace
        if exc.__class__.__module__.startswith(("nlp.", "phase0.")):
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        raise


if __name__ == "__main__":
    raise SystemExit(main())
