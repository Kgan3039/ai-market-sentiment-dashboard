"""``make_review_sheets.py``: sample and score Phase 0 review sheets (A4, #74).

G1 theme-assignment sheets (A4a) and G2 sentence-faithfulness sheets (A4b).

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

G2 (A4b) reviews every sentence of every summary a reader may be shown, on
two days drawn from the operator's candidate days::

    # candidate days in, two eligible days drawn by the seed, census out
    python -m tools.make_review_sheets sample-sentences \\
        --database "$PHASE0_DATABASE_PATH" \\
        --window-start 2026-09-08 --window-end 2026-09-12 \\
        --seed phase0-g2 --out reviews/g2/g2.csv

    # one census round: the manifest and its one or two completed sheets
    python -m tools.make_review_sheets score-sentences \\
        --round reviews/g2/g2.manifest.json reviews/g2/g2.alice.csv \\
                reviews/g2/g2.bob.csv \\
        --adjudication reviews/g2/g2.adjudicated.csv \\
        --report reviews/g2/scorecard.json

G2 sampling resolves the production summary policy exactly as the API does,
reads SQLite read-only, and never generates, so it needs no GEMINI_API_KEY.

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

from nlp.eval import faithfulness
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


def _attestation(args: argparse.Namespace) -> OperatorAttestation | None:
    if not (args.attestation or args.attested_by):
        return None
    if not (args.attestation and args.attested_by):
        raise ReviewSamplingError("--attestation and --attested-by go together")
    return OperatorAttestation(
        attested_by=args.attested_by,
        statement=args.attestation,
        attested_at=datetime.now(timezone.utc).isoformat(),
    )


def _cmd_sample_sentences(args: argparse.Namespace) -> int:
    if bool(args.window_start) != bool(args.window_end):
        raise ReviewSamplingError("--window-start and --window-end go together")
    window = (args.window_start, args.window_end) if args.window_start else None
    # Everything the manifest would record verbatim, and every path it would
    # write, is checked before the database is read or any file is opened.
    for value, field in (
        (args.seed, "seed"),
        (args.round_id, "round_id"),
        (args.protocol, "protocol"),
        (Path(args.out).name, "output file name"),
    ):
        faithfulness.require_clean_operator_value(value, field)
    faithfulness.require_known_g2_protocol(args.protocol)
    out = Path(args.out)
    faithfulness.check_output_paths(
        [out, faithfulness.manifest_path_for(out)],
        inputs=faithfulness.protected_database_paths(args.database),
    )
    population = faithfulness.load_sentence_population(
        args.database,
        candidate_days=args.candidate_day or (),
        window=window,
        pipeline_version=args.pipeline_version,
    )
    sample = faithfulness.sample_sentences(
        population,
        seed=args.seed,
        draw_size=args.development_days,
        round_id=args.round_id,
    )
    manifest = faithfulness.build_manifest(
        sample,
        protocol_id=args.protocol,
        csv_name=out.name,
        operator_attestation=_attestation(args),
    )
    csv_path, manifest_path = faithfulness.write_sample(sample, out, manifest=manifest)
    selection = manifest["selection"]
    print(
        f"wrote {len(sample.rows)} sentences from {len(sample.artifacts)} summaries "
        f"on {', '.join(sample.selected_days)} to {csv_path}"
    )
    print(f"manifest: {manifest_path}")
    print(
        f"eligible days {len(selection['eligible_days'])} of "
        f"{len(selection['candidate_days'])} candidates; origin: "
        f"{manifest['origin']['status']}; protocol: {args.protocol}; "
        f"snapshot {manifest['snapshot']['sha256'][:12]}"
    )
    for excluded in selection["excluded_days"]:
        print(
            f"excluded {excluded['trading_day']}: {excluded['reason']}",
            file=sys.stderr,
        )
    counts = manifest["population"]["selected_days"]
    for key in ("withheld_artifact_count", "population_changed_partition_count"):
        if counts[key]:
            print(
                f"NOTE: {key} = {counts[key]}; the census is incomplete",
                file=sys.stderr,
            )
    if sample.development_override:
        print(
            "NOTE: a development draw size was used; the round cannot be gate eligible",
            file=sys.stderr,
        )
    return EXIT_PASS


def _cmd_score_sentences(args: argparse.Namespace) -> int:
    if not 2 <= len(args.round) <= 3:
        raise ReviewSamplingError(
            "--round takes a manifest and one or two completed sheets"
        )
    if args.report:
        inputs = [Path(p) for p in args.round]
        if args.adjudication:
            inputs.append(Path(args.adjudication))
        faithfulness.check_output_paths([args.report], inputs=inputs)
    manifest = faithfulness.read_manifest(args.round[0])
    result = faithfulness.score_sentence_round(
        manifest, args.round[1:], adjudicated=args.adjudication
    )
    scorecard = faithfulness.score_g2(
        result,
        development=faithfulness.G2DevelopmentOverrides(
            threshold=args.development_threshold
        ),
    )
    payload = scorecard.as_dict()
    payload["round_report"] = result.as_dict()
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        faithfulness.create_new_file(
            report, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(faithfulness.render_scorecard(scorecard))
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
        description="Sample and score Phase 0 G1 and G2 review sheets (issue #74).",
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

    sentences = sub.add_parser(
        "sample-sentences",
        help="draw two eligible days and take every current summary sentence (gate G2)",
    )
    sentences.add_argument(
        "--database", type=Path, required=True, help="Phase 0 SQLite database"
    )
    sentences.add_argument(
        "--candidate-day", action="append", help="a candidate day, repeatable"
    )
    sentences.add_argument("--window-start", help="first candidate day, inclusive")
    sentences.add_argument("--window-end", help="last candidate day, inclusive")
    sentences.add_argument(
        "--pipeline-version", help="required when the candidates span several"
    )
    sentences.add_argument(
        "--seed", required=True, help="the day draw is a pure function of this"
    )
    sentences.add_argument("--round-id", default="g2")
    sentences.add_argument(
        "--development-days",
        type=_positive_int,
        help="development only: draw this many days; forces a NOT_ELIGIBLE evaluation",
    )
    sentences.add_argument("--protocol", default=UNRATIFIED_PROTOCOL)
    sentences.add_argument(
        "--attested-by", help="who is recording an attestation (audit metadata only)"
    )
    sentences.add_argument(
        "--attestation",
        help="what the operator checked, in their words (audit metadata only)",
    )
    sentences.add_argument("--out", type=Path, required=True)

    score_sentences = sub.add_parser(
        "score-sentences",
        help="compute gate G2 from one census manifest and its completed sheets",
    )
    score_sentences.add_argument(
        "--round",
        nargs="+",
        required=True,
        metavar="PATH",
        help="the manifest followed by its one or two completed sheets",
    )
    score_sentences.add_argument(
        "--adjudication", type=Path, help="the round's adjudication sheet"
    )
    score_sentences.add_argument(
        "--development-threshold",
        type=_finite_unit_float,
        help="development only: forces a NOT_ELIGIBLE evaluation",
    )
    score_sentences.add_argument(
        "--report", type=Path, help="write the scorecard JSON here"
    )
    score_sentences.add_argument("--json", action="store_true")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_PASS
    try:
        if args.command == "sample-assignments":
            return _cmd_sample(args)
        if args.command == "sample-sentences":
            return _cmd_sample_sentences(args)
        if args.command == "score-sentences":
            return _cmd_score_sentences(args)
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
