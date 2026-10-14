#!/usr/bin/env python3
"""Black and Flake8 over the maintained Python scope (CI-1).

This file is the one definition of what CI lints.

**Scope.** Every git-tracked ``*.py`` file, minus a fixed, documented
baseline of files that did not pass when CI was introduced (main at
``be79267``).  Scope is subtractive on purpose: a new file anywhere in the
repository is linted the moment it is tracked, and nothing is excluded by
directory or pattern.

**The baseline only shrinks.**

* An entry must name a tracked file, and must still fail its tool.  When a
  baseline file starts passing, the run fails until the entry is removed,
  so the cleanup is recorded rather than silently re-allowing regressions.
* An entry must belong to ``*_BASELINE_AT_INTRODUCTION``.  Adding a path
  needs an edit to those frozen sets as well, which is visible in review
  and refused by ``tests/test_ci_python_lint.py`` unless that test's own
  pinned copy is changed too.
* With ``--changed-since REF`` (pull requests), a baseline file the change
  touches loses its exemption for that run: touching a legacy file means
  bringing it up to standard (``black <file>``, fix Flake8) and deleting its
  entry here.  Untouched legacy files are never reformatted by CI.

Exit status: ``0`` clean, ``1`` lint or baseline failure, ``2`` usage or
environment error (no git, tool missing, a ref that is not a commit).

Run locally from the repository root::

    python tools/ci_python_lint.py
    python tools/ci_python_lint.py --changed-since origin/main
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: Pre-Phase-0 sentiment/prediction application code and its ad-hoc tests.
#: Not maintained for Phase 0; spec section 2 retires prediction and numeric
#: sentiment.  ``backend/main.py`` is here because it predates Phase 0 and
#: still mounts the legacy routers.
_LEGACY_BLACK = frozenset(
    {
        "backend/app/models/schemas.py",
        "backend/app/routes/dashboard.py",
        "backend/app/routes/market.py",
        "backend/app/routes/prediction.py",
        "backend/app/routes/sentiment.py",
        "backend/app/services/data_service.py",
        "backend/app/services/prediction_service.py",
        "backend/app/services/sentiment_service.py",
        "backend/main.py",
        "backend/test_api.py",
        "backend/test_data_service_cache.py",
        "backend/test_prediction_integration.py",
        "backend/test_prediction_persistence.py",
        "data/app.py",
        "nlp/sentiment.py",
        "prediction/prediction.py",
    }
)
_LEGACY_FLAKE8 = (_LEGACY_BLACK - {"backend/app/routes/market.py"}) | {
    "backend/test_prediction_startup_prewarm.py"
}

#: Maintained Phase 0 files that already failed when CI was introduced.
#: Known debt, to be cleaned in a dedicated formatting change; touching one
#: in a pull request requires cleaning it (see ``--changed-since``).
_PHASE0_DEBT = frozenset(
    {
        "tests/test_ai_summarization.py",
        "tests/test_phase0_config.py",
        "tests/test_phase0_copy_rules.py",
        "tools/validate_phase0_config.py",
        "tools/validate_phase0_copy_rules.py",
    }
)

BLACK_BASELINE_AT_INTRODUCTION = _LEGACY_BLACK | _PHASE0_DEBT
FLAKE8_BASELINE_AT_INTRODUCTION = _LEGACY_FLAKE8 | _PHASE0_DEBT

#: The live exemptions.  Remove entries as files are cleaned; never add.
BLACK_BASELINE: frozenset[str] = BLACK_BASELINE_AT_INTRODUCTION - {
    "backend/app/services/data_service.py",
}
FLAKE8_BASELINE: frozenset[str] = FLAKE8_BASELINE_AT_INTRODUCTION - {
    "backend/app/services/data_service.py",
}


class LintEnvironmentError(RuntimeError):
    """Git or a lint tool could not be run; no verdict is possible."""


#: ``(paths) -> paths that fail``.  Injected so the policy is testable
#: without running the real tools.
FailureProbe = Callable[[Sequence[str]], frozenset[str]]


@dataclass
class ToolReport:
    tool: str
    checked: int
    failures: list[str] = field(default_factory=list)  # in-scope files failing
    unexpected_passes: list[str] = field(default_factory=list)  # stale baseline
    untracked_baseline: list[str] = field(default_factory=list)
    unregistered_baseline: list[str] = field(default_factory=list)
    touched_baseline: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.failures
            or self.unexpected_passes
            or self.untracked_baseline
            or self.unregistered_baseline
        )


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args), cwd=ROOT, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise LintEnvironmentError(f"cannot run {args[0]!r}: {exc}") from exc


def _git_paths(*args: str) -> set[str]:
    result = _run(["git", "ls-files", "-z", *args, "--", "*.py"])
    if result.returncode != 0:
        raise LintEnvironmentError(f"git ls-files failed: {result.stderr.strip()}")
    return {path for path in result.stdout.split("\0") if path}


def tracked_python_files() -> list[str]:
    """Python files present in the working tree: tracked files plus untracked,
    non-ignored ones (a new file is in scope locally before it is staged, as
    it will be in CI), minus tracked files deleted but not yet committed.

    A deleted path never reaches Black or Flake8.  A deleted *baseline*
    path is reported as an entry that names no file, so deleting or
    renaming an exempt file cannot carry its exemption anywhere.
    """
    present = _git_paths("--cached", "--others", "--exclude-standard")
    return sorted(present - _git_paths("--deleted"))


def resolve_commit(ref: str) -> str:
    """The commit SHA *ref* names, or :class:`LintEnvironmentError`.

    Refs are user input.  One that looks like an option (``--cached``) is
    refused outright, and resolution uses ``--end-of-options`` so Git can
    never read it as one; ``^{commit}`` refuses trees, blobs and unknown
    names.  Everything after this compares against the SHA, never the text.
    """
    if not ref or ref.startswith("-"):
        raise LintEnvironmentError(f"not a commit reference: {ref!r}")
    result = _run(
        ["git", "rev-parse", "--verify", "--quiet", "--end-of-options"]
        + [f"{ref}^{{commit}}"]
    )
    sha = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", sha):
        raise LintEnvironmentError(f"not a commit reference: {ref!r}")
    return sha


def changed_python_files(ref: str) -> frozenset[str]:
    """Python files added, copied, modified or renamed between the merge base
    of *ref* and the working tree (so uncommitted edits count locally; in CI
    the working tree is the commit under test).  A rename reports its new
    path, which no baseline names."""
    base = resolve_commit(ref)
    result = _run(
        ["git", "diff", "--name-only", "-z", "--diff-filter=ACMR", "--merge-base"]
        + [base, "--"]
    )
    if result.returncode != 0:
        raise LintEnvironmentError(
            f"git diff against {ref!r} ({base}) failed: {result.stderr.strip()}"
        )
    return frozenset(path for path in result.stdout.split("\0") if path.endswith(".py"))


def _relative(reported: str) -> str:
    path = Path(reported.strip())
    if path.is_absolute():
        path = path.resolve().relative_to(ROOT)
    return path.as_posix()


def black_failures(paths: Sequence[str]) -> frozenset[str]:
    if not paths:
        return frozenset()
    result = _run([sys.executable, "-m", "black", "--check", "--", *paths])
    reformat = frozenset(
        _relative(line[len("would reformat ") :])
        for line in result.stderr.splitlines()
        if line.startswith("would reformat ")
    )
    # Exit 123 with "cannot format <path>: ..." is a file Black cannot parse:
    # a lint failure of that file, not an environment problem.
    unparseable = frozenset(
        _relative(line[len("error: cannot format ") :].split(": ", 1)[0])
        for line in result.stderr.splitlines()
        if line.startswith("error: cannot format ")
    )
    failing = reformat | unparseable
    if (result.returncode == 1 and reformat and not unparseable) or (
        result.returncode == 123 and unparseable
    ):
        return failing
    if result.returncode == 0 and not failing:
        return frozenset()
    raise LintEnvironmentError(
        f"black exited {result.returncode}: {result.stderr.strip()[-2000:]}"
    )


def flake8_failures(paths: Sequence[str]) -> frozenset[str]:
    if not paths:
        return frozenset()
    result = _run([sys.executable, "-m", "flake8", "--format=%(path)s", "--", *paths])
    failing = frozenset(
        _relative(line) for line in result.stdout.splitlines() if line.strip()
    )
    if result.returncode == 1 and failing:
        return failing
    if result.returncode == 0 and not failing:
        return frozenset()
    raise LintEnvironmentError(
        f"flake8 exited {result.returncode}: "
        f"{(result.stderr or result.stdout).strip()[-2000:]}"
    )


def evaluate(
    tool: str,
    *,
    tracked: Iterable[str],
    baseline: frozenset[str],
    registered: frozenset[str],
    changed: frozenset[str],
    probe: FailureProbe,
) -> ToolReport:
    """Apply the scope and ratchet policy for one tool."""
    tracked_set = frozenset(tracked)
    touched = baseline & changed
    exempt = baseline - touched
    in_scope = sorted(tracked_set - exempt)
    report = ToolReport(
        tool=tool,
        checked=len(in_scope),
        untracked_baseline=sorted(baseline - tracked_set),
        unregistered_baseline=sorted(baseline - registered),
        touched_baseline=sorted(touched),
    )
    failing = probe(in_scope)
    report.failures = sorted(failing)
    still_exempt = sorted(exempt & tracked_set)
    # A touched baseline file that passes was cleaned in this branch; its
    # entry must go in the same change, or the next push to main would fail.
    cleaned = (touched & tracked_set) - failing
    report.unexpected_passes = sorted(
        (frozenset(still_exempt) - probe(still_exempt)) | cleaned
    )
    return report


def _print_report(report: ToolReport) -> None:
    status = "ok" if report.ok else "FAILED"
    print(f"[{report.tool}] {status}: {report.checked} files in scope")
    for path in report.touched_baseline:
        print(
            f"[{report.tool}] {path}: baseline file changed in this branch; "
            "it is linted like any maintained file"
        )
    for path in report.failures:
        hint = (
            " (changed baseline file: clean it and remove its entry from "
            "tools/ci_python_lint.py)"
            if path in report.touched_baseline
            else ""
        )
        print(f"[{report.tool}] {path}: fails {report.tool}{hint}")
    for path in report.unexpected_passes:
        print(
            f"[{report.tool}] {path}: now passes; remove it from the "
            f"{report.tool} baseline in tools/ci_python_lint.py"
        )
    for path in report.untracked_baseline:
        print(f"[{report.tool}] {path}: baseline entry is not a tracked file")
    for path in report.unregistered_baseline:
        print(
            f"[{report.tool}] {path}: baseline entry was not in the baseline "
            "at introduction; the baseline may only shrink"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--changed-since",
        metavar="REF",
        help="lift the exemption for baseline files changed since REF's merge base",
    )
    args = parser.parse_args(argv)
    try:
        tracked = tracked_python_files()
        changed = (
            changed_python_files(args.changed_since)
            if args.changed_since is not None
            else frozenset()
        )
        reports = [
            evaluate(
                "black",
                tracked=tracked,
                baseline=BLACK_BASELINE,
                registered=BLACK_BASELINE_AT_INTRODUCTION,
                changed=changed,
                probe=black_failures,
            ),
            evaluate(
                "flake8",
                tracked=tracked,
                baseline=FLAKE8_BASELINE,
                registered=FLAKE8_BASELINE_AT_INTRODUCTION,
                changed=changed,
                probe=flake8_failures,
            ),
        ]
    except LintEnvironmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for report in reports:
        _print_report(report)
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
