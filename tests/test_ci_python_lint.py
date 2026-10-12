"""CI-1: the lint scope policy and the workflow's own guarantees.

The lint policy is tested with injected failure probes, so these tests
never run Black or Flake8; the ``python-lint`` job runs the real tools.
The workflow checks pin the properties docs/ci.md promises: triggers,
least privilege, no secrets, pinned actions, and the commands each job runs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools import ci_python_lint as lint

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# A deliberate second copy of the baseline at introduction (main be79267).
# Growing the baseline must change this file too, in plain sight.
PINNED_BLACK_BASELINE = frozenset(
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
        "tests/test_ai_summarization.py",
        "tests/test_phase0_config.py",
        "tests/test_phase0_copy_rules.py",
        "tools/validate_phase0_config.py",
        "tools/validate_phase0_copy_rules.py",
    }
)
PINNED_FLAKE8_BASELINE = (PINNED_BLACK_BASELINE - {"backend/app/routes/market.py"}) | {
    "backend/test_prediction_startup_prewarm.py"
}


def probe_failing(failing: set[str]):
    calls: list[list[str]] = []

    def probe(paths):
        calls.append(list(paths))
        return frozenset(failing) & frozenset(paths)

    probe.calls = calls
    return probe


def run(*, tracked, baseline, failing, changed=(), registered=None):
    baseline = frozenset(baseline)
    return lint.evaluate(
        "black",
        tracked=tracked,
        baseline=baseline,
        registered=baseline if registered is None else frozenset(registered),
        changed=frozenset(changed),
        probe=probe_failing(set(failing)),
    )


# -- Policy --------------------------------------------------------------------


def test_clean_scope_with_failing_baseline_passes():
    report = run(
        tracked=["a.py", "legacy.py"], baseline={"legacy.py"}, failing={"legacy.py"}
    )
    assert report.ok
    assert report.checked == 1
    assert report.failures == []


def test_new_file_is_in_scope_without_registration():
    report = run(
        tracked=["a.py", "brand_new.py", "legacy.py"],
        baseline={"legacy.py"},
        failing={"legacy.py", "brand_new.py"},
    )
    assert not report.ok
    assert report.failures == ["brand_new.py"]


def test_failure_in_maintained_file_fails():
    report = run(tracked=["a.py"], baseline=set(), failing={"a.py"})
    assert not report.ok
    assert report.failures == ["a.py"]


def test_baseline_file_that_now_passes_must_be_removed():
    report = run(tracked=["legacy.py"], baseline={"legacy.py"}, failing=set())
    assert not report.ok
    assert report.unexpected_passes == ["legacy.py"]


def test_baseline_entry_must_be_tracked():
    report = run(tracked=["a.py"], baseline={"gone.py"}, failing=set())
    assert not report.ok
    assert report.untracked_baseline == ["gone.py"]


def test_baseline_cannot_grow_beyond_introduction():
    report = run(
        tracked=["a.py", "legacy.py"],
        baseline={"legacy.py", "a.py"},
        registered={"legacy.py"},
        failing={"legacy.py", "a.py"},
    )
    assert not report.ok
    assert report.unregistered_baseline == ["a.py"]


def test_changed_baseline_file_loses_its_exemption():
    report = run(
        tracked=["a.py", "legacy.py", "other_legacy.py"],
        baseline={"legacy.py", "other_legacy.py"},
        failing={"legacy.py", "other_legacy.py"},
        changed={"legacy.py"},
    )
    assert not report.ok
    assert report.touched_baseline == ["legacy.py"]
    assert report.failures == ["legacy.py"]
    assert report.unexpected_passes == []  # the untouched one still fails


def test_changed_baseline_file_once_cleaned_demands_entry_removal_in_the_pr():
    # Cleaned in the branch but left in the baseline: the pull request fails
    # now, rather than the next push to main failing after merge.
    report = run(
        tracked=["legacy.py"],
        baseline={"legacy.py"},
        failing=set(),
        changed={"legacy.py"},
    )
    assert not report.ok
    assert report.failures == []
    assert report.unexpected_passes == ["legacy.py"]


def test_untouched_legacy_files_are_never_handed_to_the_probe_for_scope():
    probe = probe_failing({"legacy.py"})
    lint.evaluate(
        "black",
        tracked=["a.py", "legacy.py"],
        baseline=frozenset({"legacy.py"}),
        registered=frozenset({"legacy.py"}),
        changed=frozenset(),
        probe=probe,
    )
    scope_call, ratchet_call = probe.calls
    assert scope_call == ["a.py"]
    assert ratchet_call == ["legacy.py"]


# -- The real baseline ---------------------------------------------------------


def test_baseline_at_introduction_is_pinned():
    assert lint.BLACK_BASELINE_AT_INTRODUCTION == PINNED_BLACK_BASELINE
    assert lint.FLAKE8_BASELINE_AT_INTRODUCTION == PINNED_FLAKE8_BASELINE


def test_live_baseline_only_shrinks():
    assert lint.BLACK_BASELINE <= lint.BLACK_BASELINE_AT_INTRODUCTION
    assert lint.FLAKE8_BASELINE <= lint.FLAKE8_BASELINE_AT_INTRODUCTION


def test_live_baseline_names_tracked_python_files():
    tracked = set(lint.tracked_python_files())
    assert lint.BLACK_BASELINE <= tracked
    assert lint.FLAKE8_BASELINE <= tracked


def test_lint_tool_is_not_exempt_from_itself():
    path = "tools/ci_python_lint.py"
    assert path not in lint.BLACK_BASELINE | lint.FLAKE8_BASELINE


# -- Workflow ------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key ``on`` as boolean true.
    document["on"] = document.pop(True, document.get("on"))
    return document


def steps_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_triggers_are_pull_requests_and_pushes_to_main(workflow):
    assert workflow["on"] == {
        "pull_request": {"branches": ["main"]},
        "push": {"branches": ["main"]},
    }


def test_least_privilege_and_no_job_widens_it(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        assert "permissions" not in job, name


def test_no_secrets_and_no_privileged_triggers():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text
    assert "pull_request_target" not in text
    assert "workflow_run" not in text


def test_actions_are_pinned_to_full_commit_shas(workflow):
    for name, job in workflow["jobs"].items():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", step["uses"]), (
                    name,
                    step["uses"],
                )


def test_checkouts_do_not_persist_credentials(workflow):
    for name, job in workflow["jobs"].items():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False, name


def test_expected_jobs_and_runtimes(workflow):
    assert set(workflow["jobs"]) == {"python-tests", "python-lint", "frontend"}
    assert workflow["env"]["PYTHON_VERSION"] == "3.11"
    assert workflow["env"]["NODE_VERSION"] == "22"
    for job in workflow["jobs"].values():
        assert job["timeout-minutes"] > 0


def test_python_job_installs_cpu_torch_first(workflow):
    job = workflow["jobs"]["python-tests"]
    text = steps_text(job)
    assert workflow["env"]["TORCH_INDEX_URL"].endswith("/whl/cpu")
    torch_at = text.index("TORCH_INDEX_URL")
    requirements_at = text.index("-r backend/requirements.txt")
    assert torch_at < requirements_at
    for requirement in ("backend/requirements.txt", "data/requirements.txt"):
        assert f"-r {requirement}" in text
    assert "torch.version.cuda is None" in text
    assert job["env"]["HF_HUB_OFFLINE"] == "1"


#: Offline backend suites CI runs by name (docs/ci.md, "Test scope").
CI_BACKEND_SCRIPTS = (
    "backend/test_data_service_cache.py",
    "backend/test_prediction_integration.py",
    "backend/test_prediction_persistence.py",
    "backend/test_prediction_startup_prewarm.py",
    "backend/test_sentiment_quality.py",
)
#: Not run by CI, with the reason.
CI_EXCLUDED_BACKEND_SCRIPTS = {
    "backend/test_api.py": "requests a live server at localhost:8000",
}


def test_python_job_runs_the_exact_test_scope(workflow):
    step = next(
        s
        for s in workflow["jobs"]["python-tests"]["steps"]
        if "pytest" in s.get("run", "")
    )
    command = step["run"].split()
    assert command[:4] == ["python", "-m", "pytest", "-q"]
    assert command[4:] == ["tests", "backend/tests", *CI_BACKEND_SCRIPTS]


def test_every_backend_test_script_is_classified():
    scripts = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "backend").glob("test_*.py")
    }
    assert scripts == set(CI_BACKEND_SCRIPTS) | set(CI_EXCLUDED_BACKEND_SCRIPTS)


def test_pull_requests_share_a_cancellable_group_and_main_runs_never_contend(
    workflow,
):
    concurrency = workflow["concurrency"]
    group = " ".join(concurrency["group"].split())
    assert group == (
        "ci-${{ github.workflow }}-${{ github.event_name == 'pull_request' "
        "&& format('pr-{0}', github.event.pull_request.number) "
        "|| format('run-{0}', github.run_id) }}"
    )
    assert concurrency["cancel-in-progress"] == (
        "${{ github.event_name == 'pull_request' }}"
    )


def test_cache_keys_name_existing_files(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            paths = step.get("with", {}).get("cache-dependency-path")
            for path in str(paths or "").split():
                assert (ROOT / path).is_file(), path


def test_lint_job_uses_the_scope_tool_with_pinned_versions(workflow):
    job = workflow["jobs"]["python-lint"]
    text = steps_text(job)
    assert "backend/requirements.txt" in text
    assert "python tools/ci_python_lint.py --changed-since" in text
    checkout = next(s for s in job["steps"] if "checkout" in s.get("uses", ""))
    assert checkout["with"]["fetch-depth"] == 0
    pins = re.findall(
        r"^(black|flake8)==\S+$",
        (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert sorted(pins) == ["black", "flake8"]


def test_frontend_job_runs_ci_lint_test_build_in_order(workflow):
    job = workflow["jobs"]["frontend"]
    assert job["defaults"]["run"]["working-directory"] == "frontend"
    commands = [step["run"] for step in job["steps"] if "run" in step]
    assert commands == ["npm ci", "npm run lint", "npm test", "npm run build"]


# -- Real Git: the tool run end to end in a temporary repository ----------------
#
# Each case builds a small repository holding the unmodified tool, the real
# .flake8, one file at every real baseline path (failing exactly the tools
# that baseline names it for) and one clean maintained file, then runs the
# tool as CI does.  Black and Flake8 are the pinned ones this suite runs on.

FAILS_BLACK = "x = 'a'\n"  # Black wants double quotes; Flake8 accepts it.
FAILS_FLAKE8 = "import os\n"  # F401 unused import; Black accepts it.
CLEAN = 'x = "a"\n'


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env=_GIT_ENV,
    )
    return result.stdout.strip()


_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "ci-test",
    "GIT_AUTHOR_EMAIL": "ci-test@example.invalid",
    "GIT_COMMITTER_NAME": "ci-test",
    "GIT_COMMITTER_EMAIL": "ci-test@example.invalid",
}


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _write(
        root, "tools/ci_python_lint.py", (ROOT / "tools/ci_python_lint.py").read_text()
    )
    _write(root, ".flake8", (ROOT / ".flake8").read_text())
    for path in lint.BLACK_BASELINE | lint.FLAKE8_BASELINE:
        text = (FAILS_BLACK if path in lint.BLACK_BASELINE else "") + (
            FAILS_FLAKE8 if path in lint.FLAKE8_BASELINE else ""
        )
        _write(root, path, text)
    _write(root, "phase0/maintained.py", CLEAN)
    _commit(root, "base")
    return root


def run_tool(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/ci_python_lint.py", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )


def test_git_baseline_repository_is_clean(repo):
    result = run_tool(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "ref", ["--cached", "--output=injected.txt", "-p", "", "nope", "HEAD^{tree}"]
)
def test_git_option_shaped_and_invalid_refs_fail(repo, ref):
    result = run_tool(repo, f"--changed-since={ref}")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "not a commit reference" in result.stderr
    assert not (repo / "injected.txt").exists()


def test_git_valid_branch_tag_and_sha_resolve(repo):
    _git(repo, "tag", "base-tag")
    sha = _git(repo, "rev-parse", "HEAD")
    for ref in ("main", "base-tag", sha, sha[:12], "HEAD"):
        result = run_tool(repo, "--changed-since", ref)
        assert result.returncode == 0, (ref, result.stdout + result.stderr)


def test_git_new_python_file_is_linted_tracked_or_not(repo):
    _write(repo, "nlp/new_module.py", FAILS_BLACK)
    untracked = run_tool(repo)
    assert untracked.returncode == 1
    assert "nlp/new_module.py: fails black" in untracked.stdout
    _commit(repo, "add module")
    tracked = run_tool(repo, "--changed-since", "main~1")
    assert tracked.returncode == 1
    assert "nlp/new_module.py: fails black" in tracked.stdout


def test_git_uncommitted_deletion_of_a_maintained_file_is_not_a_failure(repo):
    (repo / "phase0/maintained.py").unlink()
    result = run_tool(repo, "--changed-since", "main")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "maintained.py" not in result.stdout + result.stderr


def test_git_committed_deletion_of_a_maintained_file_is_not_a_failure(repo):
    _git(repo, "rm", "-q", "phase0/maintained.py")
    _commit(repo, "delete")
    result = run_tool(repo, "--changed-since", "main~1")
    assert result.returncode == 0, result.stdout + result.stderr


def test_git_deleting_an_exempt_file_requires_removing_its_entry(repo):
    (repo / "data/app.py").unlink()
    result = run_tool(repo)
    assert result.returncode == 1
    assert "data/app.py: baseline entry is not a tracked file" in result.stdout


def test_git_renaming_an_exempt_file_does_not_carry_its_exemption(repo):
    _git(repo, "checkout", "-q", "-b", "feature")
    _git(repo, "mv", "data/app.py", "data/renamed_app.py")
    _commit(repo, "rename")
    result = run_tool(repo, "--changed-since", "main")
    assert result.returncode == 1
    assert "data/renamed_app.py: fails black" in result.stdout
    assert "data/app.py: baseline entry is not a tracked file" in result.stdout


def test_git_changed_exempt_file_loses_its_exemption(repo):
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "data/app.py", (repo / "data/app.py").read_text() + "y = 'b'\n")
    _commit(repo, "touch legacy")
    result = run_tool(repo, "--changed-since", "main")
    assert result.returncode == 1
    assert "data/app.py: fails black (changed baseline file" in result.stdout
    # Without the base, as on a push, the untouched-exemption still applies.
    assert run_tool(repo).returncode == 0


def test_git_detached_pull_request_merge_state(repo):
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "data/app.py", (repo / "data/app.py").read_text() + "y = 'b'\n")
    _commit(repo, "touch legacy")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "phase0/other.py", CLEAN)
    _commit(repo, "main moves on")
    # actions/checkout on pull_request: a detached merge of the PR into base.
    _git(repo, "checkout", "-q", "--detach", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "feature")
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    result = run_tool(repo, "--changed-since", "main")
    assert result.returncode == 1
    assert "data/app.py: fails black (changed baseline file" in result.stdout
    assert "phase0/other.py" not in result.stdout


def test_git_attempted_exemption_expansion_is_refused(repo):
    _write(repo, "phase0/sloppy.py", FAILS_BLACK)
    tool = repo / "tools/ci_python_lint.py"
    source = tool.read_text()
    expanded = source.replace(
        "BLACK_BASELINE: frozenset[str] = BLACK_BASELINE_AT_INTRODUCTION",
        "BLACK_BASELINE: frozenset[str] = BLACK_BASELINE_AT_INTRODUCTION"
        ' | {"phase0/sloppy.py"}',
    )
    assert expanded != source
    tool.write_text(expanded)
    result = run_tool(repo)
    assert result.returncode == 1
    assert (
        "phase0/sloppy.py: baseline entry was not in the baseline at introduction"
        in result.stdout
    )
