# Continuous integration (CI-1)

`.github/workflows/ci.yml` tests every pull request to `main` and every push
to `main`. It is CI only: it deploys nothing, and there is no CD workflow.

## Jobs

| Job (check name) | What it runs | Typical time |
|---|---|---|
| `Python tests (3.11)` | CPU-only torch, the pinned requirements, an environment check, then `python -m pytest` over the test scope below | ~4 min of tests plus install (cold cache several minutes; torch is the bulk) |
| `Python lint (Black + Flake8, maintained scope)` | `tools/ci_python_lint.py` with the pinned Black and Flake8 | < 1 min |
| `Frontend (Node 22)` | `npm ci`, `npm run lint`, `npm test`, `npm run build` in `frontend/` | ~1 min |

Any failing install, test, lint or build step fails its job.

**Concurrency.** GitHub allows at most one running and one pending run per
concurrency group; a newer pending run replaces an older pending one, and
`cancel-in-progress` also cancels the running one. Pull-request runs share a
group per PR number with `cancel-in-progress`, so a new push to a PR cancels
its obsolete run. Each push to `main` gets a group of its own
(`run-<github.run_id>`, unique per run), so every push to `main` runs to
completion: none waits behind, replaces, or cancels another.

## Test scope

The `Python tests (3.11)` job runs, by explicit path:

- `tests/` and `backend/tests/`;
- these `backend/test_*.py` suites, each checked to run offline: Yahoo
  fetches are replaced with fakes, the legacy sentiment test forces its
  lexicon fallback (no model), and the prediction tests write artifacts only
  under `tmp_path`. Together they are 17 tests, and they pass with outbound
  sockets blocked:
  - `backend/test_data_service_cache.py`
  - `backend/test_prediction_integration.py`
  - `backend/test_prediction_persistence.py`
  - `backend/test_prediction_startup_prewarm.py`
  - `backend/test_sentiment_quality.py`

Not run: `backend/test_api.py`, which sends requests to a live server at
`localhost:8000` (and its helper `test_endpoint` is not a pytest test), and
`backend/test_api_curl.sh`, a shell script for the same live server.

The backend files are listed, never globbed. `tests/test_ci_python_lint.py`
fails if a `backend/test_*.py` file is added without being classified as run
or excluded, or if the workflow's test command drifts from that list.

## Security

- `permissions: contents: read` for the whole workflow; no job widens it.
- Triggers are `pull_request` and `push` only. Never `pull_request_target`
  or `workflow_run`, which run fork code with secrets or write tokens.
- The workflow references no secrets, and fork pull requests receive none.
- Every action is pinned to a full commit SHA, with its tag in a comment.
- `actions/checkout` runs with `persist-credentials: false`.

`tests/test_ci_python_lint.py` checks each of these properties.

## Environment assumptions

- **Runner:** `ubuntu-24.04`, a non-root user. The SQLite permission tests
  (`needs_permissions`) skip under root, so do not move the job into a
  root container without accounting for that.
- **Python 3.11.** `numpy==1.24.3` has no wheels for 3.12+. The job asserts
  the interpreter version.
- **SQLite ≥ 3.38** (`phase0.schema.MINIMUM_SQLITE_VERSION`). The job
  asserts it against the runner's Python build before any test runs.
- **PyTorch:** not pinned by the project; it arrives via
  `sentence-transformers`. CI installs `torch==2.13.0` (the version in the
  development environment CI was introduced against) from the PyTorch CPU
  index *before* the requirements, so pip keeps it and never downloads the
  CUDA build. The job asserts `torch.version.cuda is None`. To change the
  version, edit `TORCH_VERSION` in the workflow.
- **Dependencies:** `backend/requirements.txt`, `data/requirements.txt` and
  `requirements-dev.txt`, followed by `pip check`. Transitive dependencies are
  not locked (there is no lock file), so a new upstream release can change
  what CI installs. Cached wheels are keyed on those files and the workflow.
- **No network services:** Gemini, Yahoo and RSS are faked in tests.
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` make any attempt to load the
  real embedding model fail instead of downloading it. No `GEMINI_API_KEY`
  or other credential is set.
- **Import path:** tests import `phase0`, `tools`, `ai` and `nlp` from the
  repository root, so they run as `python -m pytest` from the root (which
  puts the root on `sys.path`).
- **Timezone:** runners use UTC. The pipeline derives its day in
  `America/New_York` through `zoneinfo` (`tzdata` is pinned as a fallback).
- **Node 22** with `npm ci` against `frontend/package-lock.json`. Vite 8
  needs Node 20.19+ or 22.12+.

## Python lint scope

`tools/ci_python_lint.py` is the single definition of what is linted.

- **In scope:** every tracked `*.py` file present in the working tree (plus
  untracked, non-ignored ones when run locally), with nothing excluded by
  directory. New files are linted automatically. A tracked file deleted but
  not yet committed is left out rather than handed to Black or Flake8.
- **Baseline:** the files that already failed Black or Flake8 on `main` at
  `be79267`, listed in the script, in two groups:
  - *legacy*: the pre-Phase-0 sentiment/prediction backend, `data/app.py`,
    `nlp/sentiment.py`, `prediction/prediction.py`, and their ad-hoc tests;
  - *Phase 0 debt*: `tools/validate_phase0_{config,copy_rules}.py`,
    `tests/test_phase0_{config,copy_rules}.py` and
    `tests/test_ai_summarization.py`.
- **The baseline only shrinks:**
  - every entry must name a present, tracked file and must still fail its
    tool, so deleting or renaming an exempt file fails until its entry is
    removed (a renamed file is linted under its new name);
  - when an entry starts passing, the run fails until the entry is removed;
  - entries outside the baseline at introduction are refused, and that set
    is pinned again in `tests/test_ci_python_lint.py`.
- **Touching a baseline file in a pull request ends its exemption.** The
  lint job diffs against the PR's base. A changed baseline file is linted
  like any other file, and the PR must also delete its baseline entry.
  Untouched legacy files are never reformatted.
- **`--changed-since REF` must name a commit.** The ref is resolved with
  `git rev-parse --verify --end-of-options REF^{commit}`, and anything that
  looks like an option (starts with `-`), is empty, is unknown, or names a
  non-commit object fails with exit `2`. The diff then uses the resolved SHA,
  never the raw text.

To remove an entry, subtract it from `BLACK_BASELINE` or `FLAKE8_BASELINE`,
for example `BLACK_BASELINE = BLACK_BASELINE_AT_INTRODUCTION - {"path.py"}`.
Never edit the `*_AT_INTRODUCTION` sets.

Exit status: `0` clean, `1` lint or baseline failure, `2` environment or
usage error (no git, a tool missing, a ref that is not a commit).

## Running the jobs locally

```bash
# Python tests (from the repository root)
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests backend/tests \
  backend/test_data_service_cache.py backend/test_prediction_integration.py \
  backend/test_prediction_persistence.py backend/test_prediction_startup_prewarm.py \
  backend/test_sentiment_quality.py

# Python lint, as on a pull request
.venv/bin/python tools/ci_python_lint.py --changed-since origin/main

# Frontend
cd frontend && npm ci && npm run lint && npm test && npm run build
```
