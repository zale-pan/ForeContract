# Contributing

Copyright 2026 Zale Pan.

## Development install

Install into an environment outside the repository.

```powershell
$venvRoot = '<outside-the-repo>/forecontract-venv'
$python = Join-Path $venvRoot 'Scripts/python.exe'
$env:FORECONTRACT_TEST_SCRATCH = '<outside-the-repo>/forecontract-scratch'
python -m venv $venvRoot
& $python -m pip install -r requirements-dev.txt
& $python -m pip install -e packages/core -e packages/adapters/fake -e packages/adapters/web -e packages/planners/scripted -e packages/planners/llm
```

## Test groups

* offline unit and integration tests: the frozen 54-file selection whose complete, executable
  PowerShell block is in [examples/README.offline.zh-CN.md](examples/README.offline.zh-CN.md);
* Web reference tests need a separately installed Chromium build and are listed on their own;
* the release engineering tests in `tests/release` are internal release checks.

Run that block from the project root; do not replace it with `pytest tests` or with a bare
directory list.

## Temporary directories

Keep every temporary file, virtual environment, build output and wheel outside this repository.
Point the test scratch environment variables at a directory of your own; do not commit generated
artifacts, caches or virtual environments.

## Submitting changes

1. Keep each change minimal and scoped to one concern.
2. Do not commit credentials, personal absolute paths or generated output.
3. Reproduce the offline test selection locally before submitting.
4. Describe what you verified and what you did not; a not-run check is never a pass.
