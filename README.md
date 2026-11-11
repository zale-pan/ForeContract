# ForeContract

Copyright 2026 Zale Pan. Licensed under the Apache License 2.0 (see `LICENSE`).

ForeContract is an experimental, local-first runtime for executing a declared task against an
environment, verifying the effect from evidence, and recording what really happened. It is a
source candidate, not a production service.

## The five packages

| Package | Directory | Responsibility |
|---|---|---|
| `forecontract-core` (import `forecontract`) | `packages/core` | runtime engine, contract builder, step verifier, task oracle, recovery engine, tracing |
| `forecontract-fake` | `packages/adapters/fake` | deterministic Fake environment adapter for offline runs |
| `forecontract-web` | `packages/adapters/web` | Web/browser adapter with process-isolated control |
| `forecontract-scripted` | `packages/planners/scripted` | scripted planner used by fixed offline scenarios |
| `forecontract-llm` | `packages/planners/llm` | LLM planner, retrieval planner, evidence reader and audit |

The Core package does not import the adapters or the planners; they depend on Core.

## Quick start

Keep the virtual environment and every output outside this repository, and point the test
scratch variable at an external writable directory.

```powershell
$venvRoot = '<outside-the-repo>/forecontract-venv'
$python = Join-Path $venvRoot 'Scripts/python.exe'
$env:FORECONTRACT_TEST_SCRATCH = '<outside-the-repo>/forecontract-scratch'
python -m venv $venvRoot
& $python -m pip install -r requirements-dev.txt
& $python -m pip install -e packages/core -e packages/adapters/fake -e packages/adapters/web -e packages/planners/scripted -e packages/planners/llm
```

& $python -B examples/offline_runtime_demo.py --output-dir '<outside-the-repo>/demo'

## Tests

The public offline selection is a frozen 54-file list with exactly one deselected internal
node; it is not `pytest tests`. The complete, executable PowerShell block — an array of the 54
files passed to pytest with the single `--deselect`, run from the project root — is in
[examples/README.offline.zh-CN.md](examples/README.offline.zh-CN.md).

The Web group needs a separately installed Chromium build and is listed on its own.

The local Web reference runs that passed 136 tests in an earlier round are a historical
reference, not a result of this round.

## Capability claims

* **Historical independent results** — offline runtime, retrieval and evidence paths
  were verified in earlier rounds; the local Web reference had 136 passing tests.
  These are historical results, reused only when the relevant inputs remain unchanged.
* **This documentation correction** — no product, installation, browser or API run
  is claimed for this round. Independent review of the local source candidate is pending.
* **Not run / unknown in this round** — real API execution is not authorized;
  cleanup cancellation and hang behavior remain unexercised or unknown.
* **Future work** — wider platform, provider and website coverage. This is not a
  production reliability or general compatibility claim.

## Limitations

* Experimental interfaces; no compatibility promise and no production reliability claim.
* No claim of coverage for arbitrary websites, of model quality or benefit, or of any operating
  system beyond the one used for validation.
* Behaviour of a provider that hangs during cleanup remains unknown.
* There is no operating-system sandbox; repository checks are not isolation.

See `RELEASE_NOTES.md`, `SECURITY.md`, `CONTRIBUTING.md`, `THIRD_PARTY_NOTICES.md` and
`examples/LLM_API_SMOKE.md`.
