# DeepSeek host smoke

This example adds one host integration for the existing planner stack. It does **not** replace the
Core runtime, the planner, the evidence recorder or the acceptance platform.

## Files

| File | Role |
|---|---|
| `deepseek_provider.py` | Async transport: one bounded POST to `https://api.deepseek.com/chat/completions`, mapping the provider envelope back to the existing `ModelResponse`. |
| `llm_api_smoke.py` | Host assembly and CLI: `run_case`, `load_cases`, `validate_case_evidence`, `main`. |
| `llm_api_smoke_cases.v1.json` | Versioned public synthetic fixture: `direct`, `retrieval-pending`, `retrieval-ready`. Not holdout data and not business data. |
| `llm_api_smoke.requirements.txt` | The host dependency, pinned to the offline-verified `httpx==0.28.1`. |

## Running

```powershell
$python = "<outside-the-repo>/venv/Scripts/python.exe"   # any interpreter with the host dependencies

# Offline: no socket, no credential read.
& $python -B examples/llm_api_smoke.py --mode offline `
    --cases examples/llm_api_smoke_cases.v1.json `
    --output-dir "<仓库之外的输出目录>/offline-001"
```

`--mode offline` is the default and never reads `DEEPSEEK_API_KEY`. Live mode must be selected
explicitly and additionally requires `--model`:

```powershell
& $python -B examples/llm_api_smoke.py --mode live --model deepseek-flash `
    --max-requests 3 --budget-cny 1 `
    --cases examples/llm_api_smoke_cases.v1.json `
    --output-dir "<仓库之外的输出目录>/live-001"
```

## Credentials

The key is injected only by the operator, through the local process environment variable
`DEEPSEEK_API_KEY`. It is never written to argv, logs, summaries, fixtures or this repository, and
no tool reads it on your behalf. When the variable is absent the live run records
`status=not_run`, `reason=missing_local_credentials`, `actual_http_requests=0` and exits non-zero:
a missing key is never reported as a passing live result, and an offline run is never relabelled as
live.

## What the evidence means

Each case directory contains:

* `summary.json` – the case identity, mode, terminal status, executed action, provider call count,
  close result and the paths of the evidence it names;
* `trace.jsonl` – the real Core trace; the last record must be `run_finished` with
  `terminal_status=success` for a completed case;
* `provider-events.jsonl` – one redacted record per provider request (request body, response body,
  sizes, timestamps); no authorization header and no raw transport exception text;
* `snapshots/` – the real host snapshot written and registered before the retrieval read.

`validate_case_evidence(dir)` re-reads those files and returns a list of problems, so a summary
claiming success over a failed trace, or an offline run carrying a live label, is reported instead
of trusted.

## Deliberately not claimed

* The transport bounds size, time and retries; it does not claim a supplier-side hard cost limit.
* JSON mode is not business-schema validation: the content is decoded by the existing planner.
* `deepseek-flash` is recorded as the requested model name; it is not a proof of a fixed model
  version behind the provider alias.
* A real browser integration is out of scope; the real API call, when authorized, drives only the
  synthetic Fake scenarios.
* Network access, API keys and production systems are never used by the offline path.

## Bounded cleanup (SR03)

The live group issues at most three real requests under one 90 s absolute group deadline. Its
AsyncClient is created and closed on the same event loop. The close itself has its own explicit
finite bound of **5 s** (CLOSE_TIMEOUT_SECONDS), enforced with asyncio.wait_for, so a transport
that never finishes closing cannot hold the group open and is never retried:

* a normal close returns immediately;
* a close that raises is recorded as a bounded failure and only a failed root summary is written,
  with the real actual_http_requests and the already-completed case evidence preserved;
* a close that exceeds 5 s is cancelled by the bound and treated the same way;
* a caller cancellation is re-raised unchanged and is recorded as cancelled_during_live_group,
  never as completed;
* bounded cleanup is allowed to cross the 90 s group deadline, and once it does the outcome is
  failed; it must still finish before the 100 s outer Windows Job timeout, which is a last resort
  and not the mechanism relied on.

The offline default never opens a socket, so its close is an immediate local call.

### Cancellation during the close phase (SR03)

The close call sits in the group `finally`, so a `CancelledError` raised by the close itself
would otherwise skip every failed-root branch. It is now caught at that exact point
(`_live_main.group` finally block), recorded, and the group writes a real failed root summary
before returning its integer non-zero status: `mode=live`, `status=failed`, a non-empty
`reason` (`cancelled_during_client_close`), `actual_http_requests` from the real send count,
and the already-completed `direct` and `retrieval-pending` cases with their summary, trace and
provider records untouched. An outer guard around the group coroutine writes
`cancelled_during_cleanup` if a cancellation still escapes. The 5 s finite close bound is
unchanged, and the underlying cancellation semantics are preserved.
