# Offline Web Trace & Verification Evidence Replay (`examples/replay_trace.py`)

`examples/replay_trace.py` provides an offline, read-only entry point for validating a recorded Web runtime evidence bundle (`schema_version = "0.1"`, `kind = "web_runtime_evidence_index"`), re-evaluating every recorded step and task verification snapshot against the pure Web verifiers (`WebStepVerifier` and `WebTaskCompletionOracle`), computing structured JSON Pointer diffs, and summarizing non-passing verification and trace facts.

## 1. Interfaces

### Python Function

```python
from pathlib import Path
from replay_trace import ReplayError, replay_evidence

report = replay_evidence(Path("path/to/evidence-profile"))
```

- `replay_evidence(root: Path) -> dict` is strictly read-only: it never creates, modifies, or deletes files, and importing `replay_trace` performs no file reads, subprocess execution, or network activity.
- Input or output errors raise `ReplayError(code, message)`, where `code` belongs to the closed set `{"REPLAY_INPUT_INVALID", "REPLAY_OUTPUT_INVALID"}` and `message` is a fixed diagnostic string that never echoes untrusted input values or raw exception text.
- `KeyboardInterrupt` and `SystemExit` are never swallowed or converted into `ReplayError`.

### Command-Line Interface

```powershell
python -B examples/replay_trace.py --input <evidence_dir> --output <new_report.json>
```

#### Exit Codes

| Exit Code | Error Code / Status | Meaning |
| --- | --- | --- |
| `0` | `replay_status = "matched"` | Complete valid input bundle re-evaluated; all recomputed verification results match the recorded results (`difference_count == 0`). Historical runs whose terminal status is `failed`, `aborted`, `timeout`, or `cancelled` also exit `0` when their recorded verifications match recomputation. |
| `2` | `replay_status = "different"` | Complete valid input bundle re-evaluated; at least one verification result differs (`difference_count > 0`). The full deterministic report is written to `--output`. |
| `3` | `REPLAY_INPUT_INVALID` | Input bundle is missing files, has unlisted/extra/duplicate entries, unsupported `schema_version`/`kind`/`verifier`, broken associations, hash/byte mismatch, BOM, duplicate JSON keys, non-finite numbers (`NaN`/`Infinity`), path traversal/absolute paths, symlinks/Windows junctions/reparse points, or concurrent modification during replay. No output file is created. |
| `4` | `REPLAY_OUTPUT_INVALID` | Output path already exists, resides inside the `--input` tree, traverses a symlink or Windows reparse point/junction, or cannot be written. Existing files are never overwritten and no partial report is left behind. |

## 2. Input Format & Validation Rules

1. **Evidence Index**: `<input>/evidence-index.json` must declare `schema_version = "0.1"` and `kind = "web_runtime_evidence_index"`.
2. **Portable Relative Paths**: `trace.path`, `working_memory.path`, and all snapshot/association `path` fields must be relative POSIX paths confined within `<input>`. Absolute paths, drive letters, backslashes, `..` segments, symbolic links, and Windows junctions/reparse points anywhere in the input tree or ancestor chain are rejected before any verifier runs.
3. **Strict JSON / JSONL Parsing**: `evidence-index.json`, `working-memory.json`, `snapshots/*.json`, and every line of `trace.jsonl` must be valid UTF-8 without BOM, newline-terminated (`\n`), free of duplicate JSON object keys, and free of non-finite numbers (`NaN`, `Infinity`, `-Infinity`).
4. **Full Inventory & Association Verification**: All checks from `examples/web_runtime_evidence.py:verify_evidence_index` remain enforced (exact snapshot directory inventory, SHA-256 and byte counts, contiguous ordinals, contract/attempt/verification/observation/working-memory/terminal cross-checks), and all `ActionContract`, `VerificationEvidence`, and `TaskCompletionCriteria` payloads are validated before invoking `WebStepVerifier` or `WebTaskCompletionOracle`.
5. **Pre/Post Read Consistency**: The complete input tree digest is captured before validation and re-checked after recomputation; any concurrent mutation aborts report publication with `REPLAY_INPUT_INVALID` (exit `3`).

## 3. Output Report Schema

Reports are serialized deterministically as UTF-8 JSON (`ensure_ascii=False`, `sort_keys=True`, `allow_nan=False`, `indent=2`, LF `\n` line endings, trailing newline) with no timestamps, random identifiers, or output paths.

Top-level fields (`schema_version = "0.1"`, `kind = "verification_replay_report"`):

- `input_index_sha256`: SHA-256 hex digest of `<input>/evidence-index.json`.
- `run_id`, `profile`: Copied from the validated index.
- `verifier_sources`: `{ "core_version": ..., "web_version": ..., "sha256": { ... } }` computed from the live `forecontract`, `forecontract_web`, and `examples/web_runtime_evidence.py` source files.
- `recorded_terminal`: Exact `terminal` object (`terminal_status`, `reason_code`, `source`) from the index.
- `replay_status`: `"matched"` when `difference_count == 0`, else `"different"`.
- `verification_count`, `difference_count`: Total verification snapshots evaluated and count of rows with `equal == False`.
- `verifications`: Array sorted by `(kind, ordinal, source_path)`. Each item contains:
  - `source_path`, `source_sha256`, `kind`, `ordinal`, `trace_fact`, `event_record_id`, `step_id`, `contract_id`, `attempt_id`, `observation_id` (`null` when not applicable; recovery re-verifications preserve `trace_fact = null` and `event_record_id = null`).
  - `recorded`, `recomputed`: Full `VerificationResult` JSON objects.
  - `equal`: Boolean equality of `recorded` and `recomputed`.
  - `differences`: JSON Pointer (`RFC 6901`) difference entries `{ "path", "recorded_present", "recomputed_present", "recorded", "recomputed" }` sorted by `path`. Missing keys and `null` values are distinguished.
- `analysis`:
  - `nonpassing_source_paths`: `source_path` values (in `verifications` order) whose recomputed `status` is `"fail"` or `"inconclusive"`.
  - `terminal_reason_code`: `recorded_terminal.reason_code`.
  - `failure_event_record_ids`: Trace `record_id` values (in trace sequence order) where `status` is `"fail"`/`"inconclusive"`, `execution_status` is `"failed"`/`"execution_unknown"`, or `kind` is `"execution_unknown"`.

## 4. Evidence Boundaries

- **Offline Pure Re-evaluation Only**: Replays saved verification evidence against `WebStepVerifier` and `WebTaskCompletionOracle`. It never launches a browser, adapter, runtime engine, planner, subprocess, or network connection.
- **Factual Failure Summary**: `analysis` reports only explicit non-passing verification snapshots, the recorded terminal reason code, and non-passing/unknown trace event IDs. It does not claim unproven root causes and never rewrites a failed, timed-out, cancelled, or aborted run into task completion.
- **Scope Boundary**: Contributes offline trace/evidence replay for `P5-T01` (`G-REPLAY-001-AC01` partial contribution). Composed/model-based verifiers (`P5-T02`), extended memory (`P5-T03`), and live API execution are separate and not exercised here.
