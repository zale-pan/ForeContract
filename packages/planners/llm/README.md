# forecontract-llm

`forecontract-llm==0.1.0` is the ForeContract **offline structured LLMPlanner**. It depends
only on `forecontract-core>=0.1,<0.2` and `pydantic>=2,<3`, requires Python 3.12 or newer and
imports as `forecontract_llm`.

## What it provides

- `LLMPlanner`, a Core-compatible planner with exactly `async plan(observation) -> Forecast`,
  plus `aclose() -> PlannerCloseResult`, `calls_used` and `state`.
- `ModelProvider`, the single injected transport protocol: one async `complete(request)`.
- `PlannerContext`, `PlannerConfig`, `ModelRequest`, `ModelResponse`, `ModelUsage`,
  `PlannerCloseResult` and `PlannerError`.
- Internal `forecontract_llm.codec.build_request` / `decode_response`, the strict boundary
  between the model's structured output and the Core action/observation vocabulary.

## Offline protocol

The host injects the provider. This distribution contains **no network client, no HTTP call,
no API-key read, no environment-variable secret lookup and no vendor SDK import**. There is no
production transport in this round: a real API/Web end-to-end is a later task. Controlled
evidence retrieval is provided by the `forecontract_llm.retrieval` subpackage documented below.
`forecontract_llm.provider.OfflineProvider` is the deterministic, in-process reference double
used by the tests; it replays a finite sequence of already-validated responses and performs no
I/O.

One planning call is:

1. revalidate the observation and check its environment against the declared capabilities;
2. build the structured request from the trusted context, the complete current observation,
   the response schema and the caller's limits, then enforce the byte, depth and node limits;
3. check the call allowance and the absolute deadline, then charge exactly one call;
4. make exactly one provider call, with no hidden retry and no JSON repair;
5. bind the response to the current request identifier, refuse a refusal, enforce the response
   byte, depth and node limits, parse one strict JSON object, and validate the action and
   outcomes against the declared capabilities and the declared token budget;
6. return one Core `Forecast` whose `forecast_id` is the current request identifier.

Any malformed, out-of-vocabulary or oversized answer becomes a fixed `PlannerError`. The
planner never coerces a bad answer into an action and never returns a partial forecast.

## Host responsibilities

- Create a fresh planner for each Run and keep `PlannerContext.run_id`, `goal_id` and
  `task_criteria` equal to that Run's `RunRequest`. Core does not hand the planner a
  `RunRequest`, so the planner cannot authenticate that binding and does not claim to.
- Supply `action_guidance` and `outcome_guidance` whose keys are exactly the adapter's declared
  capability names and whose values are non-empty JSON objects describing each payload. They
  are format documentation, not an executed JSON Schema.
- Ensure the provider is non-blocking and cooperatively asynchronous. The planner starts no
  thread and no process.
- In a `finally` block call `await planner.aclose()` and then close the provider. `aclose`
  reports `drained` and `pending_calls` honestly; it does not promise to kill uncooperative
  code.

## Limitations

- One single-step proposal only. The proposal still has to pass the Contract Builder and the
  domain gates; a correct JSON document or a declared capability name does not make an action
  executable.
- `task_completion_hint` is advisory. It can never complete a goal; only the Core task oracle
  decides completion.
- `max_output_tokens` is sent as a constraint and a declared overrun is refused, but without a
  tokenizer this package claims only byte, depth, node, call-count and time limits. A declared
  usage is the provider's statement, not a proof of cost.
- Calls, refusals, malformed output, provider failures, timeouts and cancellations are charged
  and are never refunded.
- The context binding is a host duty, not a verified fact.

## Install and test

```console
python -m pip install forecontract-llm-0.1.0-py3-none-any.whl
```

The test suite builds the current sources in an external directory, installs the wheel into
fresh environments (Core-only and Core+LLM) without any source injection and drives the
planner against the real Core Runtime with an offline provider double. The same installed
Core+LLM environment runs one retrieval consumer: it registers a real snapshot, performs one
authorized read and one forecast, and proves that an unauthorized read, a cross-Run scope and an
escaping snapshot path are still refused.

## Controlled evidence retrieval (`forecontract_llm.retrieval`)

The subpackage adds one independently imported entry point next to the original planner; the
original six modules, the root exports, `ModelProvider`, `ModelRequest`, `ModelResponse`, the
package version and the dependencies are unchanged.

Public entry points (exactly these thirteen): `EvidenceLLMPlanner`, `RetrievalConfig`,
`RetrievalModelRequest`, `RetrievalProvider`, `EvidenceReader`, `EvidenceScope`,
`EvidenceReadRequest`, `EvidenceReadResult`, `EvidenceGrant`, `RegisteredEvidenceReader`,
`RetrievalAuditSink`, `RetrievalAuditEvent`, `RetrievalError`.

`EvidenceLLMPlanner` implements the same Core `Planner` contract as `LLMPlanner`
(`async plan(observation) -> Forecast`) plus `async aclose() -> PlannerCloseResult`, `calls_used`,
`state` and `audit_unconverged`. One instance is bound to one Run and supports no concurrent
plan. A plan opens a frozen evidence scope, then permits at most two authorized reads and at most
`min(max_model_calls, max_reads + 1)` provider calls inside one monotonic deadline.

### Message protocol

The provider receives one measured `RetrievalModelRequest`: the original Core request (context
and complete observation), the frozen scope, every read result already obtained in this plan and
the response schema. The answer must be exactly one strict JSON object of one of two shapes, with
all common fields required and the other branch's field forbidden:

```text
{schema_version, run_id, plan_id, turn, request_id, kind: "forecast", forecast: {...}}
{schema_version, run_id, plan_id, turn, request_id, kind: "read_evidence",
 read: {evidence_id, view_id, text_range: null | {start, end}}}
```

Duplicate keys, non-finite numbers, lone surrogates, code fences, trailing text, an extra field
or a mismatched run/plan/turn/request identity are refused. A forecast body is handed back to the
original `decode_response` through a restored request, so the original action/outcome/capability
gates still decide and `forecast_id` stays the final provider request identifier. Only the
forecast branch can complete a plan; a refusal ends it without another call.

### Budgets and measurement

| Budget | Default / hard cap |
| --- | --- |
| Reads per plan | 2 (configurable 0..2) |
| Model calls per plan | 3 (configurable 1..3), also `max_reads + 1` |
| Instance model calls | `base.max_calls`, default 20 |
| Plan deadline / cleanup window | `base.call_timeout_ms` (30,000 ms) / `base.cleanup_timeout_ms` (100 ms) |
| One raw snapshot | 1,048,576 bytes (read at most cap + 1) |
| Registered snapshots | 8,388,608 bytes / 64, reader lifetime |
| Views per snapshot | 8 |
| Scope directory | 32,768 bytes |
| One / cumulative read result | 65,536 / 131,072 bytes |
| One / cumulative request | `min(base.max_request_bytes, 262,144)` / 786,432 bytes |
| One / cumulative response content | `min(base.max_response_bytes, 65,536)` / 196,608 bytes |
| Requested output tokens | `min(base.max_output_tokens, 2,048)` |
| JSON depth / nodes | `min(base.max_json_depth, 32)` / `min(base.max_json_nodes, 10,000)` |

Canonical measurement JSON is UTF-8, `ensure_ascii=false`, `sort_keys=true`, compact separators,
`allow_nan=false`, no trailing newline; it is this protocol's rule only and not the event-protocol
RFC 8785 form. A missing provider usage is recorded as unknown, never as zero; a declared
`output_tokens` above the request value is refused. There is no silent truncation, no automatic
summary, no hidden cache and no retry.

### Host responsibilities

- Register evidence only from the host. `RegisteredEvidenceReader(root, run_id, config)` takes an
  explicit root, `register_observation(observation, entry, grants)` is a synchronous host-only
  call, and the model can never reach it. The entry must be the real `EvidenceRecorder` entry: its
  bounded bytes are re-hashed, strictly parsed, checked against the snapshot schema, kind, Run,
  Observation, ordinal/call ordinal and associations, and its payload must equal the actual
  Observation. Only the declared snapshot member path under the explicit root is opened; an
  absolute path, a `..` escape, a symlink or a reparse component is refused. `ArtifactRef.uri` is
  data, never a read location.
- Supply the grants. An `EvidenceGrant` names one `evidence_id`, one `view_id`, one
  `format` (`json` or `text`) and one selector (`state` or `summary`, at most 32 bounded segments,
  no expressions, wildcards, coercion or URI dereference). Granting a view grants the whole
  selected value; the host must review nested fields and keep credentials, oracles, evaluation
  answers and future steps out of them. Duplicate evidence or view identifiers and more than eight
  views per snapshot are refused, and a failed registration leaves no half-registered entry.
- Read snapshots as registration-time copies. The validated bytes are kept as an independent
  copy; a later file change does not change the registered evidence, and the audit states that
  this is a registration-time snapshot rather than live disk state. The mechanism does not claim
  to defeat a malicious concurrent OS replacement of a file while it is being read and provides
  no OS sandbox.
- Treat reader results as untrusted records. The returned result is copied from its declared
  storage and revalidated; measurement, the audit record and the next request all use that one
  detached copy, never the caller's object or an overridable serializer. The reader's `read`
  must be an async callable: a synchronous shape is refused at construction, and an exception
  raised while creating the awaitable is normalised to `reader_failed` without echoing facility
  text. A `RetrievalError` raised by an injected interface while that interface is being
  evaluated (the provider's `complete`, the reader's `open_scope`/`read`, the sink's `append`)
  is an external failure, not this planner's own classification: it is normalised to
  `provider_failed`, `reader_failed` or `audit_failed` under the planner's own request
  identifier. The facility's code, stage, message and request identifier are never re-raised or
  echoed; an external `CancelledError` keeps its identity, and the planner's own budget,
  binding and selection failures keep their own codes.
- Enforce the directory budget at the planner. The frozen scope is measured against its
  32,768-byte canonical directory limit before any provider call; a reader-side check is not
  treated as the planner's trust boundary and an oversized scope is refused, never truncated.
- Persist the audit trail. `RetrievalAuditSink.append(event)` is the persistence boundary; an
  in-memory list is not enough for the formal demo. Every event binds its own canonical payload
  bytes and SHA-256, every external call has its intent persisted before the call, and a failed
  result record forbids the next call and forbids returning a Forecast. A provider value that
  arrives but fails its binding is still recorded as a real return (bounded, or as an explicit
  rejection), never disguised as an absent return. A terminal record after a cancellation or
  timeout is only ever attempted inside the bounded cleanup window; the same absolute cleanup
  deadline is re-read after the sink is looked up, the event is prepared and its awaitable is
  created, and before any task is scheduled or awaited. A window that expired during that
  synchronous preparation never starts the record: the unscheduled coroutine is closed and the
  record is reported as `audit_unconverged` instead of being retried with a new window. The demo's
  `FileAuditSink` writes and fsyncs a private temporary file, then publishes the final name
  atomically and without overwriting, so an interrupted write leaves no partial `*.json`.
- Re-check the deadline after synchronous preparation. Every provider, reader and sink interface
  is evaluated synchronously before its awaitable exists; the remaining plan deadline is re-read
  after that step and before the task is scheduled. An operation whose deadline already expired
  is not started, its prepared coroutine is closed and the plan reports `timeout`/`deadline`.
- Close honestly. `aclose()` cancels outstanding operations inside one cleanup window, reports
  `drained` and `pending_calls` (all unconverged external operations, not only model calls) and
  never fabricates a drain. An operation that does not settle keeps a strong reference, poisons
  the planner and is refused by a later plan. A host `CancelledError` always keeps its identity;
  only the close's own cancellation is reported as `invalid_state`.

### Limitations

- This is an offline, host-authorized retrieval capability over local registered snapshots. It is
  not a real model transport, not a vector store, not an OCR or multi-modal reader, not a Runtime
  or a verification authority, and it claims no browser/API end-to-end behaviour.
- Python's cooperative cancellation does not pre-empt a blocking OS call; a cancelled read or
  audit append that ignores cancellation is retained and reported, not killed.
- A `not_found` means the source was valid but the path key was missing or the array index was out
  of range. An illegal path, a type mismatch along the path or a non-string text selection is an
  error, never disguised as a missing value.
- A JSON selector is a projection; `partial` only reports that a text interval is not the whole
  string. A full interval does not turn a projection into a complete source snapshot.
- Evidence content can never widen authorization: the planner's scope is the sole read
  authority, and the final action still has to pass the Contract Builder and the domain gates.

