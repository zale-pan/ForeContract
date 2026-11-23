"""``CodingAdapter``: implements Core ``Adapter`` with 5 strict tools and JUnit acceptance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from forecontract.models.actions import ActionSpec, AdapterCapabilities
from forecontract.models.observations import ArtifactRef, Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo, ExecutionStatus
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
)
from pydantic import ValidationError

from forecontract_coding.errors import (
    CODING_INPUT_INVALID,
    CODING_OBSERVATION_OVERFLOW,
    CODING_PROCESS_CLEANUP_FAILED,
    CODING_PROCESS_TIMEOUT,
    CODING_SCOPE_VIOLATION,
    CODING_VERIFICATION_INVALID,
    CodingError,
)
from forecontract_coding.models import (
    CODING_ADAPTER_NAME,
    CODING_ENVIRONMENT_TYPE,
    SCHEMA_VERSION,
    SUPPORTED_ACTIONS,
    SUPPORTED_CONDITION_TYPES,
    SUPPORTED_OUTCOME_TYPES,
    AcceptanceSummary,
    ChangedPathEntry,
    CodingConfig,
    CodingTask,
    CommandSpec,
    ReadFilePayload,
    RunCommandPayload,
    ScopeViolationEntry,
    SearchTextPayload,
    VerifyTaskPayload,
    WriteFilePayload,
    WriteTarget,
    thaw_json_value,
    utf8_byte_length,
    validate_safe_single_segment,
)
from forecontract_coding.process import run_command_in_job
from forecontract_coding.workspace import (
    FileSnapshot,
    ManagedWorkspace,
    ScopeViolation,
    collect_and_verify_acceptance_sources,
    ensure_clean_subdir,
    sha256_bytes,
    validate_task_and_config_paths,
)

__all__ = ["CodingAdapter"]

_PAYLOAD_MODELS: dict[str, type] = {
    "read_file": ReadFilePayload,
    "search_text": SearchTextPayload,
    "write_file": WriteFilePayload,
    "run_command": RunCommandPayload,
    "verify_task": VerifyTaskPayload,
}


def _truncate_utf8(text: str, limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    cut = encoded[:limit]
    while cut:
        try:
            return cut.decode("utf-8"), True
        except UnicodeDecodeError:
            cut = cut[:-1]
    return "", True


def _resolve_result_dir_arg(spec: CommandSpec, result_dir: Path) -> tuple[tuple[str, ...], Path]:
    """Bind ``{result_dir}`` only in command args and resolve JUnit XML report location."""
    bound_args: list[str] = []
    junit_path: Path | None = None
    result_str = str(result_dir)
    for arg in spec.args:
        replaced = arg.replace("{result_dir}", result_str)
        bound_args.append(replaced)
        if "{result_dir}" in arg:
            candidate_str = replaced
            if candidate_str.startswith("--junitxml="):
                candidate_str = candidate_str.split("=", 1)[1]
            if candidate_str.endswith(".xml"):
                junit_path = Path(candidate_str)
    if junit_path is None:
        junit_path = result_dir / "junit.xml"
    # Verify resolved report path sits strictly inside result_dir
    real_dir = Path(os.path.realpath(result_dir))
    real_junit = Path(os.path.realpath(junit_path))
    if real_junit == real_dir or real_dir not in real_junit.parents:
        raise CodingError(
            CODING_VERIFICATION_INVALID,
            "JUnit report path must reside inside the allocated result_dir",
            location=("command", spec.command_id, "junit_path"),
        )
    return (spec.executable, *bound_args), junit_path


def _verify_junit_report(
    *,
    junit_path: Path,
    spec: CommandSpec,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Parse JUnit XML using stdlib ElementTree and enforce exact non-empty test-set match."""
    if not junit_path.exists() or junit_path.is_symlink() or not junit_path.is_file():
        return False, "missing_junit_report", {"report_valid": False, "tests_executed": 0}
    try:
        raw = junit_path.read_bytes()
        root = ET.fromstring(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ET.ParseError):
        return False, "malformed_junit_report", {"report_valid": False, "tests_executed": 0}

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        return False, "unexpected_junit_root", {"report_valid": False, "tests_executed": 0}

    for suite in suites:
        for attr in ("failures", "errors", "skipped"):
            raw_count = suite.attrib.get(attr)
            if raw_count is not None:
                try:
                    if int(raw_count) != 0:
                        return (
                            False,
                            f"junit_suite_{attr}_nonzero",
                            {"report_valid": True, "tests_executed": len(root.findall(".//testcase"))},
                        )
                except ValueError:
                    return False, "invalid_junit_suite_count", {"report_valid": False}

    testcases = root.findall(".//testcase")
    if not testcases:
        return False, "zero_tests_executed", {"report_valid": True, "tests_executed": 0}

    observed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    failures = 0
    errors = 0
    skipped = 0
    for tc in testcases:
        cls_name = tc.attrib.get("classname", "")
        tc_name = tc.attrib.get("name", "")
        if not cls_name or not tc_name:
            return (
                False,
                "missing_testcase_identity",
                {"report_valid": False, "tests_executed": len(testcases)},
            )
        key = (cls_name, tc_name)
        if key in seen:
            return (
                False,
                "duplicate_testcase_identity",
                {"report_valid": False, "tests_executed": len(testcases)},
            )
        seen.add(key)
        observed.append(key)
        if tc.find("failure") is not None:
            failures += 1
        if tc.find("error") is not None:
            errors += 1
        if tc.find("skipped") is not None:
            skipped += 1

    expected = {(item.classname, item.name) for item in spec.expected_tests}
    if failures > 0 or errors > 0:
        return (
            False,
            "testcase_failures_present",
            {
                "report_valid": True,
                "tests_executed": len(observed),
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
            },
        )
    if skipped > 0:
        return (
            False,
            "skipped_tests_forbidden",
            {
                "report_valid": True,
                "tests_executed": len(observed),
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
            },
        )
    if seen != expected:
        return (
            False,
            "expected_test_set_mismatch",
            {
                "report_valid": True,
                "tests_executed": len(observed),
                "expected_count": len(expected),
                "observed_count": len(seen),
            },
        )

    report_sha = sha256_bytes(raw)
    return (
        True,
        None,
        {
            "report_valid": True,
            "report_sha256": report_sha,
            "tests_executed": len(observed),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "verified_tests": [
                {"classname": c, "name": n} for c, n in sorted(expected)
            ],
        },
    )


class CodingAdapter:
    """Real Windows file and Job-Object subprocess Adapter implementing Core's ``Adapter``."""

    def __init__(self, *, task: CodingTask, config: CodingConfig) -> None:
        self._task = CodingTask.model_validate(task)
        self._config = CodingConfig.model_validate(config)
        validate_task_and_config_paths(self._task, self._config)
        src_hashes, by_cmd = collect_and_verify_acceptance_sources(
            self._task, self._config
        )
        self._acceptance_source_baseline: dict[str, str] = src_hashes
        self._acceptance_sources_by_cmd: dict[str, tuple[str, ...]] = by_cmd
        self._session_id: str = uuid4().hex[:12]
        self._write_map: dict[str, WriteTarget] = {
            target.path: target for target in self._task.write_paths
        }
        self._read_set: frozenset[str] = frozenset(self._task.read_paths)
        self._workspace: ManagedWorkspace | None = None
        self._scope_baseline: dict[str, str] = {}
        self._baseline_snapshots: dict[str, FileSnapshot] = {}
        self._baseline_revision: str | None = None
        self._executed_action_ids: set[str] = set()
        self._sequence: int = 0
        self._history: list[dict[str, Any]] = []
        self._last_tool_result: dict[str, Any] | None = None
        self._acceptance = AcceptanceSummary(
            status="not_run",
            revision=None,
            action_id=None,
            command_ids=list(self._task.verification_command_ids),
            commands_digest=self._task.commands_digest(),
            stale_reason=None,
            results=[],
        )
        self._started = False
        self._stopped = False

    @property
    def name(self) -> str:
        return CODING_ADAPTER_NAME

    @property
    def baseline_revision(self) -> str | None:
        return self._baseline_revision

    def current_workspace_revision(self) -> str:
        """Compute current workspace revision consistent with ``observe()``."""
        if self._workspace is None:
            ws = ManagedWorkspace(task=self._task, config=self._config)
            baseline = ws.capture_scope_baseline()
            _, snaps, _ = ws.snapshot()
            return ws.compute_full_revision(
                snapshots=snaps, inventory=baseline, scope_ok=True
            )
        rev, _, _, _ = self._inspect_workspace_state()
        return rev

    @property
    def acceptance_summary(self) -> AcceptanceSummary:
        return self._acceptance.model_copy(deep=True)

    def current_changed_paths(self) -> list[ChangedPathEntry]:
        """Compute sorted baseline-to-current diffs for managed writable paths."""
        if self._workspace is None:
            return []
        _, snapshots, _, _ = self._inspect_workspace_state()
        return self._compute_changed_paths(snapshots)

    def _compute_changed_paths(
        self, snapshots: tuple[FileSnapshot, ...]
    ) -> list[ChangedPathEntry]:
        snap_by_path = {item.path: item for item in snapshots}
        entries: list[ChangedPathEntry] = []
        for path in sorted(self._write_map):
            target = self._write_map[path]
            before = self._baseline_snapshots.get(path)
            after = snap_by_path.get(path)
            before_exists = before.exists if before is not None else False
            after_exists = after.exists if after is not None else False
            before_sha = before.sha256 if before is not None else None
            after_sha = after.sha256 if after is not None else None
            if before_exists != after_exists or before_sha != after_sha:
                entries.append(
                    ChangedPathEntry(
                        path=path,
                        operation=target.operation,
                        before_exists=before_exists,
                        after_exists=after_exists,
                        before_sha256=before_sha,
                        after_sha256=after_sha,
                    )
                )
        return entries

    def capabilities(self) -> AdapterCapabilities:
        """Pure declaration of supported Coding capabilities (zero I/O)."""
        return AdapterCapabilities(
            schema_version=SCHEMA_VERSION,
            adapter_name=CODING_ADAPTER_NAME,
            environment_type=CODING_ENVIRONMENT_TYPE,
            supported_actions=list(SUPPORTED_ACTIONS),
            supported_observation_types=["coding_state"],
            supported_condition_types=list(SUPPORTED_CONDITION_TYPES),
            supported_outcome_types=list(SUPPORTED_OUTCOME_TYPES),
        )

    def validate_action(self, action: ActionSpec) -> None:
        """Synchronous, I/O-free validation of action type, unique ``action_id``, and payload."""
        if not isinstance(action, ActionSpec):
            raise InvalidActionPayloadError("ActionSpec instance is required")
        try:
            validated = ActionSpec.model_validate(action.model_dump(mode="python"))
        except ValidationError as exc:
            raise InvalidActionPayloadError("invalid ActionSpec structure") from exc

        if validated.action_type not in SUPPORTED_ACTIONS:
            raise UnsupportedActionError(
                f"unsupported coding action_type: {validated.action_type}"
            )

        meta = validated.metadata
        if not isinstance(meta, Mapping) or set(meta.keys()) != {"action_id"}:
            raise InvalidActionPayloadError(
                "ActionSpec.metadata must contain only non-blank action_id"
            )
        action_id = meta.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise InvalidActionPayloadError("ActionSpec.metadata.action_id must be non-blank")
        try:
            validate_safe_single_segment(action_id, field_name="action_id")
        except (TypeError, ValueError) as exc:
            raise InvalidActionPayloadError(str(exc)) from None
        if action_id in self._executed_action_ids:
            raise InvalidActionPayloadError(
                f"duplicate/replayed action_id is forbidden: {action_id}"
            )

        payload_model = _PAYLOAD_MODELS[validated.action_type]
        raw_payload = (
            thaw_json_value(validated.payload)
            if isinstance(action.payload, Mapping) and not isinstance(action.payload, dict)
            else validated.payload
        )
        try:
            parsed = payload_model.model_validate(raw_payload)
        except ValidationError as exc:
            raise InvalidActionPayloadError(
                f"invalid payload for {validated.action_type}"
            ) from exc

        if validated.action_type == "read_file":
            assert isinstance(parsed, ReadFilePayload)
            if parsed.path not in self._read_set:
                raise InvalidActionPayloadError("read_file path is outside task.read_paths")
        elif validated.action_type == "search_text":
            assert isinstance(parsed, SearchTextPayload)
            for item in parsed.paths:
                if item not in self._read_set:
                    raise InvalidActionPayloadError(
                        "search_text paths must be a subset of task.read_paths"
                    )
        elif validated.action_type == "write_file":
            assert isinstance(parsed, WriteFilePayload)
            target = self._write_map.get(parsed.path)
            if target is None:
                raise InvalidActionPayloadError("write_file path is outside task.write_paths")
            if target.operation == "create" and parsed.expected_sha256 is not None:
                raise InvalidActionPayloadError(
                    "create write_file requires expected_sha256=null"
                )
            if target.operation == "modify" and parsed.expected_sha256 is None:
                raise InvalidActionPayloadError(
                    "modify write_file requires expected_sha256"
                )
        elif validated.action_type == "run_command":
            assert isinstance(parsed, RunCommandPayload)
            if parsed.command_id not in self._task.commands:
                raise InvalidActionPayloadError("unknown command_id")

    async def start(self) -> None:
        if self._started and not self._stopped:
            return
        sources_ok, reason, _ = self._verify_acceptance_sources()
        if not sources_ok:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                f"external acceptance test source invalid at start: {reason}",
                location=("task", "verification_command_ids"),
            )
        out_root = Path(self._config.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        ensure_clean_subdir(out_root, ())
        self._workspace = ManagedWorkspace(task=self._task, config=self._config)
        self._scope_baseline = self._workspace.capture_scope_baseline()
        rev, snapshots, _, _ = self._inspect_workspace_state()
        self._baseline_snapshots = {item.path: item for item in snapshots}
        self._baseline_revision = rev
        self._started = True
        self._stopped = False

    async def stop(self) -> None:
        self._stopped = True

    def _verify_acceptance_sources(self) -> tuple[bool, str | None, dict[str, str]]:
        """Re-hash the pinned startup acceptance sources; never silently rebind or skip."""
        observed: dict[str, str] = {}
        all_ok = True
        for key, baseline_sha in self._acceptance_source_baseline.items():
            src = Path(key)
            if not os.path.lexists(src):
                all_ok = False
                continue
            try:
                st = os.lstat(src)
                attrs = getattr(st, "st_file_attributes", 0)
                if (
                    src.is_symlink()
                    or (attrs & 0x0400)
                    or not stat.S_ISREG(st.st_mode)
                    or st.st_nlink > 1
                ):
                    all_ok = False
                    continue
                live_sha = sha256_bytes(src.read_bytes())
            except OSError:
                all_ok = False
                continue
            observed[key] = live_sha
            if live_sha != baseline_sha:
                all_ok = False
        return all_ok, (None if all_ok else "test_source_changed"), observed

    def _inspect_workspace_state(
        self,
    ) -> tuple[str, tuple[FileSnapshot, ...], bool, tuple[ScopeViolation, ...]]:
        """Inspect managed slots and bounded full workspace inventory safely."""
        assert self._workspace is not None
        scope_ok, violations, live_inventory = self._workspace.inspect_scope(
            self._scope_baseline
        )
        try:
            _, snapshots, _ = self._workspace.snapshot()
        except CodingError as exc:
            scope_ok = False
            snap_list: list[FileSnapshot] = []
            for norm in self._task.managed_paths():
                wt = self._write_map.get(norm)
                live_sha = live_inventory.get(norm)
                snap_list.append(
                    FileSnapshot(
                        path=norm,
                        exists=(live_sha is not None),
                        sha256=live_sha,
                        byte_length=0,
                        readable=(norm in self._read_set),
                        writable=(wt is not None),
                        operation=(wt.operation if wt is not None else None),
                    )
                )
            snapshots = tuple(snap_list)
            if not violations:
                violations = (
                    ScopeViolation(
                        path=".",
                        reason="scope_scan_failed",
                        before_sha256=None,
                        after_sha256=None,
                    ),
                )
                _ = exc
        rev = self._workspace.compute_full_revision(
            snapshots=snapshots,
            inventory=live_inventory,
            scope_ok=scope_ok,
        )
        return rev, snapshots, scope_ok, violations

    def _invalidate_stale_acceptance(
        self,
        current_revision: str,
        *,
        sources_ok: bool = True,
        scope_ok: bool = True,
    ) -> None:
        if self._acceptance.status != "passed":
            return
        if not sources_ok:
            self._acceptance = self._acceptance.model_copy(
                update={"status": "unknown", "stale_reason": "test_source_changed"}
            )
            return
        if not scope_ok:
            self._acceptance = self._acceptance.model_copy(
                update={"status": "unknown", "stale_reason": "scope_violation"}
            )
            return
        if self._acceptance.revision != current_revision:
            self._acceptance = self._acceptance.model_copy(
                update={
                    "status": "unknown",
                    "stale_reason": "workspace_revision_changed",
                }
            )

    def _append_history(self, entry: dict[str, Any]) -> None:
        excerpt = str(entry.get("excerpt", ""))
        bounded_excerpt, excerpt_truncated = _truncate_utf8(
            excerpt, self._config.history_excerpt_bytes
        )
        stored = dict(entry)
        stored["excerpt"] = bounded_excerpt
        stored["excerpt_truncated"] = bool(entry.get("excerpt_truncated", False) or excerpt_truncated)
        self._history.append(stored)

    def _bounded_history_payload(
        self, limit: int
    ) -> tuple[list[dict[str, Any]], bool, int]:
        total = len(self._history)
        if total == 0:
            return [], False, 0
        if limit >= total:
            return list(self._history), False, 0
        selected = list(self._history[-limit:])
        # Preserve the latest structured failure if it would be dropped from tail window
        selected_seqs = {item["sequence"] for item in selected}
        failures = [item for item in self._history if not item.get("succeeded", True)]
        if failures and failures[-1]["sequence"] not in selected_seqs and limit >= 1:
            if limit == 1:
                selected = [failures[-1]]
            else:
                selected = [failures[-1], *self._history[-(limit - 1) :]]
        omitted = total - len(selected)
        return selected, omitted > 0, omitted

    def _build_observation_record(
        self,
        *,
        obs_id: str,
        now: datetime,
        state: dict[str, Any],
    ) -> Observation:
        return Observation(
            schema_version=SCHEMA_VERSION,
            observation_id=obs_id,
            environment_type=CODING_ENVIRONMENT_TYPE,
            state=state,
            timestamp=now,
            metadata={"state_type": "coding_state"},
        )

    def _compact_tool_result(self, result: dict[str, Any], max_text_bytes: int) -> dict[str, Any]:
        compacted = dict(result)
        for text_key, trunc_key in (
            ("content", "truncated"),
            ("stdout", "stdout_truncated"),
            ("stderr", "stderr_truncated"),
        ):
            val = compacted.get(text_key)
            if isinstance(val, str):
                trimmed, was_cut = _truncate_utf8(val, max_text_bytes)
                if was_cut:
                    compacted[text_key] = trimmed
                    compacted[trunc_key] = True
        if isinstance(compacted.get("matches"), list) and max_text_bytes < 512:
            compacted["matches"] = compacted["matches"][:5]
            compacted["truncated"] = True
        return compacted

    async def observe(self) -> Observation:
        if not self._started or self._workspace is None:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "CodingAdapter.start() must run before observe()",
                location=("adapter", "observe"),
            )
        rev, snapshots, scope_ok, violations = self._inspect_workspace_state()
        sources_ok, _, _ = self._verify_acceptance_sources()
        self._invalidate_stale_acceptance(
            rev, sources_ok=sources_ok, scope_ok=scope_ok
        )
        changed = [entry.model_dump(mode="json") for entry in self._compute_changed_paths(snapshots)]
        violation_dicts = [
            ScopeViolationEntry(
                path=v.path,
                reason=v.reason,  # type: ignore[arg-type]
                before_sha256=v.before_sha256,
                after_sha256=v.after_sha256,
            ).model_dump(mode="json")
            for v in violations
        ]
        files_list = [
            {
                "path": s.path,
                "exists": s.exists,
                "sha256": s.sha256,
                "bytes": s.byte_length,
                "readable": s.readable,
                "writable": s.writable,
                "operation": s.operation,
            }
            for s in snapshots
        ]

        obs_id = f"obs-{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        history_limit = self._config.max_history_entries
        text_limit = self._config.max_output_bytes
        while True:
            hist_entries, hist_trunc, omitted = self._bounded_history_payload(history_limit)
            last_result = (
                self._compact_tool_result(self._last_tool_result, text_limit)
                if self._last_tool_result is not None
                else None
            )
            state: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "workspace_revision": rev,
                "baseline_workspace_revision": self._baseline_revision,
                "scope_ok": scope_ok,
                "scope_violations": violation_dicts,
                "acceptance_sources_ok": sources_ok,
                "acceptance_source_hashes": dict(self._acceptance_source_baseline),
                "files": files_list,
                "changed_paths": changed,
                "acceptance": self._acceptance.model_dump(mode="json"),
                "last_tool_result": last_result,
                "history": hist_entries,
                "history_truncated": hist_trunc,
                "omitted_history_count": omitted,
                "observation_overflow": False,
            }
            candidate = self._build_observation_record(obs_id=obs_id, now=now, state=state)
            size = utf8_byte_length(json.dumps(candidate.model_dump(mode="json"), allow_nan=False))
            if size <= self._config.max_observation_bytes:
                return candidate
            if text_limit > 256:
                text_limit = max(256, text_limit // 4)
                continue
            if history_limit > 0:
                history_limit = 0 if history_limit == 1 else max(1, history_limit // 2)
                continue
            break

        # Irreducible state still exceeded max_observation_bytes: return explicit incomplete
        # observation rather than silently omitting authorization/acceptance proof and passing.
        minimal_last = None
        if self._last_tool_result is not None:
            minimal_last = {
                "action_id": self._last_tool_result.get("action_id"),
                "action_type": self._last_tool_result.get("action_type"),
                "tool_succeeded": self._last_tool_result.get("tool_succeeded"),
                "error_code": self._last_tool_result.get("error_code"),
                "exit_code": self._last_tool_result.get("exit_code"),
                "sequence": self._last_tool_result.get("sequence"),
            }
        overflow_violations = [
            {"path": item["path"], "reason": item["reason"]}
            for item in violation_dicts[:4]
        ]
        overflow_state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "workspace_revision": rev,
            "baseline_workspace_revision": self._baseline_revision,
            "scope_ok": scope_ok,
            "scope_violations": overflow_violations,
            "acceptance_sources_ok": sources_ok,
            "observation_overflow": True,
            "max_observation_bytes": self._config.max_observation_bytes,
            "measured_bytes": size,
            "acceptance": {
                "status": "unknown",
                "revision": self._acceptance.revision,
                "action_id": self._acceptance.action_id,
                "command_ids": list(self._acceptance.command_ids),
                "commands_digest": self._acceptance.commands_digest,
                "stale_reason": "observation_overflow",
                "results": [],
            },
            "last_tool_result": minimal_last,
            "history": [],
            "history_truncated": True,
            "omitted_history_count": len(self._history),
        }
        overflow_obs = self._build_observation_record(
            obs_id=obs_id, now=now, state=overflow_state
        )
        overflow_bytes = utf8_byte_length(
            json.dumps(overflow_obs.model_dump(mode="json"), allow_nan=False)
        )
        if overflow_bytes > self._config.max_observation_bytes:
            raise CodingError(
                CODING_OBSERVATION_OVERFLOW,
                "observation exceeds tightened max_observation_bytes even in minimal overflow form",
                location=("observation",),
            )
        return overflow_obs

    @staticmethod
    def _encode_command_segment(command_id: str) -> str:
        """Encode any logical ``command_id`` into a deterministic host-safe directory segment."""
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError("command_id must be a non-empty, non-blank string")
        digest = hashlib.sha256(command_id.encode("utf-8")).hexdigest()[:12]
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", command_id).strip("_-")[:24].rstrip("_-").lower()
        if not cleaned or cleaned in {
            "con",
            "prn",
            "aux",
            "nul",
            *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10)),
        }:
            cleaned = "cmd"
        return validate_safe_single_segment(f"cmd-{cleaned}-{digest}", field_name="command_id")

    def _allocate_result_dir(
        self, *, kind: str, action_id: str, command_id: str, seq: int
    ) -> Path:
        try:
            safe_action = validate_safe_single_segment(action_id, field_name="action_id")
            safe_cmd = self._encode_command_segment(command_id)
        except (TypeError, ValueError) as exc:
            raise CodingError(
                CODING_INPUT_INVALID,
                str(exc),
                location=("action", action_id),
            ) from None
        out_root = Path(self._config.output_root)
        kind_dir = ensure_clean_subdir(out_root, (kind,))
        result_dir = kind_dir / f"{safe_action}__{safe_cmd}__{self._session_id}__{seq:04d}"
        result_dir.mkdir(exist_ok=False)
        real_out = Path(os.path.realpath(out_root))
        real_res = Path(os.path.realpath(result_dir))
        if real_out not in real_res.parents:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "allocated result_dir escapes output_root",
                location=("output_root",),
            )
        return result_dir

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        if not self._started or self._workspace is None:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "CodingAdapter.start() must run before execute()",
                location=("adapter", "execute"),
            )
        self.validate_action(action)
        action_id = str(action.metadata["action_id"])
        # Record immediately before executing so replay is impossible even on error/cancel.
        self._executed_action_ids.add(action_id)
        self._sequence += 1
        seq = self._sequence
        started_ns = time.monotonic_ns()
        before_rev, _, _, _ = self._inspect_workspace_state()
        outcome_id = f"out-{action_id}-{seq}"

        raw_payload = thaw_json_value(action.payload)
        if action.action_type == "read_file":
            payload = ReadFilePayload.model_validate(raw_payload)
            try:
                chunk = self._workspace.read_file(
                    path=payload.path,
                    start_line=payload.start_line,
                    max_lines=payload.max_lines,
                    line_byte_offset=payload.line_byte_offset,
                    expected_sha256=payload.expected_sha256,
                )
            except CodingError as exc:
                after_rev, _, _, _ = self._inspect_workspace_state()
                dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "read_file",
                    "sequence": seq,
                    "path": payload.path,
                    "line_byte_offset": payload.line_byte_offset,
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "read_file",
                        "target": payload.path,
                        "succeeded": False,
                        "error_code": exc.code,
                        "excerpt": exc.message,
                    }
                )
                return ActualOutcome(
                    schema_version=SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    status=ExecutionStatus.FAILED,
                    state_before={"workspace_revision": before_rev},
                    state_after={"workspace_revision": after_rev},
                    error=ErrorInfo(
                        schema_version=SCHEMA_VERSION,
                        code=exc.code,
                        message=exc.message,
                        recoverable=False,
                    ),
                    duration_ms=dur,
                    metadata={"action_id": action_id, "sequence": seq},
                )
            after_rev, _, _, _ = self._inspect_workspace_state()
            dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
            self._last_tool_result = {
                "action_id": action_id,
                "action_type": "read_file",
                "sequence": seq,
                "path": chunk.path,
                "file_sha256": chunk.sha256,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "total_lines": chunk.total_lines,
                "line_byte_offset": payload.line_byte_offset,
                "content": chunk.content,
                "truncated": chunk.truncated,
                "next_start_line": chunk.next_start_line,
                "next_line_byte_offset": chunk.next_line_byte_offset,
                "before_revision": before_rev,
                "after_revision": after_rev,
                "tool_succeeded": True,
            }
            self._append_history(
                {
                    "sequence": seq,
                    "action_id": action_id,
                    "action_type": "read_file",
                    "target": chunk.path,
                    "succeeded": True,
                    "excerpt": f"read {chunk.path} lines {chunk.start_line}..{chunk.end_line} offset={payload.line_byte_offset} sha={chunk.sha256[:12]}",
                }
            )
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.SUCCEEDED,
                state_before={"workspace_revision": before_rev},
                state_after={
                    "workspace_revision": after_rev,
                    "action_id": action_id,
                    "file_sha256": chunk.sha256,
                },
                duration_ms=dur,
                metadata={"action_id": action_id, "sequence": seq},
            )

        if action.action_type == "search_text":
            payload = SearchTextPayload.model_validate(raw_payload)
            try:
                res = self._workspace.search_text(
                    query=payload.query,
                    paths=payload.paths,
                    max_matches=payload.max_matches,
                )
            except CodingError as exc:
                after_rev, _, _, _ = self._inspect_workspace_state()
                dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "search_text",
                    "sequence": seq,
                    "paths": list(payload.paths),
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "search_text",
                        "target": ",".join(payload.paths),
                        "succeeded": False,
                        "error_code": exc.code,
                        "excerpt": exc.message,
                    }
                )
                return ActualOutcome(
                    schema_version=SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    status=ExecutionStatus.FAILED,
                    state_before={"workspace_revision": before_rev},
                    state_after={"workspace_revision": after_rev},
                    error=ErrorInfo(
                        schema_version=SCHEMA_VERSION,
                        code=exc.code,
                        message=exc.message,
                        recoverable=False,
                    ),
                    duration_ms=dur,
                    metadata={"action_id": action_id, "sequence": seq},
                )
            after_rev, _, _, _ = self._inspect_workspace_state()
            dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
            match_dicts = [
                {
                    "path": m.path,
                    "line_number": m.line_number,
                    "column": m.column,
                    "line_text": m.line_text,
                    "file_sha256": m.file_sha256,
                }
                for m in res.matches
            ]
            self._last_tool_result = {
                "action_id": action_id,
                "action_type": "search_text",
                "sequence": seq,
                "query": res.query,
                "searched_paths": list(res.searched_paths),
                "matches": match_dicts,
                "truncated": res.truncated,
                "next_cursor": res.next_cursor,
                "before_revision": before_rev,
                "after_revision": after_rev,
                "tool_succeeded": True,
            }
            self._append_history(
                {
                    "sequence": seq,
                    "action_id": action_id,
                    "action_type": "search_text",
                    "target": ",".join(res.searched_paths),
                    "succeeded": True,
                    "excerpt": f"matches={len(match_dicts)} truncated={res.truncated}",
                }
            )
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.SUCCEEDED,
                state_before={"workspace_revision": before_rev},
                state_after={"workspace_revision": after_rev, "action_id": action_id},
                duration_ms=dur,
                metadata={"action_id": action_id, "sequence": seq},
            )

        if action.action_type == "write_file":
            payload = WriteFilePayload.model_validate(raw_payload)
            try:
                wo = self._workspace.write_file(
                    path=payload.path,
                    expected_sha256=payload.expected_sha256,
                    content=payload.content,
                )
            except CodingError as exc:
                after_rev, _, _, _ = self._inspect_workspace_state()
                dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "write_file",
                    "sequence": seq,
                    "path": payload.path,
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "write_file",
                        "target": payload.path,
                        "succeeded": False,
                        "error_code": exc.code,
                        "excerpt": exc.message,
                    }
                )
                return ActualOutcome(
                    schema_version=SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    status=ExecutionStatus.FAILED,
                    state_before={"workspace_revision": before_rev},
                    state_after={"workspace_revision": after_rev},
                    error=ErrorInfo(
                        schema_version=SCHEMA_VERSION,
                        code=exc.code,
                        message=exc.message,
                        recoverable=True,
                    ),
                    duration_ms=dur,
                    metadata={"action_id": action_id, "sequence": seq},
                )
            after_rev, _, write_scope_ok, _ = self._inspect_workspace_state()
            self._invalidate_stale_acceptance(after_rev, scope_ok=write_scope_ok)
            dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
            self._last_tool_result = {
                "action_id": action_id,
                "action_type": "write_file",
                "sequence": seq,
                "path": wo.path,
                "operation": wo.operation,
                "before_sha256": wo.before_sha256,
                "written_sha256": wo.written_sha256,
                "read_back_sha256": wo.read_back_sha256,
                "read_back_match": wo.read_back_match,
                "before_revision": before_rev,
                "after_revision": after_rev,
                "tool_succeeded": wo.read_back_match,
            }
            self._append_history(
                {
                    "sequence": seq,
                    "action_id": action_id,
                    "action_type": "write_file",
                    "target": wo.path,
                    "succeeded": wo.read_back_match,
                    "excerpt": f"write {wo.path} {wo.operation} sha={wo.read_back_sha256[:12]}",
                }
            )
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.SUCCEEDED
                if wo.read_back_match
                else ExecutionStatus.FAILED,
                state_before={"workspace_revision": before_rev},
                state_after={
                    "workspace_revision": after_rev,
                    "path": wo.path,
                    "written_sha256": wo.written_sha256,
                    "read_back_sha256": wo.read_back_sha256,
                    "read_back_match": wo.read_back_match,
                },
                changes={
                    "path": wo.path,
                    "operation": wo.operation,
                    "before_sha256": wo.before_sha256,
                    "after_sha256": wo.read_back_sha256,
                },
                duration_ms=dur,
                metadata={"action_id": action_id, "sequence": seq},
            )

        if action.action_type == "run_command":
            payload = RunCommandPayload.model_validate(raw_payload)
            spec = self._task.commands[payload.command_id]
            result_dir = self._allocate_result_dir(
                kind="cmd_results",
                action_id=action_id,
                command_id=spec.command_id,
                seq=seq,
            )
            argv, _ = _resolve_result_dir_arg(spec, result_dir)
            temp_dir = ensure_clean_subdir(Path(self._config.output_root), ("temp",))
            try:
                proc_res = await run_command_in_job(
                    spec=spec,
                    argv=argv,
                    temp_dir=temp_dir,
                    max_output_bytes=self._config.max_output_bytes,
                    process_cleanup_ms=self._config.process_cleanup_ms,
                )
            except asyncio.CancelledError:
                after_rev, _, after_scope_ok, _ = self._inspect_workspace_state()
                sources_ok, _, _ = self._verify_acceptance_sources()
                self._invalidate_stale_acceptance(
                    after_rev, sources_ok=sources_ok, scope_ok=after_scope_ok
                )
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "run_command",
                    "sequence": seq,
                    "command_id": spec.command_id,
                    "argv": list(argv),
                    "exit_code": None,
                    "timed_out": False,
                    "cancelled": True,
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": "CODING_PROCESS_CANCELLED",
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "run_command",
                        "target": spec.command_id,
                        "succeeded": False,
                        "error_code": "CODING_PROCESS_CANCELLED",
                        "excerpt": f"cancelled {spec.command_id}",
                    }
                )
                raise
            except CodingError as exc:
                after_rev, _, after_scope_ok, _ = self._inspect_workspace_state()
                sources_ok, _, _ = self._verify_acceptance_sources()
                self._invalidate_stale_acceptance(
                    after_rev, sources_ok=sources_ok, scope_ok=after_scope_ok
                )
                dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "run_command",
                    "sequence": seq,
                    "command_id": spec.command_id,
                    "argv": list(argv),
                    "exit_code": None,
                    "timed_out": False,
                    "cancelled": False,
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "run_command",
                        "target": spec.command_id,
                        "succeeded": False,
                        "error_code": exc.code,
                        "excerpt": exc.message,
                    }
                )
                status = (
                    ExecutionStatus.UNKNOWN
                    if exc.code == CODING_PROCESS_CLEANUP_FAILED
                    else ExecutionStatus.FAILED
                )
                return ActualOutcome(
                    schema_version=SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    status=status,
                    state_before={"workspace_revision": before_rev},
                    state_after={"workspace_revision": after_rev},
                    error=ErrorInfo(
                        schema_version=SCHEMA_VERSION,
                        code=exc.code,
                        message=exc.message,
                        recoverable=False,
                    ),
                    duration_ms=dur,
                    metadata={"action_id": action_id, "sequence": seq},
                )

            after_rev, _, after_scope_ok, _ = self._inspect_workspace_state()
            sources_ok, _, _ = self._verify_acceptance_sources()
            self._invalidate_stale_acceptance(
                after_rev, sources_ok=sources_ok, scope_ok=after_scope_ok
            )
            if proc_res.timed_out or not proc_res.cleanup_ok:
                err_code = (
                    CODING_PROCESS_CLEANUP_FAILED
                    if not proc_res.cleanup_ok
                    else CODING_PROCESS_TIMEOUT
                )
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "run_command",
                    "sequence": seq,
                    "command_id": spec.command_id,
                    "argv": list(proc_res.argv),
                    "exit_code": proc_res.exit_code,
                    "stdout": proc_res.stdout,
                    "stderr": proc_res.stderr,
                    "stdout_truncated": proc_res.stdout_truncated,
                    "stderr_truncated": proc_res.stderr_truncated,
                    "timed_out": proc_res.timed_out,
                    "cancelled": proc_res.cancelled,
                    "cleanup_ok": proc_res.cleanup_ok,
                    "before_revision": before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "error_code": err_code,
                }
                self._append_history(
                    {
                        "sequence": seq,
                        "action_id": action_id,
                        "action_type": "run_command",
                        "target": spec.command_id,
                        "succeeded": False,
                        "error_code": err_code,
                        "excerpt": f"cmd={spec.command_id} timed_out={proc_res.timed_out} cleanup_ok={proc_res.cleanup_ok}",
                    }
                )
                return ActualOutcome(
                    schema_version=SCHEMA_VERSION,
                    outcome_id=outcome_id,
                    status=ExecutionStatus.UNKNOWN,
                    state_before={"workspace_revision": before_rev},
                    state_after={"workspace_revision": after_rev, "effect_unknown": True},
                    error=ErrorInfo(
                        schema_version=SCHEMA_VERSION,
                        code=err_code,
                        message="command timed out or cleanup incomplete; side effects unknown",
                        recoverable=False,
                    ),
                    duration_ms=proc_res.duration_ms,
                    metadata={"action_id": action_id, "sequence": seq},
                )

            # Normal exit (e.g., 0 or 1): tool execution SUCCEEDED; failure diagnostics travel in last_tool_result
            self._last_tool_result = {
                "action_id": action_id,
                "action_type": "run_command",
                "sequence": seq,
                "command_id": spec.command_id,
                "argv": list(proc_res.argv),
                "exit_code": proc_res.exit_code,
                "stdout": proc_res.stdout,
                "stderr": proc_res.stderr,
                "stdout_truncated": proc_res.stdout_truncated,
                "stderr_truncated": proc_res.stderr_truncated,
                "stdout_bytes_seen": proc_res.stdout_bytes_seen,
                "stderr_bytes_seen": proc_res.stderr_bytes_seen,
                "timed_out": False,
                "cancelled": False,
                "cleanup_ok": True,
                "before_revision": before_rev,
                "after_revision": after_rev,
                "tool_succeeded": True,
            }
            diag_snippet = proc_res.stdout or proc_res.stderr
            self._append_history(
                {
                    "sequence": seq,
                    "action_id": action_id,
                    "action_type": "run_command",
                    "target": spec.command_id,
                    "succeeded": (proc_res.exit_code == 0),
                    "exit_code": proc_res.exit_code,
                    "excerpt": f"cmd={spec.command_id} exit={proc_res.exit_code} {diag_snippet}",
                }
            )
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.SUCCEEDED,
                state_before={"workspace_revision": before_rev},
                state_after={
                    "workspace_revision": after_rev,
                    "command_id": spec.command_id,
                    "exit_code": proc_res.exit_code,
                },
                duration_ms=proc_res.duration_ms,
                metadata={"action_id": action_id, "sequence": seq},
            )

        # verify_task
        VerifyTaskPayload.model_validate(raw_payload)
        verify_before_rev, _, pre_scope_ok, _ = self._inspect_workspace_state()
        pre_sources_ok, pre_source_reason, _ = self._verify_acceptance_sources()
        if not pre_sources_ok or not pre_scope_ok:
            dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
            err_code = CODING_VERIFICATION_INVALID if not pre_sources_ok else CODING_SCOPE_VIOLATION
            stale_why = pre_source_reason if not pre_sources_ok else "scope_violation"
            err_msg = (
                "external acceptance test source changed or invalid; refusing verification run"
                if not pre_sources_ok
                else "workspace scope violation detected; refusing verification run"
            )
            self._acceptance = AcceptanceSummary(
                status="unknown",
                revision=verify_before_rev,
                action_id=action_id,
                command_ids=list(self._task.verification_command_ids),
                commands_digest=self._task.commands_digest(),
                stale_reason=stale_why,
                results=[],
            )
            self._last_tool_result = {
                "action_id": action_id,
                "action_type": "verify_task",
                "sequence": seq,
                "acceptance_status": "unknown",
                "before_revision": verify_before_rev,
                "after_revision": verify_before_rev,
                "command_ids": list(self._task.verification_command_ids),
                "stdout": err_msg,
                "stdout_truncated": False,
                "tool_succeeded": False,
                "error_code": err_code,
            }
            self._append_history(
                {
                    "sequence": seq,
                    "action_id": action_id,
                    "action_type": "verify_task",
                    "target": ",".join(self._task.verification_command_ids),
                    "succeeded": False,
                    "error_code": err_code,
                    "excerpt": f"verify=unknown {stale_why}",
                }
            )
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.FAILED,
                state_before={"workspace_revision": verify_before_rev},
                state_after={
                    "workspace_revision": verify_before_rev,
                    "acceptance_status": "unknown",
                },
                error=ErrorInfo(
                    schema_version=SCHEMA_VERSION,
                    code=err_code,
                    message=err_msg,
                    recoverable=False,
                ),
                duration_ms=dur,
                metadata={"action_id": action_id, "sequence": seq},
            )

        temp_dir = ensure_clean_subdir(Path(self._config.output_root), ("temp",))
        cmd_results: list[dict[str, Any]] = []
        artifacts: list[ArtifactRef] = []
        any_unknown = False
        all_passed = True
        source_changed_unknown = False
        scope_violated_unknown = False
        failure_excerpts: list[str] = []

        for cid in self._task.verification_command_ids:
            spec = self._task.commands[cid]
            test_hashes = {
                k: self._acceptance_source_baseline[k]
                for k in self._acceptance_sources_by_cmd[cid]
            }
            step_src_ok, _, _ = self._verify_acceptance_sources()
            cmd_before_rev, _, step_scope_ok, _ = self._inspect_workspace_state()
            if not step_src_ok or not step_scope_ok or cmd_before_rev != verify_before_rev:
                any_unknown = True
                all_passed = False
                if not step_src_ok:
                    source_changed_unknown = True
                    reason = "test_source_changed"
                elif cmd_before_rev != verify_before_rev:
                    reason = "workspace_changed_during_verification"
                elif not step_scope_ok:
                    scope_violated_unknown = True
                    reason = "scope_violation"
                else:
                    reason = "workspace_changed_during_verification"
                cmd_results.append(
                    {
                        "command_id": cid,
                        "action_id": action_id,
                        "argv": [spec.executable, *spec.args],
                        "status": "unknown",
                        "reason": reason,
                        "before_revision": cmd_before_rev,
                        "after_revision": cmd_before_rev,
                        "report_valid": False,
                        "test_source_hashes": test_hashes,
                    }
                )
                failure_excerpts.append(f"{cid}: {reason}")
                continue

            result_dir = self._allocate_result_dir(
                kind="verify",
                action_id=action_id,
                command_id=cid,
                seq=seq,
            )
            argv, junit_path = _resolve_result_dir_arg(spec, result_dir)

            try:
                proc_res = await run_command_in_job(
                    spec=spec,
                    argv=argv,
                    temp_dir=temp_dir,
                    max_output_bytes=self._config.max_output_bytes,
                    process_cleanup_ms=self._config.process_cleanup_ms,
                )
            except asyncio.CancelledError:
                after_rev, _, _, _ = self._inspect_workspace_state()
                self._acceptance = AcceptanceSummary(
                    status="unknown",
                    revision=after_rev,
                    action_id=action_id,
                    command_ids=list(self._task.verification_command_ids),
                    commands_digest=self._task.commands_digest(),
                    stale_reason="verification_cancelled",
                    results=cmd_results,
                )
                self._last_tool_result = {
                    "action_id": action_id,
                    "action_type": "verify_task",
                    "sequence": seq,
                    "before_revision": verify_before_rev,
                    "after_revision": after_rev,
                    "tool_succeeded": False,
                    "cancelled": True,
                    "error_code": "CODING_PROCESS_CANCELLED",
                }
                raise
            except CodingError as exc:
                any_unknown = True
                all_passed = False
                cmd_after_rev, _, _, _ = self._inspect_workspace_state()
                cmd_results.append(
                    {
                        "command_id": cid,
                        "action_id": action_id,
                        "argv": list(argv),
                        "status": "unknown",
                        "reason": exc.code,
                        "before_revision": cmd_before_rev,
                        "after_revision": cmd_after_rev,
                        "report_valid": False,
                        "test_source_hashes": test_hashes,
                    }
                )
                failure_excerpts.append(f"{cid}: {exc.code}")
                continue

            cmd_after_rev, _, after_scope_ok, _ = self._inspect_workspace_state()
            after_src_ok, _, _ = self._verify_acceptance_sources()
            if (
                proc_res.timed_out
                or not proc_res.cleanup_ok
                or not after_src_ok
                or not after_scope_ok
                or cmd_after_rev != verify_before_rev
            ):
                any_unknown = True
                all_passed = False
                if not after_src_ok:
                    source_changed_unknown = True
                    reason = "test_source_changed"
                elif cmd_after_rev != verify_before_rev:
                    reason = "workspace_changed_during_verification"
                elif not after_scope_ok:
                    scope_violated_unknown = True
                    reason = "scope_violation"
                elif proc_res.timed_out:
                    reason = "command_timed_out"
                else:
                    reason = "cleanup_incomplete"
                cmd_results.append(
                    {
                        "command_id": cid,
                        "action_id": action_id,
                        "argv": list(proc_res.argv),
                        "exit_code": proc_res.exit_code,
                        "status": "unknown",
                        "reason": reason,
                        "before_revision": cmd_before_rev,
                        "after_revision": cmd_after_rev,
                        "report_valid": False,
                        "stdout": proc_res.stdout,
                        "stderr": proc_res.stderr,
                        "stdout_truncated": proc_res.stdout_truncated,
                        "stderr_truncated": proc_res.stderr_truncated,
                        "test_source_hashes": test_hashes,
                    }
                )
                failure_excerpts.append(f"{cid}: {reason}")
                continue

            report_ok, report_reason, report_details = _verify_junit_report(
                junit_path=junit_path, spec=spec
            )
            cmd_passed = bool(proc_res.exit_code == 0 and report_ok)
            if not cmd_passed:
                all_passed = False
                failure_excerpts.append(
                    f"{cid} exit={proc_res.exit_code} reason={report_reason or 'exit_nonzero'}\n{proc_res.stdout}\n{proc_res.stderr}"
                )
            if junit_path.exists() and junit_path.is_file() and not junit_path.is_symlink():
                artifacts.append(
                    ArtifactRef(
                        schema_version=SCHEMA_VERSION,
                        artifact_id=f"junit-{action_id}-{cid}",
                        artifact_type="junit_xml",
                        uri=str(junit_path),
                        metadata={"command_id": cid, "action_id": action_id},
                    )
                )
            cmd_results.append(
                {
                    "command_id": cid,
                    "action_id": action_id,
                    "argv": list(proc_res.argv),
                    "exit_code": proc_res.exit_code,
                    "status": "passed" if cmd_passed else "failed",
                    "reason": report_reason
                    if not report_ok
                    else (None if proc_res.exit_code == 0 else "exit_nonzero"),
                    "before_revision": cmd_before_rev,
                    "after_revision": cmd_after_rev,
                    "stdout": proc_res.stdout,
                    "stderr": proc_res.stderr,
                    "stdout_truncated": proc_res.stdout_truncated,
                    "stderr_truncated": proc_res.stderr_truncated,
                    "test_source_hashes": test_hashes,
                    **report_details,
                }
            )

        verify_after_rev, _, end_scope_ok, _ = self._inspect_workspace_state()
        end_src_ok, _, _ = self._verify_acceptance_sources()
        dur = (time.monotonic_ns() - started_ns) / 1_000_000.0
        if not end_src_ok:
            source_changed_unknown = True
            any_unknown = True
            all_passed = False
        if not end_scope_ok:
            scope_violated_unknown = True
            any_unknown = True
            all_passed = False
        if verify_after_rev != verify_before_rev:
            any_unknown = True
            all_passed = False

        if any_unknown:
            acc_status = "unknown"
            if source_changed_unknown:
                stale_reason = "test_source_changed"
            elif verify_after_rev != verify_before_rev:
                stale_reason = "workspace_changed_during_verification"
            elif scope_violated_unknown:
                stale_reason = "scope_violation"
            else:
                stale_reason = "verification_execution_unknown"
        elif all_passed:
            acc_status = "passed"
            stale_reason = None
        else:
            acc_status = "failed"
            stale_reason = None

        self._acceptance = AcceptanceSummary(
            status=acc_status,
            revision=verify_after_rev,
            action_id=action_id,
            command_ids=list(self._task.verification_command_ids),
            commands_digest=self._task.commands_digest(),
            stale_reason=stale_reason,
            results=cmd_results,
        )

        stdout_summary = "\n---\n".join(failure_excerpts) if failure_excerpts else "all acceptance commands passed"
        bounded_diag, diag_trunc = _truncate_utf8(stdout_summary, self._config.max_output_bytes)
        self._last_tool_result = {
            "action_id": action_id,
            "action_type": "verify_task",
            "sequence": seq,
            "acceptance_status": acc_status,
            "before_revision": verify_before_rev,
            "after_revision": verify_after_rev,
            "command_ids": list(self._task.verification_command_ids),
            "stdout": bounded_diag,
            "stdout_truncated": diag_trunc,
            "tool_succeeded": not any_unknown,
        }
        self._append_history(
            {
                "sequence": seq,
                "action_id": action_id,
                "action_type": "verify_task",
                "target": ",".join(self._task.verification_command_ids),
                "succeeded": (acc_status == "passed"),
                "excerpt": f"verify={acc_status} {bounded_diag}",
            }
        )

        if any_unknown:
            return ActualOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_id=outcome_id,
                status=ExecutionStatus.UNKNOWN,
                state_before={"workspace_revision": verify_before_rev},
                state_after={"workspace_revision": verify_after_rev, "acceptance_status": "unknown"},
                artifacts=artifacts,
                error=ErrorInfo(
                    schema_version=SCHEMA_VERSION,
                    code=CODING_VERIFICATION_INVALID,
                    message="verification state unknown due to workspace mutation, timeout, or cleanup error",
                    recoverable=False,
                ),
                duration_ms=dur,
                metadata={"action_id": action_id, "sequence": seq},
            )

        return ActualOutcome(
            schema_version=SCHEMA_VERSION,
            outcome_id=outcome_id,
            status=ExecutionStatus.SUCCEEDED,
            state_before={"workspace_revision": verify_before_rev},
            state_after={
                "workspace_revision": verify_after_rev,
                "acceptance_status": acc_status,
            },
            artifacts=artifacts,
            duration_ms=dur,
            metadata={"action_id": action_id, "sequence": seq},
        )
