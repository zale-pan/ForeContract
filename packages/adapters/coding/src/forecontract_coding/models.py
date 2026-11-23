"""Strict Pydantic records, closed vocabularies, and fixed v1 limits for Coding."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, Final, Literal

from forecontract.models.base import (
    NonBlankStr,
    SchemaVersion,
    StrictBool,
    StrictInt,
    StrictStr,
)
from forecontract.runtime.models import RunResult
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CODING_ADAPTER_NAME",
    "CODING_ENVIRONMENT_TYPE",
    "CODING_VERIFIER_TYPE",
    "HISTORY_EXCERPT_BYTES",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_HISTORY_ENTRIES",
    "MAX_OBSERVATION_BYTES",
    "MAX_OUTPUT_BYTES",
    "MAX_PROCESS_CLEANUP_MS",
    "MAX_READ_CHUNK_BYTES",
    "MAX_TOTAL_FILE_BYTES",
    "SCHEMA_VERSION",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_CONDITION_TYPES",
    "SUPPORTED_OUTCOME_TYPES",
    "AcceptanceSummary",
    "ChangedPathEntry",
    "CodingConfig",
    "CodingFileWrittenPayload",
    "CodingRunResult",
    "CodingScopeValidPayload",
    "CodingTask",
    "CodingTaskVerifiedPayload",
    "CodingToolResultPayload",
    "CodingWritePreconditionPayload",
    "CommandSpec",
    "ExpectedTestCase",
    "ReadFilePayload",
    "RunCommandPayload",
    "ScopeViolationEntry",
    "SearchTextPayload",
    "VerifyTaskPayload",
    "WriteFilePayload",
    "WriteTarget",
    "canonicalize_relative_path",
    "is_valid_sha256_hex",
    "utf8_byte_length",
    "validate_safe_single_segment",
]

SCHEMA_VERSION: Final[str] = "0.1"
CODING_ENVIRONMENT_TYPE: Final[str] = "coding"
CODING_ADAPTER_NAME: Final[str] = "coding_workspace"
CODING_VERIFIER_TYPE: Final[str] = "coding_rule"

SUPPORTED_ACTIONS: Final[tuple[str, ...]] = (
    "read_file",
    "search_text",
    "write_file",
    "run_command",
    "verify_task",
)
SUPPORTED_CONDITION_TYPES: Final[tuple[str, ...]] = (
    "coding_scope_valid",
    "coding_write_precondition",
)
SUPPORTED_OUTCOME_TYPES: Final[tuple[str, ...]] = (
    "coding_tool_result",
    "coding_file_written",
    "coding_task_verified",
)

MAX_FILES: Final[int] = 200
MAX_TOTAL_FILE_BYTES: Final[int] = 4 * 1024 * 1024
MAX_FILE_BYTES: Final[int] = 64 * 1024
MAX_READ_CHUNK_BYTES: Final[int] = 16 * 1024
MAX_OBSERVATION_BYTES: Final[int] = 64 * 1024
MAX_OUTPUT_BYTES: Final[int] = 32 * 1024
MAX_HISTORY_ENTRIES: Final[int] = 8
HISTORY_EXCERPT_BYTES: Final[int] = 2 * 1024
MAX_PROCESS_CLEANUP_MS: Final[int] = 2000

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_BRACE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\{[^{}]*\}")
_WINDOWS_RESERVED_STEMS: Final[frozenset[str]] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)
_CREDENTIAL_ENV_EXACT: Final[frozenset[str]] = frozenset(
    {
        "API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AUTHORIZATION",
        "BEARER_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
)
_CREDENTIAL_ENV_SUFFIXES: Final[tuple[str, ...]] = (
    "_API_KEY",
    "_SECRET",
    "_SECRET_KEY",
    "_TOKEN",
    "_PASSWORD",
)

_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)


class _StrictCodingModel(BaseModel):
    model_config = _STRICT_CONFIG


PositiveStrictInt = Annotated[int, StrictInt, Field(gt=0)]
NonNegativeStrictInt = Annotated[int, StrictInt, Field(ge=0)]


def utf8_byte_length(text: str) -> int:
    """Return the exact UTF-8 encoded byte length of ``text``."""
    return len(text.encode("utf-8"))


def is_valid_sha256_hex(value: object) -> bool:
    """Return True if ``value`` is a lowercase 64-character SHA-256 hex digest."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def canonicalize_relative_path(raw: object) -> str:
    """Validate and canonicalize a safe workspace-relative POSIX file path.

    Rejects empty/non-string paths, NUL/control characters, absolute paths, drive prefixes,
    UNC prefixes, Alternate Data Streams (``:``), ``.`` / ``..`` segments, trailing spaces or dots
    on any segment, and Windows reserved device names (with or without extensions).
    """
    if not isinstance(raw, str):
        raise TypeError("path must be a strict string")
    if not raw or not raw.strip():
        raise ValueError("path must be non-blank")
    if "\x00" in raw:
        raise ValueError("path contains a NUL byte")
    if any(ord(ch) < 32 for ch in raw):
        raise ValueError("path contains control characters")
    if raw.startswith(("/", "\\")):
        raise ValueError("absolute or rooted paths are forbidden")
    if ":" in raw:
        raise ValueError("drive prefixes and NTFS alternate data streams are forbidden")
    win = PureWindowsPath(raw)
    if win.drive or win.root or win.is_absolute():
        raise ValueError("absolute, drive-qualified, or UNC paths are forbidden")
    normalized = raw.replace("\\", "/")
    if "//" in normalized or normalized.endswith("/"):
        raise ValueError("empty path segments or directory-trailing slashes are forbidden")
    parts = normalized.split("/")
    for part in parts:
        if not part or part in (".", ".."):
            raise ValueError("relative traversal segments ('.', '..') are forbidden")
        if part.endswith((" ", ".")):
            raise ValueError("trailing spaces or dots on Windows path segments are forbidden")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_STEMS:
            raise ValueError(f"Windows reserved device name '{part}' is forbidden")
    return "/".join(parts)


def validate_safe_single_segment(raw: object, *, field_name: str) -> str:
    """Validate an identifier used as one safe directory segment on Windows."""
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a string")
    if not raw or not raw.strip():
        raise ValueError(f"{field_name} must be non-blank")
    if raw != raw.strip():
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")
    if len(raw) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    if "\x00" in raw or any(ord(ch) < 32 for ch in raw):
        raise ValueError(f"{field_name} must not contain NUL or control characters")
    if any(ch in raw for ch in '/\\:*?"<>|'):
        raise ValueError(f"{field_name} contains forbidden path characters")
    if raw in (".", "..") or ".." in raw:
        raise ValueError(f"{field_name} must not contain traversal segments")
    if raw.endswith((" ", ".")):
        raise ValueError(f"{field_name} must not end with space or dot on Windows")
    stem = raw.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_STEMS:
        raise ValueError(f"{field_name} must not use Windows reserved device name '{raw}'")
    return raw


def _validate_absolute_local_path(raw: object, *, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if "\x00" in raw:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    if any(ord(ch) < 32 for ch in raw):
        raise ValueError(f"{field_name} must not contain control characters")
    if raw.startswith(("\\\\", "//")):
        raise ValueError(f"{field_name} must not be a UNC path")
    win = PureWindowsPath(raw)
    if not win.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    for part in win.parts:
        if part in (".", ".."):
            raise ValueError(f"{field_name} must not contain '.' or '..' segments")
    return str(Path(raw))


def _ensure_no_credential_env(env: dict[str, str]) -> None:
    for key, val in env.items():
        if not key or not key.strip() or "=" in key or "\x00" in key or "\x00" in val:
            raise ValueError("invalid environment variable name or value")
        upper = key.upper()
        if upper in _CREDENTIAL_ENV_EXACT or upper.endswith(_CREDENTIAL_ENV_SUFFIXES):
            raise ValueError(f"credential-like environment variable '{key}' is forbidden")


def _validate_command_arg(arg: str) -> None:
    if "\x00" in arg:
        raise ValueError("command arguments must not contain NUL bytes")
    for match in _BRACE_TOKEN_RE.finditer(arg):
        if match.group(0) != "{result_dir}":
            raise ValueError("only the fixed {result_dir} placeholder is permitted")
    if "{result_dir}" in arg:
        after = arg.split("{result_dir}", 1)[1]
        if ".." in after.replace("\\", "/").split("/"):
            raise ValueError("{result_dir} suffix must not contain '..'")


class WriteTarget(_StrictCodingModel):
    """Exact authorization for a writable relative path."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    path: NonBlankStr
    operation: Literal["create", "modify"]

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return canonicalize_relative_path(value)


class ExpectedTestCase(_StrictCodingModel):
    """Exact `(classname, name)` testcase identity expected in JUnit XML."""

    classname: NonBlankStr
    name: NonBlankStr


class CommandSpec(_StrictCodingModel):
    """Host-approved command specification executed without shell interpolation."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    command_id: NonBlankStr
    executable: NonBlankStr
    args: Annotated[list[StrictStr], Field(strict=True)] = Field(default_factory=list)
    cwd: NonBlankStr
    env: Annotated[dict[StrictStr, StrictStr], Field(strict=True)] = Field(default_factory=dict)
    timeout_ms: PositiveStrictInt
    purpose: Literal["diagnostic", "acceptance"]
    expected_tests: Annotated[list[ExpectedTestCase], Field(strict=True)] = Field(
        default_factory=list
    )

    @field_validator("executable")
    @classmethod
    def _check_executable(cls, value: str) -> str:
        return _validate_absolute_local_path(value, field_name="executable")

    @field_validator("cwd")
    @classmethod
    def _check_cwd(cls, value: str) -> str:
        return _validate_absolute_local_path(value, field_name="cwd")

    @field_validator("args")
    @classmethod
    def _check_args(cls, value: list[str]) -> list[str]:
        for item in value:
            _validate_command_arg(item)
        return value

    @field_validator("env")
    @classmethod
    def _check_env(cls, value: dict[str, str]) -> dict[str, str]:
        _ensure_no_credential_env(value)
        return value

    @model_validator(mode="after")
    def _check_purpose_and_tests(self) -> CommandSpec:
        if self.purpose == "acceptance":
            if not self.expected_tests:
                raise ValueError("acceptance commands must declare non-empty expected_tests")
            seen: set[tuple[str, str]] = set()
            for item in self.expected_tests:
                key = (item.classname, item.name)
                if key in seen:
                    raise ValueError("expected_tests must not contain duplicate test cases")
                seen.add(key)
            if not any("{result_dir}" in arg for arg in self.args):
                raise ValueError(
                    "acceptance commands must bind {result_dir} in args for isolated JUnit output"
                )
        elif self.expected_tests:
            raise ValueError("diagnostic commands must have empty expected_tests")
        return self


class CodingConfig(_StrictCodingModel):
    """Fixed-ceiling configuration for a Coding run; callers may tighten, never relax."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    output_root: NonBlankStr
    max_files: Annotated[int, StrictInt, Field(ge=1, le=MAX_FILES)] = MAX_FILES
    max_total_file_bytes: Annotated[
        int, StrictInt, Field(ge=1, le=MAX_TOTAL_FILE_BYTES)
    ] = MAX_TOTAL_FILE_BYTES
    max_file_bytes: Annotated[int, StrictInt, Field(ge=1, le=MAX_FILE_BYTES)] = MAX_FILE_BYTES
    max_observation_bytes: Annotated[
        int, StrictInt, Field(ge=1, le=MAX_OBSERVATION_BYTES)
    ] = MAX_OBSERVATION_BYTES
    max_output_bytes: Annotated[int, StrictInt, Field(ge=1, le=MAX_OUTPUT_BYTES)] = (
        MAX_OUTPUT_BYTES
    )
    max_history_entries: Annotated[
        int, StrictInt, Field(ge=1, le=MAX_HISTORY_ENTRIES)
    ] = MAX_HISTORY_ENTRIES
    history_excerpt_bytes: Annotated[
        int, StrictInt, Field(ge=1, le=HISTORY_EXCERPT_BYTES)
    ] = HISTORY_EXCERPT_BYTES
    process_cleanup_ms: Annotated[
        int, StrictInt, Field(ge=1, le=MAX_PROCESS_CLEANUP_MS)
    ] = MAX_PROCESS_CLEANUP_MS

    @field_validator("output_root")
    @classmethod
    def _check_output_root(cls, value: str) -> str:
        return _validate_absolute_local_path(value, field_name="output_root")


class CodingTask(_StrictCodingModel):
    """Authoritative host task definition for one bounded Coding run."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: NonBlankStr
    goal_id: NonBlankStr
    goal_text: NonBlankStr
    workspace_root: NonBlankStr
    read_paths: Annotated[list[NonBlankStr], Field(strict=True)]
    write_paths: Annotated[list[WriteTarget], Field(strict=True)]
    commands: Annotated[dict[NonBlankStr, CommandSpec], Field(strict=True)]
    verification_command_ids: Annotated[list[NonBlankStr], Field(strict=True)]
    max_steps: PositiveStrictInt
    max_planning_calls: PositiveStrictInt
    max_replans: NonNegativeStrictInt
    run_timeout_ms: PositiveStrictInt

    @field_validator("workspace_root")
    @classmethod
    def _check_workspace_root(cls, value: str) -> str:
        return _validate_absolute_local_path(value, field_name="workspace_root")

    @field_validator("read_paths")
    @classmethod
    def _check_read_paths(cls, value: list[str]) -> list[str]:
        canonical: list[str] = []
        seen_exact: set[str] = set()
        seen_folded: set[str] = set()
        for item in value:
            norm = canonicalize_relative_path(item)
            folded = norm.casefold()
            if norm in seen_exact:
                raise ValueError(f"duplicate read_paths entry: {norm}")
            if folded in seen_folded:
                raise ValueError(f"case-insensitive collision in read_paths: {norm}")
            seen_exact.add(norm)
            seen_folded.add(folded)
            canonical.append(norm)
        return canonical

    @field_validator("write_paths")
    @classmethod
    def _check_write_paths(cls, value: list[WriteTarget]) -> list[WriteTarget]:
        seen_exact: set[str] = set()
        seen_folded: set[str] = set()
        for item in value:
            folded = item.path.casefold()
            if item.path in seen_exact:
                raise ValueError(f"duplicate write_paths entry: {item.path}")
            if folded in seen_folded:
                raise ValueError(f"case-insensitive collision in write_paths: {item.path}")
            seen_exact.add(item.path)
            seen_folded.add(folded)
        return value

    @field_validator("commands")
    @classmethod
    def _check_commands(cls, value: dict[str, CommandSpec]) -> dict[str, CommandSpec]:
        seen_folded: set[str] = set()
        for key, spec in value.items():
            if key != spec.command_id:
                raise ValueError(
                    f"commands key '{key}' does not match CommandSpec.command_id '{spec.command_id}'"
                )
            folded = key.casefold()
            if folded in seen_folded:
                raise ValueError(f"case-insensitive duplicate command_id: {key}")
            seen_folded.add(folded)
        return value

    @model_validator(mode="after")
    def _check_cross_fields(self) -> CodingTask:
        # Check cross-list case collisions between read_paths and write_paths
        read_by_folded = {path.casefold(): path for path in self.read_paths}
        managed_exact: set[str] = set(self.read_paths)
        for target in self.write_paths:
            folded = target.path.casefold()
            if folded in read_by_folded and read_by_folded[folded] != target.path:
                raise ValueError(
                    f"case collision between read_paths and write_paths: {target.path}"
                )
            managed_exact.add(target.path)
        if not managed_exact:
            raise ValueError("task must declare at least one managed file")
        if len(managed_exact) > MAX_FILES:
            raise ValueError(f"managed file count exceeds v1 ceiling of {MAX_FILES}")

        if not self.verification_command_ids:
            raise ValueError("verification_command_ids must be non-empty")
        seen_verify: set[str] = set()
        for cid in self.verification_command_ids:
            if cid in seen_verify:
                raise ValueError(f"duplicate verification_command_ids entry: {cid}")
            seen_verify.add(cid)
            spec = self.commands.get(cid)
            if spec is None:
                raise ValueError(f"verification_command_ids references unknown command: {cid}")
            if spec.purpose != "acceptance":
                raise ValueError(
                    f"verification_command_ids entry '{cid}' must have purpose='acceptance'"
                )
        return self

    def managed_paths(self) -> tuple[str, ...]:
        """Return sorted union of ``read_paths`` and ``write_paths``."""
        paths = set(self.read_paths) | {item.path for item in self.write_paths}
        return tuple(sorted(paths))

    def commands_digest(self) -> str:
        """Return a deterministic SHA-256 over the verification commands and test sets."""
        canonical = [
            self.commands[cid].model_dump(mode="json")
            for cid in self.verification_command_ids
        ]
        raw = json.dumps(
            {"verification_command_ids": self.verification_command_ids, "commands": canonical},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class ReadFilePayload(_StrictCodingModel):
    """Exact payload for ``read_file``."""

    path: NonBlankStr
    start_line: PositiveStrictInt
    max_lines: Annotated[int, StrictInt, Field(ge=1, le=200)]
    line_byte_offset: NonNegativeStrictInt = 0
    expected_sha256: NonBlankStr | None = None

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return canonicalize_relative_path(value)

    @field_validator("expected_sha256")
    @classmethod
    def _check_expected_sha(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_sha256_hex(value):
            raise ValueError("expected_sha256 must be null or 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _check_offset_sha(self) -> ReadFilePayload:
        if self.line_byte_offset > 0 and self.expected_sha256 is None:
            raise ValueError("expected_sha256 is required when line_byte_offset > 0")
        return self


class SearchTextPayload(_StrictCodingModel):
    """Exact payload for ``search_text``."""

    query: Annotated[str, StrictStr, Field(min_length=1)]
    paths: Annotated[list[NonBlankStr], Field(strict=True, min_length=1)]
    max_matches: Annotated[int, StrictInt, Field(ge=1, le=50)]

    @field_validator("paths")
    @classmethod
    def _check_paths(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in value:
            norm = canonicalize_relative_path(raw)
            if norm in seen:
                raise ValueError(f"duplicate search path: {norm}")
            seen.add(norm)
            result.append(norm)
        return result


class WriteFilePayload(_StrictCodingModel):
    """Exact payload for ``write_file``."""

    path: NonBlankStr
    expected_sha256: NonBlankStr | None
    content: StrictStr

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return canonicalize_relative_path(value)

    @field_validator("expected_sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_sha256_hex(value):
            raise ValueError("expected_sha256 must be null or 64 lowercase hex characters")
        return value


class RunCommandPayload(_StrictCodingModel):
    """Exact payload for ``run_command`` (model chooses ``command_id`` only)."""

    command_id: NonBlankStr


class VerifyTaskPayload(_StrictCodingModel):
    """Exact payload for ``verify_task`` (must be empty)."""


class CodingScopeValidPayload(_StrictCodingModel):
    """Exact payload for ``coding_scope_valid`` precondition."""


class CodingWritePreconditionPayload(_StrictCodingModel):
    """Exact payload for ``coding_write_precondition`` precondition."""

    path: NonBlankStr
    operation: Literal["create", "modify"]
    expected_sha256: NonBlankStr | None

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return canonicalize_relative_path(value)

    @field_validator("expected_sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_sha256_hex(value):
            raise ValueError("expected_sha256 must be null or 64 lowercase hex characters")
        return value


class CodingToolResultPayload(_StrictCodingModel):
    """Exact payload for ``coding_tool_result`` expected outcome."""

    action_type: Literal["read_file", "search_text", "write_file", "run_command", "verify_task"]
    path: NonBlankStr | None = None
    command_id: NonBlankStr | None = None
    require_success: StrictBool = True

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str | None) -> str | None:
        return canonicalize_relative_path(value) if value is not None else None


class CodingFileWrittenPayload(_StrictCodingModel):
    """Exact payload for ``coding_file_written`` expected outcome."""

    path: NonBlankStr
    expected_after_sha256: NonBlankStr | None = None

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return canonicalize_relative_path(value)

    @field_validator("expected_after_sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_sha256_hex(value):
            raise ValueError("expected_after_sha256 must be null or 64 lowercase hex characters")
        return value


class CodingTaskVerifiedPayload(_StrictCodingModel):
    """Exact payload for ``coding_task_verified`` expected outcome and task criteria."""

    verification_command_ids: Annotated[list[NonBlankStr], Field(strict=True, min_length=1)] | None = (
        None
    )
    commands_digest: NonBlankStr | None = None

    @field_validator("verification_command_ids")
    @classmethod
    def _check_cids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("verification_command_ids must not contain duplicates")
        return value

    @field_validator("commands_digest")
    @classmethod
    def _check_digest(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_sha256_hex(value):
            raise ValueError("commands_digest must be a 64-char lowercase hex digest")
        return value


class ChangedPathEntry(_StrictCodingModel):
    """Exact baseline-to-final diff entry for one managed writable file."""

    path: NonBlankStr
    operation: Literal["create", "modify"]
    before_exists: StrictBool
    after_exists: StrictBool
    before_sha256: NonBlankStr | None
    after_sha256: NonBlankStr | None


class ScopeViolationEntry(_StrictCodingModel):
    """Structured workspace scope breach relative to authorized ``write_paths``."""

    path: NonBlankStr
    reason: Literal[
        "unauthorized_create",
        "unauthorized_modify",
        "unauthorized_delete",
        "scope_scan_failed",
    ]
    before_sha256: NonBlankStr | None = None
    after_sha256: NonBlankStr | None = None


class AcceptanceSummary(_StrictCodingModel):
    """Final acceptance association summary (never synthesized by the model)."""

    status: Literal["not_run", "failed", "passed", "unknown"]
    revision: NonBlankStr | None = None
    action_id: NonBlankStr | None = None
    command_ids: Annotated[list[NonBlankStr], Field(strict=True)]
    commands_digest: NonBlankStr
    stale_reason: StrictStr | None = None
    results: Annotated[list[dict[str, Any]], Field(strict=True)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _finite_results(self) -> AcceptanceSummary:
        raw = json.dumps(self.results, allow_nan=False)
        if math.isnan(len(raw)):
            raise ValueError("unreachable")
        return self


class CodingRunResult(_StrictCodingModel):
    """Safe, reviewable result returned by ``run_coding_task``."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    run_result: RunResult
    baseline_workspace_revision: NonBlankStr
    final_workspace_revision: NonBlankStr
    changed_paths: Annotated[list[ChangedPathEntry], Field(strict=True)]
    acceptance: AcceptanceSummary
    terminal_status: NonBlankStr
    reason_code: NonBlankStr


def thaw_json_value(value: Any) -> Any:
    """Recursively unfreeze ``Mapping``/``tuple`` payloads produced by frozen contracts."""
    if isinstance(value, Mapping):
        return {k: thaw_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json_value(item) for item in value]
    return value
