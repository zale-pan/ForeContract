"""Offline read-only Web verification evidence replay and failure analysis."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
for _rel in ("packages/core/src", "packages/adapters/web/src", "examples"):
    _candidate = str(_REPO_ROOT / _rel)
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import forecontract
import forecontract_web
from forecontract.contracts import ActionContract
from forecontract.verification import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_web.verification import (
    WebStepVerifier,
    WebTaskCompletionOracle,
)
from pydantic import ValidationError
from web_runtime_evidence import EvidenceError, verify_evidence_index

SCHEMA_VERSION: Final[str] = "0.1"
INDEX_KIND: Final[str] = "web_runtime_evidence_index"
REPORT_KIND: Final[str] = "verification_replay_report"
INDEX_FILENAME: Final[str] = "evidence-index.json"

REPLAY_INPUT_INVALID: Final[str] = "REPLAY_INPUT_INVALID"
REPLAY_OUTPUT_INVALID: Final[str] = "REPLAY_OUTPUT_INVALID"
REPLAY_ERROR_CODES: Final[frozenset[str]] = frozenset({
    REPLAY_INPUT_INVALID,
    REPLAY_OUTPUT_INVALID,
})

_DECLARED_VERIFIERS: Final[dict[str, str]] = {
    "step_verification": "web_step_verifier",
    "task_verification": "web_task_completion_oracle",
}


class ReplayError(Exception):
    """Bounded replay error with a closed code and fixed message."""

    def __init__(self, code: str, message: str) -> None:
        if code not in REPLAY_ERROR_CODES:
            code = REPLAY_INPUT_INVALID
        self.code: str = code
        self.message: str = message
        super().__init__(f"{code}: {message}")


def _fail_input(message: str = "Invalid replay input") -> NoReturn:
    raise ReplayError(REPLAY_INPUT_INVALID, message)


def _fail_output(message: str = "Invalid replay output path") -> NoReturn:
    raise ReplayError(REPLAY_OUTPUT_INVALID, message)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_link_or_reparse(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _check_directory_ancestors(path: Path, code: str) -> None:
    try:
        candidates = [path, *path.parents]
    except OSError:
        raise ReplayError(code, "Unusable directory path") from None
    for current in candidates:
        try:
            st = os.lstat(current)
        except OSError:
            raise ReplayError(code, "Directory does not exist or is unreadable") from None
        if _is_link_or_reparse(st) or not stat.S_ISDIR(st.st_mode):
            raise ReplayError(code, "Directory path traverses a link or non-directory")


def _scan_input_tree(root: Path) -> dict[str, tuple[int, str]]:
    """Verify the input tree contains only regular directories/files and return its digest map."""
    if not isinstance(root, (str, os.PathLike)):
        _fail_input("Input root path is not usable")
    raw_root = str(root)
    if not raw_root or "\x00" in raw_root:
        _fail_input("Input root path is empty or malformed")
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = Path.cwd() / root_path
    _check_directory_ancestors(root_path, REPLAY_INPUT_INVALID)

    inventory: dict[str, tuple[int, str]] = {}
    stack: list[tuple[Path, str]] = [(root_path, "")]
    while stack:
        current_dir, prefix = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                entries = sorted(it, key=lambda item: item.name)
        except OSError:
            _fail_input("Input directory could not be listed")
        for entry in entries:
            try:
                st = os.lstat(entry.path)
            except OSError:
                _fail_input("Input entry could not be inspected")
            if _is_link_or_reparse(st):
                _fail_input("Input tree contains a link or reparse point")
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if stat.S_ISDIR(st.st_mode):
                stack.append((Path(entry.path), rel))
            elif stat.S_ISREG(st.st_mode):
                try:
                    data = Path(entry.path).read_bytes()
                except OSError:
                    _fail_input("Input file could not be read")
                inventory[rel] = (len(data), _sha256_bytes(data))
            else:
                _fail_input("Input tree contains a non-regular file")
    return inventory


def _validate_relative_member(root: Path, raw_path: object) -> Path:
    """Validate that a referenced member path is a clean relative POSIX path inside root."""
    if type(raw_path) is not str or not raw_path or "\x00" in raw_path:
        _fail_input("Referenced path is missing or invalid")
    if "\\" in raw_path or ":" in raw_path or raw_path.startswith("/"):
        _fail_input("Referenced path is not a portable relative path")
    posix = PurePosixPath(raw_path)
    win_path = Path(raw_path)
    if posix.is_absolute() or win_path.is_absolute() or win_path.drive or win_path.root:
        _fail_input("Referenced path must not be absolute")
    parts = posix.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        _fail_input("Referenced path contains illegal traversal segments")

    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            st = os.lstat(current)
        except OSError:
            _fail_input("Referenced file does not exist")
        if _is_link_or_reparse(st):
            _fail_input("Referenced path traverses a link or reparse point")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(st.st_mode):
                _fail_input("Referenced parent component is not a directory")
        else:
            if not stat.S_ISREG(st.st_mode):
                _fail_input("Referenced path is not a regular file")

    try:
        base_resolved = root.resolve()
        target_resolved = current.resolve()
    except OSError:
        _fail_input("Referenced path could not be resolved")
    if base_resolved not in target_resolved.parents:
        _fail_input("Referenced path escapes the input root")
    return current


def _strict_json_loads_str(text: str) -> Any:
    if not text or text.startswith("\ufeff"):
        _fail_input("JSON input carries a BOM or is empty")

    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail_input("Duplicate JSON object key is not permitted")
            result[key] = value
        return result

    def _reject_constants(_token: str) -> NoReturn:
        _fail_input("Non-finite JSON number is not permitted")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constants,
        )
    except ReplayError:
        raise
    except ValueError:
        _fail_input("Malformed JSON content")


def _strict_json_object_bytes(raw: bytes) -> dict[str, Any]:
    if raw.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        _fail_input("BOM is not permitted in JSON files")
    if not raw.endswith(b"\n"):
        _fail_input("JSON file is not newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail_input("File is not valid UTF-8")
    parsed = _strict_json_loads_str(text)
    if not isinstance(parsed, dict):
        _fail_input("JSON root must be an object")
    return parsed


def _strict_int(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _hex_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _compute_differences(
    recorded: Any,
    recomputed: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    if type(recorded) is dict and type(recomputed) is dict:
        result: list[dict[str, Any]] = []
        for key in sorted(set(recorded) | set(recomputed)):
            escaped = key.replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped}"
            if key not in recorded or key not in recomputed:
                result.append({
                    "path": child_path,
                    "recorded_present": key in recorded,
                    "recomputed_present": key in recomputed,
                    "recorded": copy.deepcopy(recorded.get(key)),
                    "recomputed": copy.deepcopy(recomputed.get(key)),
                })
            else:
                result.extend(_compute_differences(recorded[key], recomputed[key], child_path))
        return result
    if type(recorded) is list and type(recomputed) is list:
        result = []
        limit = max(len(recorded), len(recomputed))
        for index in range(limit):
            child_path = f"{path}/{index}"
            if index >= len(recorded) or index >= len(recomputed):
                rec_in = index < len(recorded)
                recomp_in = index < len(recomputed)
                result.append({
                    "path": child_path,
                    "recorded_present": rec_in,
                    "recomputed_present": recomp_in,
                    "recorded": copy.deepcopy(recorded[index]) if rec_in else None,
                    "recomputed": copy.deepcopy(recomputed[index]) if recomp_in else None,
                })
            else:
                result.extend(_compute_differences(recorded[index], recomputed[index], child_path))
        return result
    if type(recorded) is type(recomputed) and recorded == recomputed:
        return []
    return [{
        "path": path,
        "recorded_present": True,
        "recomputed_present": True,
        "recorded": copy.deepcopy(recorded),
        "recomputed": copy.deepcopy(recomputed),
    }]


def _compute_verifier_sources() -> dict[str, Any]:
    core_dir = _REPO_ROOT / "packages/core/src/forecontract"
    web_dir = _REPO_ROOT / "packages/adapters/web/src/forecontract_web"
    evidence_file = _REPO_ROOT / "examples/web_runtime_evidence.py"
    try:
        files = sorted(
            [*core_dir.rglob("*.py"), *web_dir.rglob("*.py"), evidence_file],
            key=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
        )
        digests = {
            p.relative_to(_REPO_ROOT).as_posix(): _sha256_bytes(p.read_bytes())
            for p in files
        }
    except OSError:
        _fail_input("Verifier source files could not be read")
    return {
        "core_version": forecontract.__version__,
        "web_version": forecontract_web.__version__,
        "sha256": digests,
    }


def _prevalidate_and_load(
    root: Path,
) -> tuple[
    str,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, ActionContract],
    list[tuple[dict[str, Any], dict[str, Any], VerificationEvidence, TaskCompletionCriteria | None]],
]:
    """Validate all files, hashes, schemas and models before invoking any verifier."""
    index_path = _validate_relative_member(root, INDEX_FILENAME)
    try:
        index_raw = index_path.read_bytes()
    except OSError:
        _fail_input("Evidence index could not be read")
    index_sha256 = _sha256_bytes(index_raw)
    index_doc = _strict_json_object_bytes(index_raw)

    if index_doc.get("schema_version") != SCHEMA_VERSION:
        _fail_input("Unsupported evidence index schema version")
    if index_doc.get("kind") != INDEX_KIND:
        _fail_input("Unsupported evidence index kind")

    # Validate trace file strictly before delegating to verify_evidence_index.
    trace_decl = index_doc.get("trace")
    if not isinstance(trace_decl, dict):
        _fail_input("Missing trace declaration")
    trace_path = _validate_relative_member(root, trace_decl.get("path"))
    if not _hex_digest(trace_decl.get("sha256")) or not _strict_int(trace_decl.get("bytes"), 0):
        _fail_input("Invalid trace digest or byte count declaration")
    try:
        trace_raw = trace_path.read_bytes()
    except OSError:
        _fail_input("Trace file could not be read")
    if trace_raw.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        _fail_input("BOM is not permitted in trace file")
    if not trace_raw.endswith(b"\n"):
        _fail_input("Trace file is not newline-terminated")
    if len(trace_raw) != trace_decl["bytes"] or _sha256_bytes(trace_raw) != trace_decl["sha256"]:
        _fail_input("Trace file does not match its declared hash or size")
    try:
        trace_text = trace_raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail_input("Trace file is not valid UTF-8")
    if trace_text.startswith("\ufeff"):
        _fail_input("BOM is not permitted in trace file")
    trace_lines = [line for line in trace_text.splitlines() if line.strip()]
    trace_records: list[dict[str, Any]] = []
    for line in trace_lines:
        rec = _strict_json_loads_str(line)
        if not isinstance(rec, dict):
            _fail_input("Trace line is not a JSON object")
        trace_records.append(rec)

    # Validate working memory file strictly.
    memory_decl = index_doc.get("working_memory")
    if not isinstance(memory_decl, dict):
        _fail_input("Missing working memory declaration")
    memory_path = _validate_relative_member(root, memory_decl.get("path"))
    if not _hex_digest(memory_decl.get("sha256")) or not _strict_int(memory_decl.get("bytes"), 0):
        _fail_input("Invalid working memory digest or byte count declaration")
    try:
        memory_raw = memory_path.read_bytes()
    except OSError:
        _fail_input("Working memory file could not be read")
    if (
        len(memory_raw) != memory_decl["bytes"]
        or _sha256_bytes(memory_raw) != memory_decl["sha256"]
    ):
        _fail_input("Working memory file does not match its declared hash or size")
    _strict_json_object_bytes(memory_raw)

    # Validate snapshot entries and all files in snapshots/ strictly.
    snapshots_decl = index_doc.get("snapshots")
    if not isinstance(snapshots_decl, list):
        _fail_input("Missing snapshots list")
    snapshot_docs: dict[str, dict[str, Any]] = {}
    for entry in snapshots_decl:
        if not isinstance(entry, dict):
            _fail_input("Snapshot entry is not an object")
        rel_path = entry.get("path")
        snap_path = _validate_relative_member(root, rel_path)
        if not _hex_digest(entry.get("sha256")) or not _strict_int(entry.get("bytes"), 1):
            _fail_input("Snapshot entry has invalid hash or byte count")
        try:
            snap_raw = snap_path.read_bytes()
        except OSError:
            _fail_input("Snapshot file could not be read")
        if len(snap_raw) != entry["bytes"] or _sha256_bytes(snap_raw) != entry["sha256"]:
            _fail_input("Snapshot file does not match its declared hash or size")
        snap_doc = _strict_json_object_bytes(snap_raw)
        if isinstance(rel_path, str):
            snapshot_docs[rel_path] = snap_doc

    for list_key in ("contracts", "attempts", "verifications"):
        items = index_doc.get(list_key)
        if not isinstance(items, list):
            _fail_input("Association list is missing")
        for item in items:
            if not isinstance(item, dict):
                _fail_input("Association entry is not an object")
            _validate_relative_member(root, item.get("path"))

    # Run existing full evidence index verification.
    try:
        validated_index = verify_evidence_index(root)
    except (EvidenceError, ValueError, KeyError, TypeError, OSError):
        _fail_input("Evidence index verification failed")

    # Pre-validate all contract models and verification payloads before calling any verifier.
    contracts_by_id: dict[str, ActionContract] = {}
    for contract_entry in validated_index["contracts"]:
        c_path = contract_entry["path"]
        c_doc = snapshot_docs.get(c_path)
        if not isinstance(c_doc, dict) or not isinstance(c_doc.get("payload"), dict):
            _fail_input("Contract snapshot payload is missing")
        try:
            contracts_by_id[contract_entry["contract_id"]] = ActionContract.model_validate(
                c_doc["payload"]
            )
        except ValidationError:
            _fail_input("Contract snapshot payload failed model validation")

    prepared_verifications: list[
        tuple[dict[str, Any], dict[str, Any], VerificationEvidence, TaskCompletionCriteria | None]
    ] = []
    for v_entry in validated_index["verifications"]:
        v_kind = v_entry.get("kind")
        if v_kind not in _DECLARED_VERIFIERS:
            _fail_input("Unsupported verification kind")
        if v_entry.get("verifier") != _DECLARED_VERIFIERS[v_kind]:
            _fail_input("Unsupported verifier identifier")
        v_path = v_entry["path"]
        v_doc = snapshot_docs.get(v_path)
        if not isinstance(v_doc, dict):
            _fail_input("Verification snapshot is missing")
        payload = v_doc.get("payload")
        if not isinstance(payload, dict):
            _fail_input("Verification payload is not an object")
        recorded_result = payload.get("result")
        if not isinstance(recorded_result, dict):
            _fail_input("Verification recorded result is missing")
        try:
            evidence_model = VerificationEvidence.model_validate(payload.get("evidence"))
        except ValidationError:
            _fail_input("Verification evidence payload failed model validation")

        criteria_model: TaskCompletionCriteria | None = None
        if v_kind == "step_verification":
            c_id = v_entry.get("contract_id")
            if type(c_id) is not str or c_id not in contracts_by_id:
                _fail_input("Step verification references an unknown contract")
        else:
            try:
                criteria_model = TaskCompletionCriteria.model_validate(payload.get("criteria"))
            except ValidationError:
                _fail_input("Task completion criteria failed model validation")

        prepared_verifications.append(
            (v_entry, copy.deepcopy(recorded_result), evidence_model, criteria_model)
        )

    return (
        index_sha256,
        validated_index,
        trace_records,
        contracts_by_id,
        prepared_verifications,
    )


def replay_evidence(root: Path) -> dict[str, Any]:
    """Pure read-only offline replay and failure analysis of a Web evidence directory."""
    if not isinstance(root, (str, os.PathLike)):
        _fail_input("Input path is not usable")
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = Path.cwd() / root_path

    before_inventory = _scan_input_tree(root_path)
    (
        index_sha256,
        validated_index,
        trace_records,
        contracts_by_id,
        prepared_verifications,
    ) = _prevalidate_and_load(root_path)

    contract_step_by_id: dict[str, str | None] = {
        c["contract_id"]: c.get("step_id")
        for c in validated_index["contracts"]
        if isinstance(c, dict) and isinstance(c.get("contract_id"), str)
    }

    step_verifier = WebStepVerifier()
    task_oracle = WebTaskCompletionOracle()

    verification_rows: list[dict[str, Any]] = []
    for v_entry, recorded_result, evidence_model, criteria_model in prepared_verifications:
        v_kind = v_entry["kind"]
        if v_kind == "step_verification":
            contract_model = contracts_by_id[v_entry["contract_id"]]
            recomputed_model = step_verifier.verify_step(
                contract=contract_model,
                evidence=evidence_model,
            )
        else:
            assert criteria_model is not None
            recomputed_model = task_oracle.evaluate_completion(
                criteria=criteria_model,
                evidence=evidence_model,
            )

        recomputed_json = copy.deepcopy(recomputed_model.model_dump(mode="json"))
        recorded_json = copy.deepcopy(recorded_result)
        diffs = sorted(
            _compute_differences(recorded_json, recomputed_json),
            key=lambda item: item["path"],
        )
        equal = recorded_json == recomputed_json

        contract_id = v_entry.get("contract_id")
        explicit_step = v_entry.get("step_id")
        step_id = (
            explicit_step
            if explicit_step is not None
            else (contract_step_by_id.get(contract_id) if isinstance(contract_id, str) else None)
        )

        verification_rows.append({
            "source_path": v_entry["path"],
            "source_sha256": v_entry["sha256"],
            "kind": v_kind,
            "ordinal": v_entry["ordinal"],
            "trace_fact": v_entry.get("trace_fact"),
            "event_record_id": v_entry.get("event_record_id"),
            "step_id": step_id,
            "contract_id": contract_id,
            "attempt_id": v_entry.get("attempt_id"),
            "observation_id": v_entry.get("observation_id"),
            "recorded": recorded_json,
            "recomputed": recomputed_json,
            "equal": equal,
            "differences": diffs,
        })

    verification_rows.sort(
        key=lambda row: (row["kind"], row["ordinal"], row["source_path"])
    )

    after_inventory = _scan_input_tree(root_path)
    if after_inventory != before_inventory:
        _fail_input("Input tree changed during replay")

    verifier_sources = _compute_verifier_sources()
    difference_count = sum(1 for row in verification_rows if not row["equal"])

    analysis = {
        "nonpassing_source_paths": [
            row["source_path"]
            for row in verification_rows
            if row["recomputed"].get("status") in ("fail", "inconclusive")
        ],
        "terminal_reason_code": validated_index["terminal"]["reason_code"],
        "failure_event_record_ids": [
            rec["record_id"]
            for rec in trace_records
            if rec.get("status") in ("fail", "inconclusive")
            or rec.get("execution_status") in ("failed", "execution_unknown")
            or rec.get("kind") == "execution_unknown"
        ],
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "input_index_sha256": index_sha256,
        "run_id": validated_index["run_id"],
        "profile": validated_index["profile"],
        "verifier_sources": verifier_sources,
        "recorded_terminal": copy.deepcopy(validated_index["terminal"]),
        "replay_status": "different" if difference_count > 0 else "matched",
        "verification_count": len(verification_rows),
        "difference_count": difference_count,
        "verifications": verification_rows,
        "analysis": analysis,
    }
    try:
        json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        _fail_input("Report contains non-serializable values")
    return copy.deepcopy(report)


def _validate_output_path(output_path: Path, input_root: Path) -> Path:
    raw_out = str(output_path)
    if not raw_out or "\x00" in raw_out:
        _fail_output("Output path is empty or malformed")
    out = Path(output_path)
    if not out.is_absolute():
        out = Path.cwd() / out

    if os.path.lexists(out):
        _fail_output("Output file already exists")

    try:
        ancestors = [out.parent, *out.parent.parents]
    except OSError:
        _fail_output("Output parent chain is invalid")

    for anc in ancestors:
        if os.path.lexists(anc):
            try:
                st = os.lstat(anc)
            except OSError:
                _fail_output("Output parent cannot be inspected")
            if _is_link_or_reparse(st):
                _fail_output("Output path traverses a link or reparse point")
            if not stat.S_ISDIR(st.st_mode):
                _fail_output("Output parent path is not a directory")

    in_path = Path(input_root)
    if not in_path.is_absolute():
        in_path = Path.cwd() / in_path
    try:
        in_resolved = in_path.resolve()
        out_resolved = out.resolve()
    except OSError:
        _fail_output("Output or input path could not be resolved")

    if out_resolved == in_resolved or in_resolved in out_resolved.parents:
        _fail_output("Output path must not reside inside the input root")

    return out


def _write_report_atomic(output_path: Path, input_root: Path, report: dict[str, Any]) -> None:
    out = _validate_output_path(output_path, input_root)
    try:
        serialized = (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail_output("Report serialization failed")

    staging: Path | None = None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        _check_directory_ancestors(out.parent, REPLAY_OUTPUT_INVALID)
        staging = out.parent / f".{out.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        with open(staging, "xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(out):
            _fail_output("Output path already exists")
        os.rename(staging, out)
    except ReplayError:
        if staging is not None:
            with contextlib.suppress(OSError):
                staging.unlink()
        raise
    except OSError:
        if staging is not None:
            with contextlib.suppress(OSError):
                staging.unlink()
        _fail_output("Output report could not be written")


class _ReplayArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ReplayError(REPLAY_INPUT_INVALID, "Invalid command-line arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _ReplayArgumentParser(
        prog="replay_trace.py",
        description="Offline read-only Web verification evidence replay.",
        add_help=True,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input evidence directory")
    parser.add_argument("--output", required=True, type=Path, help="New output report JSON path")

    try:
        args = parser.parse_args(argv)
        out_path = _validate_output_path(args.output, args.input)
        report = replay_evidence(args.input)
        _write_report_atomic(out_path, args.input, report)
    except ReplayError as err:
        sys.stderr.write(f"{err.code}: {err.message}\n")
        return 4 if err.code == REPLAY_OUTPUT_INVALID else 3

    return 2 if report["difference_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
