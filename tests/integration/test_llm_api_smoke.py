"""Integration tests for the host smoke entry through the real Core runtime.

Each case runs the real RuntimeEngine, ContractBuilder, StepVerifier, TaskCompletionOracle,
RecoveryEngine, Fake adapter and (for retrieval) the real RegisteredEvidenceReader. Evidence is
read back from the persisted files, never from a summary's own ``passed`` claim.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = ("packages/core/src", "packages/adapters/fake/src", "packages/adapters/web/src",
            "packages/planners/scripted/src", "packages/planners/llm/src", "examples")

CASES = REPO_ROOT / "examples/llm_api_smoke_cases.v1.json"


def _host():
    """Import the host lazily; collection must not preload or disturb other suites."""

    for candidate in _SOURCES:
        path = str(REPO_ROOT / candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
    import llm_api_smoke as module

    return module


def _fail_type() -> type:
    return _host().SmokeError


def case(case_id: str) -> dict:
    return next(item for item in _host().load_cases(CASES) if item["case_id"] == case_id)


def run(tmp_path: Path, case_id: str) -> dict:
    host = _host()
    client, provider = host._offline_client()
    try:
        return asyncio.run(host.run_case(case(case_id), tmp_path / case_id,
                                         provider=provider, mode="offline"))
    finally:
        asyncio.run(client.aclose())


def test_direct_case_completes_with_one_call(tmp_path):
    summary = run(tmp_path, "direct")
    assert summary["terminal_status"] == "completed"
    assert summary["provider_calls"] == 1
    assert summary["read_values"] == []
    assert summary["close"]["drained"] is True
    assert summary["close"]["pending_calls"] == 0
    trace = [json.loads(line) for line in (tmp_path / "direct/trace.jsonl").read_bytes().splitlines()]
    assert trace[-1]["kind"] == "run_finished"
    assert trace[-1]["terminal_status"] == "success"
    assert _host().validate_case_evidence(tmp_path / "direct") == []


@pytest.mark.parametrize("case_id,expected", [("retrieval-pending", "pending"),
                                              ("retrieval-ready", "ready")])
def test_retrieval_cases_read_then_act(tmp_path, case_id, expected):
    summary = run(tmp_path, case_id)
    assert summary["terminal_status"] == "completed"
    assert summary["provider_calls"] == 2
    assert summary["read_values"] == [expected]
    assert summary["executed_action"]["payload"]["observed"] == expected
    assert _host().validate_case_evidence(tmp_path / case_id) == []


def test_two_retrieval_cases_differ(tmp_path):
    pending = run(tmp_path, "retrieval-pending")
    ready = run(tmp_path, "retrieval-ready")
    assert pending["executed_action"] != ready["executed_action"]
    assert pending["run_id"] != ready["run_id"]


def test_occupied_directory_is_refused_before_any_call(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "retained.bin").write_bytes(b"preserve")
    for shape in (occupied, tmp_path / "empty"):
        if shape == tmp_path / "empty":
            shape.mkdir()
        with pytest.raises(_fail_type()):
            asyncio.run(_host().run_case(case("direct"), shape,
                                      provider=_host()._OfflineProvider(), mode="offline"))
    assert (occupied / "retained.bin").read_bytes() == b"preserve"


def test_fixture_validation_refuses_bad_sets(tmp_path):
    raw = json.loads(CASES.read_bytes())
    assert len(_host().load_cases(CASES)) == 3
    variants = {}
    missing = json.loads(json.dumps(raw))
    missing["cases"].pop()
    variants["missing"] = missing
    extra = json.loads(json.dumps(raw))
    extra["cases"].append(dict(extra["cases"][0], case_id="extra"))
    variants["extra"] = extra
    duplicate = json.loads(json.dumps(raw))
    duplicate["cases"][1] = duplicate["cases"][0]
    variants["duplicate"] = duplicate
    version = json.loads(json.dumps(raw))
    version["schema_version"] = "unknown"
    variants["version"] = version
    for name, document in variants.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(_fail_type()):
            _host().load_cases(path)


def test_forged_evidence_is_detected(tmp_path):
    summary = run(tmp_path, "direct")
    target = Path(summary["evidence_root"])
    records = [json.loads(line) for line in (target / "trace.jsonl").read_bytes().splitlines()]
    records[-1]["terminal_status"] = "failure"
    (target / "trace.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    assert _host().validate_case_evidence(target) != []

    forged = tmp_path / "forged-mode"
    forged.mkdir()
    for name in ("summary.json", "trace.jsonl", "provider-events.jsonl"):
        (forged / name).write_bytes((target / name).read_bytes())
    document = json.loads((forged / "summary.json").read_bytes())
    document["mode"] = "live"
    (forged / "summary.json").write_text(json.dumps(document), encoding="utf-8")
    assert _host().validate_case_evidence(forged) != []
