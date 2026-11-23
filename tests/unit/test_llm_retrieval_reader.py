"""Acceptance tests for host-side registration and bounded evidence reads.

C03/C05/C06: the reader registers one real recorder entry after verifying its bounded bytes,
keeps an independent copy, refuses every path escape and every damaged association, and serves
only the frozen scope's authorized views. Missing values are reported as ``not_found``; illegal
paths, illegal text selections and damaged sources are errors.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from forecontract.models import Observation
from forecontract_llm.models import PlannerConfig
from forecontract_llm.retrieval import (
    EvidenceGrant,
    EvidenceReadRequest,
    RegisteredEvidenceReader,
    RetrievalConfig,
    RetrievalError,
)
from forecontract_llm.retrieval import reader as reader_module

SCHEMA = "0.1"
RUN = "run-1"
OBS = "obs-1"


def _observation(state: dict | None = None, observation_id: str = OBS) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="offline",
                       state=state if state is not None else {"done": False},
                       timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))


def _config(**overrides: object) -> RetrievalConfig:
    base = {"schema_version": SCHEMA}
    base.update(overrides.pop("base", {}))
    return RetrievalConfig(schema_version=SCHEMA, base=PlannerConfig(**base), **overrides)


def _grant(*, observation_id: str = OBS, evidence_id: str = "evidence-1",
           view_id: str = "status", selector: list[object] | None = None,
           format: str = "text", run_id: str = RUN) -> EvidenceGrant:
    return EvidenceGrant(
        schema_version=SCHEMA, run_id=run_id, evidence_id=evidence_id,
        observation_id=observation_id, view_id=view_id,
        selector=selector or ["state", "order", "status"], format=format)


def _snapshot_document(observation: Observation, *, run_id: str = RUN, ordinal: int = 1,
                       call_ordinal: int = 1) -> dict:
    return {
        "schema_version": SCHEMA,
        "kind": "observation",
        "run_id": run_id,
        "ordinal": ordinal,
        "recorded_at": datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc).isoformat(),
        "associations": {"call_ordinal": call_ordinal,
                         "observation_id": observation.observation_id},
        "payload": observation.model_dump(mode="json"),
    }


def _write_snapshot(root: Path, observation: Observation, *, run_id: str = RUN,
                    ordinal: int = 1, call_ordinal: int = 1,
                    document: dict | None = None) -> dict:
    document = document if document is not None else _snapshot_document(
        observation, run_id=run_id, ordinal=ordinal, call_ordinal=call_ordinal)
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    relative = f"snapshots/observation-{ordinal:04d}.json"
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {
        "kind": "observation",
        "ordinal": ordinal,
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "call_ordinal": call_ordinal,
        "observation_id": observation.observation_id,
    }


def _reader(tmp_path: Path, root_name: str = "evidence") -> tuple[RegisteredEvidenceReader, Path]:
    root = tmp_path / root_name
    root.mkdir()
    return RegisteredEvidenceReader(root, RUN, _config()), root


def _register(reader: RegisteredEvidenceReader, root: Path, observation: Observation,
              grants: list[EvidenceGrant] | None = None, **snapshot_kwargs: object) -> dict:
    entry = _write_snapshot(root, observation, **snapshot_kwargs)
    reader.register_observation(
        observation, entry, grants if grants is not None else [_grant()])
    return entry


def _scope(reader: RegisteredEvidenceReader, observation_id: str = OBS):
    return reader.open_scope(RUN, observation_id)


def _read_request(scope, *, evidence_id: str = "evidence-1", view_id: str = "status",
                  text_range: dict | None = None, run_id: str = RUN,
                  scope_id: str | None = None, turn: int = 1) -> EvidenceReadRequest:
    from forecontract_llm.retrieval.models import TextRange

    return EvidenceReadRequest(
        schema_version=SCHEMA,
        scope_id=scope_id or scope.scope_id,
        run_id=run_id,
        plan_id="plan-1",
        turn=turn,
        request_id="request-1",
        evidence_id=evidence_id,
        view_id=view_id,
        text_range=None if text_range is None else TextRange(schema_version=SCHEMA,
                                                             **text_range),
    )


def _read(reader: RegisteredEvidenceReader, scope, **kwargs: object):
    import asyncio

    return asyncio.run(reader.read(_read_request(scope, **kwargs)))


def _error(call) -> RetrievalError:
    with pytest.raises(RetrievalError) as info:
        call()
    return info.value


# --------------------------------------------------------------------------- registration


def test_a_registered_snapshot_becomes_a_directory_entry_without_a_path(tmp_path: Path) -> None:
    """C03: the entry names identifiers and digests, never a local path."""
    reader, root = _reader(tmp_path)
    observation = _observation({"order": {"status": "pending"}})
    _register(reader, root, observation)
    scope = _scope(reader)
    assert len(scope.entries) == 1
    entry = scope.entries[0]
    assert (entry.evidence_id, entry.source_observation_id, entry.kind) == (
        "evidence-1", OBS, "observation")
    assert entry.raw_bytes > 0
    assert "path" not in scope.model_dump(mode="json")["entries"][0]
    assert (reader.snapshot_count, reader.snapshot_bytes) == (1, entry.raw_bytes)


def test_the_scope_id_is_the_reader_side_identity(tmp_path: Path) -> None:
    """C03: the scope is a frozen directory bound to this Run and this Observation."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation())
    scope = _scope(reader)
    assert (scope.run_id, scope.observation_id) == (RUN, OBS)
    assert len(scope.scope_id) == 36


def test_zero_grants_are_registered_but_produce_an_empty_directory(tmp_path: Path) -> None:
    """C03: zero grants is legal; the scope is empty and a direct forecast stays legal."""
    reader, root = _reader(tmp_path)
    entry = _write_snapshot(root, _observation(), ordinal=1, call_ordinal=1)
    reader.register_observation(_observation(), entry, [])
    scope = _scope(reader)
    assert scope.entries == []
    assert reader.snapshot_count == 1


def test_a_registration_failure_leaves_no_half_registered_entry(tmp_path: Path) -> None:
    """C03: every check runs before the commit, so a failure changes nothing."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    before = (reader.snapshot_count, reader.snapshot_bytes, len(_scope(reader).entries))
    second = _observation({"order": {"status": "ready"}}, observation_id="obs-2")
    entry = _write_snapshot(root, second, ordinal=2, call_ordinal=1)
    entry["sha256"] = "b" * 64
    error = _error(lambda: reader.register_observation(
        second, entry, [_grant(observation_id="obs-2", evidence_id="evidence-2")]))
    assert (error.code, error.stage) == ("source_invalid", "source")
    assert (reader.snapshot_count, reader.snapshot_bytes, len(_scope(reader).entries)) == before


@pytest.mark.parametrize("mutation", [
    {"kind": "forecast"},
    {"run_id": "run-2"},
    {"ordinal": 2},
    {"associations": {"call_ordinal": 9, "observation_id": OBS}},
    {"associations": {"call_ordinal": 1, "observation_id": "obs-2"}},
    {"schema_version": "0.2"},
    {"payload": {"schema_version": SCHEMA, "observation_id": "other",
                 "environment_type": "offline", "state": {},
                 "timestamp": "2026-11-03T12:00:00+00:00"}},
])
def test_a_snapshot_document_must_declare_the_registered_observation(
        tmp_path: Path, mutation: dict) -> None:
    """C03: schema, kind, Run, ordinal, associations and payload all have to agree."""
    reader, root = _reader(tmp_path)
    observation = _observation({"order": {"status": "pending"}})
    document = _snapshot_document(observation)
    document.update(mutation)
    entry = _write_snapshot(root, observation, document=document)
    error = _error(lambda: reader.register_observation(
        observation, entry, [_grant()]))
    assert (error.code, error.stage) == ("source_invalid", "source")


def test_a_snapshot_changed_during_registration_is_refused(tmp_path: Path) -> None:
    """C03: the bytes read must still match the declared entry hash and length."""
    reader, root = _reader(tmp_path)
    observation = _observation()
    entry = _write_snapshot(root, observation)
    entry["bytes"] = entry["bytes"] + 1
    assert _error(lambda: reader.register_observation(
        observation, entry, [_grant()])).code == "source_invalid"
    entry = _write_snapshot(root, observation, ordinal=2, call_ordinal=2)
    (root / entry["path"]).write_bytes(b'{"tampered": true}')
    assert _error(lambda: reader.register_observation(
        observation, entry, [_grant()])).code == "source_invalid"


@pytest.mark.parametrize("path", [
    "C:/Windows/win.ini",
    "../outside.json",
    "snapshots/../snapshots/observation-0001.json",
    "other/observation-0001.json",
    "snapshots/observation-0002.json",
    "snapshots\\observation-0001.json",
])
def test_an_absolute_or_escaping_entry_path_is_refused(tmp_path: Path, path: str) -> None:
    """C03: only the declared snapshot member path is a read location."""
    reader, root = _reader(tmp_path)
    observation = _observation()
    entry = _write_snapshot(root, observation)
    entry["path"] = path
    error = _error(lambda: reader.register_observation(
        observation, entry, [_grant()]))
    assert (error.code, error.stage) == ("source_invalid", "source")


def test_a_symlinked_snapshot_member_is_refused(tmp_path: Path) -> None:
    """C03: a symlink/reparse component is refused even when it resolves inside the root."""
    reader, root = _reader(tmp_path)
    observation = _observation()
    entry = _write_snapshot(root, observation)
    target = root / entry["path"]
    real = root / "snapshots" / "real.json"
    target.replace(real)
    try:
        os.symlink(real, target)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a symlink in the test directory")
    assert _error(lambda: reader.register_observation(
        observation, entry, [_grant()])).code == "source_invalid"


def test_a_duplicate_evidence_or_view_identifier_is_refused(tmp_path: Path) -> None:
    """C03: a duplicate evidence id or a duplicate view id in one grant set is refused."""
    reader, root = _reader(tmp_path)
    first = _observation({"order": {"status": "pending"}})
    _register(reader, root, first)
    second = _observation({"order": {"status": "ready"}}, observation_id="obs-2")
    entry = _write_snapshot(root, second, ordinal=2, call_ordinal=1)
    error = _error(lambda: reader.register_observation(
        second, entry, [_grant(observation_id="obs-2")]))
    assert error.code == "invalid_input"
    entry = _write_snapshot(root, second, ordinal=3, call_ordinal=2)
    error = _error(lambda: reader.register_observation(
        second, entry, [_grant(observation_id="obs-2", evidence_id="evidence-3"),
                        _grant(observation_id="obs-2", evidence_id="evidence-4")]))
    assert error.code == "invalid_input"


@pytest.mark.parametrize("grant_kwargs", [
    {"run_id": "run-2"},
    {"observation_id": "obs-2"},
])
def test_a_grant_for_another_run_or_observation_is_refused(
        tmp_path: Path, grant_kwargs: dict) -> None:
    """C03: the authorization must match the snapshot's Run and Observation."""
    reader, root = _reader(tmp_path)
    observation = _observation()
    entry = _write_snapshot(root, observation)
    error = _error(lambda: reader.register_observation(
        observation, entry, [_grant(**grant_kwargs)]))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")


# --------------------------------------------------------------------------- scope


def test_an_unknown_current_observation_or_another_run_is_refused(tmp_path: Path) -> None:
    """C03: a scope exists only for a registered current Observation of this Run."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation())
    assert _error(lambda: reader.open_scope(RUN, "obs-unknown")).code == "access_denied"
    assert _error(lambda: reader.open_scope("run-2", OBS)).code == "binding_mismatch"


def test_a_scope_is_frozen_against_later_registrations(tmp_path: Path) -> None:
    """C03: a later registration does not widen an already open scope."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    second = _observation({"order": {"status": "ready"}}, observation_id="obs-2")
    _register(reader, root, second,
              grants=[_grant(observation_id="obs-2", evidence_id="evidence-2")],
              ordinal=2, call_ordinal=1)
    assert [entry.evidence_id for entry in scope.entries] == ["evidence-1"]


# --------------------------------------------------------------------------- reads


def test_a_json_view_returns_the_selected_value_and_its_digest(tmp_path: Path) -> None:
    """C03: the found value is measured by its canonical JSON bytes."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"},
                                          "count": 3}))
    scope = _scope(reader)
    result = _read(reader, scope)
    assert (result.status, result.value) == ("found", "pending")
    assert result.value_bytes == len(b'"pending"')
    assert result.value_sha256 == hashlib.sha256(b'"pending"').hexdigest()
    assert (result.partial, result.text_range) == (False, None)


def test_a_text_view_returns_the_whole_string_and_a_half_open_slice(tmp_path: Path) -> None:
    """C03: text intervals are code-point ranges and partial is only about the interval."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    whole = _read(reader, scope)
    assert (whole.value, whole.partial) == ("pending", False)
    slice_ = _read(reader, scope, text_range={"start": 1, "end": 4})
    assert (slice_.value, slice_.partial) == ("end", True)
    empty = _read(reader, scope, text_range={"start": 2, "end": 2})
    assert (empty.value, empty.partial) == ("", True)
    tail = _read(reader, scope, text_range={"start": 0, "end": 7})
    assert (tail.value, tail.partial) == ("pending", False)


def test_a_text_range_beyond_the_string_is_refused(tmp_path: Path) -> None:
    """C03: an out-of-range interval is a selection error, not a shorter value."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    error = _error(lambda: _read(reader, scope, text_range={"start": 0, "end": 8}))
    assert (error.code, error.stage) == ("selection_invalid", "selection")


def test_a_text_range_on_a_json_view_is_refused(tmp_path: Path) -> None:
    """C03: a JSON view is a projection; it takes no text interval."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}),
              grants=[_grant(format="json")])
    scope = _scope(reader)
    error = _error(lambda: _read(reader, scope, text_range={"start": 0, "end": 1}))
    assert (error.code, error.stage) == ("selection_invalid", "selection")


def test_a_declared_view_whose_path_is_missing_is_not_found_with_nulls(tmp_path: Path) -> None:
    """C03: a missing path is not_found with null value/hash and zero value bytes."""
    reader, root = _reader(tmp_path)
    observation = _observation({"order": {"status": "pending"}})
    _register(reader, root, observation,
              grants=[_grant(selector=["state", "order", "absent"])])
    scope = _scope(reader)
    result = _read(reader, scope)
    assert (result.status, result.value, result.value_sha256, result.value_bytes) == (
        "not_found", None, None, 0)
    assert result.partial is False


def test_a_found_json_null_is_not_a_missing_value(tmp_path: Path) -> None:
    """C03: a present null is found with the digest of canonical null."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}, "note": None}),
              grants=[_grant(selector=["state", "note"], format="json")])
    scope = _scope(reader)
    result = _read(reader, scope)
    assert (result.status, result.value, result.value_bytes) == ("found", None, 4)
    assert result.value_sha256 == hashlib.sha256(b"null").hexdigest()


@pytest.mark.parametrize("selector", [
    ["state", 0],
    ["state", "order", 0],
    ["state", "done", "nested"],
    ["state", "order", "status", "deeper"],
])
def test_a_type_mismatch_along_the_path_is_an_error(tmp_path: Path, selector: list) -> None:
    """C03: an illegal path is selection_invalid, never disguised as missing."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"},
                                          "done": False}),
              grants=[_grant(selector=selector, format="json")])
    scope = _scope(reader)
    error = _error(lambda: _read(reader, scope))
    assert (error.code, error.stage) == ("selection_invalid", "selection")


def test_a_text_view_of_a_non_string_is_refused(tmp_path: Path) -> None:
    """C03: text selects a string; selecting a number is an error."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"count": 3}),
              grants=[_grant(selector=["state", "count"], format="text")])
    scope = _scope(reader)
    assert _error(lambda: _read(reader, scope)).code == "selection_invalid"


def test_a_read_identity_is_checked_against_the_scope(tmp_path: Path) -> None:
    """C03/C06: identity is refused before existence, and a stale scope is unusable."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    assert _error(lambda: _read(reader, scope, run_id="run-2")).code == "binding_mismatch"
    assert _error(lambda: _read(reader, scope, evidence_id="absent")).code == "access_denied"
    assert _error(lambda: _read(reader, scope, view_id="absent")).code == "access_denied"
    assert _error(lambda: _read(reader, scope, scope_id="absent")).code == "invalid_state"


def test_a_closed_scope_cannot_publish_a_late_result(tmp_path: Path) -> None:
    """C06: close is synchronous and idempotent, and a closed scope serves nothing."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    reader.close_scope(scope.scope_id)
    reader.close_scope(scope.scope_id)
    error = _error(lambda: _read(reader, scope))
    assert (error.code, error.stage) == ("invalid_state", "scope")


def test_a_scope_id_from_another_reader_is_not_a_token(tmp_path: Path) -> None:
    """C03: a scope id is not a bearer token; another reader does not know it."""
    first, root = _reader(tmp_path)
    _register(first, root, _observation({"order": {"status": "pending"}}))
    foreign_scope = _scope(first)
    second = RegisteredEvidenceReader(root, RUN, _config())
    error = _error(lambda: _read(second, foreign_scope))
    assert (error.code, error.stage) == ("invalid_state", "scope")


def test_a_registered_snapshot_is_a_registration_time_copy(tmp_path: Path) -> None:
    """C03: a later file change does not change the registered evidence."""
    reader, root = _reader(tmp_path)
    observation = _observation({"order": {"status": "pending"}})
    entry = _register(reader, root, observation)
    (root / entry["path"]).write_bytes(b'{"changed": true}')
    scope = _scope(reader)
    result = _read(reader, scope)
    assert (result.status, result.value) == ("found", "pending")
    assert result.raw_sha256 == entry["sha256"]


# --------------------------------------------------------------------------- budgets


def test_every_view_budget_is_enforced_before_registration(tmp_path: Path) -> None:
    """C05: at most eight views per snapshot."""
    reader, root = _reader(tmp_path)
    observation = _observation()
    entry = _write_snapshot(root, observation)
    grants = [_grant(view_id=f"view-{index}") for index in range(9)]
    error = _error(lambda: reader.register_observation(observation, entry, grants))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")


def test_the_registered_snapshot_count_is_capped(tmp_path: Path) -> None:
    """C05: the reader lifetime count is capped and never evicts automatically."""
    reader, root = _reader(tmp_path)
    for ordinal in range(1, 65):
        observation = _observation({"ordinal": ordinal},
                                   observation_id=f"obs-{ordinal:04d}")
        entry = _write_snapshot(root, observation, ordinal=ordinal, call_ordinal=1)
        grants = [_grant(observation_id=observation.observation_id,
                         evidence_id=f"evidence-{ordinal:04d}")]
        reader.register_observation(observation, entry, grants)
    observation = _observation({"ordinal": 65}, observation_id="obs-0065")
    entry = _write_snapshot(root, observation, ordinal=65, call_ordinal=1)
    error = _error(lambda: reader.register_observation(
        observation, entry,
        [_grant(observation_id="obs-0065", evidence_id="evidence-0065")]))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")


def test_a_single_snapshot_over_its_byte_limit_is_refused(tmp_path: Path) -> None:
    """C05: at most one mebibyte of raw snapshot bytes is read."""
    reader, root = _reader(tmp_path)
    observation = _observation({"blob": "x" * 1_100_000})
    entry = _write_snapshot(root, observation)
    error = _error(lambda: reader.register_observation(observation, entry, [_grant()]))
    assert error.code == "budget_exceeded"


def test_the_total_registered_bytes_are_capped(tmp_path: Path, monkeypatch) -> None:
    """C05: the lifetime byte total is capped, not evicted."""
    reader, root = _reader(tmp_path)
    prepared = []
    for ordinal in range(1, 4):
        observation = _observation({"blob": "x" * 400},
                                   observation_id=f"obs-{ordinal:04d}")
        entry = _write_snapshot(root, observation, ordinal=ordinal, call_ordinal=1)
        grants = [_grant(observation_id=observation.observation_id,
                         evidence_id=f"evidence-{ordinal:04d}")]
        prepared.append((observation, entry, grants))
    monkeypatch.setattr(reader_module, "REGISTERED_MAX_BYTES",
                        prepared[0][1]["bytes"] * 2 + 1)
    for observation, entry, grants in prepared[:2]:
        reader.register_observation(observation, entry, grants)
    observation, entry, grants = prepared[2]
    error = _error(lambda: reader.register_observation(observation, entry, grants))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")


def test_a_scope_over_its_directory_budget_is_refused_not_truncated(tmp_path: Path) -> None:
    """C05: the frozen directory is measured and refused whole when it is too large."""
    reader, root = _reader(tmp_path)
    huge = ["state"] + ["x" * 128] * 31
    for ordinal in range(1, 9):
        observation = _observation({"ordinal": ordinal},
                                   observation_id=f"obs-{ordinal:04d}")
        entry = _write_snapshot(root, observation, ordinal=ordinal, call_ordinal=1)
        reader.register_observation(
            observation, entry,
            [_grant(observation_id=observation.observation_id,
                    evidence_id=f"evidence-{ordinal:04d}", selector=huge)])
    error = _error(lambda: reader.open_scope(RUN, "obs-0001"))
    assert (error.code, error.stage) == ("budget_exceeded", "scope")


def test_the_single_result_budget_is_enforced(tmp_path: Path) -> None:
    """C05: one complete read result is measured and refused above 65,536 bytes."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"blob": "x" * 70_000}),
              grants=[_grant(selector=["state", "blob"], format="text")])
    scope = _scope(reader)
    error = _error(lambda: _read(reader, scope))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")


def test_the_partial_text_value_is_a_projection_not_a_snapshot_claim(tmp_path: Path) -> None:
    """C03: the selector is preserved on every result and partial only marks the interval."""
    reader, root = _reader(tmp_path)
    _register(reader, root, _observation({"order": {"status": "pending"}}))
    scope = _scope(reader)
    result = _read(reader, scope, text_range={"start": 0, "end": 3})
    assert list(result.selector) == ["state", "order", "status"]
    assert result.partial is True
