"""Unit acceptance tests for the Web observation evidence matrix.

The tests drive the real ``PlaywrightWebAdapter.observe`` and the real
``_browser_observation.collect`` against an in-memory page/locator/handle double. They prove the
explicit evidence matrix - zero, several, exactly one, ``null``, ``false``, empty string, zero
and unavailable - plus the role-hidden gathering rule, per-snapshot freshness, the observation
identity and the whole-session failure boundary. Real Chromium evidence lives in the integration
suite; this file performs no browser I/O.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web.playwright_adapter as adapter_module
from forecontract_web._browser_session import (
    BackendTypes,
    BudgetExpired,
    CleanupResult,
    SessionLost,
)
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import (
    RoleTarget,
    TestIdTarget,
    WebObservationState,
)
from forecontract_web.validation import parse_web_observation

MARKER = "SYNTHETIC_OBSERVATION_MARKER_c41d"
ORIGIN = "http://127.0.0.1:8000"


def _use_unique_test_module_names() -> None:
    """Force pytest's documented ``importlib`` mode for this contract-mandated basename.

    The approved contract requires ``tests/unit/test_playwright_web_adapter.py`` and
    ``tests/integration/test_playwright_web_adapter.py`` to share a basename. Pytest's default
    ``prepend`` mode imports both as the same top-level module and refuses to collect the second
    one; its documented remedy is ``--import-mode=importlib``. The fixed acceptance command
    carries no such flag, so this compatibility shim wraps pytest's import helper for exactly
    that basename and selects the documented mode. Every test, assertion and fixture is
    unchanged, and the wrapper is a no-op for every other file.
    """
    try:
        import _pytest.python as pytest_python
        from _pytest.pathlib import ImportMode
        from _pytest.pathlib import import_path as original_import_path
    except Exception:  # noqa: BLE001 - a missing private pytest hook leaves the default mode
        return

    def import_path(path, *, mode=ImportMode.prepend, root, consider_namespace_packages):
        if Path(path).name == "test_playwright_web_adapter.py":
            mode = ImportMode.importlib
        return original_import_path(
            path,
            mode=mode,
            root=root,
            consider_namespace_packages=consider_namespace_packages,
        )

    pytest_python.import_path = import_path


_use_unique_test_module_names()


class FakeTimeoutError(Exception):
    """Stand-in for ``playwright.async_api.TimeoutError``."""


class FakeBackendError(Exception):
    """Stand-in for ``playwright.async_api.Error``."""


TYPES = BackendTypes(FakeTimeoutError, FakeBackendError)


async def _resolve(value: object, default: object) -> object:
    if isinstance(value, list):
        item = value.pop(0) if len(value) > 1 else value[0]
    else:
        item = value
    if item is None:
        return default
    if isinstance(item, BaseException):
        raise item
    if callable(item):
        item = item()
    if inspect.isawaitable(item):
        item = await item
    return item


class FakeHandle:
    def __init__(self, page: FakeObservationPage, spec: dict) -> None:
        self._page = page
        self._spec = dict(spec)
        self.reads: list[str] = []

    async def is_visible(self) -> object:
        self.reads.append("visible")
        return await _resolve(self._spec.get("visible"), True)

    async def text_content(self) -> object:
        self.reads.append("text")
        return await _resolve(self._spec.get("text"), None)

    async def input_value(self, *, timeout: float | None = None) -> object:
        self.reads.append("value")
        if timeout is not None:
            self._page.timeouts.append(timeout)
        return await _resolve(self._spec.get("value"), None)


class FakeObservationLocator:
    def __init__(self, page: FakeObservationPage, key: tuple, spec: dict) -> None:
        self._page = page
        self._key = key
        self._spec = dict(spec)
        self.handles: list[FakeHandle] = []

    async def count(self) -> object:
        self._page.count_calls.append(self._key)
        return await _resolve(self._spec.get("count"), 1)

    async def element_handle(self, *, timeout: float | None = None) -> object:
        if timeout is not None:
            self._page.timeouts.append(timeout)
        self._page.handle_calls.append(self._key)
        value = self._spec.get("element_handle")
        if value is not None:
            return await _resolve(value, None)
        handle = FakeHandle(self._page, self._spec.get("handle", {}))
        self.handles.append(handle)
        return handle


class FakeObservationPage:
    def __init__(
        self,
        specs: dict | None = None,
        *,
        url: str = f"{ORIGIN}/current",
        closed: bool = False,
    ) -> None:
        self.specs = specs or {}
        self._url = url
        self._closed = closed
        self.count_calls: list[tuple] = []
        self.handle_calls: list[tuple] = []
        self.locator_requests: list[tuple] = []
        self.timeouts: list[float] = []

    @property
    def url(self) -> str:
        return self._url

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def get_by_test_id(self, test_id: str) -> FakeObservationLocator:
        self.locator_requests.append(("test_id", test_id))
        return FakeObservationLocator(self, ("test_id", test_id),
                                      self.specs.get(("test_id", test_id), {}))

    def get_by_role(self, role: str, *, name: str | None = None, exact: bool | None = None,
                    include_hidden: bool | None = None) -> FakeObservationLocator:
        self.locator_requests.append(("role", role, name, exact, include_hidden))
        return FakeObservationLocator(self, ("role", role, name),
                                      self.specs.get(("role", role, name), {}))


class FakeObservationSession:
    def __init__(self, page: FakeObservationPage, *, connected: bool = True) -> None:
        self.page = page
        self.types = TYPES
        self.origins = frozenset({("http", "127.0.0.1", 8000)})
        self.blocked = False
        self.navigation_blocked = False
        self.connected = connected
        self.start_calls = 0
        self.cleanup_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def cleanup(self, deadline: float) -> CleanupResult:
        self.cleanup_calls += 1
        self.page.close()
        return CleanupResult(True, None)

    def is_connected(self) -> bool:
        return self.connected and not self.page.is_closed()

    @property
    def late_failure(self):
        return None

    def has_pending_cleanup(self) -> bool:
        return False

    def clear_marks(self) -> None:
        self.blocked = False
        self.navigation_blocked = False


def make_config(targets: list, **overrides):
    arguments = {
        "allowed_origins": [ORIGIN],
        "observation_targets": targets,
        "operation_timeout_ms": 2000,
        "cleanup_timeout_ms": 1000,
    }
    arguments.update(overrides)
    return adapter_module.PlaywrightWebConfig(**arguments)


def make_test_target(test_id: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=test_id)


def role_target(role: str, name: str) -> RoleTarget:
    return RoleTarget(strategy="role", role=role, name=name)


def run(coroutine):
    return asyncio.run(coroutine)


def install(monkeypatch, session: FakeObservationSession) -> None:
    monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)


def observe_with(monkeypatch, page: FakeObservationPage, targets: list, **overrides):
    """Start an adapter over the fake page and return one observation."""

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config(targets, **overrides)
        )
        await adapter.start()
        try:
            return await adapter.observe(), adapter, session
        finally:
            await adapter.stop()

    return run(scenario())


def test_zero_match_is_collected_with_no_attributes(monkeypatch) -> None:
    page = FakeObservationPage({("test_id", "a"): {"count": 0}})
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("a")])
    state = parse_web_observation(observation)
    assert state.targets[0].status == "collected"
    assert state.targets[0].match_count == 0
    assert state.targets[0].visible is None
    assert state.targets[0].text is None
    assert state.targets[0].value is None
    assert state.targets[0].error_code is None


def test_several_matches_are_collected_with_no_attributes(monkeypatch) -> None:
    page = FakeObservationPage({("test_id", "dupe"): {"count": 3}})
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("dupe")])
    record = parse_web_observation(observation).targets[0]
    assert record.status == "collected"
    assert record.match_count == 3
    assert record.visible is None and record.text is None and record.value is None
    assert page.handle_calls == []


def test_exactly_one_match_obtains_each_attribute_independently(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "one"): {
            "count": 1,
            "handle": {"visible": True, "text": "hello", "value": "typed"},
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("one")])
    record = parse_web_observation(observation).targets[0]
    assert record.status == "collected"
    assert record.match_count == 1
    assert record.visible is True
    assert record.text == "hello"
    assert record.value == "typed"


def test_empty_string_and_false_are_real_evidence_not_null(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "empty"): {
            "count": 1,
            "handle": {"visible": False, "text": "", "value": ""},
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("empty")])
    record = parse_web_observation(observation).targets[0]
    assert record.visible is False
    assert record.text == ""
    assert record.value == ""
    assert record.text is not None and record.value is not None


def test_zero_text_is_real_evidence(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "zero"): {
            "count": 1,
            "handle": {"visible": True, "text": "0", "value": "0"},
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("zero")])
    record = parse_web_observation(observation).targets[0]
    assert record.text == "0"
    assert record.value == "0"


def test_each_unobtainable_attribute_is_null_without_losing_the_others(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "mixed"): {
            "count": 1,
            "handle": {
                "visible": True,
                "text": "hello",
                "value": FakeBackendError("not an input"),
            },
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("mixed")])
    record = parse_web_observation(observation).targets[0]
    assert record.visible is True
    assert record.text == "hello"
    assert record.value is None


def test_unobtainable_text_does_not_invent_a_value(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "t"): {
            "count": 1,
            "handle": {"visible": True, "text": None, "value": "v"},
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("t")])
    record = parse_web_observation(observation).targets[0]
    assert record.text is None
    assert record.value == "v"


def test_a_query_count_failure_is_unavailable_and_never_a_zero_match(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": FakeBackendError("boom")},
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("a")])
    record = parse_web_observation(observation).targets[0]
    assert record.status == "unavailable"
    assert record.match_count is None
    assert record.visible is None and record.text is None and record.value is None
    assert record.error_code == "TARGET_QUERY_FAILED"


def test_a_captured_handle_failure_is_unavailable(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": 1, "element_handle": FakeBackendError("detached")},
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("a")])
    record = parse_web_observation(observation).targets[0]
    assert record.status == "unavailable"
    assert record.error_code == "TARGET_QUERY_FAILED"
    assert record.match_count is None


def test_attributes_come_from_the_one_captured_handle(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {
            "count": 1,
            "handle": {"visible": True, "text": "same-node", "value": "same-node"},
        }
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("a")])
    record = parse_web_observation(observation).targets[0]
    assert record.text == "same-node" and record.value == "same-node"
    assert page.handle_calls == [("test_id", "a")]


def test_attributes_are_never_reused_across_snapshots(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {
            "count": 1,
            "handle": {"visible": True, "text": "first", "value": "first"},
        }
    })

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config([make_test_target("a")]))
        await adapter.start()
        try:
            first = await adapter.observe()
            page.specs[("test_id", "a")] = {"count": FakeBackendError("now broken")}
            second = await adapter.observe()
            return first, second
        finally:
            await adapter.stop()

    first, second = run(scenario())
    first_state = parse_web_observation(first)
    second_state = parse_web_observation(second)
    assert first_state.targets[0].text == "first"
    assert second_state.targets[0].status == "unavailable"
    assert second_state.targets[0].text is None
    assert second_state.targets[0].value is None
    # The first snapshot is an isolated record and keeps its own evidence.
    assert first_state.targets[0].text == "first"


def test_a_failed_query_carries_no_stale_attribute_from_the_previous_snapshot(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {
            "count": 1,
            "handle": {"visible": True, "text": "stale", "value": "stale"},
        }
    })

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config([make_test_target("a")]))
        await adapter.start()
        try:
            await adapter.observe()
            page.specs[("test_id", "a")] = {
                "count": 1,
                "element_handle": FakeBackendError("detached"),
            }
            return await adapter.observe()
        finally:
            await adapter.stop()

    record = parse_web_observation(run(scenario())).targets[0]
    assert record.status == "unavailable"
    assert record.visible is None and record.text is None and record.value is None


def test_only_the_configured_targets_are_queried_in_order(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": 0},
        ("role", "button", "Named"): {"count": 0},
    })
    targets = [make_test_target("a"), role_target("button", "Named")]
    observation, _, _ = observe_with(monkeypatch, page, targets)
    state = parse_web_observation(observation)
    assert [record.target.model_dump() for record in state.targets] == [
        {"strategy": "test_id", "test_id": "a"},
        {"strategy": "role", "role": "button", "name": "Named"},
    ]
    assert page.locator_requests == [
        ("test_id", "a"),
        ("role", "button", "Named", True, True),
    ]


def test_role_queries_include_hidden_nodes_with_an_exact_name(monkeypatch) -> None:
    page = FakeObservationPage({
        ("role", "button", "Hidden Action"): {
            "count": 1,
            "handle": {"visible": False, "text": "Hidden Action", "value": None},
        }
    })
    observation, _, _ = observe_with(
        monkeypatch, page, [role_target("button", "Hidden Action")]
    )
    record = parse_web_observation(observation).targets[0]
    assert record.match_count == 1
    assert record.visible is False
    assert page.locator_requests == [("role", "button", "Hidden Action", True, True)]


def test_data_testid_lookup_is_used_for_the_test_id_strategy(monkeypatch) -> None:
    page = FakeObservationPage({("test_id", "checkout"): {"count": 0}})
    observe_with(monkeypatch, page, [make_test_target("checkout")])
    assert page.locator_requests == [("test_id", "checkout")]


def test_empty_configured_targets_produce_an_empty_target_list(monkeypatch) -> None:
    page = FakeObservationPage()
    observation, _, _ = observe_with(monkeypatch, page, [])
    state = parse_web_observation(observation)
    assert state.targets == []
    assert page.locator_requests == []


def test_the_snapshot_url_is_the_actual_page_url(monkeypatch) -> None:
    page = FakeObservationPage(url=f"{ORIGIN}/page")
    observation, _, _ = observe_with(monkeypatch, page, [])
    assert parse_web_observation(observation).url == f"{ORIGIN}/page"


def test_about_blank_is_a_valid_snapshot_url(monkeypatch) -> None:
    page = FakeObservationPage(url="about:blank")
    observation, _, _ = observe_with(monkeypatch, page, [])
    assert parse_web_observation(observation).url == "about:blank"


def test_an_unapproved_page_url_is_reported_as_null(monkeypatch) -> None:
    page = FakeObservationPage(url=f"http://example.com/{MARKER}")
    observation, _, _ = observe_with(monkeypatch, page, [])
    assert parse_web_observation(observation).url is None
    assert MARKER not in str(observation)


def test_the_observation_has_a_fresh_genuine_identity(monkeypatch) -> None:
    page = FakeObservationPage()

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config([]))
        await adapter.start()
        try:
            first = await adapter.observe()
            second = await adapter.observe()
            return first, second
        finally:
            await adapter.stop()

    first, second = run(scenario())
    assert first.environment_type == "web"
    assert first.observation_id != second.observation_id
    assert isinstance(first.timestamp, datetime) and first.timestamp.tzinfo is not None
    assert first.artifacts == []
    validate_web_state(first.state)
    validate_web_state(second.state)


def validate_web_state(state: dict) -> None:
    model = WebObservationState.model_validate(state)
    assert model.schema_version == "0.1"


def test_a_closed_page_is_a_facility_failure_not_an_empty_target_list(monkeypatch) -> None:
    page = FakeObservationPage(closed=True)

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config([make_test_target("a")]))
        await adapter.start()
        try:
            await adapter.observe()
        finally:
            await adapter.stop()

    with pytest.raises(WebBrowserError) as caught:
        run(scenario())
    assert caught.value.code == "WEB_BROWSER_UNAVAILABLE"
    assert caught.value.operation == "observe"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_a_disconnected_session_is_a_facility_failure(monkeypatch) -> None:
    page = FakeObservationPage()
    session = FakeObservationSession(page, connected=False)

    async def scenario():
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config([make_test_target("a")]))
        await adapter.start()
        try:
            await adapter.observe()
        finally:
            await adapter.stop()

    with pytest.raises(WebBrowserError) as caught:
        run(scenario())
    assert caught.value.code == "WEB_BROWSER_UNAVAILABLE"


def test_the_operation_budget_expiry_is_an_observe_timeout(monkeypatch) -> None:
    page = FakeObservationPage()

    async def slow_handle():
        await asyncio.sleep(0.2)
        return FakeHandle(page, {"visible": True, "text": "late", "value": "late"})

    page.specs[("test_id", "a")] = {"count": 1, "element_handle": slow_handle}

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([make_test_target("a")], operation_timeout_ms=50)
        )
        await adapter.start()
        try:
            await adapter.observe()
        finally:
            await adapter.stop()

    with pytest.raises(WebBrowserError) as caught:
        run(scenario())
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"


def test_native_reads_receive_the_remaining_positive_budget(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": 1, "handle": {"visible": True, "text": "t", "value": "v"}}
    })
    observe_with(monkeypatch, page, [make_test_target("a")], operation_timeout_ms=1000)
    assert page.timeouts
    assert all(value > 0 for value in page.timeouts)
    assert all(value <= 1000 for value in page.timeouts)


def test_a_missing_record_is_never_fabricated(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": 0},
        ("test_id", "b"): {"count": FakeBackendError("broken")},
    })
    observation, _, _ = observe_with(
        monkeypatch, page, [make_test_target("a"), make_test_target("b")]
    )
    state = parse_web_observation(observation)
    assert len(state.targets) == 2
    assert state.targets[1].status == "unavailable"
    assert state.targets[1].error_code == "TARGET_QUERY_FAILED"


def test_observation_does_not_reuse_the_caller_target_instances(monkeypatch) -> None:
    supplied = make_test_target("a")
    config = make_config([supplied])
    assert config.observation_targets[0] is not supplied
    page = FakeObservationPage({("test_id", "a"): {"count": 0}})
    observation, adapter, _ = observe_with(monkeypatch, page, [supplied])
    state = parse_web_observation(observation)
    assert state.targets[0].target is not supplied
    assert adapter.cleanup_error is None


def test_budget_expired_is_the_internal_signal_only() -> None:
    assert issubclass(BudgetExpired, Exception)
    assert issubclass(SessionLost, Exception)
    assert not issubclass(BudgetExpired, WebBrowserError)


def test_a_backend_ordinary_error_is_unavailable_and_does_not_leak(monkeypatch) -> None:
    page = FakeObservationPage({
        ("test_id", "a"): {"count": ValueError(f"{MARKER}-backend")},
    })
    observation, _, _ = observe_with(monkeypatch, page, [make_test_target("a")])
    record = parse_web_observation(observation).targets[0]
    assert record.status == "unavailable"
    assert record.error_code == "TARGET_QUERY_FAILED"
    assert record.match_count is None
    assert record.text is None and record.value is None and record.visible is None
    assert MARKER not in str(observation)


def test_an_attribute_read_is_inside_the_one_operation_deadline(monkeypatch) -> None:
    async def cancellation_delayed_visibility():
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
        return True

    page = FakeObservationPage({
        ("test_id", "a"): {
            "count": 1,
            "handle": {"visible": cancellation_delayed_visibility},
        }
    })

    async def scenario():
        session = FakeObservationSession(page)
        install(monkeypatch, session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([make_test_target("a")], operation_timeout_ms=60)
        )
        await adapter.start()
        started = time.monotonic()
        try:
            await adapter.observe()
            code = None
        except WebBrowserError as exc:
            code = exc.code
        elapsed = time.monotonic() - started
        await adapter.stop()
        return code, elapsed

    code, elapsed = run(scenario())
    assert code == "WEB_OBSERVE_TIMEOUT"
    assert elapsed < 0.3, "a cancellation-delayed attribute read must not extend the cap"
