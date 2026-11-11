"""Unit acceptance of the optional Web single-target observation budget.

The tests drive the real ``PlaywrightWebConfig``, the real ``PlaywrightWebAdapter`` and the real
``_browser_observation`` code against an in-memory Playwright-shaped double. No browser is
started here. They prove the default absence, the strict configuration rules, the one-budget-per
-target-group rule, the distinction between a bounded local failure and a whole operation
timeout, the session-loss propagation, the bounded cancellation wind-down and the freshness of
every snapshot.

``pytest-asyncio`` is not installed in this environment, so every asynchronous scenario runs
through ``asyncio.run`` like the existing suites do.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web.playwright_adapter as adapter_module
from forecontract_web._browser_session import BackendTypes, CleanupResult
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import TestIdTarget, WebObservationState

ORIGIN = "http://127.0.0.1:8000"


class FakeTimeoutError(Exception):
    """Stand-in for ``playwright.async_api.TimeoutError``."""


class FakeBackendError(Exception):
    """Stand-in for ``playwright.async_api.Error``."""


TYPES = BackendTypes(FakeTimeoutError, FakeBackendError)


async def _resolve(value: object, default: object) -> object:
    """Resolve one scripted value: exception, callable, awaitable or plain value."""
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
    def __init__(self, page: FakePage, spec: dict) -> None:
        self._page = page
        self._spec = spec

    async def is_visible(self) -> object:
        self._page.reads.append("visible")
        return await _resolve(self._spec.get("visible"), True)

    async def text_content(self) -> object:
        self._page.reads.append("text")
        return await _resolve(self._spec.get("text"), None)

    async def input_value(self, *, timeout: float | None = None) -> object:
        self._page.reads.append("value")
        return await _resolve(self._spec.get("value"), None)


class FakeLocator:
    def __init__(self, page: FakePage, key: tuple, spec: dict) -> None:
        self._page = page
        self._key = key
        self._spec = spec

    async def count(self) -> object:
        self._page.count_calls.append(self._key)
        return await _resolve(self._spec.get("count"), 1)

    async def element_handle(self, *, timeout: float | None = None) -> object:
        self._page.handle_calls.append((self._key, timeout))
        value = self._spec.get("element_handle")
        if value is not None:
            return await _resolve(value, None)
        return FakeHandle(self._page, self._spec.get("handle", {}))


class FakePage:
    def __init__(self, specs: dict | None = None, *,
                 url: str = f"{ORIGIN}/current", closed: bool = False) -> None:
        self.specs = specs or {}
        self._url = url
        self._closed = closed
        self.count_calls: list[tuple] = []
        self.handle_calls: list[tuple] = []
        self.reads: list[str] = []

    @property
    def url(self) -> str:
        return self._url

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def get_by_test_id(self, test_id: str) -> FakeLocator:
        key = ("test_id", test_id)
        return FakeLocator(self, key, self.specs.get(key, {}))

    def get_by_role(self, role: str, *, name: str | None = None, exact: bool | None = None,
                    include_hidden: bool | None = None) -> FakeLocator:
        key = ("role", role, name)
        return FakeLocator(self, key, self.specs.get(key, {}))


class FakeSession:
    def __init__(self, page: FakePage, *, connected: bool = True) -> None:
        self.page = page
        self.types = TYPES
        self.origins = frozenset({("http", "127.0.0.1", 8000)})
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


def make_config(targets: list, **overrides) -> PlaywrightWebConfig:
    arguments = {
        "allowed_origins": [ORIGIN],
        "observation_targets": targets,
        "operation_timeout_ms": 2000,
        "cleanup_timeout_ms": 1000,
    }
    arguments.update(overrides)
    return PlaywrightWebConfig(**arguments)


def _target(test_id: str) -> TestIdTarget:
    """A plain helper: only functions named ``test_*`` are collected as nodes."""
    return TestIdTarget(strategy="test_id", test_id=test_id)


def run(coroutine):
    return asyncio.run(coroutine)


def observe_once(monkeypatch, page: FakePage, targets: list, **overrides):
    """Start one adapter over the fake page, observe once and stop it."""

    async def scenario():
        session = FakeSession(page)
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(make_config(targets, **overrides))
        await adapter.start()
        try:
            observation = await adapter.observe()
        finally:
            await adapter.stop()
        return WebObservationState.model_validate(observation.state)

    return run(scenario())


async def never() -> object:
    """An awaitable that never completes and propagates its cancellation."""
    await asyncio.Event().wait()


def slow(seconds: float):
    """Return a callable whose awaitable completes only after ``seconds``."""

    async def body():
        await asyncio.sleep(seconds)
        return 1

    return body


def slow_text():
    """Return a callable whose awaitable returns a text value after a delay."""

    async def body():
        await asyncio.sleep(0.06)
        return "text"

    return body


# --------------------------------------------------------------------------- configuration


def test_the_optional_target_budget_defaults_to_none_without_changing_repr() -> None:
    config = PlaywrightWebConfig(
        [ORIGIN], [_target("a")], 3000, 1000)
    assert config.observation_target_timeout_ms is None
    assert repr(config) == "PlaywrightWebConfig(<4 validated fields>)"
    # The four historical positional arguments and every default accessor are unchanged.
    assert config.operation_timeout_ms == 3000.0
    assert config.cleanup_timeout_ms == 1000.0
    assert [target.test_id for target in config.observation_targets] == ["a"]


def test_a_declared_target_budget_is_stored_detached_and_revalidated(monkeypatch) -> None:
    config = make_config([], observation_target_timeout_ms=500)
    assert config.observation_target_timeout_ms == 500.0
    # The adapter revalidates through the public constructor and keeps the separated value.
    session = FakeSession(FakePage())
    monkeypatch.setattr(adapter_module, "_create_session", lambda supplied: session)
    adapter = adapter_module.PlaywrightWebAdapter(config)
    assert adapter._config.observation_target_timeout_ms == 500.0
    assert adapter._config.operation_timeout_ms == 2000.0
    # A mutated backing slot is refused by the same revalidation, not trusted.
    broken = make_config([], observation_target_timeout_ms=500)
    object.__setattr__(broken, "_observation_target_timeout_ms", 5000.0)
    with pytest.raises(WebBrowserError) as caught:
        adapter_module.PlaywrightWebAdapter(broken)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


@pytest.mark.parametrize("value", [True, False, "500", 0, -1, -0.5,
                                   float("nan"), float("inf"), 2000, 3000])
def test_an_illegal_target_budget_is_refused(value: object) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config([], observation_target_timeout_ms=value)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert caught.value.operation == "config"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_an_unknown_configuration_keyword_is_still_refused() -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config([], observation_target_timeout_msec=500)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


# --------------------------------------------------------------------------- one budget per target


def test_a_slow_target_is_unavailable_while_a_fast_target_keeps_its_content(monkeypatch) -> None:
    page = FakePage({
        ("test_id", "fast"): {"count": 1,
                              "handle": {"visible": True, "text": "hello", "value": "typed"}},
        ("test_id", "slow"): {"count": never},
    })
    started = time.monotonic()
    state = observe_once(monkeypatch, page, [_target("fast"), _target("slow")],
                         observation_target_timeout_ms=100)
    elapsed = time.monotonic() - started
    by_id = {record.target.test_id: record for record in state.targets}
    assert by_id["fast"].status == "collected"
    assert by_id["fast"].match_count == 1
    assert by_id["fast"].visible is True
    assert by_id["fast"].text == "hello"
    assert by_id["fast"].value == "typed"
    assert by_id["slow"].status == "unavailable"
    assert by_id["slow"].match_count is None
    assert by_id["slow"].visible is None
    assert by_id["slow"].text is None
    assert by_id["slow"].value is None
    assert by_id["slow"].error_code == "TARGET_QUERY_FAILED"
    # The whole observation still returned inside the historical operation allowance.
    assert elapsed < 2.0


def test_a_target_timeout_is_not_a_zero_match(monkeypatch) -> None:
    page = FakePage({("test_id", "slow"): {"count": never}})
    state = observe_once(monkeypatch, page, [_target("slow")],
                         observation_target_timeout_ms=50)
    record = state.targets[0]
    assert record.status == "unavailable"
    assert record.match_count is None
    assert record.error_code == "TARGET_QUERY_FAILED"
    # A fabricated zero match would be a collected record with count 0.
    assert not (record.status == "collected" and record.match_count == 0)


def test_all_attributes_of_one_target_share_its_budget(monkeypatch) -> None:
    page = FakePage({
        ("test_id", "creeping"): {
            "count": lambda: 1,
            "handle": {"visible": slow(0.06), "text": slow_text(), "value": slow(0.06)},
        },
    })
    state = observe_once(monkeypatch, page, [_target("creeping")],
                         observation_target_timeout_ms=100)
    record = state.targets[0]
    assert record.status == "unavailable"
    assert record.match_count is None
    assert record.visible is None and record.text is None and record.value is None


def test_every_observation_rebuilds_its_targets_from_the_current_page(monkeypatch) -> None:
    handle = {"visible": True, "text": "first", "value": "first"}
    page = FakePage({("test_id", "a"): {"count": 1, "handle": handle}})

    async def scenario():
        session = FakeSession(page)
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([_target("a")], observation_target_timeout_ms=1000))
        await adapter.start()
        try:
            first = WebObservationState.model_validate((await adapter.observe()).state)
            handle["text"] = "second"
            handle["value"] = "second"
            second = WebObservationState.model_validate((await adapter.observe()).state)
        finally:
            await adapter.stop()
        return first, second

    first, second = run(scenario())
    assert first.targets[0].text == "first"
    assert first.targets[0].value == "first"
    assert second.targets[0].text == "second"
    assert second.targets[0].value == "second"
    # The repeated observation really queried the page again instead of reusing anything.
    assert len(page.count_calls) == 2


# --------------------------------------------------------------------------- failure boundaries


def test_a_closed_page_is_a_facility_failure_not_a_partial_snapshot(monkeypatch) -> None:
    page = FakePage({("test_id", "a"): {"count": 1}}, closed=True)
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("a")], observation_target_timeout_ms=500)
    assert caught.value.code == "WEB_BROWSER_UNAVAILABLE"


def test_a_disconnected_browser_is_a_facility_failure(monkeypatch) -> None:
    page = FakePage({("test_id", "a"): {"count": 1}})

    async def scenario():
        session = FakeSession(page, connected=False)
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([_target("a")], observation_target_timeout_ms=500))
        await adapter.start()
        try:
            with pytest.raises(WebBrowserError) as caught:
                await adapter.observe()
        finally:
            await adapter.stop()
        return caught.value

    error = run(scenario())
    assert error.code == "WEB_BROWSER_UNAVAILABLE"


def test_a_cancellation_that_does_not_converge_fails_the_observation(monkeypatch) -> None:
    async def stubborn():
        # Swallow the first cancellation once and keep running past the local wind-down. The
        # adapter must not continue collecting or publish a success from this late result.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)
        return 1

    page = FakePage({("test_id", "stuck"): {"count": stubborn}})
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("stuck")],
                     observation_target_timeout_ms=100)
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"
    assert elapsed < 2.0


def test_the_default_path_still_uses_the_whole_operation_budget(monkeypatch) -> None:
    page = FakePage({("test_id", "slow"): {"count": never}})
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("slow")],
                     operation_timeout_ms=200)
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"
    assert elapsed < 2.0


# ---------------------------------------------------------------- F08: one absolute deadline


def test_a_delayed_cancellation_cannot_absorb_a_second_target_allowance(monkeypatch) -> None:
    """F08: the settling of a target shares its one deadline instead of resetting it.

    The historical defect granted a fresh ``target_seconds`` after the target's first slice, so a
    count that settled 140 ms after cancellation still returned a partial snapshot near 340 ms.
    The corrected adapter must end the observation inside the one approved allowance.
    """

    async def delayed():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.14)
        return 1

    page = FakePage({
        ("test_id", "slow"): {"count": delayed},
        ("test_id", "fast"): {"count": 1},
    })
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("slow"), _target("fast")],
                     observation_target_timeout_ms=200)
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"
    # Well below the ~340 ms the reset second allowance produced, and below 2x the 200 ms budget.
    assert elapsed < 0.30
    # The unsettled group is never followed by the next target.
    assert page.count_calls == [("test_id", "slow")]


def test_a_cooperative_settle_inside_the_one_allowance_is_a_bounded_local_failure(
        monkeypatch) -> None:
    page = FakePage({
        ("test_id", "slow"): {"count": never},
        ("test_id", "fast"): {"count": 1,
                              "handle": {"visible": True, "text": "fresh", "value": "fresh"}},
    })
    started = time.monotonic()
    state = observe_once(monkeypatch, page, [_target("slow"), _target("fast")],
                         observation_target_timeout_ms=200)
    elapsed = time.monotonic() - started
    by_id = {record.target.test_id: record for record in state.targets}
    assert by_id["slow"].status == "unavailable"
    assert by_id["slow"].error_code == "TARGET_QUERY_FAILED"
    assert by_id["fast"].status == "collected"
    assert elapsed < 0.30


# ---------------------------------------------------------------- F09: raised failure boundaries


def test_a_session_lost_while_cancellation_settles_is_a_facility_failure(monkeypatch) -> None:
    async def scenario():
        session = FakeSession(FakePage())

        async def lost():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                session.connected = False
                raise

        session.page = FakePage({("test_id", "a"): {"count": lost}})
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([_target("a")], observation_target_timeout_ms=100))
        await adapter.start()
        try:
            with pytest.raises(WebBrowserError) as caught:
                await adapter.observe()
        finally:
            await adapter.stop()
        return caught.value

    error = run(scenario())
    assert error.code == "WEB_BROWSER_UNAVAILABLE"


def test_a_process_control_result_while_settling_propagates(monkeypatch) -> None:
    async def controlled():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise KeyboardInterrupt from None
        return 1

    page = FakePage({("test_id", "a"): {"count": controlled}})
    with pytest.raises(KeyboardInterrupt):
        observe_once(monkeypatch, page, [_target("a")], observation_target_timeout_ms=100)


def test_an_internal_defect_while_settling_is_not_an_unavailable_target(monkeypatch) -> None:
    async def defective_query(session, page, target, deadline, types):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("internal protocol defect") from None

    # The defect is injected at the internal query boundary, outside the per-native-call
    # conversion: only the settling layer can see it, and it must not become TARGET_QUERY_FAILED.
    monkeypatch.setattr(adapter_module._browser_observation, "query_target", defective_query)
    page = FakePage({("test_id", "a"): {"count": 1}})
    with pytest.raises(RuntimeError) as caught:
        observe_once(monkeypatch, page, [_target("a")], observation_target_timeout_ms=100)
    # An internal defect stays an internal defect; it is never relabelled TARGET_QUERY_FAILED.
    assert not isinstance(caught.value, WebBrowserError)


def test_a_live_session_is_revalidated_before_a_successful_snapshot(monkeypatch) -> None:
    holder: dict = {}

    def disconnect_after_read() -> bool:
        holder["session"].connected = False
        return True

    page = FakePage({
        ("test_id", "a"): {"count": 1, "handle": {"visible": disconnect_after_read}},
    })

    async def scenario():
        session = FakeSession(page)
        holder["session"] = session
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([_target("a")], observation_target_timeout_ms=500))
        await adapter.start()
        try:
            with pytest.raises(WebBrowserError) as caught:
                await adapter.observe()
        finally:
            await adapter.stop()
        return caught.value

    error = run(scenario())
    assert error.code == "WEB_BROWSER_UNAVAILABLE"


# ------------------------------------------------- C05-2: the separate startup allowance


def _session_with_slow_start(seconds: float) -> FakeSession:
    """A fake session whose real resource acquisition takes ``seconds``."""
    session = FakeSession(FakePage())

    async def start() -> None:
        session.start_calls += 1
        await asyncio.sleep(seconds)

    session.start = start  # type: ignore[method-assign]
    return session


def test_the_startup_allowance_is_keyword_only_and_absent_by_default() -> None:
    # The five historical positional arguments remain accepted: four required fields plus the
    # optional per-target allowance. The startup allowance itself is keyword-only.
    historical = PlaywrightWebConfig([ORIGIN], [], 3000, 1000, 500)
    assert historical.observation_target_timeout_ms == 500.0
    assert historical.startup_timeout_ms is None
    # An absent startup value keeps the historical behaviour: the operation allowance bounds start.
    assert historical.effective_startup_timeout_ms == 3000.0

    with pytest.raises(TypeError):
        PlaywrightWebConfig([ORIGIN], [], 3000, 1000, None, 10000)  # type: ignore[misc]

    declared = PlaywrightWebConfig([ORIGIN], [], 3000, 1000, startup_timeout_ms=10000)
    assert declared.startup_timeout_ms == 10000.0
    assert declared.effective_startup_timeout_ms == 10000.0
    # Declaring a startup value leaves every historical accessor and the fixed repr unchanged.
    assert declared.operation_timeout_ms == 3000.0
    assert declared.cleanup_timeout_ms == 1000.0
    assert repr(declared) == "PlaywrightWebConfig(<4 validated fields>)"


@pytest.mark.parametrize("value", [True, False, "10000", 0, -1, -0.5,
                                   float("nan"), float("inf")])
def test_an_illegal_startup_allowance_is_refused_with_the_fixed_code(value: object) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config([], startup_timeout_ms=value)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert caught.value.operation == "config"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_a_declared_startup_allowance_is_stored_detached_and_revalidated() -> None:
    config = make_config([], startup_timeout_ms=10000)
    assert config.startup_timeout_ms == 10000.0
    # The adapter rebuilds the configuration through the public constructor and keeps the
    # separated startup value; the operation allowance is untouched.
    adapter = adapter_module.PlaywrightWebAdapter(config)
    assert adapter._config.startup_timeout_ms == 10000.0
    assert adapter._config.effective_startup_timeout_ms == 10000.0
    assert adapter._config.operation_timeout_ms == 2000.0
    # A mutated or forged startup slot is refused by the same revalidation, not trusted.
    broken = make_config([], startup_timeout_ms=10000)
    object.__setattr__(broken, "_startup_timeout_ms", 0.0)
    with pytest.raises(WebBrowserError) as caught:
        adapter_module.PlaywrightWebAdapter(broken)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


def test_an_absent_startup_allowance_keeps_the_historical_operation_budget(monkeypatch) -> None:
    session = _session_with_slow_start(0.5)
    monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
    adapter = adapter_module.PlaywrightWebAdapter(make_config([], operation_timeout_ms=200))
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        run(adapter.start())
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_BROWSER_START_FAILED"
    # The historical fallback really applied: the 200 ms operation allowance bounded the start.
    assert elapsed < 0.8
    assert session.start_calls == 1
    assert adapter._state != "running"


def test_an_explicit_startup_allowance_longer_than_the_operation_budget_is_used(monkeypatch) -> None:
    session = _session_with_slow_start(0.25)
    monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
    adapter = adapter_module.PlaywrightWebAdapter(
        make_config([], operation_timeout_ms=100, startup_timeout_ms=500))

    async def scenario() -> None:
        await adapter.start()
        assert adapter._state == "running"
        await adapter.stop()

    run(scenario())
    assert session.start_calls == 1
    assert adapter.cleanup_error is None


def test_a_declared_startup_allowance_never_widens_the_observation_budget(monkeypatch) -> None:
    page = FakePage({("test_id", "slow"): {"count": never}})
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("slow")],
                     operation_timeout_ms=150, startup_timeout_ms=10000)
    elapsed = time.monotonic() - started
    # observe still ends at the 150 ms operation allowance, never at the startup allowance.
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"
    assert elapsed < 0.5


def test_a_declared_startup_allowance_never_widens_the_single_target_budget(monkeypatch) -> None:
    """F08 under the new parameter: one absolute target deadline still holds."""

    async def delayed():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.14)
        return 1

    page = FakePage({
        ("test_id", "slow"): {"count": delayed},
        ("test_id", "fast"): {"count": 1},
    })
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        observe_once(monkeypatch, page, [_target("slow"), _target("fast")],
                     observation_target_timeout_ms=200, startup_timeout_ms=10000)
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_OBSERVE_TIMEOUT"
    assert elapsed < 0.30
    assert page.count_calls == [("test_id", "slow")]


def test_a_declared_startup_allowance_keeps_the_settled_cancellation_classification(
        monkeypatch) -> None:
    """F09 under the new parameter: a lost session never becomes an unavailable target."""

    async def scenario():
        session = FakeSession(FakePage())

        async def lost():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                session.connected = False
                raise

        session.page = FakePage({("test_id", "a"): {"count": lost}})
        monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config([_target("a")], observation_target_timeout_ms=100,
                        startup_timeout_ms=10000))
        await adapter.start()
        try:
            with pytest.raises(WebBrowserError) as caught:
                await adapter.observe()
        finally:
            await adapter.stop()
        return caught.value

    error = run(scenario())
    assert error.code == "WEB_BROWSER_UNAVAILABLE"


def test_a_start_that_outlives_the_startup_allowance_never_becomes_running(monkeypatch) -> None:
    holder: dict = {}

    async def stubborn_start() -> None:
        # Ignore the abandonment once and keep running: the adapter must not publish RUNNING later.
        holder["session"].start_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)

    session = FakeSession(FakePage())
    holder["session"] = session
    session.start = stubborn_start  # type: ignore[method-assign]
    monkeypatch.setattr(adapter_module, "_create_session", lambda config: session)
    adapter = adapter_module.PlaywrightWebAdapter(
        make_config([], operation_timeout_ms=2000, startup_timeout_ms=100))
    started = time.monotonic()
    with pytest.raises(WebBrowserError) as caught:
        run(adapter.start())
    elapsed = time.monotonic() - started
    assert caught.value.code == "WEB_BROWSER_START_FAILED"
    # The one startup deadline expired; the bounded failed-start cleanup ran, and no late RUNNING
    # state was established from the abandoned acquisition.
    assert elapsed < 0.9
    assert session.start_calls == 1
    assert session.cleanup_calls == 1
    assert adapter._state != "running"
    with pytest.raises(WebBrowserError) as caught_observe:
        run(adapter.observe())
    assert caught_observe.value.code == "WEB_ADAPTER_NOT_RUNNING"

