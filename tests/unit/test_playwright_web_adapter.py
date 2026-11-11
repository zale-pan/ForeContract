"""Unit acceptance tests for the Playwright Web adapter (config, lifecycle, budget, outcomes).

The tests drive the real ``playwright_adapter``, the real ``_browser_session`` cleanup code and
the real ``_browser_actions`` code against an in-memory Playwright-shaped double. No browser is
started here: the doubles prove configuration strictness, the no-I/O constructor, the lifecycle
state table, partial-acquisition release, close ordering, bounded budgets and outcome
classification. The real headless Chromium evidence lives in the integration suite.

``pytest-asyncio`` is not installed in this environment, so every asynchronous scenario runs
through ``asyncio.run`` exactly like the existing suites do.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web._browser_actions as actions_module
import forecontract_web._browser_observation as observation_module
import forecontract_web.playwright_adapter as adapter_module
from forecontract.models.actions import ActionSpec
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
)
from forecontract_web._browser_session import (
    BackendTypes,
    BrowserSession,
    CleanupResult,
)
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import RoleTarget, TestIdTarget

MARKER = "SYNTHETIC_P3T02_MARKER_77af"

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
    """Stand-in for ``playwright.async_api.TimeoutError`` in the doubles."""


class FakeBackendError(Exception):
    """Stand-in for ``playwright.async_api.Error`` in the doubles."""


TYPES = BackendTypes(FakeTimeoutError, FakeBackendError)


# ---------------------------------------------------------------------------- doubles


async def _resolve(value: object, default: object) -> object:
    """Resolve one scripted value: exception, callable, awaitable or plain value."""
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


class FakeResource:
    def __init__(self, backend: FakeBackend, name: str) -> None:
        self._backend = backend
        self._name = name
        self.closed = False

    async def close(self) -> None:
        await self._backend.close_resource(self._name)
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


class FakeDriver(FakeResource):
    async def stop(self) -> None:
        await self.close()


class FakeBrowser(FakeResource):
    def is_connected(self) -> bool:
        return not self.closed


class FakeContext(FakeResource):
    def __init__(self, backend: FakeBackend) -> None:
        super().__init__(backend, "context")
        self.handlers: dict[str, object] = {}

    def on(self, event: str, handler: object) -> None:
        self.handlers[event] = handler


class FakePage(FakeResource):
    def __init__(self, backend: FakeBackend, specs: dict, url: str) -> None:
        super().__init__(backend, "page")
        self.specs = specs
        self._url = url
        self.calls: list[tuple] = []
        self.timeouts: list[float] = []
        self.locator_requests: list[tuple] = []

    @property
    def url(self) -> str:
        return self._url

    def is_closed(self) -> bool:
        return self.closed

    async def goto(self, url: str, *, wait_until: str | None = None,
                   timeout: float | None = None) -> object:
        self.calls.append(("goto", url, wait_until))
        if timeout is not None:
            self.timeouts.append(timeout)
        return await _resolve(self.specs.get("goto"), None)

    def _locator(self, key: tuple, default: dict) -> FakeLocator:
        return FakeLocator(self, dict(self.specs.get(key, default)), key)

    def get_by_test_id(self, test_id: str) -> FakeLocator:
        self.locator_requests.append(("test_id", test_id))
        return self._locator(("test_id", test_id), {})

    def get_by_role(self, role: str, *, name: str | None = None, exact: bool | None = None,
                    include_hidden: bool | None = None) -> FakeLocator:
        self.locator_requests.append(("role", role, name, exact, include_hidden))
        return self._locator(("role", role, name), {})


class FakeLocator:
    def __init__(self, page: FakePage, spec: dict, identity: tuple) -> None:
        self.page = page
        self.spec = spec
        self.identity = identity

    async def _run(self, method: str, default: object, timeout: float | None = None) -> object:
        self.page.calls.append((method, self.identity))
        if timeout is not None:
            self.page.timeouts.append(timeout)
        return await _resolve(self.spec.get(method), default)

    async def count(self) -> object:
        return await self._run("count", 1)

    async def is_visible(self, *, timeout: float | None = None) -> object:
        return await self._run("visible", True, timeout)

    async def is_enabled(self, *, timeout: float | None = None) -> object:
        return await self._run("enabled", True, timeout)

    async def is_editable(self, *, timeout: float | None = None) -> object:
        return await self._run("editable", True, timeout)

    async def click(self, *, timeout: float | None = None) -> object:
        return await self._run("click", None, timeout)

    async def fill(self, value: str, *, timeout: float | None = None) -> object:
        self.page.calls.append(("fill-value", value))
        return await self._run("fill", None, timeout)

    async def press(self, key: str, *, timeout: float | None = None) -> object:
        self.page.calls.append(("press-key", key))
        return await self._run("press", None, timeout)

    async def select_option(self, value: str, *, timeout: float | None = None) -> object:
        self.page.calls.append(("select-value", value))
        return await self._run("select", None, timeout)

    async def wait_for(self, *, state: str | None = None,
                       timeout: float | None = None) -> object:
        self.page.calls.append(("wait-state", state))
        return await self._run("wait_for", None, timeout)


class FakeBackend:
    """A configurable Playwright-shaped double used with the real ``BrowserSession``."""

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        entered: asyncio.Event | None = None,
        gate: asyncio.Event | None = None,
        close_behavior: dict | None = None,
        page_specs: dict | None = None,
        page_url: str = f"{ORIGIN}/",
        connected: bool = True,
    ) -> None:
        self.origins = frozenset({("http", "127.0.0.1", 8000)})
        self.types = TYPES
        self.blocked = False
        self.navigation_blocked = False
        self.log: list[str] = []
        self.close_log: list[str] = []
        self.fail_at = fail_at
        self.entered = entered
        self.gate = gate
        self.close_behavior = close_behavior or {}
        self.close_entered: dict[str, asyncio.Event] = {}
        self.page_specs = page_specs or {}
        self.page_url = page_url
        self.connected = connected
        self.page: FakePage | None = None
        self.driver = FakeDriver(self, "driver")
        self.browser = FakeBrowser(self, "browser")
        self.context = FakeContext(self)

    def clear_marks(self) -> None:
        self.blocked = False
        self.navigation_blocked = False

    async def open_driver(self) -> FakeDriver:
        self.log.append("open_driver")
        if self.fail_at == "driver":
            raise FakeBackendError("synthetic launch failure")
        return self.driver

    async def open_browser(self, driver: object) -> FakeBrowser:
        self.log.append("open_browser")
        if self.fail_at == "browser":
            raise FakeBackendError("synthetic browser failure")
        if not self.connected:
            raise FakeBackendError("synthetic disconnect")
        return self.browser

    async def open_context(self, browser: object) -> FakeContext:
        self.log.append("open_context")
        if self.fail_at == "context":
            raise FakeBackendError("synthetic context failure")
        return self.context

    async def open_page(self, context: object) -> FakePage:
        self.log.append("open_page")
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_at == "page":
            raise FakeBackendError("synthetic page failure")
        self.page = FakePage(self, self.page_specs, self.page_url)
        return self.page

    async def close_resource(self, name: str) -> None:
        self.close_log.append(name)
        entered = self.close_entered.get(name)
        if entered is not None:
            entered.set()
        behavior = self.close_behavior.get(name)
        delay = 0.0
        if isinstance(behavior, dict):
            if behavior.get("fail_first") and not behavior.get("failed_once"):
                behavior["failed_once"] = True
                raise FakeBackendError("synthetic close failure")
            delay = float(behavior.get("delay", 0.0))
            if behavior.get("hang") and not behavior.get("hung_once"):
                behavior["hung_once"] = True
                delay = 30.0
        elif behavior == "error":
            raise FakeBackendError("synthetic close failure")
        elif behavior == "hang":
            delay = 30.0
        elif isinstance(behavior, (int, float)):
            delay = float(behavior)
        if delay:
            await asyncio.sleep(delay)


def make_config(**overrides) -> PlaywrightWebConfig:
    arguments = {
        "allowed_origins": [ORIGIN],
        "observation_targets": [],
        "operation_timeout_ms": 2000,
        "cleanup_timeout_ms": 1000,
    }
    arguments.update(overrides)
    return PlaywrightWebConfig(**arguments)


def action(action_type: str, payload: dict) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type, payload=payload)


def target(test_id: str) -> dict:
    return {"strategy": "test_id", "test_id": test_id}


def install(monkeypatch, backend: FakeBackend) -> list[BrowserSession]:
    created: list[BrowserSession] = []

    def factory(config: PlaywrightWebConfig) -> BrowserSession:
        session = BrowserSession(backend)
        created.append(session)
        return session

    monkeypatch.setattr(adapter_module, "_create_session", factory)
    return created


def run(coroutine):
    return asyncio.run(coroutine)


# ---------------------------------------------------------------------------- B02 config


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "https://localhost:443",
        "https://localhost",
        "http://[::1]:9000",
        "http://LOCALHOST:8080/",
        "http://127.0.0.1",
    ],
)
def test_config_accepts_loopback_origins_with_an_effective_port(origin: str) -> None:
    config = make_config(allowed_origins=[origin])
    assert config.allowed_origins == [origin]
    assert config.operation_timeout_ms == 2000.0
    assert config.cleanup_timeout_ms == 1000.0
    assert config.observation_targets == []


@pytest.mark.parametrize(
    "origin",
    [
        f"http://example.com:80/{MARKER}",
        f"https://localhost:8443/{MARKER}",
        "ftp://127.0.0.1:21",
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000/path",
        "http://127.0.0.1:8000?query=1",
        "http://127.0.0.1:8000#fragment",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:8000\\evil",
        "http://127.0.0.1:8000 ",
        "http://[::2]:8000",
        "http://192.168.1.10:8000",
        "127.0.0.1:8000",
        "",
    ],
)
def test_config_rejects_an_unapproved_or_malformed_origin(origin: str) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(allowed_origins=[origin])
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert caught.value.operation == "config"
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "origins",
    [
        [],
        ("http://127.0.0.1:8000",),
        "http://127.0.0.1:8000",
        ["http://127.0.0.1:8000", "http://127.0.0.1:8000"],
        ["http://127.0.0.1:8000", "http://127.0.0.1:8000/"],
        [1],
    ],
)
def test_config_requires_a_nonempty_unique_origin_list(origins: object) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(allowed_origins=origins)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


@pytest.mark.parametrize(
    "targets",
    [
        [{"strategy": "css", "selector": "#a"}],
        [{"strategy": "test_id", "test_id": ""}],
        [{"strategy": "test_id", "test_id": "a", "extra": 1}],
        [{"strategy": "role", "role": "button", "name": "x", "nth": 0}],
        [{"strategy": "test_id", "test_id": "a"}, {"strategy": "test_id", "test_id": "a"}],
        [{"strategy": "test_id", "test_id": "a"}, {"strategy": "role", "role": "link", "name": ""}],
        "not-a-list",
        [42],
    ],
)
def test_config_rejects_malformed_targets(targets: object) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=targets)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


@pytest.mark.parametrize("field", ["operation_timeout_ms", "cleanup_timeout_ms"])
@pytest.mark.parametrize("value", [True, False, "1000", 0, -1, -0.5, float("nan"), float("inf")])
def test_config_rejects_nonpositive_or_coerced_timeouts(field: str, value: object) -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(**{field: value})
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


@pytest.mark.parametrize("value", [1, 1500, 0.5, 2500.75])
def test_config_accepts_finite_positive_json_numbers(value: object) -> None:
    config = make_config(operation_timeout_ms=value, cleanup_timeout_ms=value)
    assert config.operation_timeout_ms == float(value)


def test_config_requires_all_four_fields_and_rejects_extra_keywords() -> None:
    with pytest.raises(WebBrowserError) as caught:
        PlaywrightWebConfig(allowed_origins=[ORIGIN], observation_targets=[],
                            operation_timeout_ms=1000)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    with pytest.raises(WebBrowserError) as caught:
        make_config(unexpected=1)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


def test_config_rejects_a_bypass_constructed_target() -> None:
    forged = TestIdTarget.model_construct(strategy="test_id", test_id="   ")
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


def test_config_rejects_a_bypass_constructed_configuration_at_the_adapter() -> None:
    forged = object.__new__(PlaywrightWebConfig)
    with pytest.raises(WebBrowserError) as caught:
        adapter_module.PlaywrightWebAdapter(forged)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


def test_config_rejects_a_mutated_backing_slot_at_the_adapter() -> None:
    config = make_config()
    object.__setattr__(config, "_allowed_origins", ("http://example.com:80",))
    with pytest.raises(WebBrowserError) as caught:
        adapter_module.PlaywrightWebAdapter(config)
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"


def test_config_is_detached_from_the_caller_list_and_targets() -> None:
    origins = [ORIGIN]
    locator = TestIdTarget(strategy="test_id", test_id="checkout")
    config = make_config(allowed_origins=origins, observation_targets=[locator])
    origins.append("http://127.0.0.1:8001")
    assert config.allowed_origins == [ORIGIN]
    returned = config.allowed_origins
    returned.append("http://127.0.0.1:8002")
    assert config.allowed_origins == [ORIGIN]
    targets = config.observation_targets
    targets.clear()
    assert [item.test_id for item in config.observation_targets] == ["checkout"]


def test_config_is_immutable_and_has_a_value_free_repr() -> None:
    config = make_config()
    with pytest.raises(AttributeError):
        config.allowed_origins = ["http://127.0.0.1:9000"]  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del config.allowed_origins  # type: ignore[misc]
    assert MARKER not in repr(config)
    assert ORIGIN not in repr(config)


# ---------------------------------------------------------------- B01 declaration and purity


def test_constructor_does_no_io_and_declares_the_fixed_surface(monkeypatch) -> None:
    def forbidden(config: PlaywrightWebConfig) -> BrowserSession:  # pragma: no cover
        raise AssertionError("the constructor must not create a session")

    monkeypatch.setattr(adapter_module, "_create_session", forbidden)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())
    assert adapter.name == "playwright_web"
    capabilities = adapter.capabilities()
    assert capabilities.adapter_name == "playwright_web"
    assert capabilities.environment_type == "web"
    assert capabilities.supported_actions == [
        "web.navigate", "web.click", "web.fill", "web.press", "web.select", "web.wait",
    ]
    assert capabilities.supported_observation_types == ["web.snapshot"]
    rule_types = [
        "web.url_match",
        "web.element_visible",
        "web.element_hidden",
        "web.text_contains",
        "web.value_equals",
    ]
    assert capabilities.supported_condition_types == rule_types
    assert capabilities.supported_outcome_types == rule_types
    assert adapter.capabilities() is not capabilities
    assert adapter.cleanup_error is None


def test_validate_action_is_pure_and_keeps_the_contract_exceptions(monkeypatch) -> None:
    def forbidden(config: PlaywrightWebConfig) -> BrowserSession:  # pragma: no cover
        raise AssertionError("validation must not create a session")

    monkeypatch.setattr(adapter_module, "_create_session", forbidden)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())
    adapter.validate_action(action("web.click", {"target": target("a")}))
    with pytest.raises(UnsupportedActionError):
        adapter.validate_action(action("web.drag", {}))
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(action("web.press", {"target": target("a"), "key": "Control+A"}))
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(action("web.fill", {"target": target("a")}))


_NO_PLAYWRIGHT_PROBE = f"""
import asyncio, json, sys
CORE = {str(CORE_SRC)!r}
WEB = {str(WEB_SRC)!r}
sys.path[:0] = [CORE, WEB]


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "playwright" or name.startswith("playwright."):
            raise ModuleNotFoundError("blocked by the synthetic probe")
        return None


sys.meta_path.insert(0, Blocker())

from forecontract.models.actions import ActionSpec
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.playwright_adapter import PlaywrightWebAdapter

config = PlaywrightWebConfig(allowed_origins=["http://127.0.0.1:8000"], observation_targets=[],
                             operation_timeout_ms=1000, cleanup_timeout_ms=1000)
adapter = PlaywrightWebAdapter(config)
adapter.validate_action(ActionSpec(schema_version="0.1", action_type="web.click",
                                   payload={{"target": {{"strategy": "test_id",
                                                         "test_id": "a"}}}}))


async def main():
    report = {{"name": adapter.name,
              "actions": list(adapter.capabilities().supported_actions)}}
    try:
        await adapter.start()
    except BaseException as exc:
        report.update({{"type": type(exc).__name__, "code": getattr(exc, "code", None),
                       "cause": exc.__cause__ is not None,
                       "context": exc.__context__ is not None,
                       "text": str(exc)}})
    report["playwright_imported"] = any(
        name == "playwright" or name.startswith("playwright.") for name in sys.modules)
    return report


print(json.dumps(asyncio.run(main())))
"""


def test_browser_modules_import_and_construct_without_playwright() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _NO_PLAYWRIGHT_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["name"] == "playwright_web"
    assert report["actions"] == [
        "web.navigate", "web.click", "web.fill", "web.press", "web.select", "web.wait",
    ]
    assert report["type"] == "WebBrowserError"
    assert report["code"] == "WEB_BROWSER_DEPENDENCY_MISSING"
    assert report["cause"] is False
    assert report["context"] is False
    assert "blocked by the synthetic probe" not in report["text"]
    assert "No module named" not in report["text"]
    assert report["playwright_imported"] is False


_IMPORT_ONLY_PROBE = f"""
import json, sys
CORE = {str(CORE_SRC)!r}
WEB = {str(WEB_SRC)!r}
sys.path[:0] = [CORE, WEB]
import forecontract_web.playwright_adapter
import forecontract_web.browser_config
import forecontract_web.browser_errors
import forecontract_web._browser_session
import forecontract_web._browser_actions
import forecontract_web._browser_observation
print(json.dumps({{"playwright": any(name == "playwright" or name.startswith("playwright.")
                                    for name in sys.modules)}}))
"""


def test_importing_the_browser_modules_does_not_import_playwright() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_ONLY_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {"playwright": False}


# ---------------------------------------------------------------------------- B04 lifecycle


def test_new_start_acquires_in_order_and_stop_releases_in_order(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        assert adapter._state == "running"
        assert backend.log == ["open_driver", "open_browser", "open_context", "open_page"]
        assert backend.close_log == []
        await adapter.stop()
        assert adapter._state == "stopped"
        assert backend.close_log == ["page", "context", "browser", "driver"]
        assert adapter.cleanup_error is None

    run(scenario())


def test_repeated_start_is_idempotent_and_creates_no_second_resource_set(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        await adapter.start()
        assert backend.log.count("open_driver") == 1
        assert backend.log.count("open_page") == 1
        await adapter.stop()
        await adapter.stop()
        assert backend.close_log == ["page", "context", "browser", "driver"]

    run(scenario())


def test_start_during_starting_is_busy_and_observe_execute_are_busy(monkeypatch) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def scenario() -> None:
        backend = FakeBackend(entered=entered, gate=gate)
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config())
        starting = asyncio.ensure_future(adapter.start())
        await entered.wait()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_ADAPTER_BUSY"
        with pytest.raises(WebBrowserError) as caught:
            await adapter.observe()
        assert caught.value.code == "WEB_ADAPTER_BUSY"
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(action("web.click", {"target": target("a")}))
        assert caught.value.code == "WEB_ADAPTER_BUSY"
        await adapter.stop()
        assert adapter._state == "stopped"
        assert backend.close_log == ["context", "browser", "driver"]
        with pytest.raises(WebBrowserError) as caught:
            await starting
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert adapter._state == "stopped"

    run(scenario())


def test_stop_during_starting_releases_partial_resources_without_fresh_launch(monkeypatch) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def scenario() -> None:
        backend = FakeBackend(entered=entered, gate=gate)
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config())
        starting = asyncio.ensure_future(adapter.start())
        await entered.wait()
        await adapter.stop()
        with pytest.raises(WebBrowserError) as caught:
            await starting
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert adapter._state == "stopped"
        assert backend.close_log == ["context", "browser", "driver"]
        assert backend.log.count("open_page") == 1

    run(scenario())


@pytest.mark.parametrize("operation", ["observe", "execute"])
def test_operations_require_running_and_check_before_io(monkeypatch, operation: str) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        with pytest.raises(WebBrowserError) as caught:
            if operation == "observe":
                await adapter.observe()
            else:
                await adapter.execute(action("web.click", {"target": target("a")}))
        assert caught.value.code == "WEB_ADAPTER_NOT_RUNNING"
        assert backend.page is None
        await adapter.start()
        await adapter.stop()
        backend.page.calls.clear()
        with pytest.raises(WebBrowserError) as caught:
            if operation == "observe":
                await adapter.observe()
            else:
                await adapter.execute(action("web.click", {"target": target("a")}))
        assert caught.value.code == "WEB_ADAPTER_NOT_RUNNING"
        assert backend.page is not None and backend.page.calls == []

    run(scenario())


def test_new_stop_is_a_noop_and_stopped_is_not_restartable(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.stop()
        assert adapter._state == "stopped"
        await adapter.stop()
        assert backend.log == []
        assert backend.close_log == []
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_ADAPTER_STOPPED"

    run(scenario())


@pytest.mark.parametrize("fail_at", ["driver", "browser", "context", "page"])
def test_start_failure_releases_the_acquired_partial_resources(monkeypatch, fail_at: str) -> None:
    backend = FakeBackend(fail_at=fail_at)
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        expected = {
            "driver": [],
            "browser": ["driver"],
            "context": ["browser", "driver"],
            "page": ["context", "browser", "driver"],
        }[fail_at]
        assert backend.close_log == expected

    run(scenario())


def test_cleanup_failure_is_bounded_and_exposed_then_a_new_stop_retries(monkeypatch) -> None:
    backend = FakeBackend(close_behavior={"context": {"fail_first": True}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.stop()
        assert caught.value.code == "WEB_BROWSER_CLEANUP_FAILED"
        assert adapter._state == "failed"
        assert adapter.cleanup_error is not None
        assert adapter.cleanup_error.code == "WEB_BROWSER_CLEANUP_FAILED"
        assert backend.close_log == ["page", "context", "browser", "driver"]
        await adapter.stop()
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        # Each attempt closes every remaining owned handle once; the closed page is not retried.
        assert backend.close_log.count("page") == 1
        assert backend.close_log.count("context") == 2
        assert backend.close_log.count("browser") == 1
        assert backend.close_log.count("driver") == 1

    run(scenario())


def test_cleanup_timeout_is_one_bounded_allowance(monkeypatch) -> None:
    backend = FakeBackend(close_behavior={"page": {"hang": True}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config(cleanup_timeout_ms=200))

    async def scenario() -> None:
        await adapter.start()
        started = time.monotonic()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.stop()
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert caught.value.code == "WEB_BROWSER_CLEANUP_TIMEOUT"
        assert adapter.cleanup_error is not None
        assert adapter.cleanup_error.code == "WEB_BROWSER_CLEANUP_TIMEOUT"
        assert adapter._state == "failed"
        await adapter.stop()
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None

    run(scenario())


def test_concurrent_stop_joins_the_same_cleanup_attempt(monkeypatch) -> None:
    backend = FakeBackend(close_behavior={"page": 0.1, "context": 0.1})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        await asyncio.gather(adapter.stop(), adapter.stop())
        assert adapter._state == "stopped"
        assert backend.close_log == ["page", "context", "browser", "driver"]

    run(scenario())


def test_host_cancellation_during_start_propagates_and_still_releases(monkeypatch) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def scenario() -> None:
        backend = FakeBackend(entered=entered, gate=gate)
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config())
        starting = asyncio.ensure_future(adapter.start())
        await entered.wait()
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert adapter._state != "running"
        assert backend.close_log == ["context", "browser", "driver"]
        await adapter.stop()
        assert backend.close_log == ["context", "browser", "driver"]

    run(scenario())


def test_start_is_bounded_by_the_one_operation_budget(monkeypatch) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def scenario() -> None:
        backend = FakeBackend(entered=entered, gate=gate)
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config(operation_timeout_ms=150))
        starting = asyncio.ensure_future(adapter.start())
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        started = time.monotonic()
        with pytest.raises(WebBrowserError) as caught:
            await starting
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        assert backend.close_log == ["context", "browser", "driver"]

    run(scenario())


def test_cancelled_execute_propagates_without_a_forged_outcome(monkeypatch) -> None:
    async def hang():
        await asyncio.sleep(30)

    backend = FakeBackend(page_specs={("test_id", "a"): {"click": hang}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        executing = asyncio.ensure_future(
            adapter.execute(action("web.click", {"target": target("a")}))
        )
        await asyncio.sleep(0.05)
        executing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await executing
        assert adapter._active is False
        await adapter.stop()
        assert adapter._state == "stopped"

    run(scenario())


def test_internal_semantic_errors_are_not_swallowed(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()

        def broken_collect(session, targets, deadline):
            raise RuntimeError("internal observation defect")

        monkeypatch.setattr(observation_module, "collect", broken_collect)
        with pytest.raises(RuntimeError, match="internal observation defect"):
            await adapter.observe()

        def broken_perform(session, parsed, deadline, started):
            raise RuntimeError("internal action defect")

        monkeypatch.setattr(actions_module, "perform", broken_perform)
        with pytest.raises(RuntimeError, match="internal action defect"):
            await adapter.execute(action("web.click", {"target": target("a")}))
        await adapter.stop()

    run(scenario())


# ---------------------------------------------------------------------------- B10 budget


def test_operation_budget_is_one_monotonic_deadline_per_call(monkeypatch) -> None:
    async def slow_count():
        await asyncio.sleep(0.2)
        return 1

    async def slow_true():
        await asyncio.sleep(0.1)
        return True

    async def slow_enabled():
        await asyncio.sleep(0.15)
        return True

    backend = FakeBackend(
        page_specs={
            ("test_id", "a"): {
                "count": [slow_count, 1],
                "visible": slow_true,
                "enabled": slow_enabled,
            }
        },
    )
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config(operation_timeout_ms=1000))

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        assert result.status.value == "succeeded"
        assert backend.page is not None
        timeouts = backend.page.timeouts
        assert timeouts, "native calls must receive a finite timeout"
        assert all(value > 0 for value in timeouts)
        assert all(value <= 1000 for value in timeouts)
        # The click receives only the remaining budget: roughly 1000 - 200 - 100 - 150 ms.
        assert timeouts[-1] < timeouts[0] - 150
        assert timeouts[-1] < 700
        await adapter.stop()

    run(scenario())


def test_wait_timeout_is_classified_from_a_fresh_count(monkeypatch) -> None:
    backend = FakeBackend(
        page_specs={("test_id", "a"): {"count": [0, 0], "wait_for": FakeTimeoutError("t")}},
    )
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(
            action("web.wait", {"target": target("a"), "state": "visible"})
        )
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_NOT_FOUND"
        await adapter.stop()

    run(scenario())


# ---------------------------------------------------------------------------- B05/B08 actions


def test_successful_actions_report_execution_facts_only(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        for action_type, payload in (
            ("web.navigate", {"url": f"{ORIGIN}/ok"}),
            ("web.click", {"target": target("a")}),
            ("web.fill", {"target": target("a"), "text": "hello"}),
            ("web.press", {"target": target("a"), "key": "Enter"}),
            ("web.select", {"target": target("a"), "value": "b"}),
            ("web.wait", {"target": target("a"), "state": "attached"}),
        ):
            result = await adapter.execute(action(action_type, payload))
            assert result.status.value == "succeeded", action_type
            assert result.error is None
            assert result.outcome_id
            assert result.state_before is None and result.state_after is None
            assert result.artifacts == []
            assert result.changes == {}
            assert result.duration_ms is not None and result.duration_ms >= 0
        await adapter.stop()

    run(scenario())


def test_zero_match_is_a_deterministic_failed_refusal(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "missing"): {"count": 0}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("missing")}))
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_NOT_FOUND"
        assert backend.page is not None
        assert ("click", ("test_id", "missing")) not in backend.page.calls
        await adapter.stop()

    run(scenario())


def test_multiple_matches_are_a_deterministic_failed_refusal(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "dupe"): {"count": 2}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("dupe")}))
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_AMBIGUOUS"
        assert backend.page is not None
        assert ("click", ("test_id", "dupe")) not in backend.page.calls
        await adapter.stop()

    run(scenario())


@pytest.mark.parametrize("spec", [{"visible": False}, {"enabled": False}])
def test_non_actionable_target_is_a_deterministic_failed_refusal(monkeypatch, spec: dict) -> None:
    backend = FakeBackend(page_specs={("test_id", "a"): spec})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_NOT_ACTIONABLE"
        assert backend.page is not None
        assert ("click", ("test_id", "a")) not in backend.page.calls
        await adapter.stop()

    run(scenario())


def test_fill_rejects_a_non_editable_target_before_the_action(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "a"): {"editable": False}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.fill", {"target": target("a"), "text": "x"}))
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_NOT_ACTIONABLE"
        assert backend.page is not None
        assert ("fill", ("test_id", "a")) not in backend.page.calls
        await adapter.stop()

    run(scenario())


def test_empty_fill_and_select_values_are_accepted_and_passed_verbatim(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        filled = await adapter.execute(action("web.fill", {"target": target("a"), "text": ""}))
        selected = await adapter.execute(action("web.select", {"target": target("a"), "value": ""}))
        assert filled.status.value == "succeeded"
        assert selected.status.value == "succeeded"
        assert backend.page is not None
        assert ("fill-value", "") in backend.page.calls
        assert ("select-value", "") in backend.page.calls
        await adapter.stop()

    run(scenario())


def test_click_timeout_is_unknown_and_never_retried(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "a"): {"click": FakeTimeoutError("t")}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        assert result.status.value == "unknown"
        assert result.error is not None and result.error.code == "WEB_ACTION_TIMEOUT"
        assert backend.page is not None
        assert backend.page.calls.count(("click", ("test_id", "a"))) == 1
        await adapter.stop()

    run(scenario())


def test_click_backend_error_is_unknown_interrupted_and_never_retried(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "a"): {"click": FakeBackendError("e")}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        assert result.status.value == "unknown"
        assert result.error is not None and result.error.code == "WEB_ACTION_INTERRUPTED"
        assert backend.page is not None
        assert backend.page.calls.count(("click", ("test_id", "a"))) == 1
        await adapter.stop()

    run(scenario())


def test_navigate_timeout_is_unknown(monkeypatch) -> None:
    backend = FakeBackend(page_specs={"goto": FakeTimeoutError("t")})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.navigate", {"url": f"{ORIGIN}/slow"}))
        assert result.status.value == "unknown"
        assert result.error is not None and result.error.code == "WEB_ACTION_TIMEOUT"
        await adapter.stop()

    run(scenario())


def test_navigate_to_an_unapproved_origin_is_blocked_before_backend(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(
                action("web.navigate", {"url": f"http://example.com/{MARKER}"})
            )
        assert caught.value.code == "WEB_ORIGIN_BLOCKED"
        assert MARKER not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert backend.page is not None
        assert not [call for call in backend.page.calls if call[0] == "goto"]
        await adapter.stop()

    run(scenario())


def test_navigate_redirect_blocked_by_policy_is_origin_blocked(monkeypatch) -> None:
    backend = FakeBackend()

    def blocked_goto():
        backend.navigation_blocked = True
        raise FakeBackendError("blocked")

    backend.page_specs["goto"] = blocked_goto
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(action("web.navigate", {"url": f"{ORIGIN}/redirect"}))
        assert caught.value.code == "WEB_ORIGIN_BLOCKED"
        await adapter.stop()

    run(scenario())


def test_detached_wait_zero_match_is_a_success(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "gone"): {"count": 0}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(
            action("web.wait", {"target": target("gone"), "state": "detached"})
        )
        assert result.status.value == "succeeded"
        await adapter.stop()

    run(scenario())


def test_wait_multiple_matches_never_picks_one(monkeypatch) -> None:
    backend = FakeBackend(page_specs={("test_id", "dupe"): {"count": 2}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(
            action("web.wait", {"target": target("dupe"), "state": "visible"})
        )
        assert result.status.value == "failed"
        assert result.error is not None and result.error.code == "WEB_TARGET_AMBIGUOUS"
        await adapter.stop()

    run(scenario())


def test_role_target_uses_exact_name_and_hidden_nodes(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(
            action("web.click", {"target": {"strategy": "role", "role": "button",
                                            "name": "Exact Name"}})
        )
        assert result.status.value == "succeeded"
        assert backend.page is not None
        assert ("role", "button", "Exact Name", True, True) in backend.page.locator_requests
        assert not [item for item in backend.page.locator_requests
                    if item[0] == "test_id"]
        await adapter.stop()

    run(scenario())


def test_execute_never_mutates_the_caller_action(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        supplied = action("web.fill", {"target": target("a"), "text": "x"})
        before = supplied.model_dump(mode="json")
        result = await adapter.execute(supplied)
        assert result.status.value == "succeeded"
        assert supplied.model_dump(mode="json") == before
        await adapter.stop()

    run(scenario())


def test_validation_exceptions_survive_inside_execute(monkeypatch) -> None:
    backend = FakeBackend()
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(UnsupportedActionError):
            await adapter.execute(action("web.drag", {}))
        with pytest.raises(InvalidActionPayloadError):
            await adapter.execute(action("web.press", {"target": target("a"), "key": "Fn"}))
        assert backend.page is not None
        assert backend.page.calls == []
        await adapter.stop()

    run(scenario())


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(monkeypatch, interrupt: type) -> None:
    def raised():
        raise interrupt("host interrupt")

    backend = FakeBackend(page_specs={("test_id", "a"): {"click": raised}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(interrupt):
            await adapter.execute(action("web.click", {"target": target("a")}))
        assert adapter._active is False
        await adapter.stop()

    run(scenario())


def test_public_diagnostics_never_carry_backend_text_or_raw_input(monkeypatch) -> None:
    backend = FakeBackend(
        page_specs={("test_id", "a"): {"fill": FakeBackendError(f"{MARKER}-backend")}},
    )
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(
            action("web.fill", {"target": target("a"), "text": MARKER})
        )
        assert result.status.value == "unknown"
        assert result.error is not None
        rendered = str(result)
        assert MARKER not in rendered
        assert result.error.message == "the action result could not be confirmed"
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(
                action("web.navigate", {"url": f"http://example.com/{MARKER}"})
            )
        assert caught.value.code == "WEB_ORIGIN_BLOCKED"
        assert MARKER not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        await adapter.stop()

    run(scenario())


def test_concurrent_operations_are_rejected_busy_before_a_second_backend_call(monkeypatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_click():
        entered.set()
        await release.wait()

    backend = FakeBackend(page_specs={("test_id", "a"): {"click": gated_click}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        first = asyncio.ensure_future(
            adapter.execute(action("web.click", {"target": target("a")}))
        )
        await entered.wait()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.observe()
        assert caught.value.code == "WEB_ADAPTER_BUSY"
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(action("web.click", {"target": target("a")}))
        assert caught.value.code == "WEB_ADAPTER_BUSY"
        assert backend.page is not None
        assert backend.page.calls.count(("click", ("test_id", "a"))) == 1
        release.set()
        result = await first
        assert result.status.value == "succeeded"
        await adapter.stop()

    run(scenario())


def test_cleanup_result_shape_and_error_labels_are_fixed() -> None:
    error = WebBrowserError("WEB_BROWSER_CLEANUP_FAILED", "cleanup")
    assert error.code == "WEB_BROWSER_CLEANUP_FAILED"
    assert error.operation == "cleanup"
    assert error.message == "the browser resources could not be released"
    assert CleanupResult(True, None).complete is True
    with pytest.raises(ValueError):
        WebBrowserError("NOT_A_CODE", "cleanup")
    with pytest.raises(ValueError):
        WebBrowserError("WEB_BROWSER_CLEANUP_FAILED", "not-an-operation")


# ------------------------------------------------------- F01 caller-owned config reads


class ForgedOrigin(str):
    """An origin string whose ``__str__`` claims a different, accepted value."""

    def __str__(self) -> str:
        return ORIGIN


class HiddenOrigin(str):
    """A valid origin whose ``__str__`` claims a different, rejected value."""

    def __str__(self) -> str:
        return "http://example.invalid"


class ExplosiveStr(str):
    """A string whose ``__str__`` raises an ordinary caller exception."""

    def __str__(self) -> str:
        raise ValueError(MARKER)


class RenamedKey(str):
    """An unknown mapping key whose ``__str__`` claims to be ``strategy``."""

    def __str__(self) -> str:
        return "strategy"


class EqualityBomb(str):
    """A strategy value whose equality comparison raises an ordinary caller exception."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise ValueError(MARKER)

    def __ne__(self, other: object) -> bool:
        raise ValueError(MARKER)


class UnreadableMapping(dict):
    """A caller-owned mapping whose entry read raises an ordinary exception."""

    def items(self):
        raise ValueError(MARKER)


class PlainSubclass(str):
    """A normal string subclass with no behavioural override."""


def test_f01_config_reads_the_original_origin_text_without_caller_str() -> None:
    for supplied in (ForgedOrigin("https://example.invalid"), ExplosiveStr("ftp://127.0.0.1")):
        with pytest.raises(WebBrowserError) as caught:
            make_config(allowed_origins=[supplied])
        assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
        assert caught.value.operation == "config"
        assert MARKER not in str(caught.value)
        assert "example.invalid" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    # The original characters are validated in the accepted direction too: a valid stored
    # origin is accepted even when ``__str__`` reports something else.
    accepted = make_config(allowed_origins=[HiddenOrigin(ORIGIN), PlainSubclass("http://localhost:9000")])
    assert accepted.allowed_origins == [ORIGIN, "http://localhost:9000"]
    assert all(type(origin) is str for origin in accepted.allowed_origins)


def test_f01_config_reads_the_original_mapping_key_without_caller_str() -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(
            observation_targets=[{RenamedKey("unknown_key"): "test_id", "test_id": "x"}]
        )
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_f01_config_converts_a_caller_equality_failure_to_the_bounded_error() -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(
            observation_targets=[{"strategy": EqualityBomb("test_id"), "test_id": "x"}]
        )
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_f01_config_converts_a_caller_mapping_read_failure() -> None:
    with pytest.raises(WebBrowserError) as caught:
        make_config(
            observation_targets=[UnreadableMapping(strategy="test_id", test_id="x")]
        )
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_f01_config_converts_a_caller_model_read_failure() -> None:
    forged = object.__new__(TestIdTarget)  # no __init__: the stored-field read is incomplete
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_f01_config_keeps_plain_string_mapping_and_model_controls() -> None:
    config = make_config(
        allowed_origins=[PlainSubclass(ORIGIN)],
        observation_targets=[
            {"strategy": "test_id", "test_id": "plain-mapping"},
            TestIdTarget(strategy="test_id", test_id="plain-model"),
            {"strategy": "role", "role": "button", "name": "Plain Role"},
        ],
    )
    assert config.allowed_origins == [ORIGIN]
    assert [item.model_dump() for item in config.observation_targets] == [
        {"strategy": "test_id", "test_id": "plain-mapping"},
        {"strategy": "test_id", "test_id": "plain-model"},
        {"strategy": "role", "role": "button", "name": "Plain Role"},
    ]


# ------------------------------------------------- F02 whole-operation and cleanup bounds


async def _cancellation_delayed(delay: float, pause: float, value: object):
    """Sleep, then keep sleeping even when cancelled before returning ``value``."""
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        await asyncio.sleep(pause)
    return value


def test_f02_count_and_attribute_reads_stay_inside_the_one_operation_deadline(
    monkeypatch,
) -> None:
    backend = FakeBackend(
        page_specs={("test_id", "a"): {"count": lambda: _cancellation_delayed(0.3, 0.2, 0)}}
    )
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config(operation_timeout_ms=60))

    async def scenario() -> None:
        await adapter.start()
        started = time.monotonic()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        elapsed = time.monotonic() - started
        assert result.status.value == "unknown"
        assert result.error is not None and result.error.code == "WEB_ACTION_TIMEOUT"
        assert elapsed < 0.3
        await adapter.stop()

    run(scenario())


def test_f02_joining_stop_does_not_reset_the_one_cleanup_bound(monkeypatch) -> None:
    backend = FakeBackend(close_behavior={"page": "hang"})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config(cleanup_timeout_ms=300))

    async def scenario() -> None:
        await adapter.start()
        started = time.monotonic()
        first = asyncio.ensure_future(adapter.stop())
        # One explicit barrier: the first stop has created the one cleanup attempt before the
        # joining stops are issued, so the same-attempt oracle below is deterministic.
        await asyncio.sleep(0)
        attempt = adapter._cleanup_task
        deadline = adapter._cleanup_deadline
        assert attempt is not None and deadline is not None
        await asyncio.sleep(0.1)
        second = asyncio.ensure_future(adapter.stop())
        await asyncio.sleep(0.1)
        third = asyncio.ensure_future(adapter.stop())
        results = await asyncio.gather(first, second, third, return_exceptions=True)
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, "a joining stop must not receive a fresh cleanup allowance"
        assert adapter._cleanup_task is attempt, "a joining stop must join the same attempt"
        assert adapter._cleanup_deadline == deadline, "a joining stop must not reset the deadline"
        assert all(
            isinstance(item, WebBrowserError)
            and item.code == "WEB_BROWSER_CLEANUP_TIMEOUT"
            for item in results
        )
        # The one attempt spends its budget on the hanging page. Any later handle is attempted
        # only when a positive remainder of that same deadline is left, so its presence is not a
        # budget-reset oracle; each handle may be attempted at most once by the one attempt.
        assert backend.close_log.count("page") == 1
        assert backend.close_log[0] == "page"
        assert len(backend.close_log) == len(set(backend.close_log))
        assert adapter._state == "failed"

    run(scenario())


def test_f02_cancellation_delayed_start_never_becomes_running(monkeypatch) -> None:
    backend = FakeBackend()
    real_open_page = backend.open_page

    async def delayed_open_page(context: object) -> object:
        await _cancellation_delayed(0.3, 0.2, None)
        return await real_open_page(context)

    backend.open_page = delayed_open_page
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(
        make_config(operation_timeout_ms=60, cleanup_timeout_ms=60)
    )

    async def scenario() -> None:
        started = time.monotonic()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        elapsed = time.monotonic() - started
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert elapsed < 0.3
        assert adapter._state != "running"
        await asyncio.sleep(0.45)
        assert adapter._state != "running"
        session = adapter._session
        assert session is not None
        assert session.page is None
        assert session._page is None
        assert "page" in backend.close_log
        with pytest.raises(WebBrowserError) as stopped:
            await adapter.start()
        assert stopped.value.code == "WEB_ADAPTER_STOPPED"

    run(scenario())


# ----------------------------------------------------- F03/F04 post-interaction UNKNOWN


def test_f03_post_interaction_disconnect_is_unknown_and_never_replayed(monkeypatch) -> None:
    backend = FakeBackend()
    applied: list[str] = []

    async def applied_then_disconnect():
        applied.append("side-effect")
        backend.browser.closed = True
        raise FakeBackendError(f"{MARKER}-backend")

    backend.page_specs = {("test_id", "a"): {"click": applied_then_disconnect}}
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        result = await adapter.execute(action("web.click", {"target": target("a")}))
        assert result.status.value == "unknown"
        assert result.error is not None and result.error.code == "WEB_ACTION_INTERRUPTED"
        assert MARKER not in str(result)
        assert applied == ["side-effect"]
        assert backend.page is not None
        assert backend.page.calls.count(("click", ("test_id", "a"))) == 1
        await adapter.stop()

    run(scenario())


def test_f04_backend_ordinary_error_is_converted_at_the_call_boundary(monkeypatch) -> None:
    backend = FakeBackend(
        page_specs={("test_id", "a"): {"count": ValueError(f"{MARKER}-backend")}}
    )
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(WebBrowserError) as caught:
            await adapter.execute(action("web.click", {"target": target("a")}))
        assert caught.value.code == "WEB_BROWSER_UNAVAILABLE"
        assert caught.value.operation == "execute"
        assert MARKER not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        await adapter.stop()

    run(scenario())


def test_f04_cancelled_error_from_a_backend_call_propagates(monkeypatch) -> None:
    async def cancelled_count():
        raise asyncio.CancelledError()

    backend = FakeBackend(page_specs={("test_id", "a"): {"count": cancelled_count}})
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config())

    async def scenario() -> None:
        await adapter.start()
        with pytest.raises(asyncio.CancelledError):
            await adapter.execute(action("web.click", {"target": target("a")}))
        assert adapter._active is False
        await adapter.stop()

    run(scenario())


# ------------------------------------------- F01/F02 round-003 remaining corrections


class HostileExtraStore(dict):
    """A stored extra store whose truthiness raises an ordinary caller exception."""

    def __bool__(self) -> bool:
        raise ValueError(MARKER)


class HostileExtraReadTestId(TestIdTarget):
    """A stored locator whose extra-slot read raises an ordinary caller exception."""

    @property
    def __pydantic_extra__(self) -> object:
        raise ValueError(MARKER)

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value: object) -> None:
        pass


class HostileExtraReadRole(RoleTarget):
    """The role-target form of the raising extra-slot read."""

    @property
    def __pydantic_extra__(self) -> object:
        raise ValueError(MARKER)

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value: object) -> None:
        pass


class FalseTruthinessExtraStore(dict):
    """A stored extra store that really holds a key but reports itself empty.

    Both ``__bool__`` and ``__len__`` are caller-owned answers: either one returning ``False``
    or ``0`` must never be accepted as proof that the store holds no entry.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.bool_reads = 0
        self.len_reads = 0

    def __bool__(self) -> bool:
        self.bool_reads += 1
        return False

    def __len__(self) -> int:
        self.len_reads += 1
        return 0


class HiddenStoredFields(dict):
    """A stored field mapping whose announced view hides an entry that is really stored.

    ``keys``/``__iter__`` are caller-owned views, so ``dict(stored)`` silently omits the hidden
    entry; the real underlying dict entries are the only trustworthy source.
    """

    def __init__(self, visible: list[str], **fields: object) -> None:
        super().__init__(**fields)
        self._visible = list(visible)

    def __iter__(self):
        return iter(self._visible)

    def keys(self):
        return list(self._visible)


class CleanStoredFields(dict):
    """A normal dict subclass with no behavioural override at all."""


class CleanTestIdSubclass(TestIdTarget):
    """A normal locator subclass with no behavioural override at all."""


def _assert_bounded_config_error(caught) -> None:
    """Every hostile stored read is the fixed error with no raw value and no exception chain."""
    assert caught.value.code == "WEB_BROWSER_CONFIG_INVALID"
    assert caught.value.operation == "config"
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "forged",
    [
        TestIdTarget.model_construct(strategy="test_id", test_id="hostile"),
        RoleTarget.model_construct(strategy="role", role="button", name="Hostile"),
    ],
    ids=["test_id", "role"],
)
def test_f01_config_converts_a_hostile_stored_extra_truthiness(forged) -> None:
    """X01: the extra store's own truthiness is caller code and must run inside the boundary."""
    object.__setattr__(forged, "__pydantic_extra__", HostileExtraStore(hidden="synthetic"))
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    _assert_bounded_config_error(caught)


@pytest.mark.parametrize(
    "forged",
    [
        HostileExtraReadTestId(strategy="test_id", test_id="hostile"),
        HostileExtraReadRole(strategy="role", role="button", name="Hostile"),
    ],
    ids=["test_id", "role"],
)
def test_f01_config_converts_a_hostile_stored_extra_read(forged) -> None:
    """A stored slot read that raises a caller exception is the same bounded configuration error."""
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    _assert_bounded_config_error(caught)


@pytest.mark.parametrize("empty_extra", [{}, None], ids=["empty-mapping", "none"])
def test_f01_config_accepts_a_stored_model_with_a_clean_extra_store(empty_extra) -> None:
    """The positive control: the two ordinary "no extra keys" shapes stay accepted."""
    forged = TestIdTarget.model_construct(strategy="test_id", test_id="clean")
    object.__setattr__(forged, "__pydantic_extra__", empty_extra)
    accepted = make_config(observation_targets=[forged])
    assert [item.model_dump() for item in accepted.observation_targets] == [
        {"strategy": "test_id", "test_id": "clean"}
    ]


@pytest.mark.parametrize(
    "forged",
    [
        TestIdTarget.model_construct(strategy="test_id", test_id="hostile"),
        RoleTarget.model_construct(strategy="role", role="button", name="Hostile"),
    ],
    ids=["test_id", "role"],
)
def test_f01_config_rejects_a_stored_extra_store_that_reports_itself_empty(forged) -> None:
    """X01b: false truthiness/length cannot prove a non-empty extra store empty.

    The stored store really holds ``hidden``; its ``__bool__`` returns ``False`` and its
    ``__len__`` returns ``0``. Presence must be read from the real entries, so the locator is
    rejected with the fixed bounded error and neither caller hook is even consulted.
    """
    store = FalseTruthinessExtraStore(hidden=MARKER)
    assert list(dict.items(store)) == [("hidden", MARKER)], "the fixture really stores the key"
    object.__setattr__(forged, "__pydantic_extra__", store)
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    _assert_bounded_config_error(caught)
    assert store.bool_reads == 0, "truthiness is not evidence that the store holds no entry"
    assert store.len_reads == 0, "length is not evidence that the store holds no entry"


@pytest.mark.parametrize(
    "forged, fields",
    [
        (
            TestIdTarget.model_construct(strategy="test_id", test_id="hostile"),
            {"strategy": "test_id", "test_id": "hostile"},
        ),
        (
            RoleTarget.model_construct(strategy="role", role="button", name="Hostile"),
            {"strategy": "role", "role": "button", "name": "Hostile"},
        ),
    ],
    ids=["test_id", "role"],
)
def test_f01_config_rejects_a_stored_copy_that_omits_a_real_unknown_field(
    forged, fields
) -> None:
    """X01c: the real stored mapping, not a detached ``dict(stored)`` copy, is revalidated.

    The stored mapping really holds ``hidden`` while its ``keys``/``__iter__`` announce only the
    declared fields; ``dict(stored)`` silently drops the hidden entry, which is exactly the
    round-003 bypass. The real entries are read instead, so the locator is rejected.
    """
    stored = HiddenStoredFields(list(fields), **fields, hidden=MARKER)
    assert list(dict.items(stored)) == [*fields.items(), ("hidden", MARKER)]
    assert sorted(dict(stored)) == sorted(fields), "the detached copy loses the hidden entry"
    object.__setattr__(forged, "__dict__", stored)
    with pytest.raises(WebBrowserError) as caught:
        make_config(observation_targets=[forged])
    _assert_bounded_config_error(caught)


@pytest.mark.parametrize(
    "stored",
    [
        {"strategy": "test_id", "test_id": "clean"},
        CleanStoredFields(strategy="test_id", test_id="clean"),
    ],
    ids=["plain-dict", "dict-subclass"],
)
def test_f01_config_accepts_a_clean_stored_mapping(stored) -> None:
    """The positive control: clean plain dicts and clean dict subclasses stay accepted."""
    forged = TestIdTarget.model_construct(strategy="test_id", test_id="clean")
    object.__setattr__(forged, "__dict__", stored)
    accepted = make_config(observation_targets=[forged])
    assert [item.model_dump() for item in accepted.observation_targets] == [
        {"strategy": "test_id", "test_id": "clean"}
    ]


@pytest.mark.parametrize(
    "empty_extra",
    [{}, FalseTruthinessExtraStore(), None],
    ids=["empty-mapping", "empty-false-truthiness", "none"],
)
def test_f01_config_accepts_a_genuinely_empty_or_absent_extra_store(empty_extra) -> None:
    """Empty and absent extra stores are decided by their real entries, never by truthiness."""
    forged = TestIdTarget.model_construct(strategy="test_id", test_id="clean")
    object.__setattr__(forged, "__pydantic_extra__", empty_extra)
    accepted = make_config(observation_targets=[forged])
    assert [item.model_dump() for item in accepted.observation_targets] == [
        {"strategy": "test_id", "test_id": "clean"}
    ]


def test_f01_config_accepts_a_clean_model_subclass_and_plain_mapping() -> None:
    """The remaining positives: a clean model subclass and plain Mapping inputs stay accepted."""
    accepted = make_config(
        observation_targets=[
            CleanTestIdSubclass(strategy="test_id", test_id="clean-subclass"),
            {"strategy": "role", "role": "button", "name": "Plain Role"},
        ]
    )
    assert [item.model_dump() for item in accepted.observation_targets] == [
        {"strategy": "test_id", "test_id": "clean-subclass"},
        {"strategy": "role", "role": "button", "name": "Plain Role"},
    ]


def test_f02_host_cancel_with_a_concurrent_stop_keeps_the_cancellation(monkeypatch) -> None:
    """X02: the original host CancelledError wins over a concurrent stop's start failure."""
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def scenario() -> None:
        backend = FakeBackend(entered=entered, gate=gate)
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config())
        starting = asyncio.ensure_future(adapter.start())
        await entered.wait()
        stopping = asyncio.ensure_future(adapter.stop())
        # No suspension between the concurrent stop and the host cancellation: whichever runs
        # first, the host cancellation must keep its type and never become a start failure.
        starting.cancel("host cancellation")
        with pytest.raises(asyncio.CancelledError):
            await starting
        await stopping
        assert adapter._state == "stopped"
        assert backend.page is None
        assert backend.close_log == ["context", "browser", "driver"]

    run(scenario())


def test_f02_stop_cancels_and_bounds_the_owned_active_operation(monkeypatch) -> None:
    """X03: stop owns the in-flight execute await; no late SUCCEEDED and no late side effect."""
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    effects: list[str] = []

    async def gated_click() -> None:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        effects.append("applied")

    async def scenario() -> None:
        backend = FakeBackend(page_specs={("test_id", "a"): {"click": gated_click}})
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(make_config())
        await adapter.start()
        operation = asyncio.ensure_future(
            adapter.execute(action("web.click", {"target": target("a")}))
        )
        await entered.wait()
        active_task = adapter._active_task
        assert active_task is operation, "the public execute await is the owned active task"
        assert not active_task.done()
        await adapter.stop()
        assert adapter._state == "stopped"
        assert adapter._active_task is None
        assert active_task.done()
        assert cancelled.is_set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert effects == []
        # Releasing the gate afterwards cannot turn the stopped operation into a success.
        release.set()
        await asyncio.sleep(0)
        assert effects == []
        assert adapter._state == "stopped"
        assert backend.close_log == ["page", "context", "browser", "driver"]

    run(scenario())


def test_f02_stop_cancels_the_owned_active_observation(monkeypatch) -> None:
    """The same owned-await cancellation covers an in-flight observe call."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_count() -> int:
        entered.set()
        await release.wait()
        return 0

    async def scenario() -> None:
        configured = TestIdTarget(strategy="test_id", test_id="a")
        backend = FakeBackend(page_specs={("test_id", "a"): {"count": gated_count}})
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config(observation_targets=[configured])
        )
        await adapter.start()
        observation = asyncio.ensure_future(adapter.observe())
        await entered.wait()
        active_task = adapter._active_task
        assert active_task is observation, "the public observe await is the owned active task"
        assert not active_task.done()
        await adapter.stop()
        assert adapter._state == "stopped"
        assert active_task.done()
        with pytest.raises(asyncio.CancelledError):
            await observation
        release.set()
        await asyncio.sleep(0)
        assert adapter._state == "stopped"
        assert backend.close_log == ["page", "context", "browser", "driver"]

    run(scenario())


def test_f02_host_cancel_during_stop_keeps_the_type_and_finishes_the_cleanup(
    monkeypatch,
) -> None:
    """A host cancellation of stop propagates while the one bounded cleanup still completes."""
    entered = asyncio.Event()
    backend = FakeBackend(close_behavior={"page": {"delay": 0.2}})
    backend.close_entered["page"] = entered
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(make_config(cleanup_timeout_ms=1000))

    async def scenario() -> None:
        await adapter.start()
        stopping = asyncio.ensure_future(adapter.stop())
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        attempt = adapter._cleanup_task
        deadline = adapter._cleanup_deadline
        assert attempt is not None and deadline is not None
        stopping.cancel("host cancellation during stop")
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert backend.page is not None
        assert adapter._state == "stopping"
        assert adapter._cleanup_task is attempt
        assert adapter._cleanup_deadline == deadline
        await asyncio.wait({attempt}, timeout=2.0)
        assert attempt.done()
        assert backend.page.closed is True
        assert adapter.cleanup_error is None
        await adapter.stop()
        assert adapter._state == "stopped"
        assert backend.close_log == ["page", "context", "browser", "driver"]

    run(scenario())


def test_f02_stop_is_bounded_when_the_operation_delays_its_own_cancellation(
    monkeypatch,
) -> None:
    """A cancellation-delayed active call cannot extend stop or produce a visible success."""
    entered = asyncio.Event()
    finished = asyncio.Event()

    async def delayed_click() -> None:
        entered.set()
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
        finished.set()

    async def scenario() -> None:
        backend = FakeBackend(page_specs={("test_id", "a"): {"click": delayed_click}})
        install(monkeypatch, backend)
        adapter = adapter_module.PlaywrightWebAdapter(
            make_config(cleanup_timeout_ms=100)
        )
        await adapter.start()
        operation = asyncio.ensure_future(
            adapter.execute(action("web.click", {"target": target("a")}))
        )
        await entered.wait()
        started = time.monotonic()
        await adapter.stop()
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, "stop must not wait for cancellation-delayed work"
        assert adapter._state == "stopped"
        with pytest.raises(asyncio.CancelledError):
            await operation
        abandoned = set(adapter._late_tasks)
        assert abandoned, "the resisting backend await must stay tracked"
        await asyncio.wait(abandoned, timeout=2.0)
        await asyncio.sleep(0)
        assert finished.is_set() is True
        # The abandoned call may wind down internally, but it never reaches the stopped caller.
        assert adapter._state == "stopped"

    run(scenario())


def test_f02_late_driver_close_failure_is_visible_and_retryable(monkeypatch) -> None:
    """X04b: a late failed driver close is not consumed; the handle stays retryable by stop."""
    release = asyncio.Event()
    backend = FakeBackend(close_behavior={"driver": {"fail_first": True}})
    real_open_driver = backend.open_driver

    async def delayed_open_driver():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return await real_open_driver()

    backend.open_driver = delayed_open_driver
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(
        make_config(operation_timeout_ms=50, cleanup_timeout_ms=50)
    )

    async def scenario() -> None:
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        session = adapter._session
        assert session is not None
        pending = set(adapter._late_tasks)
        assert pending, "the abandoned acquisition must be tracked"
        release.set()
        await asyncio.wait(pending, timeout=3.0)
        await asyncio.sleep(0)  # let the owned-task callbacks report the late failure
        assert adapter._state == "failed"
        assert adapter.cleanup_error is not None
        assert adapter.cleanup_error.code == "WEB_BROWSER_CLEANUP_FAILED"
        assert session.has_pending_cleanup() is True
        assert len(session._late_handles) == 1
        assert backend.driver.closed is False
        assert backend.close_log.count("driver") == 1
        await adapter.stop()
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        assert session.has_pending_cleanup() is False
        assert backend.driver.closed is True
        assert backend.close_log.count("driver") == 2

    run(scenario())


def test_f02_late_page_close_failure_is_visible_and_retryable(monkeypatch) -> None:
    """A late failed page close keeps the page and the bounded error for a later stop retry."""
    release = asyncio.Event()
    backend = FakeBackend(close_behavior={"page": {"fail_first": True}})
    real_open_page = backend.open_page

    async def delayed_open_page(context: object):
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return await real_open_page(context)

    backend.open_page = delayed_open_page
    install(monkeypatch, backend)
    adapter = adapter_module.PlaywrightWebAdapter(
        make_config(operation_timeout_ms=50, cleanup_timeout_ms=50)
    )

    async def scenario() -> None:
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        session = adapter._session
        assert session is not None
        pending = set(adapter._late_tasks)
        assert pending, "the abandoned acquisition must be tracked"
        release.set()
        await asyncio.wait(pending, timeout=3.0)
        await asyncio.sleep(0)  # let the owned-task callbacks report the late failure
        assert adapter._state == "failed"
        assert adapter.cleanup_error is not None
        assert adapter.cleanup_error.code == "WEB_BROWSER_CLEANUP_FAILED"
        assert session.has_pending_cleanup() is True
        assert len(session._late_handles) == 1
        assert backend.page is not None
        assert backend.page.closed is False
        assert backend.close_log.count("page") == 1
        await adapter.stop()
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        assert session.has_pending_cleanup() is False
        assert backend.page.closed is True
        assert backend.close_log.count("page") == 2

    run(scenario())
