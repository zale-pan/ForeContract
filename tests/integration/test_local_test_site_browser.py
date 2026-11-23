"""Real headless Chromium acceptance of the five scenarios.

The real ``PlaywrightWebAdapter`` drives the running site through the existing six actions and
two locator strategies only. Evidence is read through the existing Web observation vocabulary
(``parse_web_observation``): match counts, visibility, text and value. The site's host oracle is
compared against literal counts derived from the contract; the test never asks the site to
calculate an expected result.

No browser path is skipped. A missing Playwright package or Chromium binary fails the test.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_SRC = REPO_ROOT / "packages" / "core" / "src"
WEB_SRC = REPO_ROOT / "packages" / "adapters" / "web" / "src"
for candidate in (REPO_ROOT, CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.models.actions import ActionSpec
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import RoleTarget, TestIdTarget
from forecontract_web.playwright_adapter import PlaywrightWebAdapter
from forecontract_web.validation import parse_web_observation

from test_site import (
    LocalTestSite,
    Scenario,
    SiteConfig,
    SiteEvent,
)

USERNAME = "test-user"
PASSWORD = "test-password"
WRONG_USERNAME = "wrong-user"
WRONG_PASSWORD = "wrong-password"
SUCCESS_TEXT = "登录成功"
ERROR_TEXT = "用户名或密码错误"
SUBMIT_NAME = "登录"


def running_site(
    scenario: Scenario,
    *,
    request_timeout_ms: int = 15000,
    cleanup_timeout_ms: int = 5000,
) -> LocalTestSite:
    return LocalTestSite(
        SiteConfig(
            scenario=scenario,
            request_timeout_ms=request_timeout_ms,
            cleanup_timeout_ms=cleanup_timeout_ms,
        )
    ).start()


def make_adapter(
    site: LocalTestSite,
    targets: list[TestIdTarget | RoleTarget],
    *,
    operation_timeout_ms: int = 8000,
) -> PlaywrightWebAdapter:
    return PlaywrightWebAdapter(
        PlaywrightWebConfig(
            allowed_origins=[site.origin],
            observation_targets=list(targets),
            operation_timeout_ms=operation_timeout_ms,
            cleanup_timeout_ms=5000,
        )
    )


def act(action_type: str, payload: dict) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type, payload=payload)


def by_test_id(name: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=name)


def role_target(role: str, name: str) -> RoleTarget:
    return RoleTarget(strategy="role", role=role, name=name)


def payload(target: TestIdTarget | RoleTarget) -> dict:
    return target.model_dump()


def standard_targets() -> list[TestIdTarget | RoleTarget]:
    return [
        by_test_id("username"),
        by_test_id("password"),
        by_test_id("submit"),
        by_test_id("login-error"),
        by_test_id("overlay"),
        by_test_id("welcome"),
        role_target("button", SUBMIT_NAME),
        role_target("heading", SUCCESS_TEXT),
    ]


def evidence(observation) -> tuple[object, dict[str, object]]:
    state = parse_web_observation(observation)
    records: dict[str, object] = {}
    for record in state.targets:
        if isinstance(record.target, TestIdTarget):
            records[f"test_id:{record.target.test_id}"] = record
        else:
            records[f"role:{record.target.role}:{record.target.name}"] = record
    return state, records


def run(coroutine):
    return asyncio.run(coroutine)


def test_browser_normal_login_uses_visible_text_and_value_evidence() -> None:
    site = running_site(Scenario.NORMAL)
    try:
        async def scenario() -> None:
            adapter = make_adapter(site, standard_targets())
            await adapter.start()
            try:
                navigated = await adapter.execute(
                    act("web.navigate", {"url": f"{site.origin}/login"})
                )
                assert navigated.status.value == "succeeded"

                filled_user = await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("username")),
                            "text": USERNAME,
                        },
                    )
                )
                assert filled_user.status.value == "succeeded"
                filled_password = await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("password")),
                            "text": PASSWORD,
                        },
                    )
                )
                assert filled_password.status.value == "succeeded"

                state, records = evidence(await adapter.observe())
                assert state.url.endswith("/login")
                username = records["test_id:username"]
                assert username.status == "collected"
                assert username.match_count == 1
                assert username.visible is True
                assert username.value == USERNAME
                password = records["test_id:password"]
                assert password.match_count == 1
                assert password.visible is True
                assert password.value == PASSWORD
                submit = records["test_id:submit"]
                assert submit.match_count == 1
                assert submit.visible is True
                assert submit.text == SUBMIT_NAME
                error = records["test_id:login-error"]
                assert error.match_count == 1
                assert error.visible is False
                assert error.text == ERROR_TEXT
                overlay = records["test_id:overlay"]
                assert overlay.match_count == 1
                assert overlay.visible is False
                welcome = records["test_id:welcome"]
                assert welcome.status == "collected"
                assert welcome.match_count == 0
                button = records[f"role:button:{SUBMIT_NAME}"]
                assert button.match_count == 1
                assert button.visible is True
                assert button.text == SUBMIT_NAME
                assert records[f"role:heading:{SUCCESS_TEXT}"].match_count == 0

                clicked = await adapter.execute(
                    act("web.click", {"target": payload(by_test_id("submit"))})
                )
                assert clicked.status.value == "succeeded"
                waited = await adapter.execute(
                    act(
                        "web.wait",
                        {
                            "target": payload(by_test_id("welcome")),
                            "state": "visible",
                        },
                    )
                )
                assert waited.status.value == "succeeded"

                state, records = evidence(await adapter.observe())
                assert state.url.endswith("/dashboard")
                welcome = records["test_id:welcome"]
                assert welcome.match_count == 1
                assert welcome.visible is True
                assert welcome.text == SUCCESS_TEXT
                heading = records[f"role:heading:{SUCCESS_TEXT}"]
                assert heading.match_count == 1
                assert heading.visible is True
                assert heading.text == SUCCESS_TEXT
                assert records["test_id:login-error"].match_count == 0
                assert records["test_id:username"].match_count == 0
                assert records["test_id:password"].match_count == 0

                snapshot = site.snapshot_oracle()
                assert snapshot.login_requests == 1
                assert snapshot.auth_successes == 1
                assert snapshot.auth_failures == 0
                assert snapshot.commits == 1
                assert snapshot.authenticated_sessions == 1
            finally:
                await adapter.stop()
                assert adapter.cleanup_error is None

        run(scenario())
    finally:
        result = site.stop()
        assert result.complete, result


def test_browser_wrong_credentials_fail_in_all_four_modes() -> None:
    for scenario in Scenario:
        site = running_site(scenario)
        try:
            async def wrong_case(
                current: Scenario = scenario,
                current_site: LocalTestSite = site,
            ) -> None:
                adapter = make_adapter(current_site, standard_targets())
                await adapter.start()
                try:
                    await adapter.execute(
                        act("web.navigate", {"url": f"{current_site.origin}/login"})
                    )
                    if current is Scenario.BLOCKED_SUBMIT:
                        current_site.release()
                        await adapter.execute(
                            act(
                                "web.navigate",
                                {"url": f"{current_site.origin}/login"},
                            )
                        )
                    await adapter.execute(
                        act(
                            "web.fill",
                            {
                                "target": payload(by_test_id("username")),
                                "text": WRONG_USERNAME,
                            },
                        )
                    )
                    await adapter.execute(
                        act(
                            "web.fill",
                            {
                                "target": payload(by_test_id("password")),
                                "text": WRONG_PASSWORD,
                            },
                        )
                    )
                    clicked = await adapter.execute(
                        act("web.click", {"target": payload(by_test_id("submit"))})
                    )
                    assert clicked.status.value == "succeeded", current
                    waited = await adapter.execute(
                        act(
                            "web.wait",
                            {
                                "target": payload(by_test_id("login-error")),
                                "state": "visible",
                            },
                        )
                    )
                    assert waited.status.value == "succeeded", current

                    state, records = evidence(await adapter.observe())
                    assert state.url.endswith("/login"), current
                    error = records["test_id:login-error"]
                    assert error.match_count == 1
                    assert error.visible is True
                    assert error.text == ERROR_TEXT
                    assert records["test_id:welcome"].match_count == 0

                    snapshot = current_site.snapshot_oracle()
                    assert snapshot.login_requests == 1, current
                    assert snapshot.auth_successes == 0, current
                    assert snapshot.auth_failures == 1, current
                    assert snapshot.commits == 0, current
                    assert snapshot.authenticated_sessions == 0, current
                    assert snapshot.response_blocked is False, current
                finally:
                    await adapter.stop()
                    assert adapter.cleanup_error is None

            run(wrong_case())
        finally:
            result = site.stop()
            assert result.complete, result


def test_browser_delayed_login_gate_release_yields_one_commit() -> None:
    site = running_site(Scenario.DELAYED_LOGIN)
    try:
        async def scenario() -> None:
            adapter = make_adapter(site, standard_targets())
            await adapter.start()
            try:
                await adapter.execute(
                    act("web.navigate", {"url": f"{site.origin}/login"})
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("username")),
                            "text": USERNAME,
                        },
                    )
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("password")),
                            "text": PASSWORD,
                        },
                    )
                )

                before: list[object] = []
                released: list[bool] = []

                def host_release() -> None:
                    if not site.wait_for(
                        event=SiteEvent.LOGIN_RECEIVED, timeout_ms=7000
                    ):
                        return
                    if not site.wait_for(
                        event=SiteEvent.RESPONSE_BLOCKED, timeout_ms=7000
                    ):
                        return
                    before.append(site.snapshot_oracle())
                    site.release()
                    released.append(True)

                host = threading.Thread(target=host_release)
                host.start()
                clicked = await adapter.execute(
                    act("web.click", {"target": payload(by_test_id("submit"))})
                )
                host.join(8.0)
                assert host.is_alive() is False
                assert released == [True]
                assert len(before) == 1
                blocked = before[0]
                assert blocked.login_requests == 1
                assert blocked.auth_successes == 0
                assert blocked.commits == 0
                assert blocked.response_blocked is True
                assert clicked.status.value == "succeeded"

                waited = await adapter.execute(
                    act(
                        "web.wait",
                        {
                            "target": payload(by_test_id("welcome")),
                            "state": "visible",
                        },
                    )
                )
                assert waited.status.value == "succeeded"
                state, records = evidence(await adapter.observe())
                assert state.url.endswith("/dashboard")
                assert records["test_id:welcome"].visible is True

                snapshot = site.snapshot_oracle()
                assert snapshot.login_requests == 1
                assert snapshot.auth_successes == 1
                assert snapshot.commits == 1
                assert snapshot.authenticated_sessions == 1
                assert snapshot.response_blocked is False
            finally:
                await adapter.stop()
                assert adapter.cleanup_error is None

        run(scenario())
    finally:
        result = site.stop()
        assert result.complete, result


def test_browser_overlay_blocks_pointer_then_release_recovers() -> None:
    site = running_site(Scenario.BLOCKED_SUBMIT)
    try:
        async def scenario() -> None:
            adapter = make_adapter(site, standard_targets(), operation_timeout_ms=3000)
            await adapter.start()
            try:
                await adapter.execute(
                    act("web.navigate", {"url": f"{site.origin}/login"})
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("username")),
                            "text": USERNAME,
                        },
                    )
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("password")),
                            "text": PASSWORD,
                        },
                    )
                )
                _, records = evidence(await adapter.observe())
                assert records["test_id:overlay"].visible is True

                blocked = await adapter.execute(
                    act("web.click", {"target": payload(by_test_id("submit"))})
                )
                assert blocked.status.value == "unknown"
                assert blocked.error is not None
                assert blocked.error.code == "WEB_ACTION_TIMEOUT"
                snapshot = site.snapshot_oracle()
                assert snapshot.login_requests == 0
                assert snapshot.auth_successes == 0
                assert snapshot.commits == 0
                assert snapshot.authenticated_sessions == 0
                assert snapshot.active_requests == 0

                site.release()
                reloaded = await adapter.execute(
                    act("web.navigate", {"url": f"{site.origin}/login"})
                )
                assert reloaded.status.value == "succeeded"
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("username")),
                            "text": USERNAME,
                        },
                    )
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("password")),
                            "text": PASSWORD,
                        },
                    )
                )
                _, records = evidence(await adapter.observe())
                assert records["test_id:overlay"].match_count == 1
                assert records["test_id:overlay"].visible is False
                clicked = await adapter.execute(
                    act("web.click", {"target": payload(by_test_id("submit"))})
                )
                assert clicked.status.value == "succeeded"
                waited = await adapter.execute(
                    act(
                        "web.wait",
                        {
                            "target": payload(by_test_id("welcome")),
                            "state": "visible",
                        },
                    )
                )
                assert waited.status.value == "succeeded"

                snapshot = site.snapshot_oracle()
                assert snapshot.login_requests == 1
                assert snapshot.auth_successes == 1
                assert snapshot.commits == 1
                assert snapshot.authenticated_sessions == 1
            finally:
                await adapter.stop()
                assert adapter.cleanup_error is None

        run(scenario())
    finally:
        result = site.stop()
        assert result.complete, result


def test_browser_commit_then_stall_is_unknown_without_replay() -> None:
    site = running_site(Scenario.COMMIT_THEN_STALL)
    try:
        async def scenario() -> None:
            adapter = make_adapter(site, standard_targets(), operation_timeout_ms=2500)
            await adapter.start()
            try:
                await adapter.execute(
                    act("web.navigate", {"url": f"{site.origin}/login"})
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("username")),
                            "text": USERNAME,
                        },
                    )
                )
                await adapter.execute(
                    act(
                        "web.fill",
                        {
                            "target": payload(by_test_id("password")),
                            "text": PASSWORD,
                        },
                    )
                )
                stalled = await adapter.execute(
                    act("web.click", {"target": payload(by_test_id("submit"))})
                )
                assert stalled.status.value == "unknown"
                assert stalled.error is not None
                assert stalled.error.code == "WEB_ACTION_TIMEOUT"

                assert site.wait_for(
                    event=SiteEvent.LOGIN_COMMITTED, timeout_ms=3000
                ) is True
                assert site.wait_for(
                    event=SiteEvent.RESPONSE_BLOCKED, timeout_ms=3000
                ) is True
                before = site.snapshot_oracle()
                assert before.login_requests == 1
                assert before.auth_successes == 1
                assert before.commits == 1
                assert before.authenticated_sessions == 1
                assert before.response_blocked is True
                assert before.active_requests == 1

                try:
                    await adapter.observe()
                except WebBrowserError as error:
                    assert error.code in (
                        "WEB_OBSERVE_TIMEOUT",
                        "WEB_BROWSER_UNAVAILABLE",
                    )
                after_observation = site.snapshot_oracle()
                assert after_observation.commits == 1
                assert after_observation.login_requests == 1
                assert after_observation.authenticated_sessions == 1

                site.release()
            finally:
                await adapter.stop()
                assert adapter.cleanup_error is None

            final = site.snapshot_oracle()
            assert final.commits == 1
            assert final.login_requests == 1
            assert final.authenticated_sessions == 1

        run(scenario())
    finally:
        result = site.stop()
        assert result.complete, result
