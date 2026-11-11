"""Real Chromium observation followed by pure Web rule verification.

One small ephemeral loopback fixture is served from inside the test process. The real
``PlaywrightWebAdapter`` opens headless Chromium, navigates to that page and collects the
existing ``web.snapshot`` evidence. The fixture's own HTML and the server request counter are
the independent oracle: the expected label, input value, visibility, empty value, zero match,
many matches and hidden text are declared here, not read back from the observation under test.

The browser is stopped **before** any rule runs, so a passing verification cannot have queried a
live page. The same observed ``Observation`` then drives the Web step verifier and the
independent task oracle for a PASS, a FAIL and an INCONCLUSIVE case. A second node states the
complete negative/inconclusive matrix over a synthetic but structurally valid snapshot, because
a healthy single query cannot produce an unavailable record, a null text at all.

No browser path is skipped: an unavailable dependency or binary fails the test. This is not a
full-site, Runtime or end-to-end claim.
"""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts.models import (
    ActionContract,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import ActionSpec, ExpectedOutcome
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import TestIdTarget
from forecontract_web.playwright_adapter import PlaywrightWebAdapter
from forecontract_web.rule_models import UrlMatchPayload, WebRuleEvaluation
from forecontract_web.rules import (
    WebRuleEvaluator,
    WebRuleInvalid,
    WebVerificationInputInvalid,
)
from forecontract_web.validation import parse_web_observation
from forecontract_web.verification import WebStepVerifier, WebTaskCompletionOracle

SCHEMA = "0.1"
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

PAGE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>p3 rule fixture</title></head>
<body>
<span data-testid="message">Order Success #42</span>
<input data-testid="name" value="seed">
<button data-testid="visible">visible</button>
<span data-testid="hidden" style="display:none">hidden text</span>
<input data-testid="blank" value="">
<span data-testid="empty"></span>
<button data-testid="dup">one</button>
<button data-testid="dup">two</button>
</body></html>"""

#: The independent DOM oracle: declared here, never read back from the observation.
EXPECTED_TEXT = "Order Success #42"
EXPECTED_VALUE = "seed"
HIDDEN_TEXT = "hidden text"


def id_target(test_id: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=test_id)


def target_payload(target: TestIdTarget) -> dict:
    return {"target": target.model_dump()}


def record(target: TestIdTarget, *, count=1, visible=None, text=None, value=None) -> dict:
    return {
        "target": target.model_dump(),
        "status": "collected",
        "match_count": count,
        "visible": visible,
        "text": text,
        "value": value,
        "error_code": None,
    }


def unavailable(target: TestIdTarget) -> dict:
    return {
        "target": target.model_dump(),
        "status": "unavailable",
        "match_count": None,
        "visible": None,
        "text": None,
        "value": None,
        "error_code": "TARGET_QUERY_FAILED",
    }


def evidence(observation: Observation, *, complete: bool = True) -> VerificationEvidence:
    return VerificationEvidence(
        schema_version=SCHEMA, observation=observation, state_complete=complete
    )


def expected(rule_type: str, payload: dict, *, required: bool = True) -> ExpectedOutcome:
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type=rule_type, payload=payload, required=required
    )


def contract(outcomes: list[ExpectedOutcome]) -> ActionContract:
    bindings = [
        VerificationBinding(
            schema_version=SCHEMA, outcome_index=index, verifier_type="web_rule", config={}
        )
        for index in range(len(outcomes))
    ]
    return ActionContract(
        schema_version=SCHEMA,
        contract_id="integration-contract",
        goal_id="integration-goal",
        forecast_id="integration-forecast",
        observation_id="owned-by-evidence",
        adapter_name="playwright_web",
        environment_type="web",
        action=ActionSpec(
            schema_version=SCHEMA,
            action_type="web.navigate",
            payload={"url": "http://127.0.0.1:1/"},
        ),
        preconditions=[],
        expected_outcomes=outcomes,
        verification_policy=VerificationPolicy(schema_version=SCHEMA, bindings=bindings),
        recovery_policy=RecoveryPolicy(schema_version=SCHEMA),
        timeout_ms=1000.0,
        created_at=NOW,
    )


def criteria(conditions: list[ExpectedOutcome]) -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id="integration-goal", conditions=conditions
    )


# ------------------------------------------------------------------ fixture server


class _State:
    def __init__(self) -> None:
        self.hits = 0
        self.lock = threading.Lock()

    def hit(self) -> None:
        with self.lock:
            self.hits += 1

    def count(self) -> int:
        with self.lock:
            return self.hits


class FixtureServer:
    def __init__(self, server: ThreadingHTTPServer, state: _State) -> None:
        self._server = server
        self._state = state

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def hits(self) -> int:
        return self._state.count()


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            state.hit()
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@pytest.fixture()
def server():
    state = _State()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield FixtureServer(httpd, state)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def _observation_targets() -> list[TestIdTarget]:
    return [
        id_target("message"),
        id_target("name"),
        id_target("visible"),
        id_target("hidden"),
        id_target("blank"),
        id_target("empty"),
        id_target("dup"),
        id_target("absent"),
    ]


def _observe(server: FixtureServer) -> tuple[Observation, str]:
    adapter = PlaywrightWebAdapter(
        PlaywrightWebConfig(
            allowed_origins=[server.origin],
            observation_targets=_observation_targets(),
            operation_timeout_ms=20000,
            cleanup_timeout_ms=15000,
        )
    )

    async def run() -> tuple[Observation, str]:
        await adapter.start()
        version = "unknown"
        session = adapter._session
        assert session is not None and session._browser is not None
        version = session._browser.version
        try:
            navigated = await adapter.execute(
                ActionSpec(
                    schema_version=SCHEMA,
                    action_type="web.navigate",
                    payload={"url": f"{server.origin}/"},
                )
            )
            assert navigated.status.value == "succeeded", navigated
            seen = await adapter.observe()
        finally:
            await adapter.stop()
        assert adapter.cleanup_error is None
        return seen, version

    try:
        observation, version = asyncio.run(run())
    except WebBrowserError as exc:
        pytest.fail(f"real Chromium is not available for the rule integration: {exc.code}")
    assert _VERSION_PATTERN.match(version) is not None, version
    return observation, version


# ------------------------------------------------------------------ genuine observation


def test_web_rules_verify_a_genuine_chromium_observation(server: FixtureServer) -> None:
    observation, version = _observe(server)
    print(f"real Chromium rule verification: {version}")
    assert server.hits() == 1
    assert observation.environment_type == "web"

    state = parse_web_observation(observation)
    assert state.url == server.origin + "/"
    by_id = {
        item.target.test_id: item
        for item in state.targets
        if isinstance(item.target, TestIdTarget)
    }
    # The fixture is the oracle, not the observation: every declaration below is hard-coded.
    assert set(by_id) == {target.test_id for target in _observation_targets()}
    assert by_id["message"].match_count == 1
    assert by_id["message"].visible is True
    assert by_id["message"].text == EXPECTED_TEXT
    assert by_id["message"].value is None
    assert by_id["name"].value == EXPECTED_VALUE
    assert by_id["visible"].visible is True
    assert by_id["hidden"].visible is False
    assert by_id["hidden"].text == HIDDEN_TEXT
    assert by_id["blank"].value == ""
    assert by_id["dup"].status == "collected"
    assert by_id["dup"].match_count == 2
    assert by_id["absent"].status == "collected"
    assert by_id["absent"].match_count == 0

    # The browser has already been stopped by ``_observe``; pure verification still works.
    piece = evidence(observation, complete=True)
    passing = contract([
        expected("web.url_match", {"url": server.origin + "/"}),
        expected("web.element_visible", target_payload(id_target("message"))),
        expected("web.element_hidden", target_payload(id_target("hidden"))),
        expected("web.text_contains",
                 {**target_payload(id_target("message")), "text": "Success #42"}),
        expected("web.value_equals", {**target_payload(id_target("name")), "value": EXPECTED_VALUE}),
        expected("web.value_equals", {**target_payload(id_target("blank")), "value": ""}),
    ])
    step = WebStepVerifier().verify_step(contract=passing, evidence=piece)
    assert step.status is VerificationStatus.PASS, step
    assert step.verifier == "web_step_verifier"
    assert [item.status for item in step.outcome_results] == [VerificationStatus.PASS] * 6
    assert [item.reason for item in step.outcome_results] == ["RULE_MATCH"] * 6
    assert step.metadata["rule_evidence_refs"] == [
        {"rule_type": "web.url_match", "target_index": None},
        {"rule_type": "web.element_visible", "target_index": 0},
        {"rule_type": "web.element_hidden", "target_index": 3},
        {"rule_type": "web.text_contains", "target_index": 0},
        {"rule_type": "web.value_equals", "target_index": 1},
        {"rule_type": "web.value_equals", "target_index": 4},
    ]
    assert list(step.metadata) == [
        "scope", "contract_id", "goal_id", "observation_id", "state_complete",
        "rule_evidence_refs",
    ]

    # A genuine FAIL case: the hidden node is visible=false, the empty node carries a known
    # empty text/value, and the absent target was queried successfully and matched zero nodes.
    failing = contract([
        expected("web.element_visible", target_payload(id_target("hidden"))),
        expected("web.element_hidden", target_payload(id_target("visible"))),
        expected("web.text_contains",
                 {**target_payload(id_target("message")), "text": "success #42"}),
        expected("web.value_equals", {**target_payload(id_target("name")), "value": "other"}),
        expected("web.element_visible", target_payload(id_target("absent"))),
        expected("web.text_contains",
                 {**target_payload(id_target("absent")), "text": "x"}),
        expected("web.value_equals",
                 {**target_payload(id_target("absent")), "value": "x"}),
    ])
    failed = WebStepVerifier().verify_step(contract=failing, evidence=piece)
    assert failed.status is VerificationStatus.FAIL
    assert failed.reason == "REQUIRED_FAILED"
    assert [item.status for item in failed.outcome_results] == [VerificationStatus.FAIL] * 7
    assert [item.reason for item in failed.outcome_results] == [
        "RULE_MISMATCH", "RULE_MISMATCH", "RULE_MISMATCH", "RULE_MISMATCH",
        "TARGET_NOT_FOUND", "TARGET_NOT_FOUND", "TARGET_NOT_FOUND",
    ]

    # A genuine INCONCLUSIVE case: many matches and a missing target record. ``message`` is a
    # span, so its value attribute is null rather than an empty string.
    inconclusive = contract([
        expected("web.element_visible", target_payload(id_target("dup"))),
        expected("web.element_visible", target_payload(id_target("unlisted"))),
        expected("web.value_equals", {**target_payload(id_target("message")), "value": ""}),
    ])
    unclear = WebStepVerifier().verify_step(contract=inconclusive, evidence=piece)
    assert unclear.status is VerificationStatus.INCONCLUSIVE
    assert unclear.reason == "REQUIRED_INCONCLUSIVE"
    assert [item.status for item in unclear.outcome_results] == [
        VerificationStatus.INCONCLUSIVE,
    ] * 3

    # The independent task oracle over the same genuine observation.
    passing_task = WebTaskCompletionOracle().evaluate_completion(
        criteria=criteria([
            expected("web.url_match", {"url": server.origin + "/"}),
            expected("web.text_contains",
                     {**target_payload(id_target("message")), "text": EXPECTED_TEXT}),
            expected("web.value_equals",
                     {**target_payload(id_target("name")), "value": EXPECTED_VALUE}),
        ]),
        evidence=piece,
    )
    assert passing_task.status is VerificationStatus.PASS
    assert passing_task.verifier == "web_task_completion_oracle"
    assert "contract_id" not in passing_task.metadata
    failing_task = WebTaskCompletionOracle().evaluate_completion(
        criteria=criteria([
            expected("web.element_visible", target_payload(id_target("hidden"))),
            expected("web.value_equals",
                     {**target_payload(id_target("name")), "value": EXPECTED_VALUE},
                     required=False),
        ]),
        evidence=piece,
    )
    assert failing_task.status is VerificationStatus.FAIL
    unclear_task = WebTaskCompletionOracle().evaluate_completion(
        criteria=criteria([
            expected("web.element_visible", target_payload(id_target("dup"))),
        ]),
        evidence=piece,
    )
    assert unclear_task.status is VerificationStatus.INCONCLUSIVE

    # The same pure call still produces the same verdict after the browser has been released.
    again = WebRuleEvaluator().evaluate(
        rule_type="web.text_contains",
        payload={**target_payload(id_target("message")), "text": "Success #42"},
        evidence=piece,
    )
    assert (again.status, again.reason, again.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", 0,
    )


# ------------------------------------------------------------------ full matrix, no browser


def _synthetic_evidence(*, url: str | None, complete: bool = True) -> VerificationEvidence:
    state = {
        "schema_version": SCHEMA,
        "url": url,
        "targets": [
            record(id_target("zero"), count=0),
            record(id_target("unique"), visible=True, text="abc", value=""),
            record(id_target("hidden"), visible=False, text="hidden", value="h"),
            record(id_target("many"), count=2),
            unavailable(id_target("failed")),
            record(id_target("nulls"), visible=None, text=None, value=None),
        ],
    }
    observation = Observation(
        schema_version=SCHEMA,
        observation_id="synthetic-observation",
        environment_type="web",
        state=state,
        timestamp=NOW,
    )
    return VerificationEvidence(
        schema_version=SCHEMA, observation=observation, state_complete=complete
    )


MATRIX = [
    # rule type, payload, expected status, expected reason, expected target index
    ("web.url_match", {"url": "about:blank"}, "pass", "RULE_MATCH", None),
    ("web.url_match", {"url": "about:blank"}, "fail", "RULE_MISMATCH", None),
    ("web.url_match", {"url": "about:blank"}, "inconclusive", "EVIDENCE_INSUFFICIENT", None),
    ("web.element_visible", target_payload(id_target("unique")), "pass", "RULE_MATCH", 1),
    ("web.element_visible", target_payload(id_target("hidden")), "fail", "RULE_MISMATCH", 2),
    ("web.element_visible", target_payload(id_target("zero")), "fail", "TARGET_NOT_FOUND", 0),
    ("web.element_visible", target_payload(id_target("many")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 3),
    ("web.element_visible", target_payload(id_target("failed")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 4),
    ("web.element_visible", target_payload(id_target("nulls")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 5),
    ("web.element_visible", target_payload(id_target("absent")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", None),
    ("web.element_hidden", target_payload(id_target("zero")), "pass", "RULE_MATCH", 0),
    ("web.element_hidden", target_payload(id_target("hidden")), "pass", "RULE_MATCH", 2),
    ("web.element_hidden", target_payload(id_target("unique")), "fail", "RULE_MISMATCH", 1),
    ("web.element_hidden", target_payload(id_target("many")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 3),
    ("web.element_hidden", target_payload(id_target("nulls")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 5),
    ("web.element_hidden", target_payload(id_target("failed")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", 4),
    ("web.element_hidden", target_payload(id_target("absent")), "inconclusive",
     "EVIDENCE_INSUFFICIENT", None),
    ("web.text_contains", {**target_payload(id_target("unique")), "text": "bc"}, "pass",
     "RULE_MATCH", 1),
    ("web.text_contains", {**target_payload(id_target("unique")), "text": "BC"}, "fail",
     "RULE_MISMATCH", 1),
    ("web.text_contains", {**target_payload(id_target("zero")), "text": "x"}, "fail",
     "TARGET_NOT_FOUND", 0),
    ("web.text_contains", {**target_payload(id_target("many")), "text": "x"}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 3),
    ("web.text_contains", {**target_payload(id_target("failed")), "text": "x"}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 4),
    ("web.text_contains", {**target_payload(id_target("nulls")), "text": "x"}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 5),
    ("web.text_contains", {**target_payload(id_target("absent")), "text": "x"}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", None),
    ("web.value_equals", {**target_payload(id_target("unique")), "value": ""}, "pass",
     "RULE_MATCH", 1),
    ("web.value_equals", {**target_payload(id_target("unique")), "value": "x"}, "fail",
     "RULE_MISMATCH", 1),
    ("web.value_equals", {**target_payload(id_target("zero")), "value": ""}, "fail",
     "TARGET_NOT_FOUND", 0),
    ("web.value_equals", {**target_payload(id_target("many")), "value": ""}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 3),
    ("web.value_equals", {**target_payload(id_target("failed")), "value": ""}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 4),
    ("web.value_equals", {**target_payload(id_target("nulls")), "value": ""}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", 5),
    ("web.value_equals", {**target_payload(id_target("absent")), "value": ""}, "inconclusive",
     "EVIDENCE_INSUFFICIENT", None),
]

_STATUS = {
    "pass": VerificationStatus.PASS,
    "fail": VerificationStatus.FAIL,
    "inconclusive": VerificationStatus.INCONCLUSIVE,
}


@pytest.mark.parametrize(
    ("rule_type", "payload", "status", "reason", "index"),
    MATRIX,
    ids=[f"{case[0]}-{position}-{case[2]}" for position, case in enumerate(MATRIX)],
)
def test_web_rule_matrix_covers_every_negative_and_inconclusive_shape(
    rule_type: str, payload: dict, status: str, reason: str, index: int | None
) -> None:
    if rule_type == "web.url_match" and status == "pass":
        piece = _synthetic_evidence(url="about:blank")
    elif rule_type == "web.url_match" and status == "fail":
        piece = _synthetic_evidence(url="http://127.0.0.1:8000/other")
    elif rule_type == "web.url_match":
        piece = _synthetic_evidence(url=None)
    else:
        piece = _synthetic_evidence(url="about:blank")
    result = WebRuleEvaluator().evaluate(
        rule_type=rule_type, payload=payload, evidence=piece
    )
    assert result.status is _STATUS[status]
    assert result.reason == reason
    assert result.target_index == index
    assert result.observation_id == "synthetic-observation"


def test_a_zero_match_is_local_negative_evidence_even_with_incomplete_state() -> None:
    piece = _synthetic_evidence(url=None, complete=False)
    hidden = WebRuleEvaluator().evaluate(
        rule_type="web.element_hidden",
        payload=target_payload(id_target("zero")),
        evidence=piece,
    )
    visible = WebRuleEvaluator().evaluate(
        rule_type="web.element_visible",
        payload=target_payload(id_target("zero")),
        evidence=piece,
    )
    assert hidden.status is VerificationStatus.PASS
    assert visible.status is VerificationStatus.FAIL
    assert piece.state_complete is False


# ------------------------------------------------------------------ round-002 F01/F02/F03, no browser


@pytest.mark.parametrize("expected_url", [
    pytest.param("", id="empty"),
    pytest.param("not a url", id="relative"),
    pytest.param("javascript:alert(1)", id="javascript"),
    pytest.param("https://user:pass@example.com/", id="userinfo"),
    pytest.param("http://example.com:bad", id="bad-port"),
    pytest.param("ftp://example.com/", id="other-scheme"),
    pytest.param(" http://example.com/", id="leading-space"),
])
def test_the_synthetic_snapshot_rejects_an_expected_url_outside_the_page_grammar(
    expected_url: str,
) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match",
            payload={"url": expected_url},
            evidence=_synthetic_evidence(url="about:blank"),
        )
    assert caught.value.code == "WEB_RULE_INVALID"


class _FalseExtra(dict):
    def __bool__(self) -> bool:
        return False


class _RaisingExtra(dict):
    def __bool__(self) -> bool:
        raise ValueError("SYNTHETIC_INTEGRATION_EXTRA")


class _HidingDict(dict):
    def __iter__(self):
        return iter(["url"])

    def keys(self):
        return ["url"]


def _stored_payload() -> UrlMatchPayload:
    return UrlMatchPayload(url="about:blank")


def test_a_false_extra_store_on_the_synthetic_snapshot_is_a_fixed_rule_error() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__pydantic_extra__", _FalseExtra(hidden="x"))
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload=payload,
            evidence=_synthetic_evidence(url="about:blank"),
        )


def test_a_raising_extra_store_on_the_synthetic_snapshot_is_a_fixed_rule_error() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__pydantic_extra__", _RaisingExtra(hidden="x"))
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload=payload,
            evidence=_synthetic_evidence(url="about:blank"),
        )


def test_a_lying_stored_dict_on_the_synthetic_snapshot_is_a_fixed_rule_error() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__dict__", _HidingDict(url="about:blank", hidden="x"))
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload=payload,
            evidence=_synthetic_evidence(url="about:blank"),
        )


class _RewritingEvidence(VerificationEvidence):
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["observation"]["state"]["url"] = "about:blank"
        return data


def test_an_overridden_evidence_dump_cannot_rewrite_a_synthetic_verdict() -> None:
    state = {"schema_version": SCHEMA, "url": "http://127.0.0.1:8000/other", "targets": []}
    piece = _RewritingEvidence(
        schema_version=SCHEMA,
        observation=Observation(
            schema_version=SCHEMA,
            observation_id="synthetic-observation",
            environment_type="web",
            state=state,
            timestamp=NOW,
        ),
        state_complete=True,
    )
    result = WebRuleEvaluator().evaluate(
        rule_type="web.url_match", payload={"url": "about:blank"}, evidence=piece
    )
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")


def test_the_synthetic_snapshot_keeps_a_valid_url_match_and_a_null_url_index() -> None:
    result = WebRuleEvaluator().evaluate(
        rule_type="web.url_match",
        payload={"url": "about:blank"},
        evidence=_synthetic_evidence(url="about:blank"),
    )
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", None,
    )


def test_the_public_result_schema_requires_a_null_url_target_index() -> None:
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version=SCHEMA, rule_type="web.url_match",
            status=VerificationStatus.PASS, reason="RULE_MATCH",
            observation_id="synthetic-observation", target_index=0,
        )
    valid = WebRuleEvaluation(
        schema_version=SCHEMA, rule_type="web.url_match",
        status=VerificationStatus.PASS, reason="RULE_MATCH",
        observation_id="synthetic-observation", target_index=None,
    )
    assert valid.target_index is None


def test_the_synthetic_step_and_task_oracle_still_accept_a_valid_url() -> None:
    piece = _synthetic_evidence(url="about:blank")
    match = expected("web.url_match", {"url": "about:blank"})
    step = WebStepVerifier().verify_step(contract=contract([match]), evidence=piece)
    task = WebTaskCompletionOracle().evaluate_completion(
        criteria=criteria([match]), evidence=piece
    )
    assert step.status is VerificationStatus.PASS
    assert task.status is VerificationStatus.PASS
    assert step.metadata["rule_evidence_refs"] == [
        {"rule_type": "web.url_match", "target_index": None},
    ]


# ------------------------------------------------------------------ remaining F01 (round-003)


class _HiddenPayloadDict(dict):
    """A dict-backed payload that really stores an unknown entry but hides it from ``items``."""

    def items(self):
        return [("url", "about:blank")]


class _HiddenTargetDict(dict):
    """A dict-backed locator that really stores an unknown entry but hides it from ``items``."""

    def items(self):
        return [("strategy", "test_id"), ("test_id", "unique")]


class _SecondHashKey(str):
    """A stored ``str`` key whose second hash raises the synthetic audit marker."""

    hashes = 0

    def __hash__(self):
        type(self).hashes += 1
        if type(self).hashes >= 2:
            raise ValueError("AUDIT_SYNTHETIC_RAW_MAPPING_KEY")
        return str.__hash__(self)


class _PlainKey(str):
    """A ``str`` subclass with no behavioural override (provably hook-free key)."""


def test_a_real_dict_payload_cannot_hide_an_unknown_stored_entry() -> None:
    """F01 payload/hidden-items-real-unknown on the synthetic snapshot."""
    payload = _HiddenPayloadDict(url="about:blank", hidden="x")
    assert len(dict.items(payload)) == 2
    assert list(payload.items()) == [("url", "about:blank")]
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload=payload,
            evidence=_synthetic_evidence(url="about:blank"),
        )


def test_a_real_dict_locator_cannot_hide_an_unknown_stored_entry() -> None:
    """F01 target/hidden-items-real-unknown on the synthetic snapshot."""
    locator = _HiddenTargetDict(strategy="test_id", test_id="unique", hidden="x")
    assert len(dict.items(locator)) == 3
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.element_visible", payload={"target": locator},
            evidence=_synthetic_evidence(url="about:blank"),
        )


def test_a_caller_key_hash_hook_that_runs_is_a_fixed_input_error() -> None:
    """F01 evidence/nested-mapping-key-hash-error on the synthetic snapshot."""
    _SecondHashKey.hashes = 0
    piece = _synthetic_evidence(url="about:blank")
    state = {_SecondHashKey("schema_version"): SCHEMA, "url": "about:blank", "targets": []}
    object.__getattribute__(piece.observation, "__dict__")["state"] = state
    with pytest.raises(WebVerificationInputInvalid) as caught:
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload={"url": "about:blank"}, evidence=piece
        )
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert "AUDIT_SYNTHETIC" not in str(error)
    assert error.__cause__ is None and error.__context__ is None
    assert _SecondHashKey.hashes >= 2, "the caller hash hook must really run in the product"


def test_a_provably_hook_free_stored_key_reconstructs_exactly() -> None:
    """F01 evidence/ordinary-stored-key-subclass and the safe-reconstruction branch."""
    assert _PlainKey.__hash__ is str.__hash__
    assert _PlainKey.__eq__ is str.__eq__
    key = _PlainKey("schema_version")
    piece = _synthetic_evidence(url="about:blank")
    stored = {key: SCHEMA, "url": "about:blank", "targets": []}
    object.__getattribute__(piece.observation, "__dict__")["state"] = stored
    result = WebRuleEvaluator().evaluate(
        rule_type="web.url_match", payload={"url": "about:blank"}, evidence=piece
    )
    assert result.status is VerificationStatus.PASS
    state_after = object.__getattribute__(piece.observation, "__dict__")["state"]
    assert state_after is stored
    assert list(state_after) == ["schema_version", "url", "targets"]
    assert next(iter(state_after)) is key


def test_ordinary_dict_payload_and_locator_controls_still_evaluate() -> None:
    """F01 payload/plain-dict-subclass and target/plain-dict-subclass controls."""
    piece = _synthetic_evidence(url="about:blank")
    url = WebRuleEvaluator().evaluate(
        rule_type="web.url_match", payload={"url": "about:blank"}, evidence=piece
    )
    assert (url.status, url.reason) == (VerificationStatus.PASS, "RULE_MATCH")
    locator = WebRuleEvaluator().evaluate(
        rule_type="web.element_visible",
        payload={"target": {"strategy": "test_id", "test_id": "unique"}},
        evidence=piece,
    )
    assert (locator.status, locator.reason, locator.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", 1,
    )


def test_a_plain_unknown_payload_field_is_still_a_fixed_rule_error() -> None:
    """F01 payload/plain-extra-control: an ordinary dict extra key is still refused."""
    with pytest.raises(WebRuleInvalid):
        WebRuleEvaluator().evaluate(
            rule_type="web.url_match", payload={"url": "about:blank", "hidden": "x"},
            evidence=_synthetic_evidence(url="about:blank"),
        )


def _step_contract_with_policy(policy: RecoveryPolicy) -> ActionContract:
    """Return a valid step contract whose nested recovery policy is the supplied record."""
    base = contract([expected("web.url_match", {"url": "about:blank"})])
    fields = dict(base.__dict__)
    fields["recovery_policy"] = policy
    return ActionContract.model_construct(**fields)


@pytest.mark.parametrize(("control", "supplied"), [
    pytest.param("max_retries", {"max_retries": 1}, id="abort-retries"),
    pytest.param("allow_replan", {"allow_replan": True}, id="abort-replan"),
])
def test_a_real_non_null_abort_control_is_a_fixed_input_error(
    control: str, supplied: dict
) -> None:
    """F01 step/recovery/abort-retries and step/recovery/abort-replan."""
    policy = RecoveryPolicy.model_construct(
        schema_version=SCHEMA, mode="abort", **supplied
    )
    assert getattr(policy, control) is not None
    with pytest.raises(ValidationError):
        RecoveryPolicy(schema_version=SCHEMA, mode="abort", **supplied)
    with pytest.raises(WebVerificationInputInvalid) as caught:
        WebStepVerifier().verify_step(
            contract=_step_contract_with_policy(policy),
            evidence=_synthetic_evidence(url="about:blank"),
        )
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert error.path == ("contract", "recovery_policy", control)
    assert error.__cause__ is None and error.__context__ is None


def test_an_ordinary_abort_recovery_policy_still_verifies() -> None:
    """F01 step/recovery/ordinary-abort-control."""
    piece = _synthetic_evidence(url="about:blank")
    recorded = contract([expected("web.url_match", {"url": "about:blank"})])
    assert recorded.recovery_policy.mode == "abort"
    assert recorded.recovery_policy.max_retries is None
    assert recorded.recovery_policy.allow_replan is None
    assert WebStepVerifier().verify_step(contract=recorded, evidence=piece).status is (
        VerificationStatus.PASS
    )


def test_an_ordinary_bounded_recovery_policy_still_verifies() -> None:
    """F01 step/recovery/ordinary-bounded-control."""
    piece = _synthetic_evidence(url="about:blank")
    policy = RecoveryPolicy(
        schema_version=SCHEMA, mode="bounded", max_retries=1, allow_replan=False
    )
    assert WebStepVerifier().verify_step(
        contract=_step_contract_with_policy(policy), evidence=piece
    ).status is VerificationStatus.PASS
