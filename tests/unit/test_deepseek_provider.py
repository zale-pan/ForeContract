"""Unit tests for the DeepSeek transport boundary.

Every case uses ``httpx.MockTransport``: no socket, no credential read, no environment key.
Facts are asserted from the observed request and response, never from a self-reported summary.
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
_REQUESTS_FILE = REPO_ROOT / "tests" / "fixtures" / "deepseek_requests.v1.json"


def _deps() -> tuple:
    """Resolve the transport lazily; collection must not change sys.path or import order."""

    for candidate in _SOURCES:
        path = str(REPO_ROOT / candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
    import httpx as http_client
    from deepseek_provider import DeepSeekProvider as provider_type
    from deepseek_provider import make_client as client_factory
    from forecontract_llm.models import ModelRequest as direct_type
    from forecontract_llm.retrieval.models import (
        RetrievalModelRequest as retrieval_type,
    )

    return http_client, provider_type, client_factory, direct_type, retrieval_type


class _Lazy:
    def __getattr__(self, name: str) -> object:
        return getattr(_deps()[0], name)


httpx = _Lazy()
ENDPOINT = "https://api.deepseek.com/chat/completions"


def _requests() -> dict:
    return json.loads(_REQUESTS_FILE.read_bytes())


def envelope(content: str = "{}", **changes: object) -> dict:
    value = {"id": "synthetic", "model": "deepseek-flash",
             "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": content}}],
             "usage": {"prompt_tokens": 21, "completion_tokens": 5}}
    value.update(changes)
    return value


def direct() -> object:
    return _deps()[3].model_validate(_requests()["direct"])


async def call(*, request=None, body=None, status: int = 200, payload=None, **limits):
    captured = []

    async def handle(incoming: httpx.Request) -> httpx.Response:
        captured.append({"method": incoming.method, "url": str(incoming.url),
                         "body": incoming.content.decode()})
        return httpx.Response(status, content=(body if body is not None
                                               else json.dumps(payload if payload is not None
                                                               else envelope()).encode()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False,
                                 follow_redirects=False, timeout=2) as client:
        provider = _deps()[1](client=client, model="deepseek-flash", **limits)
        try:
            response = await provider.complete(request or direct())
        except Exception as error:  # noqa: BLE001 - a bounded failure is data here
            return {"response": None, "error": type(error).__name__, "calls": captured}
    return {"response": response, "error": None, "calls": captured}


def test_posts_one_canonical_request():
    request = direct()
    result = asyncio.run(call(request=request))
    assert result["error"] is None
    assert len(result["calls"]) == 1
    assert result["calls"][0]["method"] == "POST"
    assert result["calls"][0]["url"] == ENDPOINT
    body = json.loads(result["calls"][0]["body"])
    assert json.loads(body["messages"][1]["content"]) == request.model_dump(mode="json")
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == request.max_output_tokens
    assert "json" in body["messages"][0]["content"].lower()
    assert result["response"].request_id == request.request_id
    assert result["response"].content == "{}"


def test_retrieval_request_carries_scope_and_exchanges():
    for kind in ("retrieval-pending", "retrieval-ready"):
        request = _deps()[4].model_validate(_requests()[kind])
        result = asyncio.run(call(request=request))
        assert result["error"] is None
        body = json.loads(result["calls"][0]["body"])
        assert json.loads(body["messages"][1]["content"]) == request.model_dump(mode="json")
        assert body["max_tokens"] == request.base_request.max_output_tokens


@pytest.mark.parametrize("payload", [
    {},
    {"choices": []},
    {"choices": envelope()["choices"] * 2},
    envelope(""),
    envelope("   "),
    {**envelope(), "choices": [{"index": 0, "finish_reason": "length",
                                "message": {"content": "{}"}}]},
    {**envelope(), "choices": [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"content": "{}"}}]},
])
def test_malformed_envelopes_are_refused(payload):
    result = asyncio.run(call(payload=payload))
    assert result["error"] is None
    assert result["response"].status == "refused"
    assert len(result["calls"]) == 1


@pytest.mark.parametrize("status", [401, 429, 500, 302])
def test_http_failures_are_bounded_and_never_retried(status):
    result = asyncio.run(call(status=status, body=b"SENTINEL-BODY"))
    assert result["response"].status == "refused"
    assert len(result["calls"]) == 1


def test_usage_missing_is_unknown_and_invalid_is_refused():
    missing = envelope()
    missing.pop("usage")
    result = asyncio.run(call(payload=missing))
    assert result["response"].status == "ok"
    assert result["response"].usage is None
    for field in ("prompt_tokens", "completion_tokens"):
        for value in (True, -1, "1"):
            broken = envelope()
            broken["usage"] = dict(broken["usage"])
            broken["usage"][field] = value
            refused = asyncio.run(call(payload=broken))
            assert refused["response"].status == "refused"


def test_request_and_response_byte_bounds():
    first = asyncio.run(call())
    size = len(first["calls"][0]["body"].encode())
    assert asyncio.run(call(max_request_bytes=size))["response"].status == "ok"
    over = asyncio.run(call(max_request_bytes=size - 1))
    assert over["response"].status == "refused"
    assert over["calls"] == []

    raw = json.dumps(envelope()).encode()
    assert asyncio.run(call(body=raw, max_response_bytes=len(raw)))["response"].status == "ok"
    rejected = asyncio.run(call(body=raw + b" ", max_response_bytes=len(raw)))
    assert rejected["response"].status == "refused"


def test_client_policy_is_explicit():
    client = _deps()[2](api_key="synthetic-not-a-key")
    try:
        assert client.trust_env is False
        assert client.follow_redirects is False
        phases = client.timeout.as_dict()
        assert set(phases) == {"connect", "read", "write", "pool"}
        assert all(isinstance(value, (int, float)) and value > 0 for value in phases.values())
    finally:
        asyncio.run(client.aclose())


def test_missing_model_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        _deps()[1](client=None, model="")


def test_late_success_after_the_bound_is_refused():
    async def late(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json=envelope())

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(late), trust_env=False) as client:
            provider = _deps()[1](client=client, model="deepseek-flash", timeout_seconds=0.05)
            return await asyncio.wait_for(provider.complete(direct()), 0.8)

    assert asyncio.run(scenario()).status == "refused"
