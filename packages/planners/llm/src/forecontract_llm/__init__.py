"""Public surface of the offline structured LLM planner.

Import the planner, the provider protocol and the strict request/response records from
``forecontract_llm``. The distribution depends only on ``forecontract-core`` and Pydantic. It
contains no network client, no vendor SDK, no API-key lookup and no production path; the
provider is always injected by the host and is exercised offline.

What it is: a Core-compatible planner that builds one measured structured request, makes one
bounded provider call and decodes one strict structured answer into a Core ``Forecast``.

What it is not: a real model transport, an evidence retriever, a Contract Builder, a verifier
or a Runtime. A returned forecast is a proposal only; it proves nothing about the environment
or task completion.
"""

from __future__ import annotations

from forecontract_llm.errors import PlannerError
from forecontract_llm.models import (
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PlannerCloseResult,
    PlannerConfig,
    PlannerContext,
)
from forecontract_llm.planner import LLMPlanner
from forecontract_llm.provider import ModelProvider

__version__ = "0.1.0"

__all__ = [
    "LLMPlanner",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "PlannerCloseResult",
    "PlannerConfig",
    "PlannerContext",
    "PlannerError",
]
