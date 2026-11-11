"""The offline provider boundary.

The planner depends on one structural interface only: an asynchronous ``complete`` that maps
one :class:`~forecontract_llm.models.ModelRequest` to one
:class:`~forecontract_llm.models.ModelResponse`. A real vendor transport is deliberately not
part of this package: there is no HTTP client, no SDK, no credential lookup and no network
call anywhere in this distribution.

:class:`OfflineProvider` is the deterministic, in-process implementation supplied for tests
and examples. It replays a finite sequence of already-validated responses and records the
request records it was handed; it performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from forecontract_llm.errors import PlannerError
from forecontract_llm.models import ModelRequest, ModelResponse, read_declared_copy

__all__ = ["ModelProvider", "OfflineProvider"]


class ModelProvider(Protocol):
    """The injected model transport.

    An implementation must be a cooperative, non-blocking coroutine. It owns its own
    connection resources; the planner owns only its single call task.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one response bound to ``request.request_id``."""
        ...


class OfflineProvider:
    """A deterministic in-process provider over a finite response sequence.

    Each :meth:`complete` call consumes exactly one response. The returned object is a
    detached copy, so a caller cannot mutate the provider's stored answer. An exhausted
    sequence is an ordinary provider failure, not a fabricated response.
    """

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses: tuple[ModelResponse, ...] = tuple(
            read_declared_copy(item, ModelResponse, "INPUT_INVALID") for item in responses
        )
        self._cursor = 0
        self.received: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted response for this request, or fail honestly."""
        validated = read_declared_copy(request, ModelRequest, "INPUT_INVALID")
        self.received.append(validated)
        if self._cursor >= len(self._responses):
            raise PlannerError("PROVIDER_FAILURE",
                               "the offline provider has no scripted response left",
                               location=("provider",))
        response = self._responses[self._cursor]
        self._cursor += 1
        return response.model_copy(deep=True)
