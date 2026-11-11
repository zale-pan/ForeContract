"""Public ScriptedPlanner surface.

Import the deterministic planner fixture from ``forecontract_scripted``. The distribution is
standalone: it depends only on ``forecontract-core`` and Pydantic, it never imports the Fake
adapter package, Memory or any Runtime module, and it performs no external I/O.

What it is: a replay of one versioned :class:`PlannerScript` behind one async ``plan`` method,
with explicit facility errors.

What it is not: a production planner, an LLM, a verifier, a Contract Builder, a Memory client or
a Runtime. Returning a scripted forecast proves that the fixture produced it; it proves nothing
about a real environment, a committable contract or a whole Goal.
"""

from __future__ import annotations

from forecontract_scripted.errors import (
    PlannerError,
    PlannerInputError,
    PlannerObservationMismatchError,
    PlannerScriptError,
    PlannerScriptExhaustedError,
)
from forecontract_scripted.models import PlannerScript, PlannerStep
from forecontract_scripted.planner import ScriptedPlanner

__version__ = "0.1.0"

#: The closed public surface: components, models and error classes only (F05). The error-code
#: tuple stays internal to its module, so a caller cannot come to depend on a fixture constant
#: as if it were part of the agreed interface.
__all__ = [
    "PlannerError",
    "PlannerInputError",
    "PlannerObservationMismatchError",
    "PlannerScript",
    "PlannerScriptError",
    "PlannerScriptExhaustedError",
    "PlannerStep",
    "ScriptedPlanner",
]
