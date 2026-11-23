"""Public Fake adapter surface.

Import the deterministic Fake fixture from ``forecontract_fake``. The distribution is
standalone: it depends only on ``forecontract-core`` and Pydantic, it never imports the scripted
planner, Memory or any Runtime module, and it performs no external I/O.

What it is: a replay of one versioned :class:`FakeScenario` behind the existing Core
:class:`~forecontract.protocols.adapter.Adapter` protocol, with explicit facility errors.

What it is not: a production environment adapter, a domain verifier, a recovery or retry
policy, a Runtime, a browser or LLM integration. Replaying a script proves that the fixture
produced the scripted facts; it proves nothing about a real environment or a whole Goal.
"""

from __future__ import annotations

from forecontract_fake.adapter import FakeEnvironmentAdapter
from forecontract_fake.errors import (
    FakeError,
    FakeInputError,
    FakeLifecycleError,
    FakeScenarioError,
    FakeScriptExhaustedError,
    FakeStepMismatchError,
)
from forecontract_fake.models import (
    FakeScenario,
    FakeStage,
    FakeStep,
)

__version__ = "0.1.0"

#: The closed public surface: components, models and error classes only (F05). The environment
#: type and the error-code tuple stay internal to their modules, so a caller cannot come to
#: depend on a fixture constant as if it were part of the agreed interface.
__all__ = [
    "FakeEnvironmentAdapter",
    "FakeError",
    "FakeInputError",
    "FakeLifecycleError",
    "FakeScenario",
    "FakeScenarioError",
    "FakeScriptExhaustedError",
    "FakeStage",
    "FakeStep",
    "FakeStepMismatchError",
]
