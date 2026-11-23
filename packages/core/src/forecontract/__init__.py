"""Core distribution of ForeContract.

It provides the versioned data models, the adapter, planner and domain protocol declarations,
the deterministic Contract Builder, tracing/memory primitives, the verification/recovery
building blocks and the sequential single-Run Runtime. The Runtime always prevalidates task
criteria before any resource starts and accepts an independent criteria validator; the generic
Recovery engine takes the host's declared step-verifier identity instead of a hardcoded label.
It imports no browser, planner, serving or training package, and the package root exports only
its version.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
