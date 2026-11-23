#!/usr/bin/env python
"""Host-only registration bridge between the existing evidence recorder and the reader.

:class:`RegisteringEvidenceRecorder` subclasses the already-verified
``web_runtime_evidence.EvidenceRecorder``. It preserves the parent's exact behaviour - the
same snapshot bytes, the same index entry, the same error type - and adds exactly one step
after the real snapshot was written: the host's ``grants_for(observation)`` supplies the
authorizations, and ``reader.register_observation`` validates and registers that snapshot.

Because ``web_runtime_evidence.RecordingAdapter.observe`` calls ``record_observation`` before
it returns the observation, registration is complete before the Runtime's observe call
returns. The original recorder, the original RecordingAdapter and the completed-state
``finalize``/``verify_evidence_index`` boundary are untouched; this bridge is not a substitute
for either of them.

A failed registration propagates the bounded error. The parent may already have written the raw
snapshot, which is honest: the bytes exist on disk, but they are not in the read directory and
planning must not continue.
"""

from __future__ import annotations

from pathlib import Path

from forecontract_llm.retrieval import EvidenceGrant, RegisteredEvidenceReader
from forecontract_llm.retrieval.errors import RetrievalError
from web_runtime_evidence import EvidenceError, EvidenceRecorder

__all__ = ["RegisteringEvidenceRecorder"]


class RegisteringEvidenceRecorder(EvidenceRecorder):
    """The existing recorder plus host-side registration of each real observation snapshot."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        profile: str,
        reader: RegisteredEvidenceReader,
        grants_for,
    ) -> None:
        super().__init__(root, run_id=run_id, profile=profile)
        if not isinstance(reader, RegisteredEvidenceReader):
            raise EvidenceError("the bridge requires a registered evidence reader") from None
        if not callable(grants_for):
            raise EvidenceError("the bridge requires a host grants callback") from None
        self._reader = reader
        self._grants_for = grants_for

    def record_observation(self, observation: object) -> dict:
        """Write the real snapshot exactly as the parent does, then register it."""
        entry = super().record_observation(observation)
        grants = self._grants_for(observation)
        if not isinstance(grants, (list, tuple)):
            raise EvidenceError("the host grants callback did not return a grant list") from None
        for grant in grants:
            if not isinstance(grant, EvidenceGrant):
                raise EvidenceError("a host grant is not a declared evidence grant") from None
        try:
            self._reader.register_observation(observation, entry, list(grants))
        except RetrievalError:
            raise
        except Exception:  # noqa: BLE001 - a registration failure is one bounded evidence error
            raise EvidenceError("the observation snapshot could not be registered") from None
        return entry
