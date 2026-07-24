from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterable, Protocol, TypeVar

from pydantic import ValidationError

from .contracts import (
    DeploymentProfile,
    OperationalEvent,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from .hashing import canonical_hash
from .privacy import (
    PrivacyViolation,
    assert_no_direct_identifiers,
    redacted_structure,
)
from .store import (
    AppendOnlyEvidenceStore,
    ConflictingDuplicateError,
)


ContractT = TypeVar("ContractT")


@dataclass(frozen=True)
class ContractResult(Generic[ContractT]):
    accepted: bool
    value: ContractT | None
    reasons: tuple[str, ...]
    quarantine_id: str | None = None


class ReadOnlyTelemetrySource(Protocol):
    """Source protocol intentionally exposes no write/mutation operation."""

    def read_batch(self) -> Iterable[dict[str, Any]]: ...


class ContractValidator:
    def __init__(self, store: AppendOnlyEvidenceStore) -> None:
        self.store = store

    @staticmethod
    def _reasons(error: ValidationError | PrivacyViolation) -> tuple[str, ...]:
        if isinstance(error, PrivacyViolation):
            return (str(error),)
        return tuple(
            (
                ".".join(str(part) for part in item["loc"])
                + ": "
                + item["msg"]
            )
            for item in error.errors(include_url=False)
        )

    def _quarantine(
        self,
        *,
        record_type: str,
        payload: Any,
        reasons: tuple[str, ...],
    ) -> str:
        return self.store.append_quarantine(
            record_type=record_type,
            original_payload_hash=canonical_hash(payload),
            safe_structure=redacted_structure(payload),
            reasons=reasons,
        )

    def telemetry(self, payload: dict[str, Any]) -> ContractResult[
        TelemetryObservation
    ]:
        try:
            value = TelemetryObservation.model_validate(payload)
            assert_no_direct_identifiers(value.model_dump(mode="json"))
        except (ValidationError, PrivacyViolation) as error:
            reasons = self._reasons(error)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self._quarantine(
                    record_type="telemetry",
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return ContractResult(True, value, ())

    def operational_event(self, payload: dict[str, Any]) -> ContractResult[
        OperationalEvent
    ]:
        try:
            value = OperationalEvent.model_validate(payload)
            assert_no_direct_identifiers(value.model_dump(mode="json"))
        except (ValidationError, PrivacyViolation) as error:
            reasons = self._reasons(error)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self._quarantine(
                    record_type="operational_event",
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return ContractResult(True, value, ())

    def topology_component(self, payload: dict[str, Any]) -> ContractResult[
        TopologyComponent
    ]:
        return self._typed(
            payload=payload,
            model=TopologyComponent,
            record_type="topology_component",
        )

    def topology_edge(self, payload: dict[str, Any]) -> ContractResult[
        TopologyEdge
    ]:
        return self._typed(
            payload=payload,
            model=TopologyEdge,
            record_type="topology_edge",
        )

    def deployment_profile(self, payload: dict[str, Any]) -> ContractResult[
        DeploymentProfile
    ]:
        return self._typed(
            payload=payload,
            model=DeploymentProfile,
            record_type="deployment_profile",
        )

    def _typed(
        self,
        *,
        payload: dict[str, Any],
        model,
        record_type: str,
    ):
        try:
            value = model.model_validate(payload)
            assert_no_direct_identifiers(value.model_dump(mode="json"))
        except (ValidationError, PrivacyViolation) as error:
            reasons = self._reasons(error)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self._quarantine(
                    record_type=record_type,
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return ContractResult(True, value, ())


class TelemetryMirror:
    """Append-only mirror with no capability to mutate source infrastructure."""

    source_write_capability = False

    def __init__(self, store: AppendOnlyEvidenceStore) -> None:
        self.store = store
        self.validator = ContractValidator(store)

    def ingest_telemetry(
        self,
        payload: dict[str, Any],
    ) -> ContractResult[TelemetryObservation]:
        result = self.validator.telemetry(payload)
        if not result.accepted:
            return result
        assert result.value is not None
        try:
            self.store.append_telemetry(result.value)
        except ConflictingDuplicateError as error:
            reasons = (str(error),)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self.validator._quarantine(
                    record_type="telemetry_conflicting_duplicate",
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return result

    def ingest_operational_event(
        self,
        payload: dict[str, Any],
    ) -> ContractResult[OperationalEvent]:
        result = self.validator.operational_event(payload)
        if not result.accepted:
            return result
        assert result.value is not None
        try:
            self.store.append_operational_event(result.value)
        except ConflictingDuplicateError as error:
            reasons = (str(error),)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self.validator._quarantine(
                    record_type="event_conflicting_duplicate",
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return result

    def mirror_batch(
        self,
        source: ReadOnlyTelemetrySource,
    ) -> tuple[ContractResult[TelemetryObservation], ...]:
        return tuple(
            self.ingest_telemetry(payload)
            for payload in source.read_batch()
        )


class TopologyMirror:
    """Append-only topology/profile mirror; source systems remain read-only."""

    source_write_capability = False

    def __init__(self, store: AppendOnlyEvidenceStore) -> None:
        self.store = store
        self.validator = ContractValidator(store)

    def _ingest(self, payload, validator, append, record_type):
        result = validator(payload)
        if not result.accepted:
            return result
        assert result.value is not None
        try:
            append(result.value)
        except ConflictingDuplicateError as error:
            reasons = (str(error),)
            return ContractResult(
                accepted=False,
                value=None,
                reasons=reasons,
                quarantine_id=self.validator._quarantine(
                    record_type=f"{record_type}_conflicting_duplicate",
                    payload=payload,
                    reasons=reasons,
                ),
            )
        return result

    def ingest_component(self, payload: dict[str, Any]):
        return self._ingest(
            payload,
            self.validator.topology_component,
            self.store.append_topology_component,
            "topology_component",
        )

    def ingest_edge(self, payload: dict[str, Any]):
        return self._ingest(
            payload,
            self.validator.topology_edge,
            self.store.append_topology_edge,
            "topology_edge",
        )

    def ingest_profile(self, payload: dict[str, Any]):
        return self._ingest(
            payload,
            self.validator.deployment_profile,
            self.store.append_deployment_profile,
            "deployment_profile",
        )
