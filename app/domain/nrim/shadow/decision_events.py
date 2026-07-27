from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .candidate_evidence import CandidateSet, CausalEvidenceSet
from .contracts import (
    DataQualityState,
    DecisionState,
    PredictionEnvelope,
    RankedCause,
    SnapshotMode,
)
from .hashing import canonical_hash, typed_hash
from .privacy import assert_no_direct_identifiers


SCHEMA_VERSION = "0.7.2"
ZERO_SHA256 = "0" * 64
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictEventModel(BaseModel):
    schema_version: Literal["0.7.2"] = SCHEMA_VERSION
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


class DecisionEventType(str, Enum):
    SNAPSHOT_SEALED = "snapshot_sealed"
    SUPPORT_ASSESSED = "support_assessed"
    DATA_QUALITY_ASSESSED = "data_quality_assessed"
    INCIDENT_EVIDENCE_PRODUCED = "incident_evidence_produced"
    CANDIDATE_SET_PRODUCED = "candidate_set_produced"
    RANKING_PRODUCED = "ranking_produced"
    CAUSAL_EVIDENCE_PRODUCED = "causal_evidence_produced"
    DECISION_SEALED = "decision_sealed"
    EVIDENCE_AMENDED = "evidence_amended"
    ADJUDICATION_LINKED = "adjudication_linked"


class SupportState(str, Enum):
    IN_SUPPORT = "in_support"
    LIMITED = "limited"
    UNSUPPORTED = "unsupported"


class AmendmentReason(str, Enum):
    LATE_OBSERVATION = "late_observation"
    ADJUDICATION = "adjudication"
    CORRECTION = "correction"
    SOURCE_REISSUE = "source_reissue"


class SnapshotSealedPayload(StrictEventModel):
    kind: Literal["snapshot_sealed"] = "snapshot_sealed"
    snapshot_id: str = Field(min_length=1, max_length=128)
    snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    mode: SnapshotMode
    watermark_utc: datetime
    source_digests: dict[str, str]

    @field_validator("watermark_utc")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return normalize_utc(value)

    @field_validator("source_digests")
    @classmethod
    def validate_digests(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("Snapshot source digests are required")
        if any(
            len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in value.values()
        ):
            raise ValueError("Snapshot source digest is invalid")
        return dict(sorted(value.items()))


class SupportAssessedPayload(StrictEventModel):
    kind: Literal["support_assessed"] = "support_assessed"
    state: SupportState
    axes: dict[str, float] = Field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    profile_sha256: str = Field(pattern=SHA256_PATTERN)


class DataQualityAssessedPayload(StrictEventModel):
    kind: Literal["data_quality_assessed"] = "data_quality_assessed"
    state: DataQualityState
    warning_codes: tuple[str, ...] = ()
    expected_observation_count: int = Field(ge=0)
    available_observation_count: int = Field(ge=0)
    applicability_summary: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.available_observation_count > self.expected_observation_count:
            raise ValueError("Available observations exceed expected count")
        if any(value < 0 for value in self.applicability_summary.values()):
            raise ValueError("Applicability counts cannot be negative")
        return self


class IncidentEvidenceProducedPayload(StrictEventModel):
    kind: Literal["incident_evidence_produced"] = (
        "incident_evidence_produced"
    )
    prediction: PredictionEnvelope


class CandidateSetProducedPayload(StrictEventModel):
    kind: Literal["candidate_set_produced"] = "candidate_set_produced"
    candidate_set: CandidateSet


class RankingProducedPayload(StrictEventModel):
    kind: Literal["ranking_produced"] = "ranking_produced"
    ranking: tuple[RankedCause, ...] = Field(min_length=1)
    ranker_identity_sha256: str = Field(pattern=SHA256_PATTERN)


class CausalEvidenceProducedPayload(StrictEventModel):
    kind: Literal["causal_evidence_produced"] = "causal_evidence_produced"
    evidence_set: CausalEvidenceSet


class DecisionSealedPayload(StrictEventModel):
    kind: Literal["decision_sealed"] = "decision_sealed"
    final_state: DecisionState
    activation_path: str = Field(min_length=1, max_length=64)
    prediction_envelope_sha256: str = Field(pattern=SHA256_PATTERN)
    chain_head_before_seal: str = Field(pattern=SHA256_PATTERN)


class EvidenceAmendedPayload(StrictEventModel):
    kind: Literal["evidence_amended"] = "evidence_amended"
    target_event_id: str = Field(pattern=r"^decision_event_[0-9a-f]{32}$")
    reason_code: AmendmentReason
    evidence_references: tuple[str, ...] = Field(min_length=1)
    amendment_sha256: str = Field(pattern=SHA256_PATTERN)


class AdjudicationLinkedPayload(StrictEventModel):
    kind: Literal["adjudication_linked"] = "adjudication_linked"
    adjudication_id: str = Field(min_length=1, max_length=128)
    adjudication_version: int = Field(gt=0)
    reviewed_decision_ids: tuple[str, ...] = Field(min_length=1)
    adjudication_sha256: str = Field(pattern=SHA256_PATTERN)


DecisionEventPayload = Annotated[
    SnapshotSealedPayload
    | SupportAssessedPayload
    | DataQualityAssessedPayload
    | IncidentEvidenceProducedPayload
    | CandidateSetProducedPayload
    | RankingProducedPayload
    | CausalEvidenceProducedPayload
    | DecisionSealedPayload
    | EvidenceAmendedPayload
    | AdjudicationLinkedPayload,
    Field(discriminator="kind"),
]


_RETROSPECTIVE_EVENT_TYPES = {
    DecisionEventType.EVIDENCE_AMENDED,
    DecisionEventType.ADJUDICATION_LINKED,
}


def _identity_preimage(
    *,
    event_type: DecisionEventType,
    schema_version: str,
    deployment_pseudonym: str,
    decision_id: str,
    decision_cutoff_utc: datetime,
    event_time_utc: datetime,
    recorded_at_utc: datetime,
    stream_sequence: int,
    causation_event_id: str | None,
    correlation_id: str,
    payload_sha256: str,
    previous_event_sha256: str,
) -> dict[str, object]:
    return {
        "event_type": event_type.value,
        "schema_version": schema_version,
        "deployment_pseudonym": deployment_pseudonym,
        "decision_id": decision_id,
        "decision_cutoff_utc": decision_cutoff_utc,
        "event_time_utc": event_time_utc,
        "recorded_at_utc": recorded_at_utc,
        "stream_sequence": stream_sequence,
        "causation_event_id": causation_event_id,
        "correlation_id": correlation_id,
        "payload_sha256": payload_sha256,
        "previous_event_sha256": previous_event_sha256,
    }


class DecisionEvent(StrictEventModel):
    event_id: str = Field(pattern=r"^decision_event_[0-9a-f]{32}$")
    event_type: DecisionEventType
    schema_version: Literal["0.7.2"] = SCHEMA_VERSION
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    decision_id: str = Field(pattern=r"^decision_[0-9a-z_.-]{8,120}$")
    decision_cutoff_utc: datetime
    event_time_utc: datetime
    recorded_at_utc: datetime
    stream_sequence: int = Field(gt=0)
    causation_event_id: str | None = Field(
        default=None,
        pattern=r"^decision_event_[0-9a-f]{32}$",
    )
    correlation_id: str = Field(min_length=8, max_length=128)
    payload: DecisionEventPayload
    payload_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_event_sha256: str = Field(pattern=SHA256_PATTERN)
    event_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator(
        "decision_cutoff_utc",
        "event_time_utc",
        "recorded_at_utc",
    )
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return normalize_utc(value)

    def _identity(self) -> dict[str, object]:
        return _identity_preimage(
            event_type=self.event_type,
            schema_version=self.schema_version,
            deployment_pseudonym=self.deployment_pseudonym,
            decision_id=self.decision_id,
            decision_cutoff_utc=self.decision_cutoff_utc,
            event_time_utc=self.event_time_utc,
            recorded_at_utc=self.recorded_at_utc,
            stream_sequence=self.stream_sequence,
            causation_event_id=self.causation_event_id,
            correlation_id=self.correlation_id,
            payload_sha256=self.payload_sha256,
            previous_event_sha256=self.previous_event_sha256,
        )

    @model_validator(mode="after")
    def validate_commitments(self) -> Self:
        if self.event_type.value != self.payload.kind:
            raise ValueError("Event type and payload discriminator differ")
        if self.recorded_at_utc < self.event_time_utc:
            raise ValueError("Recorded time cannot precede represented fact")
        if (
            self.event_type not in _RETROSPECTIVE_EVENT_TYPES
            and self.event_time_utc > self.decision_cutoff_utc
        ):
            raise ValueError("Original decision event contains future evidence")
        assert_no_direct_identifiers(self.payload.model_dump(mode="json"))
        expected_payload = typed_hash(
            type_name=self.event_type.value,
            schema_version=self.schema_version,
            value=self.payload,
        )
        if self.payload_sha256 != expected_payload:
            raise ValueError("Decision-event payload commitment is invalid")
        expected_id = "decision_event_" + canonical_hash(self._identity())[:32]
        if self.event_id != expected_id:
            raise ValueError("Decision-event identifier is invalid")
        expected_event = typed_hash(
            type_name="decision_event",
            schema_version=self.schema_version,
            value={
                **self._identity(),
                "event_id": self.event_id,
                "payload": self.payload,
            },
        )
        if self.event_sha256 != expected_event:
            raise ValueError("Decision-event commitment is invalid")
        if self.causation_event_id == self.event_id:
            raise ValueError("An event cannot cause itself")
        if isinstance(self.payload, DecisionSealedPayload):
            if (
                self.payload.chain_head_before_seal
                != self.previous_event_sha256
            ):
                raise ValueError("Decision seal names the wrong chain head")
        if isinstance(self.payload, EvidenceAmendedPayload):
            if self.payload.target_event_id == self.event_id:
                raise ValueError("An amendment cannot target itself")
        return self

    @classmethod
    def create(
        cls,
        *,
        event_type: DecisionEventType,
        deployment_pseudonym: str,
        decision_id: str,
        decision_cutoff_utc: datetime,
        event_time_utc: datetime,
        recorded_at_utc: datetime,
        stream_sequence: int,
        correlation_id: str,
        payload: DecisionEventPayload,
        previous_event_sha256: str = ZERO_SHA256,
        causation_event_id: str | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> DecisionEvent:
        cutoff = normalize_utc(decision_cutoff_utc)
        event_time = normalize_utc(event_time_utc)
        recorded = normalize_utc(recorded_at_utc)
        payload_sha256 = typed_hash(
            type_name=event_type.value,
            schema_version=schema_version,
            value=payload,
        )
        identity = _identity_preimage(
            event_type=event_type,
            schema_version=schema_version,
            deployment_pseudonym=deployment_pseudonym,
            decision_id=decision_id,
            decision_cutoff_utc=cutoff,
            event_time_utc=event_time,
            recorded_at_utc=recorded,
            stream_sequence=stream_sequence,
            causation_event_id=causation_event_id,
            correlation_id=correlation_id,
            payload_sha256=payload_sha256,
            previous_event_sha256=previous_event_sha256,
        )
        event_id = "decision_event_" + canonical_hash(identity)[:32]
        event_sha256 = typed_hash(
            type_name="decision_event",
            schema_version=schema_version,
            value={
                **identity,
                "event_id": event_id,
                "payload": payload,
            },
        )
        return cls(
            event_id=event_id,
            event_type=event_type,
            schema_version=schema_version,
            deployment_pseudonym=deployment_pseudonym,
            decision_id=decision_id,
            decision_cutoff_utc=cutoff,
            event_time_utc=event_time,
            recorded_at_utc=recorded,
            stream_sequence=stream_sequence,
            causation_event_id=causation_event_id,
            correlation_id=correlation_id,
            payload=payload,
            payload_sha256=payload_sha256,
            previous_event_sha256=previous_event_sha256,
            event_sha256=event_sha256,
        )
