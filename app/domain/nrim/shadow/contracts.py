from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.domain.nrim.simulation.feature_schema import (
    SIGNAL_APPLICABLE_NODE_TYPES,
)

from .hashing import canonical_hash


SCHEMA_VERSION = "0.7.0"


class StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


class Applicability(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class ObservationQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    QUARANTINED = "quarantined"


class ComponentType(str, Enum):
    SIGNAL_SOURCE = "signal_source"
    GATEWAY = "gateway"
    STREAMER = "streamer"
    MIDDLEWARE = "middleware"
    DATABASE = "database"
    CATCHUP_SERVICE = "catchup_service"
    CATCHUP_STORAGE = "catchup_storage"
    EPG_SERVICE = "epg_service"
    CORE_SWITCH = "core_switch"
    DISTRIBUTION_SWITCH = "distribution_switch"
    SMART_TV_GROUP = "smart_tv_group"


class MetricName(str, Enum):
    DISK_UTILIZATION = "system.disk.utilization"
    DISK_IO_LATENCY = "system.disk.io_latency"
    DISK_IO_ERRORS = "system.disk.io_errors"
    RECORDING_FAILURES = "iptv.catchup.recording_failures"
    PROCESS_RESTART_COUNT = "system.process.restart_count"
    ACTIVE_SESSION_COUNT = "iptv.session.active_count"
    SERVICE_AVAILABILITY = "iptv.catchup.service_availability"


class MetricUnit(str, Enum):
    PERCENT = "percent"
    MILLISECONDS = "milliseconds"
    ERRORS_PER_INTERVAL = "errors_per_interval"
    FAILURES_PER_INTERVAL = "failures_per_interval"
    RESTARTS_PER_INTERVAL = "restarts_per_interval"
    SESSIONS = "sessions"


EXPECTED_UNIT = {
    MetricName.DISK_UTILIZATION: MetricUnit.PERCENT,
    MetricName.DISK_IO_LATENCY: MetricUnit.MILLISECONDS,
    MetricName.DISK_IO_ERRORS: MetricUnit.ERRORS_PER_INTERVAL,
    MetricName.RECORDING_FAILURES: MetricUnit.FAILURES_PER_INTERVAL,
    MetricName.PROCESS_RESTART_COUNT: MetricUnit.RESTARTS_PER_INTERVAL,
    MetricName.ACTIVE_SESSION_COUNT: MetricUnit.SESSIONS,
    MetricName.SERVICE_AVAILABILITY: MetricUnit.PERCENT,
}


class DependencyType(str, Enum):
    DEPENDS_ON = "depends_on"
    SENDS_TO = "sends_to"
    AUTHENTICATES_WITH = "authenticates_with"
    STORES_ON = "stores_on"
    SERVES = "serves"
    CONNECTED_TO = "connected_to"


class DirectionSemantics(str, Enum):
    SOURCE_TO_DESTINATION = "source_to_destination"
    DESTINATION_DEPENDS_ON_SOURCE = "destination_depends_on_source"
    BIDIRECTIONAL = "bidirectional"


class DataQualityState(str, Enum):
    ACCEPTED = "accepted"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class SnapshotMode(str, Enum):
    PROSPECTIVE = "prospective"
    LATE_DATA_REPLAY = "late_data_replay"
    GOLDEN_REPLAY = "golden_replay"


class DecisionState(str, Enum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    INCIDENT = "incident"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"
    ESCALATE = "escalate"
    DATA_QUALITY_ESCALATION = "data_quality_escalation"


class ReviewAnswer(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdjudicationConfidence(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"


class TelemetryObservation(StrictContract):
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    component_pseudonym: str = Field(min_length=12, max_length=128)
    component_type: ComponentType
    metric_name: MetricName
    metric_value: float | None = None
    unit: MetricUnit
    applicability: Applicability
    quality: ObservationQuality
    event_time_utc: datetime
    ingestion_time_utc: datetime
    collector_id: str = Field(min_length=1, max_length=128)
    source_sequence_id: str = Field(min_length=1, max_length=256)
    topology_version: str = Field(min_length=1, max_length=128)
    schema_version: str = SCHEMA_VERSION

    @field_validator("event_time_utc", "ingestion_time_utc")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.unit is not EXPECTED_UNIT[self.metric_name]:
            raise ValueError(
                f"{self.metric_name.value} requires "
                f"{EXPECTED_UNIT[self.metric_name].value}"
            )
        if self.applicability is Applicability.OBSERVED:
            if self.metric_value is None or not math.isfinite(
                self.metric_value
            ):
                raise ValueError("OBSERVED requires a finite metric value")
        elif self.metric_value is not None:
            raise ValueError(
                "MISSING and NOT_APPLICABLE observations require null value"
            )

        applicable_types = SIGNAL_APPLICABLE_NODE_TYPES[
            self.metric_name.value
        ]
        if (
            self.applicability is not Applicability.NOT_APPLICABLE
            and self.component_type.value not in applicable_types
        ):
            raise ValueError(
                f"{self.metric_name.value} does not apply to "
                f"{self.component_type.value}"
            )
        if self.metric_value is not None:
            if self.metric_value < 0.0:
                raise ValueError("Metric values cannot be negative")
            if (
                self.metric_name
                in {
                    MetricName.DISK_UTILIZATION,
                    MetricName.SERVICE_AVAILABILITY,
                }
                and self.metric_value > 100.0
            ):
                raise ValueError("Percent metrics must be in [0, 100]")
        return self


class OperationalEvent(StrictContract):
    """Observable operational event; no simulator/audit category is accepted."""

    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    component_pseudonym: str | None = Field(
        default=None,
        min_length=12,
        max_length=128,
    )
    signal_name: str = Field(min_length=1, max_length=128)
    event_time_utc: datetime
    ingestion_time_utc: datetime
    source_sequence_id: str = Field(min_length=1, max_length=256)
    topology_version: str = Field(min_length=1, max_length=128)
    schema_version: str = SCHEMA_VERSION

    @field_validator("event_time_utc", "ingestion_time_utc")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _utc(value)


class TopologyComponent(StrictContract):
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    component_pseudonym: str = Field(min_length=12, max_length=128)
    component_type: ComponentType
    valid_from_utc: datetime
    valid_to_utc: datetime | None = None
    topology_version: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    schema_version: str = SCHEMA_VERSION

    @field_validator("valid_from_utc", "valid_to_utc")
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if (
            self.valid_to_utc is not None
            and self.valid_to_utc <= self.valid_from_utc
        ):
            raise ValueError("Topology validity interval must be positive")
        return self


class TopologyEdge(StrictContract):
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    source_component: str = Field(min_length=12, max_length=128)
    destination_component: str = Field(min_length=12, max_length=128)
    dependency_type: DependencyType
    direction_semantics: DirectionSemantics
    valid_from_utc: datetime
    valid_to_utc: datetime | None = None
    topology_version: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    propagation_delay_minutes: int = Field(default=0, ge=0)
    propagation_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    schema_version: str = SCHEMA_VERSION

    @field_validator("valid_from_utc", "valid_to_utc")
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.source_component == self.destination_component:
            raise ValueError("Topology self-loops are not permitted")
        if (
            self.valid_to_utc is not None
            and self.valid_to_utc <= self.valid_from_utc
        ):
            raise ValueError("Topology validity interval must be positive")
        return self


class DeploymentProfile(StrictContract):
    """Observable workload context required by the frozen v0.6 policy."""

    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    profile_version: str = Field(min_length=1, max_length=128)
    valid_from_utc: datetime
    valid_to_utc: datetime | None = None
    software_version: str = Field(min_length=1, max_length=128)
    room_count: int = Field(gt=0)
    floor_count: int = Field(gt=0)
    retention_days: int = Field(gt=0)
    base_occupancy_fraction: float = Field(ge=0.0, le=1.0)
    catchup_recording_channels: int = Field(gt=0)
    average_bitrate_mbps: float = Field(gt=0.0)
    shared_storage: bool
    redundant_middleware: bool
    collector_family: str = Field(min_length=1, max_length=128)
    schema_version: str = SCHEMA_VERSION

    @field_validator("valid_from_utc", "valid_to_utc")
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if (
            self.valid_to_utc is not None
            and self.valid_to_utc <= self.valid_from_utc
        ):
            raise ValueError("Profile validity interval must be positive")
        return self

    def inference_metadata(self, missing_fraction: float) -> dict[str, Any]:
        """Return only the numeric context fields frozen into v0.6."""

        return {
            "room_count": self.room_count,
            "floor_count": self.floor_count,
            "retention_days": self.retention_days,
            "base_occupancy_fraction": self.base_occupancy_fraction,
            "catchup_recording_channels": self.catchup_recording_channels,
            "average_bitrate_mbps": self.average_bitrate_mbps,
            "shared_storage": self.shared_storage,
            "redundant_middleware": self.redundant_middleware,
            "missing_fraction": missing_fraction,
        }


class ServingSnapshot(StrictContract):
    snapshot_id: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    observation_start_utc: datetime
    decision_cutoff_utc: datetime
    as_of_ingestion_time_utc: datetime
    topology_version: str = Field(min_length=1, max_length=128)
    profile: DeploymentProfile
    components: tuple[TopologyComponent, ...] = Field(min_length=1)
    edges: tuple[TopologyEdge, ...]
    observations: tuple[TelemetryObservation, ...]
    operational_events: tuple[OperationalEvent, ...] = ()
    data_quality_state: DataQualityState
    data_quality_warnings: tuple[str, ...] = ()
    expected_observation_count: int = Field(ge=0)
    available_observation_count: int = Field(ge=0)
    mode: SnapshotMode = SnapshotMode.PROSPECTIVE
    supersedes_prediction_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    @field_validator(
        "observation_start_utc",
        "decision_cutoff_utc",
        "as_of_ingestion_time_utc",
    )
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.observation_start_utc >= self.decision_cutoff_utc:
            raise ValueError("Observation window must be positive")
        if self.as_of_ingestion_time_utc < self.decision_cutoff_utc:
            raise ValueError("As-of ingestion time cannot precede cutoff")
        if (
            self.mode is SnapshotMode.LATE_DATA_REPLAY
            and not self.supersedes_prediction_id
        ):
            raise ValueError("Late-data replay must reference original prediction")
        if (
            self.mode is not SnapshotMode.LATE_DATA_REPLAY
            and self.supersedes_prediction_id is not None
        ):
            raise ValueError("Only late-data replay may supersede a prediction")

        deployment = self.deployment_pseudonym
        version = self.topology_version
        if self.profile.deployment_pseudonym != deployment:
            raise ValueError("Profile deployment differs from snapshot")
        if not (
            self.profile.valid_from_utc <= self.decision_cutoff_utc
            and (
                self.profile.valid_to_utc is None
                or self.decision_cutoff_utc < self.profile.valid_to_utc
            )
        ):
            raise ValueError("Profile is not valid at decision cutoff")

        component_ids = set()
        for component in self.components:
            if (
                component.deployment_pseudonym != deployment
                or component.topology_version != version
            ):
                raise ValueError("Component provenance differs from snapshot")
            if component.component_pseudonym in component_ids:
                raise ValueError("Duplicate topology component")
            component_ids.add(component.component_pseudonym)
            if not (
                component.valid_from_utc <= self.decision_cutoff_utc
                and (
                    component.valid_to_utc is None
                    or self.decision_cutoff_utc < component.valid_to_utc
                )
            ):
                raise ValueError("Component is not valid at decision cutoff")
        for edge in self.edges:
            if (
                edge.deployment_pseudonym != deployment
                or edge.topology_version != version
            ):
                raise ValueError("Edge provenance differs from snapshot")
            if (
                edge.source_component not in component_ids
                or edge.destination_component not in component_ids
            ):
                raise ValueError("Edge references unknown component")
            if not (
                edge.valid_from_utc <= self.decision_cutoff_utc
                and (
                    edge.valid_to_utc is None
                    or self.decision_cutoff_utc < edge.valid_to_utc
                )
            ):
                raise ValueError("Edge is not valid at decision cutoff")
        for observation in self.observations:
            if (
                observation.deployment_pseudonym != deployment
                or observation.topology_version != version
            ):
                raise ValueError("Observation provenance differs from snapshot")
            if observation.component_pseudonym not in component_ids:
                raise ValueError("Observation references unknown component")
            if not (
                self.observation_start_utc
                <= observation.event_time_utc
                <= self.decision_cutoff_utc
            ):
                raise ValueError("Snapshot contains future/out-of-window telemetry")
            if observation.ingestion_time_utc > self.as_of_ingestion_time_utc:
                raise ValueError("Snapshot contains not-yet-ingested telemetry")
        for event in self.operational_events:
            if event.deployment_pseudonym != deployment:
                raise ValueError("Operational-event deployment differs")
            if not (
                self.observation_start_utc
                <= event.event_time_utc
                <= self.decision_cutoff_utc
            ):
                raise ValueError("Snapshot contains future operational event")
            if event.ingestion_time_utc > self.as_of_ingestion_time_utc:
                raise ValueError("Snapshot contains not-yet-ingested event")
        return self

    @property
    def availability_fraction(self) -> float:
        if self.expected_observation_count == 0:
            return 0.0
        return (
            self.available_observation_count
            / self.expected_observation_count
        )

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("snapshot_id", None)
        return canonical_hash(payload)

    def topology_hash(self) -> str:
        return canonical_hash(
            {
                "version": self.topology_version,
                "components": self.components,
                "edges": self.edges,
            }
        )

    def profile_hash(self) -> str:
        return canonical_hash(self.profile)


class RankedCause(StrictContract):
    component_pseudonym: str = Field(min_length=12, max_length=128)
    score: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def finite_evidence(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        if any(not math.isfinite(float(item)) for item in value.values()):
            raise ValueError("Ranking evidence must be finite")
        return value


class PredictionEnvelope(StrictContract):
    prediction_id: str = Field(min_length=1, max_length=128)
    decision_cutoff_utc: datetime
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    snapshot_mode: SnapshotMode
    telemetry_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_vector_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    model_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage1_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    healthy_residual: float | None = None
    support_score: float | None = Field(default=None, ge=0.0)
    support_axes: dict[str, float] = Field(default_factory=dict)
    impact_score: float | None = Field(default=None, ge=0.0, le=1.0)
    impact_families: tuple[str, ...] = ()
    causal_consistency: bool | None = None
    fast_path_state: DecisionState | None = None
    slow_path_state: DecisionState | None = None
    dual_path_state: DecisionState | None = None
    slow_path_statistic: float | None = Field(default=None, ge=0.0)
    episode_state_before: DecisionState | None = None
    episode_state_after: DecisionState
    final_decision: DecisionState
    activation_path: str = Field(min_length=1, max_length=32)
    stage2_top_k: tuple[RankedCause, ...] = ()
    counterfactual_policy_states: dict[str, DecisionState] = Field(
        default_factory=dict
    )
    data_quality_state: DataQualityState
    data_quality_warnings: tuple[str, ...] = ()
    inference_latency_ms: float = Field(ge=0.0)
    infrastructure_version: str = Field(min_length=1, max_length=128)
    created_at_utc: datetime
    supersedes_prediction_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    @field_validator("decision_cutoff_utc", "created_at_utc")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_safe_routing(self) -> Self:
        unsupported = self.final_decision in {
            DecisionState.UNKNOWN,
            DecisionState.ESCALATE,
            DecisionState.DATA_QUALITY_ESCALATION,
        }
        if unsupported and self.stage2_top_k:
            raise ValueError("Unsupported/blocked decisions cannot expose Stage 2")
        if self.data_quality_state is DataQualityState.BLOCKED:
            if self.final_decision is not DecisionState.DATA_QUALITY_ESCALATION:
                raise ValueError("Blocked data must emit data-quality escalation")
            if any(
                value is not None
                for value in (
                    self.stage1_probability,
                    self.healthy_residual,
                    self.support_score,
                    self.impact_score,
                )
            ):
                raise ValueError("Blocked data cannot contain model outputs")
        if (
            self.snapshot_mode is SnapshotMode.LATE_DATA_REPLAY
            and not self.supersedes_prediction_id
        ):
            raise ValueError("Replay prediction must reference original")
        return self


class IncidentAdjudication(StrictContract):
    adjudication_id: str = Field(min_length=1, max_length=128)
    deployment_pseudonym: str = Field(min_length=12, max_length=128)
    review_window_start_utc: datetime
    review_window_end_utc: datetime
    customer_impact_present: ReviewAnswer
    observable_degradation_present: ReviewAnswer
    impact_onset_utc: datetime | None = None
    recovery_utc: datetime | None = None
    severity: IncidentSeverity
    confirmed_root_component: str | None = Field(
        default=None,
        min_length=12,
        max_length=128,
    )
    root_cause_family: str = Field(min_length=1, max_length=128)
    confidence: AdjudicationConfidence
    existing_monitor_detected_first: bool | None = None
    nrim_top1_useful: bool | None = None
    nrim_top3_useful: bool | None = None
    would_change_investigation_order: bool | None = None
    reviewer_id_pseudonym: str = Field(min_length=12, max_length=128)
    blinded_initial_assessment: bool
    evidence_notes: str = Field(default="", max_length=8000)
    evidence_note_source: str = Field(min_length=1, max_length=128)
    onset_time_source: str | None = Field(default=None, max_length=128)
    onset_time_confidence: AdjudicationConfidence | None = None
    adjudication_version: int = Field(ge=1)
    prediction_ids: tuple[str, ...] = ()
    created_at_utc: datetime
    schema_version: str = SCHEMA_VERSION

    @field_validator(
        "review_window_start_utc",
        "review_window_end_utc",
        "impact_onset_utc",
        "recovery_utc",
        "created_at_utc",
    )
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if self.review_window_end_utc <= self.review_window_start_utc:
            raise ValueError("Review window must be positive")
        if self.blinded_initial_assessment and any(
            value is not None
            for value in (
                self.nrim_top1_useful,
                self.nrim_top3_useful,
                self.would_change_investigation_order,
            )
        ):
            raise ValueError(
                "Blinded initial assessment cannot contain NRIM usefulness"
            )
        if (
            self.customer_impact_present is ReviewAnswer.YES
            and self.impact_onset_utc is None
        ):
            raise ValueError("Customer impact requires an onset timestamp")
        if (
            self.impact_onset_utc is not None
            and not (
                self.review_window_start_utc
                <= self.impact_onset_utc
                <= self.review_window_end_utc
            )
        ):
            raise ValueError("Impact onset is outside review window")
        if (
            self.recovery_utc is not None
            and self.impact_onset_utc is not None
            and self.recovery_utc < self.impact_onset_utc
        ):
            raise ValueError("Recovery cannot precede onset")
        return self
