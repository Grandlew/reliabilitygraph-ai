from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SimulationNodeType(str, Enum):
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


class SimulationEdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    SENDS_TO = "sends_to"
    AUTHENTICATES_WITH = "authenticates_with"
    STORES_ON = "stores_on"
    SERVES = "serves"
    CONNECTED_TO = "connected_to"


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class FailureType(str, Enum):
    HEALTHY = "healthy"
    STORAGE_CAPACITY_SATURATION = "storage_capacity_saturation"
    STORAGE_IO_DEGRADATION = "storage_io_degradation"
    CLEANUP_JOB_FAILURE = "cleanup_job_failure"
    CATCHUP_WORKER_FAILURE = "catchup_worker_failure"


class ScenarioKind(str, Enum):
    HEALTHY_CONTROL = "healthy_control"
    SINGLE_FAULT = "single_fault"


class SimulationNode(BaseModel):
    node_id: str = Field(min_length=1)
    node_type: SimulationNodeType
    name: str = Field(min_length=1)

    capacity: dict[str, float] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationEdge(BaseModel):
    edge_id: str = Field(
        default_factory=lambda: generate_id("sim_edge")
    )
    source_node_id: str
    target_node_id: str
    edge_type: SimulationEdgeType

    propagation_delay_minutes: int = Field(default=0, ge=0)
    propagation_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class DeploymentTopology(BaseModel):
    topology_id: str = Field(
        default_factory=lambda: generate_id("topology")
    )
    deployment_family: str = Field(min_length=1)
    room_count: int = Field(gt=0)
    floor_count: int = Field(gt=0)
    shared_storage: bool = True

    nodes: list[SimulationNode] = Field(min_length=1)
    edges: list[SimulationEdge] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_topology(self) -> "DeploymentTopology":
        node_ids = [node.node_id for node in self.nodes]
        node_id_set = set(node_ids)

        if len(self.nodes) != len(node_id_set):
            raise ValueError("Duplicate node IDs exist")

        for edge in self.edges:
            if edge.source_node_id not in node_id_set or edge.target_node_id not in node_id_set:
                raise ValueError(
                    f"Invalid edge endpoints for edge: {edge.edge_id}")

            if edge.source_node_id == edge.target_node_id:
                raise ValueError(f"Self-loop detected on edge: {edge.edge_id}")

        node_types = {n.node_type for n in self.nodes}
        if SimulationNodeType.CATCHUP_SERVICE not in node_types or SimulationNodeType.CATCHUP_STORAGE not in node_types:
            raise ValueError(
                "Topology must contain both CatchUPService and CatchUPStorage nodes")

        node_map = {n.node_id: n for n in self.nodes}
        for node in self.nodes:
            if node.node_type == SimulationNodeType.CATCHUP_SERVICE:
                storage_edges = [
                    e
                    for e in self.edges
                    if (
                    e.source_node_id == node.node_id and
                    e.edge_type == SimulationEdgeType.STORES_ON and
                    node_map[e.target_node_id].node_type == SimulationNodeType.CATCHUP_STORAGE
                    )
                ]
                if not storage_edges:
                    raise ValueError(
                        f"CatchUPService {node.node_id} is missing a STORES_ON connection to a storage node")

        if not self.shared_storage:
            storage_targets = [
                edge.target_node_id
                for edge in self.edges
                if edge.edge_type == SimulationEdgeType.STORES_ON
            ]
            if len(storage_targets) != len(set(storage_targets)):
                raise ValueError(
                    "Dedicated storage cannot be shared by multiple services"
                )

        return self


class OperatingRegime(BaseModel):
    regime_id: str = Field(
        default_factory=lambda: generate_id("regime")
    )

    duration_hours: int = Field(gt=0)
    sampling_interval_minutes: int = Field(gt=0)

    base_occupancy_fraction: float = Field(ge=0.0, le=1.0)
    evening_peak_multiplier: float = Field(ge=1.0)
    weekend_multiplier: float = Field(ge=0.1)

    catchup_recording_channels: int = Field(gt=0)
    average_bitrate_mbps: float = Field(gt=0.0)
    retention_days: int = Field(gt=0)

    telemetry_missing_probability: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
    )

    random_seed: int


class FaultSpecification(BaseModel):
    fault_id: str = Field(
        default_factory=lambda: generate_id("fault")
    )

    failure_type: FailureType
    target_node_id: str = Field(min_length=1)
    injection_time: datetime

    severity: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)

    reversible: bool = True
    expected_affected_service_types: list[
        SimulationNodeType
    ] = Field(default_factory=list)


class HiddenNodeState(BaseModel):
    timestamp: datetime
    node_id: str
    health_state: HealthState

    latent_capacity_factor: float = Field(
        default=1.0,
        ge=0.0,
    )
    latent_latency_factor: float = Field(
        default=1.0,
        ge=0.0,
    )
    latent_error_factor: float = Field(
        default=1.0,
        ge=0.0,
    )

    caused_by_fault_id: str | None = None


class PropagationRecord(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_id: str
    fault_id: str

    propagation_started_at: datetime
    effect_strength: float = Field(ge=0.0, le=1.0)


class ScenarioGroundTruth(BaseModel):
    scenario_id: str
    scenario_kind: ScenarioKind

    failure_type: FailureType
    root_cause_node_id: str | None
    fault_id: str | None

    injection_time: datetime | None
    incident_onset_time: datetime | None

    affected_service_node_ids: list[str] = Field(default_factory=list)
    propagation_edge_ids: list[str] = Field(default_factory=list)

    simulator_version: str
    labeling_version: str
    random_seed: int

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "ScenarioGroundTruth":

        if self.scenario_kind == ScenarioKind.HEALTHY_CONTROL:
            if any([
                self.root_cause_node_id,
                self.fault_id,
                self.injection_time,
                self.incident_onset_time,
                self.affected_service_node_ids,
                self.propagation_edge_ids
            ]):
                raise ValueError(
                    "Healthy control scenarios must not contain fault or incident data")

        if self.scenario_kind == ScenarioKind.SINGLE_FAULT:
            if not all([self.root_cause_node_id, self.fault_id, self.injection_time]):
                raise ValueError(
                    "Single fault scenarios require a root cause node, fault ID, and injection time")

        return self
