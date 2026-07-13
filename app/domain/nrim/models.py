from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .ontology import (
    EdgeType,
    EvidenceKind,
    FailurePredictability,
    GoldenSignal,
    KnowledgeStatus,
    NodeCategory,
    OutcomeStatus,
    ProductMode,
    SignalType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


STATUSES_REQUIRING_EVIDENCE = {
    KnowledgeStatus.EXTRACTED,
    KnowledgeStatus.INFERRED,
    KnowledgeStatus.ASSUMED,
    KnowledgeStatus.RECOMMENDED,
    KnowledgeStatus.OBSERVED,
    KnowledgeStatus.VERIFIED,
    KnowledgeStatus.REJECTED,
}


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("evidence"))
    kind: EvidenceKind
    statement: str = Field(min_length=1)
    source_reference: str | None = None
    value: Any = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class NRIMNode(BaseModel):
    id: str = Field(min_length=1)
    category: NodeCategory
    type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: KnowledgeStatus
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    requires_engineer_review: bool = False

    @model_validator(mode="after")
    def validate_evidence_requirement(self) -> "NRIMNode":
        if self.status in STATUSES_REQUIRING_EVIDENCE and not self.evidence:
            raise ValueError(
                f"Status {self.status} requires evidence but no evidence was provided"
            )
        return self


class NRIMEdge(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("edge"))
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    type: EdgeType
    status: KnowledgeStatus
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence_requirement(self) -> "NRIMEdge":
        if self.status in STATUSES_REQUIRING_EVIDENCE and not self.evidence:
            raise ValueError(
                f"Status {self.status} requires evidence but no evidence was provided"
            )
        return self


class OperationalObservation(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("observation"))
    deployment_id: str = Field(min_length=1)
    component_node_id: str = Field(min_length=1)
    timestamp: datetime
    signal_type: SignalType
    signal_name: str = Field(min_length=1)
    golden_signal: GoldenSignal | None = None
    value: Any
    unit: str | None = None
    collection_source: str = Field(min_length=1)
    quality: str = "unverified"
    sampling_interval_seconds: int | None = Field(default=None, gt=0)
    baseline_value: Any = None
    deviation_score: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class DiagnosticTest(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("test"))
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(min_length=1)
    estimated_minutes: int = Field(ge=0)
    cost_level: int = Field(ge=1, le=5)
    invasiveness_level: int = Field(ge=1, le=5)
    operational_risk_level: int = Field(ge=1, le=5)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    result: dict[str, Any] | None = None


class Intervention(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("intervention"))
    name: str = Field(min_length=1)
    target_node_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    rollback_plan: str | None = None
    approved_by: str | None = None
    executed_at: datetime | None = None
    outcome_status: OutcomeStatus = OutcomeStatus.UNKNOWN
    outcome_notes: str | None = None


class Forecast(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("forecast"))
    target_node_id: str
    predicted_failure_type: str = Field(min_length=1)
    predictability_class: FailurePredictability
    horizon_hours: int = Field(gt=0)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_observation_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ReliabilityCase(BaseModel):
    schema_version: str = "1.0.0"
    case_id: str = Field(default_factory=lambda: generate_id("case"))
    project_id: str = Field(min_length=1)
    deployment_id: str | None = None
    mode: ProductMode
    status: str = "draft"

    nodes: list[NRIMNode] = Field(default_factory=list)
    edges: list[NRIMEdge] = Field(default_factory=list)
    observations: list[OperationalObservation] = Field(default_factory=list)
    diagnostic_tests: list[DiagnosticTest] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    forecasts: list[Forecast] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_case_integrity(self) -> "ReliabilityCase":
        node_ids = {node.id: node for node in self.nodes}
        observation_ids = {obs.id for obs in self.observations}

        if len(self.nodes) != len(node_ids):
            raise ValueError("Duplicate node IDs found")

        for edge in self.edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError(
                    f"Invalid Edge {edge.id}: Source or target node does not exist")

            if edge.source_node_id == edge.target_node_id:
                raise ValueError(f"Self-loop edge found: {edge.id}")

        for test in self.diagnostic_tests:
            for h_id in test.hypothesis_ids:
                if h_id not in node_ids:
                    raise ValueError(f"Hypothesis ID {h_id} not found")
                if node_ids[h_id].category != NodeCategory.FAILURE:
                    raise ValueError(
                        f"Hypothesis {h_id} must reference a FAILURE node")

        for obs in self.observations:
            if obs.component_node_id not in node_ids:
                raise ValueError(
                    f"Observation {obs.id} references non-existent node {obs.component_node_id}")

        for intervention in self.interventions:
            for t_id in intervention.target_node_ids:
                if t_id not in node_ids:
                    raise ValueError(
                        f"Intervention {intervention.id} target {t_id} not found")

        for forecast in self.forecasts:
            if forecast.target_node_id not in node_ids:
                raise ValueError(
                    f"Forecast {forecast.id} target node {forecast.target_node_id} not found")
            for obs_id in forecast.supporting_observation_ids:
                if obs_id not in observation_ids:
                    raise ValueError(
                        f"Forecast {forecast.id} references non-existent observation {obs_id}")

        return self
