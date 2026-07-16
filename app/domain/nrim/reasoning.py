from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .telemetry import TelemetryQuality


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    LEADING = "leading"
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class EvidenceRole(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REQUIRED = "required"
    CONTEXT = "context"
    UNKNOWN = "unknown"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    DEFINITIVE = "definitive"


class InterventionRisk(str, Enum):
    OBSERVE_ONLY = "observe_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED_AUTONOMOUSLY = "prohibited_autonomously"


class ReasoningEvidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    source_event_id: str | None = None
    statement: str = Field(min_length=1)
    role: EvidenceRole
    strength: EvidenceStrength
    quality: TelemetryQuality
    independent_source: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class HypothesisEvidenceLink(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    evidence: ReasoningEvidence
    weight: float | None = None

    @model_validator(mode="after")
    def validate_link(self) -> "HypothesisEvidenceLink":
        if self.evidence.quality == TelemetryQuality.QUARANTINED and self.weight != 0.0:
            raise ValueError("Quarantined evidence must have a weight of 0.0")
        return self


class FailureHypothesis(BaseModel):
    hypothesis_id: str = Field(
        default_factory=lambda: generate_id("hypothesis")
    )
    failure_type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    target_node_ids: list[str] = Field(min_length=1)
    affected_service_node_ids: list[str] = Field(default_factory=list)

    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    evidence_links: list[HypothesisEvidenceLink] = Field(
        default_factory=list
    )

    ranking_score: float = 0.0
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    required_confirmation_evidence: list[str] = Field(
        default_factory=list
    )
    missing_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DiagnosticOption(BaseModel):
    diagnostic_id: str = Field(
        default_factory=lambda: generate_id("diagnostic")
    )
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(min_length=1)

    information_gain: float = Field(ge=0.0, le=1.0)
    hypothesis_separation: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    reversibility: float = Field(ge=0.0, le=1.0)
    speed: float = Field(ge=0.0, le=1.0)
    low_cost: float = Field(ge=0.0, le=1.0)

    operational_risk: InterventionRisk
    requires_engineer: bool = False
    expected_evidence: list[str] = Field(default_factory=list)
    utility_score: float = 0.0


class RecommendedIntervention(BaseModel):
    intervention_id: str = Field(
        default_factory=lambda: generate_id("intervention")
    )
    name: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    target_node_ids: list[str] = Field(min_length=1)

    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    operational_risk: InterventionRisk

    reversible: bool
    rollback_plan: str | None = None
    requires_engineer_approval: bool = True
    verification_test: str = Field(min_length=1)

    approved_by: str | None = None
    executed_at: datetime | None = None
    outcome: str | None = None

    @model_validator(mode="after")
    def validate_intervention(self) -> "RecommendedIntervention":
        if self.reversible and not self.rollback_plan:
            raise ValueError(
                "Reversible interventions must include a rollback plan")

        if self.operational_risk == InterventionRisk.PROHIBITED_AUTONOMOUSLY:
            if not self.requires_engineer_approval:
                raise ValueError(
                    "Prohibited autonomous interventions must require engineer approval")
        return self


class ReasoningResult(BaseModel):
    case_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)

    hypotheses: list[FailureHypothesis]
    ranked_diagnostics: list[DiagnosticOption]
    recommended_interventions: list[RecommendedIntervention]

    leading_hypothesis_id: str | None = None
    conclusion: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
