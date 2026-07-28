from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..hashing import canonical_hash
from .contracts import (
    GateDecision,
    LabelKind,
    SignatureMetadata,
    utc,
    verify_signature,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ReplayMetric(str, Enum):
    INCIDENT_ALIGNMENT = "incident_alignment"
    RCA_MAPPING = "rca_mapping"
    FEATURE_RECONSTRUCTABILITY = "feature_reconstructability"
    TOPOLOGY_RECONSTRUCTION = "topology_reconstruction"
    FUTURE_LEAKAGE = "future_leakage"


class CohortDefinition(StrictModel):
    cohort_id: str = Field(min_length=3, max_length=128)
    inclusion_rule: str = Field(min_length=10, max_length=1000)
    exclusion_rule: str = Field(min_length=10, max_length=1000)
    event_time_start_utc: datetime
    event_time_end_utc: datetime

    @field_validator("event_time_start_utc", "event_time_end_utc")
    @classmethod
    def normalize(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def positive_window(self) -> Self:
        if self.event_time_end_utc <= self.event_time_start_utc:
            raise ValueError("Replay cohort window must be positive")
        return self


class LabelDefinition(StrictModel):
    label_kind: LabelKind
    source_inventory_id: str = Field(min_length=3, max_length=128)
    alignment_rule: str = Field(min_length=10, max_length=1000)
    adjudication_rule: str = Field(min_length=10, max_length=1000)


class HistoricalReplayProtocol(StrictModel):
    protocol_id: str = Field(min_length=3, max_length=128)
    protocol_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    preregistered_at_utc: datetime
    cohorts: tuple[CohortDefinition, ...] = Field(min_length=1)
    labels: tuple[LabelDefinition, ...] = Field(min_length=3)
    metrics: tuple[ReplayMetric, ...] = Field(min_length=5)
    counterfactual_policy: str = Field(min_length=20, max_length=2000)
    exclusions: tuple[str, ...] = Field(min_length=3)
    event_time_cutoff_rule: str = Field(min_length=20, max_length=1000)
    no_model_tuning: bool = True
    no_threshold_tuning: bool = True
    no_feature_tuning: bool = True
    no_support_rule_tuning: bool = True
    no_watermark_tuning: bool = True
    no_episode_grouping_tuning: bool = True
    signature: SignatureMetadata

    @field_validator("preregistered_at_utc")
    @classmethod
    def normalize(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def freeze_protocol(self) -> Self:
        if {item.label_kind for item in self.labels} != set(LabelKind):
            raise ValueError("Replay protocol must separate all label kinds")
        if set(self.metrics) != set(ReplayMetric):
            raise ValueError("Replay metrics must be complete")
        commitments = (
            self.no_model_tuning,
            self.no_threshold_tuning,
            self.no_feature_tuning,
            self.no_support_rule_tuning,
            self.no_watermark_tuning,
            self.no_episode_grouping_tuning,
        )
        if not all(commitments):
            raise ValueError("Historical replay cannot authorize tuning")
        return self

    def content_hash(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"signature"})
        )

    @property
    def signature_valid(self) -> bool:
        return verify_signature(
            self.signature,
            expected_payload_sha256=self.content_hash(),
        )


def replay_readiness(
    *,
    protocol: HistoricalReplayProtocol,
    real_deployment_pack_qualified: bool,
    qualification_criteria_passed: bool,
) -> dict[str, object]:
    if not real_deployment_pack_qualified:
        decision = GateDecision.REAL_DATA_REQUIRED
        reasons = ("LAWFUL_REAL_DEPLOYMENT_PACK_REQUIRED",)
    elif not qualification_criteria_passed:
        decision = GateDecision.BLOCKED
        reasons = ("QUALIFICATION_CRITERIA_NOT_MET",)
    else:
        decision = GateDecision.REAL_REPLAY_READY
        reasons = ()
    return {
        "protocol_sha256": protocol.content_hash(),
        "decision": decision.value,
        "reason_codes": reasons,
        "tuning_authorized": False,
        "prospective_performance_claim_authorized": False,
    }
