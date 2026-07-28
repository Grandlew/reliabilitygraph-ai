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

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES

from ..hashing import canonical_hash
from .contracts import utc
from .feature_reconstruction import FeatureDryRunResult


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EvidenceCheck(StrictModel):
    check_id: str = Field(min_length=3, max_length=256)
    passed: bool
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)


class SignalMappingCheck(StrictModel):
    signal_name: str = Field(min_length=3, max_length=256)
    qualified: bool
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)


class SemanticMeasuredResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    mapping_checks: tuple[SignalMappingCheck, ...] = Field(min_length=1)
    positive_fixture_checks: tuple[EvidenceCheck, ...] = Field(min_length=1)
    mutation_checks: tuple[EvidenceCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_frozen_mapping(self) -> Self:
        names = [item.signal_name for item in self.mapping_checks]
        if len(names) != len(set(names)) or set(names) != set(SIGNAL_NAMES):
            raise ValueError(
                "Semantic result must measure every frozen signal exactly once"
            )
        return self

    @property
    def mapping_fraction(self) -> float:
        return sum(item.qualified for item in self.mapping_checks) / len(
            self.mapping_checks
        )

    @property
    def positive_fixture_fraction(self) -> float:
        return sum(item.passed for item in self.positive_fixture_checks) / len(
            self.positive_fixture_checks
        )

    @property
    def mutation_rejection_fraction(self) -> float:
        return sum(item.passed for item in self.mutation_checks) / len(
            self.mutation_checks
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class TopologyCutoffResult(StrictModel):
    cutoff_utc: datetime
    snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    deterministic: bool
    future_visible_capture_ids: tuple[str, ...] = ()

    @field_validator("cutoff_utc")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        return utc(value)


class TopologyMeasuredResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    cutoff_results: tuple[TopologyCutoffResult, ...] = Field(min_length=1)
    mutation_checks: tuple[EvidenceCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cutoffs(self) -> Self:
        cutoffs = [item.cutoff_utc for item in self.cutoff_results]
        if len(cutoffs) != len(set(cutoffs)):
            raise ValueError("Topology cutoff measurements must be unique")
        return self

    @property
    def reconstruction_fraction(self) -> float:
        return sum(item.deterministic for item in self.cutoff_results) / len(
            self.cutoff_results
        )

    @property
    def mutation_rejection_fraction(self) -> float:
        return sum(item.passed for item in self.mutation_checks) / len(
            self.mutation_checks
        )

    @property
    def future_leakage_count(self) -> int:
        return sum(
            len(item.future_visible_capture_ids)
            for item in self.cutoff_results
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class OutcomeMeasuredResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    incident_outcome_ids: tuple[str, ...]
    aligned_incident_outcome_ids: tuple[str, ...]
    adjudicated_root_cause_ids: tuple[str, ...]
    mapped_root_cause_ids: tuple[str, ...]
    inventory_complete: bool
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_subsets(self) -> Self:
        if not set(self.aligned_incident_outcome_ids).issubset(
            self.incident_outcome_ids
        ):
            raise ValueError("Aligned incidents must exist in the inventory")
        if not set(self.mapped_root_cause_ids).issubset(
            self.adjudicated_root_cause_ids
        ):
            raise ValueError("Mapped root causes must be adjudicated")
        for values in (
            self.incident_outcome_ids,
            self.aligned_incident_outcome_ids,
            self.adjudicated_root_cause_ids,
            self.mapped_root_cause_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("Outcome evidence identifiers must be unique")
        return self

    @property
    def incident_alignment_fraction(self) -> float:
        if not self.incident_outcome_ids:
            return 0.0
        return len(self.aligned_incident_outcome_ids) / len(
            self.incident_outcome_ids
        )

    @property
    def root_cause_mapping_fraction(self) -> float:
        if not self.adjudicated_root_cause_ids:
            return 0.0
        return len(self.mapped_root_cause_ids) / len(
            self.adjudicated_root_cause_ids
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class PrivacyMeasuredResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    scanned_artifact_sha256: tuple[str, ...] = Field(min_length=1)
    identifier_leakage_finding_ids: tuple[str, ...] = ()

    @field_validator("scanned_artifact_sha256")
    @classmethod
    def validate_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("Scanned artifact commitments must be unique")
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in values
        ):
            raise ValueError("Scanned artifact commitment is invalid")
        return values

    @property
    def identifier_leakage_count(self) -> int:
        return len(self.identifier_leakage_finding_ids)

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class TuningCategory(str, Enum):
    MODEL = "model"
    THRESHOLD = "threshold"
    FEATURE = "feature"
    SUPPORT_RULE = "support_rule"
    WATERMARK = "watermark"
    EPISODE_GROUPING = "episode_grouping"


class ProhibitedTuningEvent(StrictModel):
    event_id: str = Field(min_length=3, max_length=256)
    category: TuningCategory
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)


class TuningAuditResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    audited_event_sha256: tuple[str, ...] = Field(min_length=1)
    prohibited_events: tuple[ProhibitedTuningEvent, ...] = ()

    @property
    def prohibited_event_count(self) -> int:
        return len(self.prohibited_events)

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ReproducibilityMeasuredResult(StrictModel):
    result_id: str = Field(min_length=3, max_length=128)
    run_output_sha256: tuple[str, ...] = Field(min_length=2)
    tamper_checks: tuple[EvidenceCheck, ...] = Field(min_length=1)

    @property
    def determinism_fraction(self) -> float:
        first = self.run_output_sha256[0]
        return sum(
            value == first for value in self.run_output_sha256
        ) / len(self.run_output_sha256)

    @property
    def tamper_detection_fraction(self) -> float:
        return sum(item.passed for item in self.tamper_checks) / len(
            self.tamper_checks
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class QualificationMeasurements(StrictModel):
    semantic: SemanticMeasuredResult
    topology: TopologyMeasuredResult
    feature_reconstruction: FeatureDryRunResult
    outcomes: OutcomeMeasuredResult
    privacy: PrivacyMeasuredResult
    tuning_audit: TuningAuditResult
    reproducibility: ReproducibilityMeasuredResult

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
