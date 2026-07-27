from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import typed_hash


SCHEMA_VERSION = "0.7.2"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictDecisionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


class CandidateExclusionReason(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    INVALID_TOPOLOGY_VERSION = "invalid_topology_version"
    OUTSIDE_SUPPORT = "outside_support"
    DATA_UNAVAILABLE = "data_unavailable"
    POLICY_EXCLUDED = "policy_excluded"
    GENERATOR_LIMIT = "generator_limit"
    UNRESOLVED = "unresolved"


class EvidenceStatus(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNRESOLVED = "unresolved"


class ExpectedDirection(str, Enum):
    INCREASES = "increases"
    DECREASES = "decreases"
    PRESENT = "present"
    ABSENT = "absent"
    UNRESOLVED = "unresolved"


class EvidenceTimeRelation(str, Enum):
    PRECEDES = "precedes"
    COINCIDES = "coincides"
    FOLLOWS = "follows"
    WINDOW_AGGREGATE = "window_aggregate"
    UNRESOLVED = "unresolved"


class EvidenceApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class CandidateGeneratorIdentity(StrictDecisionModel):
    algorithm_version: str = Field(min_length=1, max_length=128)
    topology_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def digest(self) -> str:
        return typed_hash(
            type_name="candidate_generator_identity",
            schema_version=SCHEMA_VERSION,
            value=self,
        )


class CandidateExclusion(StrictDecisionModel):
    component_pseudonym: str = Field(min_length=12, max_length=128)
    reason: CandidateExclusionReason
    rationale_code: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")


class CandidateCoverage(StrictDecisionModel):
    eligible_true_cause_count: int = Field(ge=0)
    included_true_cause_count: int = Field(ge=0)
    denominator: int = Field(gt=0)
    adjudication_reference: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.eligible_true_cause_count != self.denominator:
            raise ValueError("Coverage denominator must equal eligible truth")
        if self.included_true_cause_count > self.eligible_true_cause_count:
            raise ValueError("Included truth cannot exceed eligible truth")
        return self


class CandidateSet(StrictDecisionModel):
    decision_cutoff_utc: datetime
    included_candidates: tuple[str, ...]
    exclusions: tuple[CandidateExclusion, ...] = ()
    generator: CandidateGeneratorIdentity
    coverage: CandidateCoverage | None = None
    schema_version: str = SCHEMA_VERSION

    @field_validator("decision_cutoff_utc")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_universe(self) -> Self:
        if tuple(sorted(self.included_candidates)) != self.included_candidates:
            raise ValueError("Candidate order must be canonical")
        if len(set(self.included_candidates)) != len(
            self.included_candidates
        ):
            raise ValueError("CandidateSet contains duplicate candidates")
        excluded = tuple(item.component_pseudonym for item in self.exclusions)
        if tuple(sorted(excluded)) != excluded:
            raise ValueError("Exclusion order must be canonical")
        if len(set(excluded)) != len(excluded):
            raise ValueError("CandidateSet contains duplicate exclusions")
        overlap = set(self.included_candidates).intersection(excluded)
        if overlap:
            raise ValueError("Included and excluded candidates overlap")
        return self

    @property
    def digest(self) -> str:
        return typed_hash(
            type_name="candidate_set",
            schema_version=self.schema_version,
            value=self,
        )


class EvidenceObligation(StrictDecisionModel):
    obligation_id: str = Field(pattern=r"^obligation_[0-9a-f]{16,64}$")
    candidate_pseudonym: str = Field(min_length=12, max_length=128)
    mechanism_code: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$"
    )
    expected_direction: ExpectedDirection
    time_relation: EvidenceTimeRelation
    applicability: EvidenceApplicability
    status: EvidenceStatus
    source_event_ids: tuple[str, ...] = ()
    rationale_code: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    observed_value: float | None = None
    standardized_value: float | None = None

    @field_validator("observed_value", "standardized_value")
    @classmethod
    def validate_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Evidence values must be finite")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (
            self.status
            in {EvidenceStatus.SUPPORTS, EvidenceStatus.CONTRADICTS}
            and not self.source_event_ids
        ):
            raise ValueError("Observed evidence requires immutable references")
        if (
            self.applicability is EvidenceApplicability.NOT_APPLICABLE
            and self.status is not EvidenceStatus.NOT_APPLICABLE
        ):
            raise ValueError("Inapplicable evidence must use its matching state")
        if (
            self.status is EvidenceStatus.NOT_APPLICABLE
            and self.applicability is not EvidenceApplicability.NOT_APPLICABLE
        ):
            raise ValueError("NOT_APPLICABLE requires inapplicability")
        return self


class AlternativeHypothesis(StrictDecisionModel):
    candidate_pseudonym: str = Field(min_length=12, max_length=128)
    alternative_candidate_pseudonym: str = Field(
        min_length=12,
        max_length=128,
    )
    discriminator_obligation_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct(self) -> Self:
        if self.candidate_pseudonym == self.alternative_candidate_pseudonym:
            raise ValueError("An alternative must identify another candidate")
        return self


class CausalEvidenceSet(StrictDecisionModel):
    ranked_candidates: tuple[str, ...]
    obligations: tuple[EvidenceObligation, ...]
    alternatives: tuple[AlternativeHypothesis, ...] = ()
    ambiguous: bool = False
    unresolved_reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$",
    )
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(set(self.ranked_candidates)) != len(self.ranked_candidates):
            raise ValueError("Ranked candidates must be unique")
        ranked = set(self.ranked_candidates)
        if any(
            item.candidate_pseudonym not in ranked
            for item in self.obligations
        ):
            raise ValueError("Evidence references an unranked candidate")
        obligation_ids = tuple(item.obligation_id for item in self.obligations)
        if len(set(obligation_ids)) != len(obligation_ids):
            raise ValueError("Evidence obligation identifiers must be unique")
        if self.ranked_candidates:
            represented = {
                item.candidate_pseudonym for item in self.obligations
            }
            if represented != ranked and self.unresolved_reason_code is None:
                raise ValueError(
                    "Every ranked candidate needs evidence or unresolved state"
                )
        elif self.obligations or self.alternatives:
            raise ValueError("Evidence cannot exist without ranked candidates")
        if self.ambiguous and len(self.ranked_candidates) > 1:
            if not self.alternatives:
                raise ValueError(
                    "Ambiguous rankings require alternative hypotheses"
                )
        valid_ids = set(obligation_ids)
        for alternative in self.alternatives:
            if (
                alternative.candidate_pseudonym not in ranked
                or alternative.alternative_candidate_pseudonym not in ranked
            ):
                raise ValueError("Alternative references an unranked candidate")
            if not set(
                alternative.discriminator_obligation_ids
            ).issubset(valid_ids):
                raise ValueError("Alternative references unknown obligations")
        return self

    def validate_event_references(
        self,
        available_event_ids: set[str],
    ) -> None:
        referenced = {
            event_id
            for obligation in self.obligations
            for event_id in obligation.source_event_ids
        }
        if not referenced.issubset(available_event_ids):
            raise ValueError("Evidence references outside the decision stream")

    @property
    def digest(self) -> str:
        return typed_hash(
            type_name="causal_evidence_set",
            schema_version=self.schema_version,
            value=self,
        )
