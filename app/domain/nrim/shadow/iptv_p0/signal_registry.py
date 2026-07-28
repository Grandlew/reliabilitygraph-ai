from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES

from ..contracts import EXPECTED_UNIT, MetricName
from ..hashing import canonical_hash
from ..semantic_acceptance import (
    CollectorAcceptanceHarness,
    CollectorSemanticContract,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class AvailabilityClass(str, Enum):
    REQUIRED = "required"
    CONDITIONAL = "conditional"
    OPTIONAL = "optional"
    UNAVAILABLE = "unavailable"


class SupportConsequence(str, Enum):
    UNKNOWN = "UNKNOWN"
    ESCALATE = "ESCALATE"
    BLOCKED = "BLOCKED"


class SignalRequirement(StrictModel):
    signal_name: str = Field(min_length=3, max_length=256)
    availability: AvailabilityClass
    canonical_unit: str = Field(min_length=1, max_length=128)
    aggregation: str = Field(min_length=3, max_length=128)
    cadence_seconds: float = Field(gt=0.0)
    applicability_predicate: str | None = Field(
        default=None,
        min_length=3,
        max_length=500,
    )
    support_consequence: SupportConsequence
    source_ids: tuple[str, ...] = ()
    evidence: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def validate_classification(self) -> Self:
        metric = MetricName(self.signal_name)
        if self.canonical_unit != EXPECTED_UNIT[metric].value:
            raise ValueError("Signal unit differs from the frozen contract")
        if (
            self.availability is AvailabilityClass.CONDITIONAL
            and self.applicability_predicate is None
        ):
            raise ValueError("Conditional signals require a predicate")
        if (
            self.availability is not AvailabilityClass.CONDITIONAL
            and self.applicability_predicate is not None
        ):
            raise ValueError("Only conditional signals have predicates")
        if (
            self.availability is AvailabilityClass.REQUIRED
            and self.support_consequence is not SupportConsequence.BLOCKED
        ):
            raise ValueError("Missing required inputs must block")
        if (
            self.availability is AvailabilityClass.UNAVAILABLE
            and self.source_ids
        ):
            raise ValueError("Unavailable signals cannot name sources")
        return self


class SignalRegistry(StrictModel):
    registry_id: str = Field(min_length=3, max_length=128)
    registry_version: str = Field(pattern=r"^0\.8\.[0-9]+$")
    requirements: tuple[SignalRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_frozen_registry(self) -> Self:
        names = [item.signal_name for item in self.requirements]
        if len(names) != len(set(names)):
            raise ValueError("Each frozen signal must be classified once")
        if set(names) != set(SIGNAL_NAMES):
            missing = sorted(set(SIGNAL_NAMES) - set(names))
            extra = sorted(set(names) - set(SIGNAL_NAMES))
            raise ValueError(
                f"Registry differs from frozen inputs; missing={missing}, "
                f"extra={extra}"
            )
        return self

    def requirement(self, signal_name: str) -> SignalRequirement:
        for item in self.requirements:
            if item.signal_name == signal_name:
                return item
        raise KeyError(signal_name)

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def route_absence(
        self,
        signal_name: str,
        *,
        predicate_value: bool | None = None,
    ) -> SupportConsequence | None:
        item = self.requirement(signal_name)
        if item.availability is AvailabilityClass.OPTIONAL:
            return None
        if item.availability is AvailabilityClass.CONDITIONAL:
            if predicate_value is False:
                return None
            if predicate_value is None:
                return SupportConsequence.BLOCKED
        return item.support_consequence


class ProfiledCollectorContract(StrictModel):
    generic_contract: CollectorSemanticContract
    signal_registry: SignalRegistry
    approved_source_id: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        generic_names = {
            item.canonical_metric.value
            for item in self.generic_contract.metric_mappings
        }
        if generic_names != {
            item.signal_name for item in self.signal_registry.requirements
        }:
            raise ValueError("Collector and availability profile differ")
        return self

    def map_record(
        self,
        record: dict[str, Any],
        *,
        harness: CollectorAcceptanceHarness,
    ):
        if record.get("source_id") != self.approved_source_id:
            raise ValueError("Record source is not approved by the profile")
        source_record = dict(record)
        source_record.pop("source_id")
        return harness.map_record(source_record)

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
