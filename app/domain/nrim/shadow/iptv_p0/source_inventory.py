from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..hashing import canonical_hash
from ..privacy import assert_no_direct_identifiers
from .contracts import LabelKind


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class SourceClassification(str, Enum):
    SYNTHETIC = "synthetic"
    PSEUDONYMIZED_OPERATIONAL = "pseudonymized_operational"
    AGGREGATED_OPERATIONAL = "aggregated_operational"


class ClockQuality(str, Enum):
    VERIFIED_SYNCHRONIZED = "verified_synchronized"
    BOUNDED_SKEW = "bounded_skew"
    UNKNOWN = "unknown"
    UNSYNCHRONIZED = "unsynchronized"


class RetentionMetadata(StrictModel):
    retention_days: int = Field(gt=0, le=3650)
    residency_region: str = Field(min_length=2, max_length=64)
    deletion_owner_role: str = Field(min_length=3, max_length=128)
    deletion_method: str = Field(min_length=10, max_length=500)


class SourceIdentity(StrictModel):
    source_id: str = Field(pattern=r"^src_[0-9a-f]{32}$")
    source_kind: str = Field(min_length=3, max_length=128)
    lineage_uri: str = Field(min_length=3, max_length=500)
    schema_version: str = Field(min_length=1, max_length=64)
    event_time_field: str = Field(min_length=1, max_length=128)
    observation_time_field: str = Field(min_length=1, max_length=128)
    ingestion_time_field: str = Field(min_length=1, max_length=128)
    clock_id: str = Field(min_length=3, max_length=128)
    clock_quality: ClockQuality
    owner_role: str = Field(min_length=3, max_length=128)
    classification: SourceClassification
    retention: RetentionMetadata

    @classmethod
    def create(
        cls,
        *,
        source_kind: str,
        lineage_uri: str,
        schema_version: str,
        event_time_field: str,
        observation_time_field: str,
        ingestion_time_field: str,
        clock_id: str,
        clock_quality: ClockQuality | str,
        owner_role: str,
        classification: SourceClassification | str,
        retention: RetentionMetadata | dict[str, object],
    ) -> SourceIdentity:
        clock = ClockQuality(clock_quality)
        source_classification = SourceClassification(classification)
        retention_model = RetentionMetadata.model_validate(retention)
        identity = {
            "source_kind": source_kind,
            "lineage_uri": lineage_uri,
            "schema_version": schema_version,
            "event_time_field": event_time_field,
            "observation_time_field": observation_time_field,
            "ingestion_time_field": ingestion_time_field,
            "clock_id": clock_id,
            "clock_quality": clock.value,
            "owner_role": owner_role,
            "classification": source_classification.value,
            "retention": retention_model.model_dump(mode="json"),
        }
        assert_no_direct_identifiers(identity)
        return cls(source_id="src_" + canonical_hash(identity)[:32], **identity)


class OutcomeSource(StrictModel):
    source_id: str = Field(pattern=r"^src_[0-9a-f]{32}$")
    label_kind: LabelKind
    alignment_key_semantics: str = Field(min_length=10, max_length=500)
    adjudication_required: bool = True
    available_record_count: int = Field(ge=0)
    aligned_record_count: int = Field(ge=0)
    adjudicated_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.aligned_record_count > self.available_record_count:
            raise ValueError("Aligned outcomes exceed available outcomes")
        if self.adjudicated_record_count > self.aligned_record_count:
            raise ValueError("Adjudicated outcomes exceed aligned outcomes")
        return self


class SourceInventory(StrictModel):
    inventory_id: str = Field(min_length=3, max_length=128)
    deployment_pseudonym: str = Field(
        pattern=r"^dep_[0-9a-f]{16,64}$"
    )
    sources: tuple[SourceIdentity, ...] = Field(min_length=1)
    outcome_sources: tuple[OutcomeSource, ...] = ()

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        ids = [item.source_id for item in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Source inventory identities must be unique")
        if any(
            item.source_id not in set(ids) for item in self.outcome_sources
        ):
            raise ValueError("Outcome inventory references unknown source")
        assert_no_direct_identifiers(self.model_dump(mode="json"))
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class Capability(str, Enum):
    OFFLINE_FILE_READ = "offline_file_read"
    NETWORK_QUERY = "network_query"
    MUTATION = "mutation"
    CREDENTIAL_USE = "credential_use"
    TICKET_CREATE = "ticket_create"
    ALARM_SUPPRESS = "alarm_suppress"


FORBIDDEN_CAPABILITIES = frozenset(Capability) - {
    Capability.OFFLINE_FILE_READ
}


class ReadOnlyCapabilityAttestation(StrictModel):
    adapter_id: str = Field(min_length=3, max_length=128)
    declared_capabilities: tuple[Capability, ...] = Field(min_length=1)
    credential_fields_present: bool = False
    active_endpoint_configured: bool = False
    forbidden_probe_results: dict[Capability, bool] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def reject_active_authority(self) -> Self:
        declared = set(self.declared_capabilities)
        if declared != {Capability.OFFLINE_FILE_READ}:
            raise ValueError("Adapter capability must be offline read only")
        if self.credential_fields_present or self.active_endpoint_configured:
            raise ValueError("Credentials and active endpoints are forbidden")
        if any(self.forbidden_probe_results.values()):
            raise ValueError("A forbidden capability probe succeeded")
        if set(self.forbidden_probe_results) != FORBIDDEN_CAPABILITIES:
            raise ValueError("Every forbidden capability must be probed")
        return self

    @property
    def passed(self) -> bool:
        return True
