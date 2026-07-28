from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.source_inventory import (
    Capability,
    ClockQuality,
    OutcomeSource,
    ReadOnlyCapabilityAttestation,
    SourceIdentity,
    SourceInventory,
)


def test_source_identity_is_deterministic(source):
    value = source.model_dump(mode="json", exclude={"source_id"})
    second = SourceIdentity.create(**value)
    assert source.source_id == second.source_id


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("source_kind", "different_kind"),
        ("lineage_uri", "approved://inventory/source-beta"),
        ("schema_version", "2.0"),
        ("event_time_field", "source_event_time"),
        ("observation_time_field", "source_observation_time"),
        ("ingestion_time_field", "source_ingestion_time"),
        ("clock_id", "clock-beta"),
        ("owner_role", "different_owner"),
    ),
)
def test_source_identity_changes_with_approved_metadata(
    source,
    field,
    replacement,
):
    value = source.model_dump(mode="json", exclude={"source_id"})
    value[field] = replacement
    changed = SourceIdentity.create(**value)
    assert changed.source_id != source.source_id


@pytest.mark.parametrize(
    "field,value",
    (
        ("lineage_uri", "mailto:operator@example.com"),
        ("owner_role", "admin@example.com"),
    ),
)
def test_source_identity_rejects_direct_identifiers(source, field, value):
    attributes = source.model_dump(mode="json", exclude={"source_id"})
    attributes[field] = value
    with pytest.raises((ValidationError, ValueError)):
        SourceIdentity.create(**attributes)


def test_inventory_hash_is_deterministic(inventory):
    assert inventory.content_hash() == inventory.content_hash()


def test_inventory_rejects_duplicate_sources(inventory, source):
    with pytest.raises(ValidationError):
        SourceInventory(
            inventory_id=inventory.inventory_id,
            deployment_pseudonym=inventory.deployment_pseudonym,
            sources=(source, source),
        )


def test_outcome_inventory_rejects_unknown_source(inventory):
    with pytest.raises(ValidationError):
        SourceInventory(
            inventory_id=inventory.inventory_id,
            deployment_pseudonym=inventory.deployment_pseudonym,
            sources=inventory.sources,
            outcome_sources=(
                OutcomeSource(
                    source_id="src_" + "f" * 32,
                    label_kind="fault_presence",
                    alignment_key_semantics=(
                        "Align by registered incident pseudonym and time."
                    ),
                    available_record_count=1,
                    aligned_record_count=1,
                    adjudicated_record_count=1,
                ),
            ),
        )


@pytest.mark.parametrize(
    "available,aligned,adjudicated",
    ((0, 1, 0), (1, 1, 2)),
)
def test_outcome_inventory_rejects_impossible_counts(
    source,
    available,
    aligned,
    adjudicated,
):
    with pytest.raises(ValidationError):
        OutcomeSource(
            source_id=source.source_id,
            label_kind="fault_presence",
            alignment_key_semantics=(
                "Align by registered incident pseudonym and time."
            ),
            available_record_count=available,
            aligned_record_count=aligned,
            adjudicated_record_count=adjudicated,
        )


@pytest.mark.parametrize(
    "capability",
    (
        Capability.NETWORK_QUERY,
        Capability.MUTATION,
        Capability.CREDENTIAL_USE,
        Capability.TICKET_CREATE,
        Capability.ALARM_SUPPRESS,
    ),
)
def test_capability_attestation_rejects_every_active_authority(capability):
    with pytest.raises(ValidationError):
        ReadOnlyCapabilityAttestation(
            adapter_id="unsafe-adapter",
            declared_capabilities=(
                Capability.OFFLINE_FILE_READ,
                capability,
            ),
        )


@pytest.mark.parametrize(
    "field",
    ("credential_fields_present", "active_endpoint_configured"),
)
def test_capability_attestation_rejects_credentials_and_endpoints(field):
    values = {
        "adapter_id": "unsafe-adapter",
        "declared_capabilities": (Capability.OFFLINE_FILE_READ,),
        field: True,
    }
    with pytest.raises(ValidationError):
        ReadOnlyCapabilityAttestation(**values)


def test_read_only_attestation_passes(capability):
    assert capability.passed
    assert capability.declared_capabilities == (
        Capability.OFFLINE_FILE_READ,
    )


def test_read_only_attestation_requires_every_forbidden_probe():
    with pytest.raises(ValidationError, match="Every forbidden"):
        ReadOnlyCapabilityAttestation(
            adapter_id="incomplete-attestation",
            declared_capabilities=(Capability.OFFLINE_FILE_READ,),
            forbidden_probe_results={Capability.MUTATION: False},
        )


def test_successful_forbidden_probe_is_rejected():
    with pytest.raises(ValidationError):
        ReadOnlyCapabilityAttestation(
            adapter_id="unsafe-adapter",
            declared_capabilities=(Capability.OFFLINE_FILE_READ,),
            forbidden_probe_results={Capability.MUTATION: True},
        )
