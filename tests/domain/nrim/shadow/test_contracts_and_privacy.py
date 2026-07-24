from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.contracts import (
    Applicability,
    ComponentType,
    DependencyType,
    DirectionSemantics,
    MetricName,
    MetricUnit,
    ObservationQuality,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from app.domain.nrim.shadow.privacy import (
    PrivacyViolation,
    Pseudonymizer,
    assert_inference_payload_is_label_free,
    assert_no_direct_identifiers,
)
from app.domain.nrim.shadow.replay import TopologyRegistry


UTC = timezone.utc
NOW = datetime(2026, 7, 24, 10, tzinfo=UTC)
DEPLOYMENT = "deployment_" + "a" * 40
SERVICE = "component_" + "b" * 40
STORAGE = "component_" + "c" * 40


def observation(**changes) -> TelemetryObservation:
    payload = {
        "deployment_pseudonym": DEPLOYMENT,
        "component_pseudonym": STORAGE,
        "component_type": ComponentType.CATCHUP_STORAGE,
        "metric_name": MetricName.DISK_UTILIZATION,
        "metric_value": 55.0,
        "unit": MetricUnit.PERCENT,
        "applicability": Applicability.OBSERVED,
        "quality": ObservationQuality.HIGH,
        "event_time_utc": NOW,
        "ingestion_time_utc": NOW + timedelta(seconds=2),
        "collector_id": "collector-a",
        "source_sequence_id": "sequence-1",
        "topology_version": "topology-v1",
    }
    payload.update(changes)
    return TelemetryObservation(**payload)


@pytest.mark.parametrize(
    ("applicability", "value"),
    [
        (Applicability.MISSING, None),
        (Applicability.NOT_APPLICABLE, None),
        (Applicability.OBSERVED, 50.0),
    ],
)
def test_applicability_semantics(
    applicability: Applicability,
    value: float | None,
) -> None:
    assert observation(
        applicability=applicability,
        metric_value=value,
    ).applicability is applicability


def test_missing_cannot_be_encoded_as_zero() -> None:
    with pytest.raises(ValidationError, match="require null"):
        observation(
            applicability=Applicability.MISSING,
            metric_value=0.0,
        )


def test_unknown_or_wrong_contract_values_fail_closed() -> None:
    payload = observation().model_dump(mode="json")
    payload["metric_name"] = "vendor.secret.metric"
    with pytest.raises(ValidationError):
        TelemetryObservation.model_validate(payload)

    payload = observation().model_dump(mode="json")
    payload["unit"] = "bytes"
    with pytest.raises(ValidationError):
        TelemetryObservation.model_validate(payload)

    payload = observation().model_dump(mode="json")
    payload["ticket_resolution"] = "disk replaced"
    with pytest.raises(ValidationError):
        TelemetryObservation.model_validate(payload)


def test_timezone_and_applicable_component_are_strict() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        observation(event_time_utc=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError, match="does not apply"):
        observation(
            component_type=ComponentType.MIDDLEWARE,
        )


def test_keyed_pseudonyms_are_deterministic_and_domain_separated() -> None:
    pseudonymizer = Pseudonymizer(b"k" * 32, key_id="pilot-key-1")
    deployment = pseudonymizer.pseudonymize(
        "Hotel Aurora",
        namespace="deployment",
    )
    assert deployment == pseudonymizer.pseudonymize(
        "Hotel Aurora",
        namespace="deployment",
    )
    assert deployment != pseudonymizer.pseudonymize(
        "Hotel Aurora",
        namespace="component",
    )
    assert "Aurora" not in deployment


def test_privacy_and_label_leakage_audits() -> None:
    assert_no_direct_identifiers(
        {"deployment_pseudonym": DEPLOYMENT}
    )
    with pytest.raises(PrivacyViolation, match="email"):
        assert_no_direct_identifiers({"notes": "alice@example.com"})
    with pytest.raises(PrivacyViolation, match="ticket_resolution"):
        assert_inference_payload_is_label_free(
            {"tensor": {"ticket_resolution": 1.0}}
        )


def test_topology_unknowns_and_overlapping_validity_fail() -> None:
    first = TopologyComponent(
        deployment_pseudonym=DEPLOYMENT,
        component_pseudonym=SERVICE,
        component_type=ComponentType.CATCHUP_SERVICE,
        valid_from_utc=NOW,
        valid_to_utc=NOW + timedelta(days=2),
        topology_version="topology-v1",
        source_system="cmdb",
    )
    overlap = first.model_copy(
        update={
            "valid_from_utc": NOW + timedelta(days=1),
            "valid_to_utc": NOW + timedelta(days=3),
            "topology_version": "topology-v2",
        }
    )
    with pytest.raises(ValueError, match="Overlapping component"):
        TopologyRegistry(
            components=(first, overlap),
            edges=(),
            profiles=(),
        )
    with pytest.raises(ValidationError):
        TopologyEdge(
            deployment_pseudonym=DEPLOYMENT,
            source_component=SERVICE,
            destination_component=STORAGE,
            dependency_type="unknown_dependency",
            direction_semantics=(
                DirectionSemantics.SOURCE_TO_DESTINATION
            ),
            valid_from_utc=NOW,
            topology_version="topology-v1",
            source_system="cmdb",
        )


def test_valid_topology_edge_contract() -> None:
    edge = TopologyEdge(
        deployment_pseudonym=DEPLOYMENT,
        source_component=SERVICE,
        destination_component=STORAGE,
        dependency_type=DependencyType.STORES_ON,
        direction_semantics=(
            DirectionSemantics.DESTINATION_DEPENDS_ON_SOURCE
        ),
        valid_from_utc=NOW,
        topology_version="topology-v1",
        source_system="cmdb",
    )
    assert edge.dependency_type is DependencyType.STORES_ON
