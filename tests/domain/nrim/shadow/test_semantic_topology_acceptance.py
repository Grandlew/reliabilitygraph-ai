from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.domain.nrim.shadow.contracts import (
    Applicability,
    ComponentType,
    DependencyType,
    DeploymentProfile,
    DirectionSemantics,
    MetricName,
    MetricUnit,
    ObservationQuality,
    TopologyComponent,
    TopologyEdge,
)
from app.domain.nrim.shadow.privacy import Pseudonymizer
from app.domain.nrim.shadow.replay import TopologyRegistry
from app.domain.nrim.shadow.semantic_acceptance import (
    AggregationSemantics,
    AggregationStatistic,
    CollectorAcceptanceHarness,
    CollectorSemanticContract,
    MetricSemanticMapping,
    ResetBehavior,
    SemanticFixture,
)
from app.domain.nrim.shadow.topology_acceptance import (
    EdgeSemanticRule,
    TopologyAcceptanceHarness,
    TopologySemanticContract,
)


UTC = timezone.utc
NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)
DEPLOYMENT = "deployment_" + "a" * 40
SERVICE = "component_" + "b" * 40
STORAGE = "component_" + "c" * 40


UNITS = {
    MetricName.DISK_UTILIZATION: MetricUnit.PERCENT,
    MetricName.DISK_IO_LATENCY: MetricUnit.MILLISECONDS,
    MetricName.DISK_IO_ERRORS: MetricUnit.ERRORS_PER_INTERVAL,
    MetricName.RECORDING_FAILURES: MetricUnit.FAILURES_PER_INTERVAL,
    MetricName.PROCESS_RESTART_COUNT: MetricUnit.RESTARTS_PER_INTERVAL,
    MetricName.ACTIVE_SESSION_COUNT: MetricUnit.SESSIONS,
    MetricName.SERVICE_AVAILABILITY: MetricUnit.PERCENT,
}


def aggregation() -> AggregationSemantics:
    return AggregationSemantics(
        statistic=AggregationStatistic.LAST,
        source_sampling_interval_seconds=60,
        target_window_seconds=3600,
        reset_behavior=ResetBehavior.GAUGE,
        numerator_semantics="Canonical observed metric value.",
        denominator_semantics="One eligible component observation.",
        boundary_rule="Event time is inside the left-open decision window.",
        late_data_rule="Late records create replay and never rewrite evidence.",
    )


def collector_contract() -> CollectorSemanticContract:
    return CollectorSemanticContract(
        contract_id="collector-contract-1",
        contract_version="1.0.0",
        deployment_pseudonym=DEPLOYMENT,
        collector_id="otel-collector-a",
        metric_field="metric",
        value_field="value",
        unit_field="unit",
        event_time_field="event_time",
        ingestion_time_field="ingestion_time",
        component_field="component",
        sequence_field="sequence",
        topology_version_field="topology_version",
        quality_field="quality",
        applicability_field="applicability",
        event_timestamp_provenance=(
            "Collector source timestamp synchronized to the registered clock."
        ),
        clock_domain="deployment-ntp-domain",
        timestamp_precision_ms=1.0,
        maximum_clock_skew_seconds=2.0,
        maximum_ingestion_delay_seconds=300.0,
        quality_mapping={
            "source-high": ObservationQuality.HIGH,
            "source-medium": ObservationQuality.MEDIUM,
            "source-low": ObservationQuality.LOW,
            "source-quarantined": ObservationQuality.QUARANTINED,
        },
        applicability_mapping={
            "source-observed": Applicability.OBSERVED,
            "source-missing": Applicability.MISSING,
            "source-na": Applicability.NOT_APPLICABLE,
        },
        metric_mappings=tuple(
            MetricSemanticMapping(
                source_metric_name=metric.value,
                canonical_metric=metric,
                expected_source_unit=UNITS[metric].value,
                canonical_unit=UNITS[metric],
                conversion_multiplier=1.0,
                aggregation=aggregation(),
                conversion_provenance=(
                    "Source and frozen canonical units are identical."
                ),
            )
            for metric in MetricName
        ),
        component_types={
            "raw-storage": ComponentType.CATCHUP_STORAGE,
            "raw-service": ComponentType.CATCHUP_SERVICE,
        },
        approved_mapping_evidence=(
            "Mapping reviewed against collector and frozen feature semantics."
        ),
    )


def fixture(metric: MetricName, index: int) -> SemanticFixture:
    storage = metric in {
        MetricName.DISK_UTILIZATION,
        MetricName.DISK_IO_LATENCY,
        MetricName.DISK_IO_ERRORS,
    }
    value = 10.0
    return SemanticFixture(
        fixture_id=f"fixture-{index}",
        raw_record={
            "metric": metric.value,
            "value": value,
            "unit": UNITS[metric].value,
            "event_time": NOW.isoformat(),
            "ingestion_time": (NOW + timedelta(seconds=1)).isoformat(),
            "component": "raw-storage" if storage else "raw-service",
            "sequence": f"sequence-{index}",
            "topology_version": "topology-v1",
            "quality": "source-low",
            "applicability": "source-observed",
        },
        expected_metric=metric,
        expected_value=value,
        expected_quality=ObservationQuality.LOW,
        expected_applicability=Applicability.OBSERVED,
    )


def test_collector_semantic_mapping_and_mutations_fail_closed() -> None:
    harness = CollectorAcceptanceHarness(
        contract=collector_contract(),
        pseudonymizer=Pseudonymizer(
            b"collector-semantic-secret-" + b"x" * 32,
            key_id="pilot-secret",
        ),
    )
    fixtures = tuple(
        fixture(metric, index)
        for index, metric in enumerate(MetricName)
    )
    report = harness.evaluate(fixtures)
    assert report.passed
    assert report.required_mapping_fraction == 1.0
    assert report.fixture_match_fraction == 1.0
    assert report.mutation_rejection_fraction == 1.0
    assert {
        item["mutation"] for item in report.mutation_results
    } == {
        "swapped_unit",
        "shifted_timestamp",
        "removed_applicability",
        "unknown_quality",
        "mutated_quality_meaning",
    }


def topology_registry() -> TopologyRegistry:
    components = (
        TopologyComponent(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=SERVICE,
            component_type=ComponentType.CATCHUP_SERVICE,
            valid_from_utc=NOW - timedelta(days=1),
            topology_version="topology-v1",
            source_system="inventory",
        ),
        TopologyComponent(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=STORAGE,
            component_type=ComponentType.CATCHUP_STORAGE,
            valid_from_utc=NOW - timedelta(days=1),
            topology_version="topology-v1",
            source_system="inventory",
        ),
    )
    edges = (
        TopologyEdge(
            deployment_pseudonym=DEPLOYMENT,
            source_component=SERVICE,
            destination_component=STORAGE,
            dependency_type=DependencyType.STORES_ON,
            direction_semantics=DirectionSemantics.SOURCE_TO_DESTINATION,
            valid_from_utc=NOW - timedelta(days=1),
            topology_version="topology-v1",
            source_system="inventory",
        ),
    )
    profiles = (
        DeploymentProfile(
            deployment_pseudonym=DEPLOYMENT,
            profile_version="profile-v1",
            valid_from_utc=NOW - timedelta(days=1),
            software_version="1.0.0",
            room_count=100,
            floor_count=10,
            retention_days=7,
            base_occupancy_fraction=0.7,
            catchup_recording_channels=100,
            average_bitrate_mbps=5.0,
            shared_storage=True,
            redundant_middleware=False,
            collector_family="otel",
        ),
    )
    return TopologyRegistry(
        components=components,
        edges=edges,
        profiles=profiles,
    )


def test_topology_lineage_is_deterministic_and_mutations_fail_closed() -> None:
    contract = TopologySemanticContract(
        contract_id="topology-contract-1",
        contract_version="1.0.0",
        deployment_pseudonym=DEPLOYMENT,
        inventory_source="approved-inventory",
        endpoint_role_provenance=(
            "Service endpoint is the writer and storage endpoint the target."
        ),
        validity_interval_provenance=(
            "Inventory changes carry effective-from and effective-to times."
        ),
        edge_rules=(
            EdgeSemanticRule(
                dependency_type=DependencyType.STORES_ON,
                direction_semantics=(
                    DirectionSemantics.SOURCE_TO_DESTINATION
                ),
                allowed_source_types=(ComponentType.CATCHUP_SERVICE,),
                allowed_destination_types=(
                    ComponentType.CATCHUP_STORAGE,
                ),
                rationale=(
                    "Catch-up service writes recordings to catch-up storage."
                ),
            ),
        ),
    )
    report = TopologyAcceptanceHarness(
        contract=contract,
        registry=topology_registry(),
    ).evaluate(cutoffs_utc=(NOW,))
    assert report.passed
    assert report.reconstruction_fraction == 1.0
    assert report.mutation_rejection_fraction == 1.0
    assert {
        item["mutation"]
        for item in report.results
        if "mutation" in item
    } == {
        "reversed_edge",
        "stale_topology",
        "changed_direction_semantics",
    }
