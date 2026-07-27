from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.shadow.compatibility_v06 import (
    CandidateContext,
    wrap_v06_prediction,
)
from app.domain.nrim.shadow.contracts import (
    Applicability,
    ComponentType,
    DataQualityState,
    DecisionState,
    DeploymentProfile,
    DependencyType,
    DirectionSemantics,
    MetricName,
    MetricUnit,
    PredictionEnvelope,
    RankedCause,
    ServingSnapshot,
    SnapshotMode,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)


UTC = timezone.utc
DECISION_START = datetime(2026, 7, 25, 8, tzinfo=UTC)
DECISION_CUTOFF = DECISION_START + timedelta(hours=2)
DECISION_DEPLOYMENT = "deployment_" + "a" * 40
DECISION_SERVICE = "component_" + "b" * 40
DECISION_STORAGE = "component_" + "c" * 40


@pytest.fixture
def decision_snapshot() -> ServingSnapshot:
    components = (
        TopologyComponent(
            deployment_pseudonym=DECISION_DEPLOYMENT,
            component_pseudonym=DECISION_SERVICE,
            component_type=ComponentType.CATCHUP_SERVICE,
            valid_from_utc=DECISION_START - timedelta(days=1),
            topology_version="topology-v1",
            source_system="cmdb",
        ),
        TopologyComponent(
            deployment_pseudonym=DECISION_DEPLOYMENT,
            component_pseudonym=DECISION_STORAGE,
            component_type=ComponentType.CATCHUP_STORAGE,
            valid_from_utc=DECISION_START - timedelta(days=1),
            topology_version="topology-v1",
            source_system="cmdb",
        ),
    )
    edge = TopologyEdge(
        deployment_pseudonym=DECISION_DEPLOYMENT,
        source_component=DECISION_SERVICE,
        destination_component=DECISION_STORAGE,
        dependency_type=DependencyType.STORES_ON,
        direction_semantics=(
            DirectionSemantics.DESTINATION_DEPENDS_ON_SOURCE
        ),
        valid_from_utc=DECISION_START - timedelta(days=1),
        topology_version="topology-v1",
        source_system="cmdb",
    )
    profile = DeploymentProfile(
        deployment_pseudonym=DECISION_DEPLOYMENT,
        profile_version="profile-v1",
        valid_from_utc=DECISION_START - timedelta(days=1),
        software_version="vendor-1.0",
        room_count=100,
        floor_count=4,
        retention_days=7,
        base_occupancy_fraction=0.7,
        catchup_recording_channels=20,
        average_bitrate_mbps=4.0,
        shared_storage=True,
        redundant_middleware=False,
        collector_family="otel-v1",
    )
    observation = TelemetryObservation(
        deployment_pseudonym=DECISION_DEPLOYMENT,
        component_pseudonym=DECISION_SERVICE,
        component_type=ComponentType.CATCHUP_SERVICE,
        metric_name=MetricName.SERVICE_AVAILABILITY,
        metric_value=91.0,
        unit=MetricUnit.PERCENT,
        applicability=Applicability.OBSERVED,
        quality="high",
        event_time_utc=DECISION_CUTOFF - timedelta(minutes=10),
        ingestion_time_utc=DECISION_CUTOFF + timedelta(minutes=1),
        collector_id="collector-a",
        source_sequence_id="sequence-1",
        topology_version="topology-v1",
    )
    return ServingSnapshot(
        snapshot_id="snapshot-decision-v072",
        deployment_pseudonym=DECISION_DEPLOYMENT,
        observation_start_utc=DECISION_START,
        decision_cutoff_utc=DECISION_CUTOFF,
        as_of_ingestion_time_utc=DECISION_CUTOFF + timedelta(minutes=2),
        topology_version="topology-v1",
        profile=profile,
        components=components,
        edges=(edge,),
        observations=(observation,),
        data_quality_state=DataQualityState.ACCEPTED,
        expected_observation_count=1,
        available_observation_count=1,
    )


@pytest.fixture
def decision_prediction() -> PredictionEnvelope:
    return PredictionEnvelope(
        prediction_id="prediction_decision_v072",
        decision_cutoff_utc=DECISION_CUTOFF,
        deployment_pseudonym=DECISION_DEPLOYMENT,
        snapshot_mode=SnapshotMode.PROSPECTIVE,
        telemetry_snapshot_hash="a" * 64,
        topology_snapshot_hash="b" * 64,
        operational_profile_hash="c" * 64,
        feature_schema_hash="d" * 64,
        feature_vector_hash="e" * 64,
        model_bundle_hash="f" * 64,
        policy_hash="1" * 64,
        stage1_probability=0.99,
        healthy_residual=5.0,
        support_score=0.1,
        support_axes={"domain": 0.1},
        impact_score=0.9,
        impact_families=("service_availability_decline",),
        causal_consistency=True,
        fast_path_state=DecisionState.INCIDENT,
        slow_path_state=DecisionState.INCIDENT,
        dual_path_state=DecisionState.INCIDENT,
        slow_path_statistic=12.0,
        episode_state_before=DecisionState.HEALTHY,
        episode_state_after=DecisionState.INCIDENT,
        final_decision=DecisionState.INCIDENT,
        activation_path="both",
        stage2_top_k=(
            RankedCause(
                component_pseudonym=DECISION_SERVICE,
                score=0.9,
                evidence={
                    "local_anomaly": 1.0,
                    "topology_residual": -0.25,
                },
            ),
            RankedCause(
                component_pseudonym=DECISION_STORAGE,
                score=0.6,
                evidence={"criticality": 0.5},
            ),
        ),
        counterfactual_policy_states={
            "F": DecisionState.INCIDENT,
            "S": DecisionState.INCIDENT,
            "D": DecisionState.INCIDENT,
        },
        data_quality_state=DataQualityState.ACCEPTED,
        inference_latency_ms=10.0,
        infrastructure_version="0.7.0",
        created_at_utc=DECISION_CUTOFF + timedelta(minutes=3),
    )


@pytest.fixture
def decision_context(
    decision_snapshot: ServingSnapshot,
    decision_prediction: PredictionEnvelope,
) -> CandidateContext:
    return CandidateContext.from_v06(
        serving_snapshot=decision_snapshot,
        prediction=decision_prediction,
    )


@pytest.fixture
def decision_events(
    decision_snapshot: ServingSnapshot,
    decision_prediction: PredictionEnvelope,
    decision_context: CandidateContext,
):
    return wrap_v06_prediction(
        serving_snapshot=decision_snapshot,
        prediction=decision_prediction,
        candidate_context=decision_context,
    )
