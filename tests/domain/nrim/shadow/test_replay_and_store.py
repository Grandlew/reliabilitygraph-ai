from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

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
    ServingSnapshot,
    SnapshotMode,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from app.domain.nrim.shadow.ingestion import TelemetryMirror, TopologyMirror
from app.domain.nrim.shadow.replay import (
    EventTimeReplayEngine,
    ServingFeatureBuilder,
    TopologyRegistry,
    WatermarkPolicy,
    assert_online_replay_parity,
)
from app.domain.nrim.shadow.store import AppendOnlyEvidenceStore


UTC = timezone.utc
START = datetime(2026, 7, 24, 0, tzinfo=UTC)
CUTOFF = START + timedelta(hours=2)
DEPLOYMENT = "deployment_" + "a" * 40
SERVICE = "component_" + "b" * 40
STORAGE = "component_" + "c" * 40


UNIT = {
    MetricName.DISK_UTILIZATION: MetricUnit.PERCENT,
    MetricName.DISK_IO_LATENCY: MetricUnit.MILLISECONDS,
    MetricName.DISK_IO_ERRORS: MetricUnit.ERRORS_PER_INTERVAL,
    MetricName.RECORDING_FAILURES: MetricUnit.FAILURES_PER_INTERVAL,
    MetricName.PROCESS_RESTART_COUNT: MetricUnit.RESTARTS_PER_INTERVAL,
    MetricName.ACTIVE_SESSION_COUNT: MetricUnit.SESSIONS,
    MetricName.SERVICE_AVAILABILITY: MetricUnit.PERCENT,
}


def registry() -> TopologyRegistry:
    components = (
        TopologyComponent(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=SERVICE,
            component_type=ComponentType.CATCHUP_SERVICE,
            valid_from_utc=START - timedelta(days=1),
            topology_version="topology-v1",
            source_system="cmdb",
        ),
        TopologyComponent(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=STORAGE,
            component_type=ComponentType.CATCHUP_STORAGE,
            valid_from_utc=START - timedelta(days=1),
            topology_version="topology-v1",
            source_system="cmdb",
        ),
    )
    edges = (
        TopologyEdge(
            deployment_pseudonym=DEPLOYMENT,
            source_component=SERVICE,
            destination_component=STORAGE,
            dependency_type=DependencyType.STORES_ON,
            direction_semantics=(
                DirectionSemantics.DESTINATION_DEPENDS_ON_SOURCE
            ),
            valid_from_utc=START - timedelta(days=1),
            topology_version="topology-v1",
            source_system="cmdb",
        ),
    )
    profiles = (
        DeploymentProfile(
            deployment_pseudonym=DEPLOYMENT,
            profile_version="profile-v1",
            valid_from_utc=START - timedelta(days=1),
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
        ),
    )
    return TopologyRegistry(
        components=components,
        edges=edges,
        profiles=profiles,
    )


def observations(*, one_late: bool = False):
    rows = (
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_UTILIZATION, 55),
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_IO_LATENCY, 10),
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_IO_ERRORS, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.RECORDING_FAILURES, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.PROCESS_RESTART_COUNT, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.ACTIVE_SESSION_COUNT, 40),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.SERVICE_AVAILABILITY, 99),
    )
    return [
        TelemetryObservation(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=component,
            component_type=component_type,
            metric_name=metric,
            metric_value=float(value),
            unit=UNIT[metric],
            applicability=Applicability.OBSERVED,
            quality="high",
            event_time_utc=CUTOFF - timedelta(minutes=10),
            ingestion_time_utc=(
                CUTOFF + timedelta(minutes=10)
                if one_late and index == 0
                else CUTOFF + timedelta(minutes=1)
            ),
            collector_id="collector-a",
            source_sequence_id=f"sequence-{index}",
            topology_version="topology-v1",
        )
        for index, (component, component_type, metric, value) in enumerate(rows)
    ]


def seeded_store(tmp_path, *, one_late: bool = False):
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    for item in observations(one_late=one_late):
        store.append_telemetry(item)
    return store


def test_late_data_creates_replay_without_changing_original(tmp_path) -> None:
    store = seeded_store(tmp_path, one_late=True)
    engine = EventTimeReplayEngine(
        store=store,
        topology_registry=registry(),
        watermark_policy=WatermarkPolicy(
            default_allowed_lateness=timedelta(minutes=5),
            minimum_available_fraction=0.80,
        ),
    )
    prospective = engine.build_snapshot(
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
    )
    assert prospective.mode is SnapshotMode.PROSPECTIVE
    assert prospective.available_observation_count == 6
    assert prospective.data_quality_state is DataQualityState.DEGRADED

    replay = engine.build_snapshot(
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
        as_of_ingestion_time_utc=CUTOFF + timedelta(minutes=20),
        mode=SnapshotMode.LATE_DATA_REPLAY,
        supersedes_prediction_id="prediction-original",
    )
    assert replay.available_observation_count == 7
    assert replay.content_hash() != prospective.content_hash()
    assert prospective.available_observation_count == 6


def test_feature_builder_is_identical_for_online_and_replay(tmp_path) -> None:
    store = seeded_store(tmp_path)
    engine = EventTimeReplayEngine(
        store=store,
        topology_registry=registry(),
        watermark_policy=WatermarkPolicy(),
    )
    snapshot = engine.build_snapshot(
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
    )
    builder = ServingFeatureBuilder()
    online = builder.build_sequence([snapshot])[0]
    replay = builder.build_sequence([snapshot])[0]
    assert_online_replay_parity(
        online_window=online,
        replay_window=replay,
    )
    assert "targets" not in online
    assert all(
        "deployment" not in name
        for name in online["node_feature_names"]
    )


def test_blocking_missingness_suppresses_inference_snapshot(tmp_path) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    store.append_telemetry(observations()[0])
    engine = EventTimeReplayEngine(
        store=store,
        topology_registry=registry(),
        watermark_policy=WatermarkPolicy(
            minimum_available_fraction=0.80
        ),
    )
    snapshot = engine.build_snapshot(
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
    )
    assert snapshot.data_quality_state is DataQualityState.BLOCKED
    with pytest.raises(ValueError, match="Blocked"):
        ServingFeatureBuilder().build_sequence([snapshot])


def test_mirror_quarantines_unknowns_and_conflicting_duplicates(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    mirror = TelemetryMirror(store)
    payload = observations()[0].model_dump(mode="json")
    assert mirror.ingest_telemetry(payload).accepted
    assert mirror.ingest_telemetry(payload).accepted
    conflicting = dict(payload)
    conflicting["metric_value"] = 99.0
    result = mirror.ingest_telemetry(conflicting)
    assert not result.accepted
    assert result.quarantine_id is not None

    invalid = dict(payload)
    invalid["unit"] = "bytes"
    invalid["source_sequence_id"] = "invalid-unit"
    result = mirror.ingest_telemetry(invalid)
    assert not result.accepted
    assert mirror.source_write_capability is False


def test_topology_mirror_is_append_only_and_unknown_types_quarantine(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    mirror = TopologyMirror(store)
    source = registry()
    for component in source.components:
        assert mirror.ingest_component(
            component.model_dump(mode="json")
        ).accepted
    for edge in source.edges:
        assert mirror.ingest_edge(edge.model_dump(mode="json")).accepted
    for profile in source.profiles:
        assert mirror.ingest_profile(
            profile.model_dump(mode="json")
        ).accepted
    rebuilt = TopologyRegistry.from_store(
        store,
        deployment_pseudonym=DEPLOYMENT,
    )
    assert len(rebuilt.components) == 2
    assert mirror.source_write_capability is False

    bad = source.edges[0].model_dump(mode="json")
    bad["dependency_type"] = "mystery"
    result = mirror.ingest_edge(bad)
    assert not result.accepted
    assert result.quarantine_id is not None


def _blocked_envelope(snapshot: ServingSnapshot) -> PredictionEnvelope:
    return PredictionEnvelope(
        prediction_id="prediction-blocked",
        decision_cutoff_utc=snapshot.decision_cutoff_utc,
        deployment_pseudonym=snapshot.deployment_pseudonym,
        snapshot_mode=snapshot.mode,
        telemetry_snapshot_hash=snapshot.content_hash(),
        topology_snapshot_hash=snapshot.topology_hash(),
        operational_profile_hash=snapshot.profile_hash(),
        feature_schema_hash="a" * 64,
        model_bundle_hash="b" * 64,
        policy_hash="c" * 64,
        episode_state_before=DecisionState.HEALTHY,
        episode_state_after=DecisionState.DATA_QUALITY_ESCALATION,
        final_decision=DecisionState.DATA_QUALITY_ESCALATION,
        activation_path="data_quality",
        data_quality_state=DataQualityState.BLOCKED,
        data_quality_warnings=("missing required telemetry",),
        inference_latency_ms=1.0,
        infrastructure_version="0.7.0",
        created_at_utc=CUTOFF + timedelta(minutes=5),
    )


def test_evidence_is_append_only_hash_chained_and_referential(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    engine = EventTimeReplayEngine(
        store=store,
        topology_registry=registry(),
        watermark_policy=WatermarkPolicy(),
    )
    snapshot = engine.build_snapshot(
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
    )
    snapshot_hash, _ = store.append_snapshot(snapshot)
    envelope = _blocked_envelope(snapshot)
    store.append_prediction(
        envelope=envelope,
        snapshot_hash=snapshot_hash,
    )
    assert store.verify_prediction_chain()
    assert store.get_prediction(envelope.prediction_id) == envelope

    connection = sqlite3.connect(store.path)
    try:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="append-only",
        ):
            connection.execute(
                "UPDATE predictions SET payload_hash = ?",
                ("0" * 64,),
            )
    finally:
        connection.close()
