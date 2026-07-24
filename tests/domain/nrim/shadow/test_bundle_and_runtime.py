from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.baselines.dual_path_episode_gate import (
    DualPathConfig,
    HealthyResidualModel,
)
from app.domain.nrim.baselines.incident_detector import IncidentDetectorModel
from app.domain.nrim.baselines.learned_fusion import RootCauseFusionModel
from app.domain.nrim.baselines.shift_sentinels import (
    DomainDiscriminator,
    FeatureSupport,
    ShiftSentinelModel,
)
from app.domain.nrim.simulation.feature_schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SIGNAL_NAMES,
    build_feature_schema,
)
from app.domain.nrim.shadow.bundle import (
    BundleIntegrityError,
    FrozenInferenceBundle,
    create_signed_bundle,
    generate_signing_keypair,
    load_signed_bundle,
    serialize_models,
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
    ServingSnapshot,
    TelemetryObservation,
    TopologyComponent,
    TopologyEdge,
)
from app.domain.nrim.shadow.hashing import canonical_hash
from app.domain.nrim.shadow.runtime import ShadowInferenceOrchestrator
from app.domain.nrim.shadow.store import AppendOnlyEvidenceStore


UTC = timezone.utc
START = datetime(2026, 7, 24, 0, tzinfo=UTC)
CUTOFF = START + timedelta(hours=2)
DEPLOYMENT = "deployment_" + "a" * 40
SERVICE = "component_" + "b" * 40
STORAGE = "component_" + "c" * 40


def models(*, unsupported: bool = False):
    incident = IncidentDetectorModel(
        feature_names=(),
        means=(),
        scales=(),
        weights=(),
        bias=5.0,
        ood_distance_threshold=100.0,
    )
    residual = HealthyResidualModel(
        feature_names=(),
        means=(),
        scales=(),
        weights=(),
        bias=0.0,
        residual_center=0.0,
        residual_scale=1.0,
    )
    supports = (
        (
            FeatureSupport(
                name="workload__log_room_count",
                axis="workload",
                lower=0.0,
                upper=1.0,
                scale=1.0,
                hard=True,
            ),
        )
        if unsupported
        else ()
    )
    sentinel = ShiftSentinelModel(
        supports=supports,
        domain_discriminator=DomainDiscriminator(
            feature_names=(),
            means=(),
            scales=(),
            weights=(),
            bias=-10.0,
            threshold=0.80,
        ),
        support_threshold=1.0,
    )
    fusion = RootCauseFusionModel(
        feature_names=(),
        means=(),
        scales=(),
        weights=(),
        bias=0.0,
        use_topology=False,
    )
    return incident, residual, sentinel, fusion


def policy() -> DualPathConfig:
    return DualPathConfig(
        fast_probability_threshold=0.90,
        fast_residual_threshold=3.0,
        require_impact_confirmation=False,
        slow_reference_kappa=1.0,
        slow_threshold=100.0,
        suspect_probability_threshold=0.65,
    )


def frozen_bundle(*, unsupported: bool = False) -> FrozenInferenceBundle:
    incident, residual, sentinel, fusion = models(
        unsupported=unsupported
    )
    selected_policy = policy()
    schema = build_feature_schema()
    policy_payload = {
        "incident_threshold": 0.5,
        "dual_path_policy": asdict(selected_policy),
        "golden_numeric_tolerance": 1e-12,
    }
    return FrozenInferenceBundle(
        incident_model=incident,
        residual_model=residual,
        support_model=sentinel,
        fusion_model=fusion,
        policy=selected_policy,
        incident_threshold=0.5,
        feature_schema=schema,
        bundle_hash="a" * 64,
        policy_hash=canonical_hash(policy_payload),
        provenance={"test": True},
        golden_snapshots=(),
    )


def bundle_artifacts():
    incident, residual, sentinel, fusion = models()
    selected_policy = policy()
    schema = build_feature_schema()
    return {
        "models.json": serialize_models(
            incident_model=incident,
            residual_model=residual,
            support_model=sentinel,
            fusion_model=fusion,
        ),
        "policy.json": {
            "incident_threshold": 0.5,
            "dual_path_policy": asdict(selected_policy),
            "golden_numeric_tolerance": 1e-12,
        },
        "feature_schema.json": schema.model_dump(mode="json"),
        "categories.json": {
            "node_types": NODE_TYPES,
            "edge_types": EDGE_TYPES,
            "signal_names": SIGNAL_NAMES,
        },
        "golden_snapshots.json": [],
        "provenance.json": {"source": "unit-test"},
    }


def test_signed_bundle_verifies_and_refuses_source_mutation(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("FROZEN = True\n", encoding="utf-8")
    private, public = generate_signing_keypair()
    bundle_path = tmp_path / "model.nrimbundle"
    public_path = tmp_path / "public.pem"
    bundle_hash = create_signed_bundle(
        output_path=bundle_path,
        public_key_path=public_path,
        private_key=private,
        public_key_pem=public,
        artifacts=bundle_artifacts(),
        source_root=tmp_path,
        source_paths=(source,),
        provenance={"source": "unit-test"},
    )
    loaded = load_signed_bundle(
        bundle_path=bundle_path,
        public_key_path=public_path,
        source_root=tmp_path,
        expected_bundle_hash=bundle_hash,
    )
    assert loaded.bundle_hash == bundle_hash
    source.write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="source differs"):
        load_signed_bundle(
            bundle_path=bundle_path,
            public_key_path=public_path,
            source_root=tmp_path,
            expected_bundle_hash=bundle_hash,
        )


def test_signed_bundle_refuses_artifact_mutation(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("FROZEN = True\n", encoding="utf-8")
    private, public = generate_signing_keypair()
    bundle_path = tmp_path / "model.nrimbundle"
    public_path = tmp_path / "public.pem"
    create_signed_bundle(
        output_path=bundle_path,
        public_key_path=public_path,
        private_key=private,
        public_key_pem=public,
        artifacts=bundle_artifacts(),
        source_root=tmp_path,
        source_paths=(source,),
        provenance={"source": "unit-test"},
    )
    with zipfile.ZipFile(bundle_path, "r") as archive:
        entries = {
            name: archive.read(name)
            for name in archive.namelist()
        }
    policy_payload = json.loads(entries["policy.json"])
    policy_payload["incident_threshold"] = 0.99
    entries["policy.json"] = json.dumps(policy_payload).encode("utf-8")
    tampered = tmp_path / "tampered.nrimbundle"
    with zipfile.ZipFile(tampered, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    with pytest.raises(BundleIntegrityError, match="artifact hash"):
        load_signed_bundle(
            bundle_path=tampered,
            public_key_path=public_path,
            source_root=tmp_path,
        )


def snapshot(
    *,
    quality: DataQualityState = DataQualityState.ACCEPTED,
) -> ServingSnapshot:
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
    edge = TopologyEdge(
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
    )
    profile = DeploymentProfile(
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
    )
    specs = (
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_UTILIZATION, MetricUnit.PERCENT, 55),
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_IO_LATENCY, MetricUnit.MILLISECONDS, 10),
        (STORAGE, ComponentType.CATCHUP_STORAGE, MetricName.DISK_IO_ERRORS, MetricUnit.ERRORS_PER_INTERVAL, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.RECORDING_FAILURES, MetricUnit.FAILURES_PER_INTERVAL, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.PROCESS_RESTART_COUNT, MetricUnit.RESTARTS_PER_INTERVAL, 0),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.ACTIVE_SESSION_COUNT, MetricUnit.SESSIONS, 40),
        (SERVICE, ComponentType.CATCHUP_SERVICE, MetricName.SERVICE_AVAILABILITY, MetricUnit.PERCENT, 99),
    )
    observations = tuple(
        TelemetryObservation(
            deployment_pseudonym=DEPLOYMENT,
            component_pseudonym=component,
            component_type=component_type,
            metric_name=metric,
            metric_value=float(value),
            unit=unit,
            applicability=Applicability.OBSERVED,
            quality="high",
            event_time_utc=CUTOFF - timedelta(minutes=10),
            ingestion_time_utc=CUTOFF + timedelta(minutes=1),
            collector_id="collector-a",
            source_sequence_id=f"sequence-{index}",
            topology_version="topology-v1",
        )
        for index, (
            component,
            component_type,
            metric,
            unit,
            value,
        ) in enumerate(specs)
    )
    return ServingSnapshot(
        snapshot_id="snapshot-runtime",
        deployment_pseudonym=DEPLOYMENT,
        observation_start_utc=START,
        decision_cutoff_utc=CUTOFF,
        as_of_ingestion_time_utc=CUTOFF + timedelta(minutes=5),
        topology_version="topology-v1",
        profile=profile,
        components=components,
        edges=(edge,),
        observations=observations if quality is not DataQualityState.BLOCKED else (),
        data_quality_state=quality,
        data_quality_warnings=(
            ("collector outage",)
            if quality is DataQualityState.BLOCKED
            else ()
        ),
        expected_observation_count=7,
        available_observation_count=(
            0 if quality is DataQualityState.BLOCKED else 7
        ),
    )


def test_runtime_runs_f_s_d_and_is_restart_idempotent(tmp_path) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    orchestrator = ShadowInferenceOrchestrator(
        bundle=frozen_bundle(),
        store=store,
    )
    envelope = orchestrator.process(snapshot())
    assert envelope.final_decision is DecisionState.INCIDENT
    assert set(envelope.counterfactual_policy_states) == {"F", "S", "D"}
    assert envelope.stage2_top_k
    assert store.verify_prediction_chain()

    restarted = ShadowInferenceOrchestrator(
        bundle=frozen_bundle(),
        store=store,
    )
    assert restarted.process(snapshot()) == envelope
    assert len(store.list_predictions()) == 1


def test_unsupported_never_enters_stage2(tmp_path) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    envelope = ShadowInferenceOrchestrator(
        bundle=frozen_bundle(unsupported=True),
        store=store,
    ).process(snapshot())
    assert envelope.final_decision is DecisionState.ESCALATE
    assert envelope.stage2_top_k == ()


def test_blocked_data_emits_quality_escalation_without_model_outputs(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    envelope = ShadowInferenceOrchestrator(
        bundle=frozen_bundle(),
        store=store,
    ).process(snapshot(quality=DataQualityState.BLOCKED))
    assert (
        envelope.final_decision
        is DecisionState.DATA_QUALITY_ESCALATION
    )
    assert envelope.stage1_probability is None
    assert envelope.stage2_top_k == ()
