from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial

import pytest

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES
from app.domain.nrim.shadow.contracts import EXPECTED_UNIT, MetricName
from app.domain.nrim.shadow.hashing import canonical_hash
from app.domain.nrim.shadow.iptv_p0.batch_adapter import BatchInputRecord
from app.domain.nrim.shadow.iptv_p0.clock_watermark import (
    ClockWatermarkPolicy,
)
from app.domain.nrim.shadow.iptv_p0.contracts import (
    CollectionMode,
    DeploymentEnvironment,
    DeploymentPack,
    DomainPack,
    IptvP0Charter,
    LabelKind,
    OperatorTask,
    OwnerRoles,
    PrivacyRetention,
    SignatureMetadata,
    SignatureState,
    create_ed25519_signature,
)
from app.domain.nrim.shadow.iptv_p0.qualification import (
    QualificationRequest,
)
from app.domain.nrim.shadow.iptv_p0.qualification_measurements import (
    EvidenceCheck,
    OutcomeMeasuredResult,
    PrivacyMeasuredResult,
    QualificationMeasurements,
    ReproducibilityMeasuredResult,
    SemanticMeasuredResult,
    SignalMappingCheck,
    TopologyCutoffResult,
    TopologyMeasuredResult,
    TuningAuditResult,
)
from app.domain.nrim.shadow.iptv_p0.feature_reconstruction import (
    reconstruct_frozen_features,
)
from app.domain.nrim.shadow.iptv_p0.replay_protocol import (
    CohortDefinition,
    HistoricalReplayProtocol,
    LabelDefinition,
    ReplayMetric,
)
from app.domain.nrim.shadow.iptv_p0.signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SignalRequirement,
    SupportConsequence,
)
from app.domain.nrim.shadow.iptv_p0.source_inventory import (
    ClockQuality,
    ReadOnlyCapabilityAttestation,
    RetentionMetadata,
    SourceClassification,
    SourceIdentity,
    SourceInventory,
)
from app.domain.nrim.shadow.iptv_p0.topology import (
    CaptureMode,
    IptvEdge,
    IptvEdgeType,
    IptvNode,
    IptvNodeType,
    TopologyCapture,
    TopologyHistory,
)
from app.domain.nrim.shadow.iptv_p0.trusted_signers import (
    TrustedSignerRegistry,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
DEPLOYMENT = "dep_" + "a" * 24


@pytest.fixture
def synthetic_signature() -> SignatureMetadata:
    return SignatureMetadata(
        state=SignatureState.NOT_SIGNED_SYNTHETIC,
        algorithm="none",
    )


@pytest.fixture
def verified_signature() -> SignatureMetadata:
    return create_ed25519_signature(
        payload_sha256="f" * 64,
        private_key_bytes=bytes(range(1, 33)),
        key_id="operator-key-01",
    )


@pytest.fixture
def charter() -> IptvP0Charter:
    return IptvP0Charter(
        charter_id="iptv-p0-charter",
        included_service_path=(
            "source_headend",
            "core_aggregation",
            "multicast_control",
            "transport",
            "access_handoff",
        ),
        explicit_exclusions=(
            "ott_abr",
            "drm",
            "model_training",
            "threshold_tuning",
            "gnn",
            "autonomous_remediation",
            "live_polling",
            "write_credentials",
        ),
        protected_operator_tasks=tuple(OperatorTask),
        label_kinds=tuple(LabelKind),
    )


@pytest.fixture
def domain_pack(
    charter: IptvP0Charter,
    synthetic_signature: SignatureMetadata,
) -> DomainPack:
    return DomainPack(
        pack_id="iptv-p0-domain",
        pack_version="0.8.0",
        issued_at_utc=BASE_TIME,
        charter=charter,
        metric_registry_sha256="1" * 64,
        topology_ontology_sha256="2" * 64,
        outcome_semantics_sha256="3" * 64,
        applicability_profile_sha256="4" * 64,
        support_policy_sha256="5" * 64,
        signature=synthetic_signature,
    )


@pytest.fixture
def source() -> SourceIdentity:
    return SourceIdentity.create(
        source_kind="offline_metric_export",
        lineage_uri="approved://inventory/source-alpha",
        schema_version="1.0",
        event_time_field="event_time_utc",
        observation_time_field="observation_time_utc",
        ingestion_time_field="ingestion_time_utc",
        clock_id="clock-alpha",
        clock_quality=ClockQuality.VERIFIED_SYNCHRONIZED,
        owner_role="telemetry_owner",
        classification=SourceClassification.SYNTHETIC,
        retention=RetentionMetadata(
            retention_days=30,
            residency_region="test-region",
            deletion_owner_role="storage_owner",
            deletion_method="Delete the isolated synthetic export.",
        ),
    )


@pytest.fixture
def inventory(source: SourceIdentity) -> SourceInventory:
    return SourceInventory(
        inventory_id="inventory-synthetic",
        deployment_pseudonym=DEPLOYMENT,
        sources=(source,),
    )


@pytest.fixture
def registry(source: SourceIdentity) -> SignalRegistry:
    return SignalRegistry(
        registry_id="frozen-v06-inputs",
        registry_version="0.8.0",
        requirements=tuple(
            SignalRequirement(
                signal_name=name,
                availability=AvailabilityClass.REQUIRED,
                canonical_unit=EXPECTED_UNIT[MetricName(name)].value,
                aggregation="latest_visible_mean",
                cadence_seconds=60.0,
                support_consequence=SupportConsequence.BLOCKED,
                source_ids=(source.source_id,),
                evidence="Synthetic contract evidence for software behavior.",
            )
            for name in SIGNAL_NAMES
        ),
    )


@pytest.fixture
def deployment_pack(
    domain_pack: DomainPack,
    source: SourceIdentity,
    synthetic_signature: SignatureMetadata,
) -> DeploymentPack:
    return DeploymentPack(
        pack_id="synthetic-deployment",
        pack_version="0.8.0",
        deployment_pseudonym=DEPLOYMENT,
        environment=DeploymentEnvironment.LAB,
        collection_modes=(
            CollectionMode.OFFLINE_JSONL,
            CollectionMode.OFFLINE_CSV,
        ),
        approved_source_ids=(source.source_id,),
        domain_pack_sha256=domain_pack.content_hash(),
        compatible_frozen_bundle_sha256="9" * 64,
        ownership=OwnerRoles(
            data_owner="synthetic_data_owner",
            security_owner="synthetic_security_owner",
            storage_owner="synthetic_storage_owner",
            adjudication_owner="synthetic_adjudication_owner",
        ),
        privacy_retention=PrivacyRetention(
            data_classification="synthetic",
            residency_region="test-region",
            retention_days=30,
            deletion_process="Delete the isolated synthetic fixture.",
        ),
        real_operator_data=False,
        lawful_scope_approved=False,
        operator_approved=False,
        signature=synthetic_signature,
    )


@pytest.fixture
def capability() -> ReadOnlyCapabilityAttestation:
    return ReadOnlyCapabilityAttestation(
        adapter_id="reference-batch-reader",
        declared_capabilities=("offline_file_read",),
        forbidden_probe_results={
            "network_query": False,
            "mutation": False,
            "credential_use": False,
            "ticket_create": False,
            "alarm_suppress": False,
        },
    )


@pytest.fixture
def clock_policy() -> ClockWatermarkPolicy:
    return ClockWatermarkPolicy(
        policy_id="clock-policy",
        allowed_lateness_seconds=120.0,
        maximum_future_skew_seconds=5.0,
        maximum_observation_delay_seconds=60.0,
    )


@pytest.fixture
def topology_capture() -> TopologyCapture:
    nodes = (
        IptvNode(
            node_id="topo_" + "1" * 12,
            node_type=IptvNodeType.SOURCE_HEADEND,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "2" * 12,
            node_type=IptvNodeType.CORE,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "3" * 12,
            node_type=IptvNodeType.AGGREGATION,
            service_path_id="path_" + "a" * 12,
            observation_component_pseudonym="cmp_" + "b" * 16,
            applicable_signals=tuple(SIGNAL_NAMES),
        ),
        IptvNode(
            node_id="topo_" + "4" * 12,
            node_type=IptvNodeType.TRANSPORT,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "5" * 12,
            node_type=IptvNodeType.ACCESS_HANDOFF,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "6" * 12,
            node_type=IptvNodeType.CUSTOMER_IMPACT,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "7" * 12,
            node_type=IptvNodeType.MULTICAST_GROUP,
            service_path_id="path_" + "a" * 12,
        ),
        IptvNode(
            node_id="topo_" + "8" * 12,
            node_type=IptvNodeType.MULTICAST_CONTROL,
            service_path_id="path_" + "a" * 12,
        ),
    )
    pairs = [
        (0, 1, IptvEdgeType.FEEDS, None),
        (1, 2, IptvEdgeType.TRANSPORTS_TO, None),
        (2, 3, IptvEdgeType.TRANSPORTS_TO, None),
        (3, 4, IptvEdgeType.TRANSPORTS_TO, None),
        (4, 5, IptvEdgeType.AFFECTS, None),
        (2, 6, IptvEdgeType.MULTICASTS_TO, "group_" + "a" * 12),
        (4, 6, IptvEdgeType.MEMBERSHIP, "group_" + "a" * 12),
        (7, 6, IptvEdgeType.CONTROLS, None),
    ]
    edges = tuple(
        IptvEdge(
            edge_id="edge_" + f"{index:012x}",
            source_node_id=nodes[source_index].node_id,
            destination_node_id=nodes[destination_index].node_id,
            edge_type=edge_type,
            multicast_group_id=group,
        )
        for index, (
            source_index,
            destination_index,
            edge_type,
            group,
        ) in enumerate(pairs, start=1)
    )
    return TopologyCapture(
        capture_id="capture_" + "a" * 12,
        deployment_pseudonym=DEPLOYMENT,
        topology_version="topology-1",
        mode=CaptureMode.FULL,
        effective_at_utc=BASE_TIME - timedelta(hours=1),
        recorded_at_utc=BASE_TIME - timedelta(minutes=55),
        nodes=nodes,
        edges=edges,
    )


@pytest.fixture
def topology_snapshot(topology_capture: TopologyCapture):
    return TopologyHistory((topology_capture,)).reconstruct(
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )


def make_record(
    *,
    source: SourceIdentity,
    metric_name: str,
    sequence: str,
    value: float = 1.0,
    event_time: datetime = BASE_TIME - timedelta(minutes=1),
    observation_time: datetime | None = None,
    ingestion_time: datetime | None = None,
    component: str = "cmp_" + "b" * 16,
    correction_of: str | None = None,
) -> BatchInputRecord:
    observation = observation_time or event_time + timedelta(seconds=1)
    ingestion = ingestion_time or observation + timedelta(seconds=1)
    return BatchInputRecord(
        source_id=source.source_id,
        source_sequence_id=sequence,
        schema_version=source.schema_version,
        metric_name=metric_name,
        metric_value=value,
        unit=EXPECTED_UNIT[MetricName(metric_name)].value,
        component_pseudonym=component,
        event_time_utc=event_time,
        observation_time_utc=observation,
        ingestion_time_utc=ingestion,
        quality="high",
        applicability="observed",
        correction_of=correction_of,
    )


@pytest.fixture
def record_factory(source: SourceIdentity):
    return partial(make_record, source=source)


@pytest.fixture
def records(source: SourceIdentity) -> tuple[BatchInputRecord, ...]:
    return tuple(
        make_record(
            source=source,
            metric_name=name,
            sequence=f"sequence-{index}",
            value=float(index),
        )
        for index, name in enumerate(SIGNAL_NAMES, start=1)
    )


@pytest.fixture
def replay_protocol(
    synthetic_signature: SignatureMetadata,
) -> HistoricalReplayProtocol:
    return HistoricalReplayProtocol(
        protocol_id="iptv-p0-historical-replay",
        protocol_version="0.8.0",
        preregistered_at_utc=BASE_TIME,
        cohorts=(
            CohortDefinition(
                cohort_id="synthetic-cohort",
                inclusion_rule="Include every qualified synthetic event.",
                exclusion_rule="Exclude quarantined or future-visible events.",
                event_time_start_utc=BASE_TIME - timedelta(days=1),
                event_time_end_utc=BASE_TIME,
            ),
        ),
        labels=tuple(
            LabelDefinition(
                label_kind=kind,
                source_inventory_id="inventory-synthetic",
                alignment_rule="Align only by registered bitemporal keys.",
                adjudication_rule="Require independent adjudication evidence.",
            )
            for kind in LabelKind
        ),
        metrics=tuple(ReplayMetric),
        counterfactual_policy=(
            "No post-hoc model, threshold, feature, support, watermark, or "
            "episode policy selection is permitted."
        ),
        exclusions=(
            "quarantined_records",
            "unsupported_semantics",
            "future_visible_evidence",
        ),
        event_time_cutoff_rule=(
            "Use only records whose event and knowledge times are visible "
            "at each preregistered cutoff."
        ),
        signature=synthetic_signature,
    )


@pytest.fixture
def measurements(
    records,
    registry,
    topology_snapshot,
) -> QualificationMeasurements:
    feature = reconstruct_frozen_features(
        records=records,
        registry=registry,
        topology=topology_snapshot,
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    semantic = SemanticMeasuredResult(
        result_id="semantic-measurement",
        mapping_checks=tuple(
            SignalMappingCheck(
                signal_name=name,
                qualified=True,
                evidence_sha256=canonical_hash(
                    {"signal": name, "qualified": True}
                ),
            )
            for name in SIGNAL_NAMES
        ),
        positive_fixture_checks=(
            EvidenceCheck(
                check_id="positive-fixture",
                passed=True,
                evidence_sha256=canonical_hash("positive-fixture"),
            ),
        ),
        mutation_checks=tuple(
            EvidenceCheck(
                check_id=f"semantic-mutation-{index}",
                passed=True,
                evidence_sha256=canonical_hash(
                    {"semantic_mutation": index}
                ),
            )
            for index in range(7)
        ),
    )
    topology = TopologyMeasuredResult(
        result_id="topology-measurement",
        cutoff_results=(
            TopologyCutoffResult(
                cutoff_utc=BASE_TIME,
                snapshot_sha256=topology_snapshot.content_hash,
                deterministic=True,
            ),
        ),
        mutation_checks=tuple(
            EvidenceCheck(
                check_id=f"topology-mutation-{index}",
                passed=True,
                evidence_sha256=canonical_hash(
                    {"topology_mutation": index}
                ),
            )
            for index in range(3)
        ),
    )
    incident_ids = tuple(f"incident-{index}" for index in range(10))
    root_cause_ids = tuple(f"root-cause-{index}" for index in range(10))
    return QualificationMeasurements(
        semantic=semantic,
        topology=topology,
        feature_reconstruction=feature,
        outcomes=OutcomeMeasuredResult(
            result_id="outcome-measurement",
            incident_outcome_ids=incident_ids,
            aligned_incident_outcome_ids=incident_ids,
            adjudicated_root_cause_ids=root_cause_ids,
            mapped_root_cause_ids=root_cause_ids,
            inventory_complete=True,
            evidence_sha256=canonical_hash("outcome-evidence"),
        ),
        privacy=PrivacyMeasuredResult(
            result_id="privacy-measurement",
            scanned_artifact_sha256=(canonical_hash("scanned-artifact"),),
        ),
        tuning_audit=TuningAuditResult(
            result_id="tuning-audit",
            audited_event_sha256=(canonical_hash("audited-events"),),
        ),
        reproducibility=ReproducibilityMeasuredResult(
            result_id="reproducibility-measurement",
            run_output_sha256=(
                canonical_hash("qualification-output"),
                canonical_hash("qualification-output"),
            ),
            tamper_checks=(
                EvidenceCheck(
                    check_id="tamper-control",
                    passed=True,
                    evidence_sha256=canonical_hash("tamper-control"),
                ),
            ),
        ),
    )


@pytest.fixture
def qualification_request(
    domain_pack: DomainPack,
    deployment_pack: DeploymentPack,
    registry: SignalRegistry,
    inventory: SourceInventory,
    capability: ReadOnlyCapabilityAttestation,
    replay_protocol: HistoricalReplayProtocol,
    measurements: QualificationMeasurements,
) -> QualificationRequest:
    return QualificationRequest(
        domain_pack=domain_pack,
        deployment_pack=deployment_pack,
        signal_registry=registry,
        source_inventory=inventory,
        capability_attestation=capability,
        replay_protocol=replay_protocol,
        measurements=measurements,
        qualification_cutoff_utc=BASE_TIME,
    )


@pytest.fixture
def trusted_signer_registry() -> TrustedSignerRegistry:
    return TrustedSignerRegistry(
        registry_id="synthetic-empty-registry",
        registry_version="0.8.0",
    )
