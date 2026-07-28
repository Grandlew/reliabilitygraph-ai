from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.domain.nrim.simulation.feature_schema import SIGNAL_NAMES

from ..contracts import EXPECTED_UNIT, MetricName
from ..hashing import canonical_hash, canonical_json, file_hash
from .batch_adapter import BatchInputRecord
from .contracts import (
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
)
from .qualification import (
    QualificationRequest,
    export_schemas,
    seal_evidence_bundle,
)
from .qualification_measurements import (
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
from .feature_reconstruction import reconstruct_frozen_features
from .replay_protocol import (
    CohortDefinition,
    HistoricalReplayProtocol,
    LabelDefinition,
    ReplayMetric,
)
from .signal_registry import (
    AvailabilityClass,
    SignalRegistry,
    SignalRequirement,
    SupportConsequence,
)
from .source_inventory import (
    Capability,
    ClockQuality,
    ReadOnlyCapabilityAttestation,
    RetentionMetadata,
    SourceClassification,
    SourceIdentity,
    SourceInventory,
)
from .topology import (
    CaptureMode,
    IptvEdge,
    IptvEdgeType,
    IptvNode,
    IptvNodeType,
    TopologyCapture,
    TopologyHistory,
)
from .trusted_signers import TrustedSignerRegistry
from .v06_stage2_parity import build_v06_stage2_example_seal


BASE_TIME = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
DEPLOYMENT = "dep_" + "a" * 24
FROZEN_BUNDLE_SHA256 = (
    "9767bc9f3184a9e68fe99e0a31c5593a610c9dbd3b524dfa155c8bb53091ecf7"
)


def _unsigned() -> SignatureMetadata:
    return SignatureMetadata(
        state=SignatureState.NOT_SIGNED_SYNTHETIC,
        algorithm="none",
    )


def _records_for_source(
    source: SourceIdentity,
) -> tuple[BatchInputRecord, ...]:
    return tuple(
        BatchInputRecord(
            source_id=source.source_id,
            source_sequence_id=f"synthetic-{index:02d}",
            schema_version=source.schema_version,
            metric_name=name,
            metric_value=float(index),
            unit=EXPECTED_UNIT[MetricName(name)].value,
            component_pseudonym="cmp_" + "b" * 16,
            event_time_utc=BASE_TIME - timedelta(minutes=1),
            observation_time_utc=BASE_TIME - timedelta(seconds=59),
            ingestion_time_utc=BASE_TIME - timedelta(seconds=58),
            quality="high",
            applicability="observed",
        )
        for index, name in enumerate(SIGNAL_NAMES, start=1)
    )


def build_synthetic_request() -> QualificationRequest:
    source = SourceIdentity.create(
        source_kind="offline_metric_export",
        lineage_uri="approved://synthetic/source-alpha",
        schema_version="1.0",
        event_time_field="event_time_utc",
        observation_time_field="observation_time_utc",
        ingestion_time_field="ingestion_time_utc",
        clock_id="synthetic-clock",
        clock_quality=ClockQuality.VERIFIED_SYNCHRONIZED,
        owner_role="synthetic_telemetry_owner",
        classification=SourceClassification.SYNTHETIC,
        retention=RetentionMetadata(
            retention_days=30,
            residency_region="test-region",
            deletion_owner_role="synthetic_storage_owner",
            deletion_method="Delete the isolated synthetic fixture.",
        ),
    )
    registry = SignalRegistry(
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
    charter = IptvP0Charter(
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
    domain = DomainPack(
        pack_id="iptv-p0-domain",
        pack_version="0.8.0",
        issued_at_utc=BASE_TIME,
        charter=charter,
        metric_registry_sha256=registry.content_hash(),
        topology_ontology_sha256=canonical_hash(
            {
                "nodes": [item.value for item in IptvNodeType],
                "edges": [item.value for item in IptvEdgeType],
            }
        ),
        outcome_semantics_sha256=canonical_hash(
            [item.value for item in LabelKind]
        ),
        applicability_profile_sha256=canonical_hash(
            {
                item.signal_name: item.applicability_predicate
                for item in registry.requirements
            }
        ),
        support_policy_sha256=canonical_hash(
            {
                item.signal_name: item.support_consequence.value
                for item in registry.requirements
            }
        ),
        signature=_unsigned(),
    )
    inventory = SourceInventory(
        inventory_id="synthetic-source-inventory",
        deployment_pseudonym=DEPLOYMENT,
        sources=(source,),
    )
    deployment = DeploymentPack(
        pack_id="synthetic-deployment",
        pack_version="0.8.0",
        deployment_pseudonym=DEPLOYMENT,
        environment=DeploymentEnvironment.LAB,
        collection_modes=(
            CollectionMode.OFFLINE_JSONL,
            CollectionMode.OFFLINE_CSV,
        ),
        approved_source_ids=(source.source_id,),
        domain_pack_sha256=domain.content_hash(),
        compatible_frozen_bundle_sha256=FROZEN_BUNDLE_SHA256,
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
        signature=_unsigned(),
    )
    protocol = HistoricalReplayProtocol(
        protocol_id="iptv-p0-historical-replay",
        protocol_version="0.8.0",
        preregistered_at_utc=BASE_TIME,
        cohorts=(
            CohortDefinition(
                cohort_id="synthetic-code-path",
                inclusion_rule="Include every qualified synthetic event.",
                exclusion_rule="Exclude quarantined or future-visible events.",
                event_time_start_utc=BASE_TIME - timedelta(days=1),
                event_time_end_utc=BASE_TIME,
            ),
        ),
        labels=tuple(
            LabelDefinition(
                label_kind=kind,
                source_inventory_id=inventory.inventory_id,
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
        signature=_unsigned(),
    )
    capability = ReadOnlyCapabilityAttestation(
        adapter_id="reference-batch-reader",
        declared_capabilities=(Capability.OFFLINE_FILE_READ,),
        forbidden_probe_results={
            item: False
            for item in Capability
            if item is not Capability.OFFLINE_FILE_READ
        },
    )
    records = _records_for_source(source)
    topology_snapshot = TopologyHistory(
        (build_synthetic_topology(),)
    ).reconstruct(
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    feature_result = reconstruct_frozen_features(
        records=records,
        registry=registry,
        topology=topology_snapshot,
        event_cutoff_utc=BASE_TIME,
        knowledge_cutoff_utc=BASE_TIME,
    )
    measurements = QualificationMeasurements(
        semantic=SemanticMeasuredResult(
            result_id="synthetic-semantic-measurement",
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
                    check_id="synthetic-positive-fixture",
                    passed=True,
                    evidence_sha256=canonical_hash(
                        "synthetic-positive-fixture"
                    ),
                ),
            ),
            mutation_checks=tuple(
                EvidenceCheck(
                    check_id=f"synthetic-semantic-mutation-{index}",
                    passed=True,
                    evidence_sha256=canonical_hash(
                        {"synthetic_semantic_mutation": index}
                    ),
                )
                for index in range(7)
            ),
        ),
        topology=TopologyMeasuredResult(
            result_id="synthetic-topology-measurement",
            cutoff_results=(
                TopologyCutoffResult(
                    cutoff_utc=BASE_TIME,
                    snapshot_sha256=topology_snapshot.content_hash,
                    deterministic=True,
                ),
            ),
            mutation_checks=tuple(
                EvidenceCheck(
                    check_id=f"synthetic-topology-mutation-{index}",
                    passed=True,
                    evidence_sha256=canonical_hash(
                        {"synthetic_topology_mutation": index}
                    ),
                )
                for index in range(3)
            ),
        ),
        feature_reconstruction=feature_result,
        outcomes=OutcomeMeasuredResult(
            result_id="synthetic-outcome-measurement",
            incident_outcome_ids=(),
            aligned_incident_outcome_ids=(),
            adjudicated_root_cause_ids=(),
            mapped_root_cause_ids=(),
            inventory_complete=False,
            evidence_sha256=canonical_hash("no-real-outcome-evidence"),
        ),
        privacy=PrivacyMeasuredResult(
            result_id="synthetic-privacy-measurement",
            scanned_artifact_sha256=(
                canonical_hash("synthetic-scanned-artifact"),
            ),
        ),
        tuning_audit=TuningAuditResult(
            result_id="synthetic-tuning-audit",
            audited_event_sha256=(
                canonical_hash("synthetic-audited-events"),
            ),
        ),
        reproducibility=ReproducibilityMeasuredResult(
            result_id="synthetic-reproducibility-measurement",
            run_output_sha256=(
                canonical_hash("synthetic-qualification-output"),
                canonical_hash("synthetic-qualification-output"),
            ),
            tamper_checks=(
                EvidenceCheck(
                    check_id="synthetic-tamper-control",
                    passed=True,
                    evidence_sha256=canonical_hash(
                        "synthetic-tamper-control"
                    ),
                ),
            ),
        ),
    )
    return QualificationRequest(
        domain_pack=domain,
        deployment_pack=deployment,
        signal_registry=registry,
        source_inventory=inventory,
        capability_attestation=capability,
        replay_protocol=protocol,
        measurements=measurements,
        qualification_cutoff_utc=BASE_TIME,
    )


def build_synthetic_trusted_signer_registry() -> TrustedSignerRegistry:
    return TrustedSignerRegistry(
        registry_id="synthetic-empty-registry",
        registry_version="0.8.0",
    )


def build_synthetic_records(
    request: QualificationRequest,
) -> tuple[BatchInputRecord, ...]:
    source = request.source_inventory.sources[0]
    return _records_for_source(source)


def build_synthetic_topology() -> TopologyCapture:
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
            observation_component_pseudonym="cmp_" + "b" * 16,
            applicable_signals=tuple(SIGNAL_NAMES),
        ),
        IptvNode(
            node_id="topo_" + "3" * 12,
            node_type=IptvNodeType.ACCESS_HANDOFF,
            service_path_id="path_" + "a" * 12,
        ),
    )
    edges = (
        IptvEdge(
            edge_id="edge_" + "1" * 12,
            source_node_id=nodes[0].node_id,
            destination_node_id=nodes[1].node_id,
            edge_type=IptvEdgeType.FEEDS,
        ),
        IptvEdge(
            edge_id="edge_" + "2" * 12,
            source_node_id=nodes[1].node_id,
            destination_node_id=nodes[2].node_id,
            edge_type=IptvEdgeType.TRANSPORTS_TO,
        ),
    )
    return TopologyCapture(
        capture_id="capture_" + "a" * 12,
        deployment_pseudonym=DEPLOYMENT,
        topology_version="synthetic-topology-1",
        mode=CaptureMode.FULL,
        effective_at_utc=BASE_TIME - timedelta(hours=1),
        recorded_at_utc=BASE_TIME - timedelta(minutes=59),
        nodes=nodes,
        edges=edges,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8") + b"\n")


def generate_fixture(root: Path) -> dict[str, object]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = build_synthetic_request()
    _write_json(
        root / "parity/v06_stage2_example.seal.json",
        build_v06_stage2_example_seal().model_dump(mode="json"),
    )
    records = build_synthetic_records(request)
    topology = build_synthetic_topology()
    (root / "SYNTHETIC_ONLY.md").write_bytes(
        (
            "# Synthetic fixture only\n\n"
            "These records prove deterministic software behavior. They are "
            "not production data and cannot satisfy the real-data gate.\n"
        ).encode("utf-8")
    )
    _write_json(
        root / "scope.json",
        {
            "schema_version": "0.8.0",
            "domain": "IPTV-P0 managed live multicast",
            "truth_status": "synthetic_software_behavior_only",
            "terminal_decision_required": "REAL_DATA_REQUIRED",
            "frozen_bundle_sha256": FROZEN_BUNDLE_SHA256,
            "research_basis": (
                "ETSI_TR_101_290",
                "IETF_RFC_3550",
                "IETF_RFC_3376",
                "IETF_RFC_4541",
                "OpenTelemetry_Semantic_Conventions",
                "NIST_Deployed_AI_Monitoring",
                "SLSA_v1.2",
            ),
        },
    )
    _write_json(
        root / "synthetic" / "request.json",
        request.model_dump(mode="json"),
    )
    records_path = root / "synthetic" / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(
        b"".join(
            canonical_json(item.model_dump(mode="json")).encode("utf-8")
            + b"\n"
            for item in records
        )
    )
    _write_json(
        root / "synthetic" / "topology.jsonl",
        topology.model_dump(mode="json"),
    )
    _write_json(
        root / "synthetic" / "mutations.json",
        {
            "registered_mutations": (
                {
                    "id": "unit_swap",
                    "expected": "PROFILE_SEMANTIC_MISMATCH",
                },
                {
                    "id": "timezone_removed",
                    "expected": "CONTRACT_INVALID",
                },
                {
                    "id": "negative_sign",
                    "expected": "PROFILE_SEMANTIC_MISMATCH",
                },
                {
                    "id": "aggregation_profile_change",
                    "expected": "CONTRACT_COMPATIBILITY_BLOCKED",
                },
                {
                    "id": "required_signal_missing",
                    "expected": "BLOCKED",
                },
                {
                    "id": "applicability_unknown",
                    "expected": "CONTRACT_INVALID",
                },
                {
                    "id": "topology_reversed",
                    "expected": "TOPOLOGY_INVALID",
                },
            ),
            "acceptance_requirement": (
                "100_percent_registered_mutation_rejection"
            ),
        },
    )
    export_schemas(root / "schemas")
    seal_evidence_bundle(
        request=request,
        trusted_signer_registry=build_synthetic_trusted_signer_registry(),
        output_directory=root / "expected",
        signature=_unsigned(),
    )
    files = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "fixture_manifest.json"
    }
    manifest = {
        "schema_version": "0.8.0",
        "truth_status": "synthetic_software_behavior_only",
        "terminal_decision": "REAL_DATA_REQUIRED",
        "files": files,
    }
    _write_json(root / "fixture_manifest.json", manifest)
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(generate_fixture(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
