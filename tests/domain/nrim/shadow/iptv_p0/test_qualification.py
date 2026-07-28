from __future__ import annotations

import json
import shutil

import pytest

from app.domain.nrim.shadow.iptv_p0.contracts import (
    CollectionMode,
    DeploymentEnvironment,
    DeploymentPack,
    DomainPack,
    GateDecision,
    SignatureState,
    create_ed25519_signature,
    verify_signature,
)
from app.domain.nrim.shadow.iptv_p0.qualification import (
    QualificationRequest,
    evaluate_qualification,
    seal_evidence_bundle,
    verify_evidence_bundle,
)
from app.domain.nrim.shadow.iptv_p0.replay_protocol import (
    HistoricalReplayProtocol,
)
from app.domain.nrim.shadow.iptv_p0.signal_registry import SignalRegistry
from app.domain.nrim.shadow.iptv_p0.source_inventory import (
    SourceClassification,
    SourceIdentity,
    SourceInventory,
)


PRIVATE_KEY = bytes(range(1, 33))


def _sign(model):
    signature = create_ed25519_signature(
        payload_sha256=model.content_hash(),
        private_key_bytes=PRIVATE_KEY,
        key_id="qualification-test-key",
    )
    value = model.model_dump(mode="json")
    value["signature"] = signature.model_dump(mode="json")
    return type(model).model_validate(value)


def _real_request(request: QualificationRequest) -> QualificationRequest:
    prior_source = request.source_inventory.sources[0]
    source_value = prior_source.model_dump(
        mode="json",
        exclude={"source_id"},
    )
    source_value["classification"] = (
        SourceClassification.PSEUDONYMIZED_OPERATIONAL
    )
    real_source = SourceIdentity.create(**source_value)
    inventory = SourceInventory(
        inventory_id=request.source_inventory.inventory_id,
        deployment_pseudonym=request.source_inventory.deployment_pseudonym,
        sources=(real_source,),
    )
    requirements = tuple(
        item.model_copy(update={"source_ids": (real_source.source_id,)})
        for item in request.signal_registry.requirements
    )
    registry = SignalRegistry(
        registry_id=request.signal_registry.registry_id,
        registry_version=request.signal_registry.registry_version,
        requirements=requirements,
    )
    domain_value = request.domain_pack.model_dump(mode="json")
    domain_value["metric_registry_sha256"] = registry.content_hash()
    domain = DomainPack.model_validate(domain_value)
    domain = _sign(domain)

    replay = _sign(request.replay_protocol)
    deployment_value = request.deployment_pack.model_dump(mode="json")
    deployment_value.update(
        {
            "environment": DeploymentEnvironment.PRODUCTION_HISTORICAL_EXPORT,
            "collection_modes": (CollectionMode.OFFLINE_JSONL,),
            "approved_source_ids": (real_source.source_id,),
            "domain_pack_sha256": domain.content_hash(),
            "privacy_retention": {
                "data_classification": "pseudonymized_operational",
                "residency_region": "approved-region",
                "retention_days": 30,
                "deletion_process": (
                    "Delete the isolated approved historical export."
                ),
                "direct_identifiers_prohibited": True,
                "raw_topology_identifiers_prohibited": True,
            },
            "real_operator_data": True,
            "lawful_scope_approved": True,
            "operator_approved": True,
        }
    )
    deployment = DeploymentPack.model_validate(deployment_value)
    deployment = _sign(deployment)
    return QualificationRequest(
        domain_pack=domain,
        deployment_pack=deployment,
        signal_registry=registry,
        source_inventory=inventory,
        capability_attestation=request.capability_attestation,
        replay_protocol=replay,
        metrics=request.metrics,
    )


def test_synthetic_pack_is_always_real_data_required(qualification_request):
    result = evaluate_qualification(qualification_request)
    assert result.decision is GateDecision.REAL_DATA_REQUIRED
    assert result.synthetic_evidence_only
    assert result.reason_codes == (
        "LAWFUL_REAL_DEPLOYMENT_PACK_REQUIRED",
    )
    assert result.claim_ceiling.value == "engineering_readiness_only"


def test_synthetic_result_never_authorizes_operational_actions(
    qualification_request,
):
    result = evaluate_qualification(qualification_request)
    assert not result.operational_write_authorized
    assert not result.model_or_threshold_tuning_authorized
    assert not result.prospective_performance_claim_authorized


def test_cryptographically_qualified_real_pack_can_reach_replay_ready(
    qualification_request,
):
    result = evaluate_qualification(_real_request(qualification_request))
    assert result.decision is GateDecision.REAL_REPLAY_READY
    assert all(result.criteria.values())
    assert result.authorized_claim == (
        "Compatible with one qualified real IPTV deployment for "
        "preregistered historical replay."
    )


@pytest.mark.parametrize(
    "field,value,criterion",
    (
        ("collector_mapping_fraction", 0.99, "collector_semantics"),
        (
            "semantic_mutation_rejection_fraction",
            0.99,
            "collector_semantics",
        ),
        ("topology_reconstruction_fraction", 0.99, "topology"),
        ("topology_mutation_rejection_fraction", 0.99, "topology"),
        (
            "required_feature_fraction",
            0.949,
            "feature_reconstructability",
        ),
        ("inference_call_count", 1, "feature_reconstructability"),
        ("incident_alignment_fraction", 0.899, "incident_alignment"),
        ("root_cause_mapping_fraction", 0.949, "root_cause_mapping"),
        ("outcome_inventory_complete", False, "outcome_inventory"),
        ("future_leakage_count", 1, "zero_leakage"),
        ("identifier_leakage_count", 1, "zero_leakage"),
        ("model_tuning_event_count", 1, "no_tuning"),
        ("threshold_tuning_event_count", 1, "no_tuning"),
        ("feature_tuning_event_count", 1, "no_tuning"),
        ("support_rule_tuning_event_count", 1, "no_tuning"),
        ("watermark_tuning_event_count", 1, "no_tuning"),
        ("episode_grouping_tuning_event_count", 1, "no_tuning"),
        (
            "evidence_determinism_fraction",
            0.99,
            "deterministic_tamper_evident_evidence",
        ),
        (
            "tamper_detection_passed",
            False,
            "deterministic_tamper_evident_evidence",
        ),
    ),
)
def test_each_external_gate_failure_blocks_with_reason_code(
    qualification_request,
    field,
    value,
    criterion,
):
    request = _real_request(qualification_request)
    metrics = request.metrics.model_copy(update={field: value})
    blocked = evaluate_qualification(
        request.model_copy(update={"metrics": metrics})
    )
    assert blocked.decision is GateDecision.BLOCKED
    assert f"CRITERION_FAILED:{criterion}" in blocked.reason_codes


def test_qualification_is_deterministic(qualification_request):
    first = evaluate_qualification(qualification_request)
    second = evaluate_qualification(qualification_request)
    assert first.content_hash == second.content_hash
    assert first.qualification_id == second.qualification_id


def test_ed25519_signature_detects_payload_change():
    signature = create_ed25519_signature(
        payload_sha256="a" * 64,
        private_key_bytes=PRIVATE_KEY,
        key_id="test-key",
    )
    assert verify_signature(signature, expected_payload_sha256="a" * 64)
    assert not verify_signature(signature, expected_payload_sha256="b" * 64)


def test_evidence_bundle_seals_and_verifies(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    manifest = seal_evidence_bundle(
        request=qualification_request,
        output_directory=tmp_path / "bundle",
        signature=synthetic_signature,
    )
    verified = verify_evidence_bundle(tmp_path / "bundle")
    assert verified == manifest
    assert manifest["decision"] == "REAL_DATA_REQUIRED"


def test_real_replay_ready_bundle_requires_and_verifies_manifest_signature(
    tmp_path,
    qualification_request,
):
    request = _real_request(qualification_request)

    def signer(payload_sha256):
        return create_ed25519_signature(
            payload_sha256=payload_sha256,
            private_key_bytes=PRIVATE_KEY,
            key_id="manifest-test-key",
        )

    manifest = seal_evidence_bundle(
        request=request,
        output_directory=tmp_path / "signed",
        signature_provider=signer,
    )
    assert manifest["decision"] == "REAL_REPLAY_READY"
    assert manifest["signature"]["state"] == SignatureState.VERIFIED.value
    assert verify_evidence_bundle(tmp_path / "signed") == manifest


def test_real_replay_ready_bundle_rejects_unsigned_manifest(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    with pytest.raises(ValueError, match="verified manifest"):
        seal_evidence_bundle(
            request=_real_request(qualification_request),
            output_directory=tmp_path / "unsigned",
            signature=synthetic_signature,
        )


def test_evidence_bundle_detects_artifact_tampering(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    directory = tmp_path / "bundle"
    seal_evidence_bundle(
        request=qualification_request,
        output_directory=directory,
        signature=synthetic_signature,
    )
    with (directory / "qualification.json").open(
        "a",
        encoding="utf-8",
        newline="",
    ) as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="differs"):
        verify_evidence_bundle(directory)


def test_evidence_bundle_survives_path_relocation(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    seal_evidence_bundle(
        request=qualification_request,
        output_directory=original,
        signature=synthetic_signature,
    )
    shutil.copytree(original, relocated)
    assert verify_evidence_bundle(relocated)["decision"] == (
        "REAL_DATA_REQUIRED"
    )


def test_bundle_manifest_rejects_path_traversal(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    directory = tmp_path / "bundle"
    manifest = seal_evidence_bundle(
        request=qualification_request,
        output_directory=directory,
        signature=synthetic_signature,
    )
    manifest["files"]["../outside.json"] = "0" * 64
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        verify_evidence_bundle(directory)


def test_bundle_outputs_are_lf_only(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    directory = tmp_path / "bundle"
    seal_evidence_bundle(
        request=qualification_request,
        output_directory=directory,
        signature=synthetic_signature,
    )
    for path in directory.rglob("*.json"):
        assert b"\r\n" not in path.read_bytes()
        assert path.read_bytes().endswith(b"\n")


def test_bundle_double_run_is_byte_stable(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        seal_evidence_bundle(
            request=qualification_request,
            output_directory=directory,
            signature=synthetic_signature,
        )
    first_bytes = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_bytes = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_bytes == second_bytes


def test_request_contract_rejects_unknown_fields(qualification_request):
    value = qualification_request.model_dump(mode="json")
    value["live_endpoint"] = "https://forbidden.invalid"
    with pytest.raises(Exception):
        QualificationRequest.model_validate(value)
