from __future__ import annotations

import json
import shutil
from datetime import timedelta

import pytest

from app.domain.nrim.shadow.hashing import canonical_hash
from app.domain.nrim.shadow.iptv_p0.contracts import (
    CollectionMode,
    DeploymentEnvironment,
    DeploymentPack,
    DomainPack,
    GateDecision,
    SignatureState,
    create_ed25519_signature,
    ed25519_public_key_base64,
    verify_signature,
)
from app.domain.nrim.shadow.iptv_p0.feature_reconstruction import (
    FeatureEvidenceState,
)
from app.domain.nrim.shadow.iptv_p0.qualification import (
    QualificationRequest,
    evaluate_qualification,
    seal_evidence_bundle,
    verify_evidence_bundle,
)
from app.domain.nrim.shadow.iptv_p0.qualification_measurements import (
    ProhibitedTuningEvent,
    QualificationMeasurements,
    TuningCategory,
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
from app.domain.nrim.shadow.iptv_p0.trusted_signers import (
    SignerRole,
    TrustedSigner,
    TrustedSignerRegistry,
)


PRIVATE_KEYS = {
    "domain": bytes([1]) * 32,
    "deployment": bytes([2]) * 32,
    "replay": bytes([3]) * 32,
    "qualification": bytes([4]) * 32,
}


def _sign(model, authority: str):
    signature = create_ed25519_signature(
        payload_sha256=model.content_hash(),
        private_key_bytes=PRIVATE_KEYS[authority],
        key_id=f"{authority}-test-key",
    )
    value = model.model_dump(mode="json")
    value["signature"] = signature.model_dump(mode="json")
    return type(model).model_validate(value)


def _trusted_registry(request: QualificationRequest) -> TrustedSignerRegistry:
    deployment = request.deployment_pack.deployment_pseudonym
    roles = {
        "domain": SignerRole.DOMAIN_AUTHORITY,
        "deployment": SignerRole.DEPLOYMENT_AUTHORITY,
        "replay": SignerRole.REPLAY_GOVERNANCE,
        "qualification": SignerRole.QUALIFICATION_AUTHORITY,
    }
    return TrustedSignerRegistry(
        registry_id="qualification-test-trust",
        registry_version="0.8.0",
        signers=tuple(
            TrustedSigner(
                signer_id=f"signer_{name}_authority",
                key_id=f"{name}-test-key",
                public_key_base64=ed25519_public_key_base64(
                    PRIVATE_KEYS[name]
                ),
                roles=(role,),
                deployment_pseudonym=(
                    None if name == "domain" else deployment
                ),
                valid_from_utc=(
                    request.qualification_cutoff_utc - timedelta(days=1)
                ),
                valid_to_utc=(
                    request.qualification_cutoff_utc + timedelta(days=1)
                ),
            )
            for name, role in roles.items()
        ),
    )


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
    domain = _sign(domain, "domain")

    replay = _sign(request.replay_protocol, "replay")
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
    deployment = _sign(deployment, "deployment")
    real_request = QualificationRequest(
        domain_pack=domain,
        deployment_pack=deployment,
        signal_registry=registry,
        source_inventory=inventory,
        capability_attestation=request.capability_attestation,
        replay_protocol=replay,
        measurements=request.measurements,
        qualification_cutoff_utc=request.qualification_cutoff_utc,
    )
    return real_request


def _measurements_with_failure(
    measurements: QualificationMeasurements,
    failure: str,
) -> QualificationMeasurements:
    if failure == "mapping":
        checks = measurements.semantic.mapping_checks
        semantic = measurements.semantic.model_copy(
            update={
                "mapping_checks": (
                    checks[0].model_copy(update={"qualified": False}),
                    *checks[1:],
                )
            }
        )
        return measurements.model_copy(update={"semantic": semantic})
    if failure in {"positive_fixture", "semantic_mutation"}:
        field = (
            "positive_fixture_checks"
            if failure == "positive_fixture"
            else "mutation_checks"
        )
        checks = getattr(measurements.semantic, field)
        semantic = measurements.semantic.model_copy(
            update={
                field: (
                    checks[0].model_copy(update={"passed": False}),
                    *checks[1:],
                )
            }
        )
        return measurements.model_copy(update={"semantic": semantic})
    if failure in {"topology_reconstruction", "future_leakage"}:
        cutoffs = measurements.topology.cutoff_results
        update = (
            {"deterministic": False}
            if failure == "topology_reconstruction"
            else {"future_visible_capture_ids": ("capture_future",)}
        )
        topology = measurements.topology.model_copy(
            update={
                "cutoff_results": (
                    cutoffs[0].model_copy(update=update),
                    *cutoffs[1:],
                )
            }
        )
        return measurements.model_copy(update={"topology": topology})
    if failure == "topology_mutation":
        checks = measurements.topology.mutation_checks
        topology = measurements.topology.model_copy(
            update={
                "mutation_checks": (
                    checks[0].model_copy(update={"passed": False}),
                    *checks[1:],
                )
            }
        )
        return measurements.model_copy(update={"topology": topology})
    if failure == "feature_coverage":
        lineage = list(measurements.feature_reconstruction.per_node_lineage)
        index = next(
            index
            for index, item in enumerate(lineage)
            if item.required_for_gate
        )
        lineage[index] = lineage[index].model_copy(
            update={
                "state": FeatureEvidenceState.MISSING,
                "source_ids": (),
                "source_record_sha256": (),
            }
        )
        values = list(measurements.feature_reconstruction.stage2.values)
        masks = list(
            measurements.feature_reconstruction.stage2.observed_mask
        )
        values[index] = None
        masks[index] = 0
        stage2 = measurements.feature_reconstruction.stage2.model_copy(
            update={
                "values": tuple(values),
                "observed_mask": tuple(masks),
            }
        )
        feature = measurements.feature_reconstruction.model_copy(
            update={
                "stage2": stage2,
                "per_node_lineage": tuple(lineage),
            }
        )
        return measurements.model_copy(
            update={"feature_reconstruction": feature}
        )
    if failure == "inference_call":
        feature = measurements.feature_reconstruction.model_copy(
            update={"inference_call_sha256": (canonical_hash("inference"),)}
        )
        return measurements.model_copy(
            update={"feature_reconstruction": feature}
        )
    if failure in {"incident_alignment", "root_cause_mapping"}:
        outcomes = measurements.outcomes
        update = (
            {
                "aligned_incident_outcome_ids": (
                    outcomes.aligned_incident_outcome_ids[:-2]
                )
            }
            if failure == "incident_alignment"
            else {
                "mapped_root_cause_ids": outcomes.mapped_root_cause_ids[:-1]
            }
        )
        return measurements.model_copy(
            update={"outcomes": outcomes.model_copy(update=update)}
        )
    if failure == "outcome_inventory":
        return measurements.model_copy(
            update={
                "outcomes": measurements.outcomes.model_copy(
                    update={"inventory_complete": False}
                )
            }
        )
    if failure == "identifier_leakage":
        privacy = measurements.privacy.model_copy(
            update={"identifier_leakage_finding_ids": ("finding-1",)}
        )
        return measurements.model_copy(update={"privacy": privacy})
    categories = {
        "model_tuning": TuningCategory.MODEL,
        "threshold_tuning": TuningCategory.THRESHOLD,
        "feature_tuning": TuningCategory.FEATURE,
        "support_rule_tuning": TuningCategory.SUPPORT_RULE,
        "watermark_tuning": TuningCategory.WATERMARK,
        "episode_grouping_tuning": TuningCategory.EPISODE_GROUPING,
    }
    if failure in categories:
        tuning = measurements.tuning_audit.model_copy(
            update={
                "prohibited_events": (
                    ProhibitedTuningEvent(
                        event_id=f"{failure}-event",
                        category=categories[failure],
                        evidence_sha256=canonical_hash(failure),
                    ),
                )
            }
        )
        return measurements.model_copy(update={"tuning_audit": tuning})
    if failure == "nondeterminism":
        reproducibility = measurements.reproducibility.model_copy(
            update={
                "run_output_sha256": (
                    measurements.reproducibility.run_output_sha256[0],
                    canonical_hash("different-output"),
                )
            }
        )
        return measurements.model_copy(
            update={"reproducibility": reproducibility}
        )
    if failure == "tamper_miss":
        checks = measurements.reproducibility.tamper_checks
        reproducibility = measurements.reproducibility.model_copy(
            update={
                "tamper_checks": (
                    checks[0].model_copy(update={"passed": False}),
                    *checks[1:],
                )
            }
        )
        return measurements.model_copy(
            update={"reproducibility": reproducibility}
        )
    raise AssertionError(f"Unknown measured-evidence failure: {failure}")


def test_synthetic_pack_is_always_real_data_required(
    qualification_request,
    trusted_signer_registry,
):
    result = evaluate_qualification(
        qualification_request,
        trusted_signer_registry=trusted_signer_registry,
    )
    assert result.decision is GateDecision.REAL_DATA_REQUIRED
    assert result.synthetic_evidence_only
    assert result.reason_codes == (
        "LAWFUL_REAL_DEPLOYMENT_PACK_REQUIRED",
    )
    assert result.claim_ceiling.value == "engineering_readiness_only"


def test_synthetic_result_never_authorizes_operational_actions(
    qualification_request,
    trusted_signer_registry,
):
    result = evaluate_qualification(
        qualification_request,
        trusted_signer_registry=trusted_signer_registry,
    )
    assert not result.operational_write_authorized
    assert not result.model_or_threshold_tuning_authorized
    assert not result.prospective_performance_claim_authorized


def test_cryptographically_qualified_real_pack_can_reach_replay_ready(
    qualification_request,
):
    request = _real_request(qualification_request)
    result = evaluate_qualification(
        request,
        trusted_signer_registry=_trusted_registry(request),
    )
    assert result.decision is GateDecision.REAL_REPLAY_READY
    assert all(result.criteria.values())
    assert result.authorized_claim == (
        "Compatible with one qualified real IPTV deployment for "
        "preregistered historical replay."
    )


@pytest.mark.parametrize(
    "failure,criterion",
    (
        ("unregistered", "verified_domain_signature"),
        ("wrong_role", "verified_domain_signature"),
        ("wrong_deployment", "verified_replay_protocol_signature"),
        ("expired", "verified_replay_protocol_signature"),
        ("revoked", "verified_deployment_signature"),
        ("wrong_public_key", "verified_domain_signature"),
    ),
)
def test_trusted_signer_authorization_failures_block(
    qualification_request,
    failure,
    criterion,
):
    request = _real_request(qualification_request)
    registry = _trusted_registry(request)
    signers = list(registry.signers)
    target = {
        "wrong_deployment": "replay-test-key",
        "expired": "replay-test-key",
        "revoked": "deployment-test-key",
    }.get(failure, "domain-test-key")
    index = next(
        index
        for index, signer in enumerate(signers)
        if signer.key_id == target
    )
    if failure == "unregistered":
        del signers[index]
    else:
        signer = signers[index]
        update = {
            "wrong_role": {
                "roles": (SignerRole.DEPLOYMENT_AUTHORITY,),
                "deployment_pseudonym": (
                    request.deployment_pack.deployment_pseudonym
                ),
            },
            "wrong_deployment": {
                "deployment_pseudonym": "dep_" + "b" * 24,
            },
            "expired": {
                "valid_to_utc": request.qualification_cutoff_utc,
            },
            "revoked": {
                "revoked_at_utc": request.qualification_cutoff_utc,
            },
            "wrong_public_key": {
                "public_key_base64": ed25519_public_key_base64(
                    bytes([9]) * 32
                ),
            },
        }[failure]
        signers[index] = TrustedSigner.model_validate(
            signer.model_copy(update=update).model_dump(mode="json")
        )
    registry = TrustedSignerRegistry(
        registry_id=registry.registry_id,
        registry_version=registry.registry_version,
        signers=tuple(signers),
    )
    result = evaluate_qualification(
        request,
        trusted_signer_registry=registry,
    )
    assert result.decision is GateDecision.BLOCKED
    assert f"CRITERION_FAILED:{criterion}" in result.reason_codes


def test_artifact_signers_must_be_distinct_even_if_multi_role_authorized(
    qualification_request,
):
    request = _real_request(qualification_request)
    replay = _sign(request.replay_protocol, "deployment")
    request = request.model_copy(update={"replay_protocol": replay})
    registry = _trusted_registry(request)
    signers = tuple(
        signer.model_copy(
            update={
                "roles": (
                    SignerRole.DEPLOYMENT_AUTHORITY,
                    SignerRole.REPLAY_GOVERNANCE,
                )
            }
        )
        if signer.key_id == "deployment-test-key"
        else signer
        for signer in registry.signers
    )
    result = evaluate_qualification(
        request,
        trusted_signer_registry=registry.model_copy(
            update={"signers": signers}
        ),
    )
    assert result.decision is GateDecision.BLOCKED
    assert (
        "CRITERION_FAILED:distinct_artifact_signers"
        in result.reason_codes
    )


def test_trusted_registry_rejects_key_aliases_for_one_identity(
    qualification_request,
):
    request = _real_request(qualification_request)
    registry = _trusted_registry(request)
    aliased = registry.signers[1].model_copy(
        update={
            "public_key_base64": registry.signers[0].public_key_base64,
        }
    )
    with pytest.raises(ValueError, match="distinct identities"):
        TrustedSignerRegistry(
            registry_id=registry.registry_id,
            registry_version=registry.registry_version,
            signers=(registry.signers[0], aliased, *registry.signers[2:]),
        )


@pytest.mark.parametrize(
    "failure,criterion",
    (
        ("mapping", "collector_semantics"),
        ("positive_fixture", "collector_semantics"),
        ("semantic_mutation", "collector_semantics"),
        ("topology_reconstruction", "topology"),
        ("topology_mutation", "topology"),
        ("feature_coverage", "feature_reconstructability"),
        ("inference_call", "feature_reconstructability"),
        ("incident_alignment", "incident_alignment"),
        ("root_cause_mapping", "root_cause_mapping"),
        ("outcome_inventory", "outcome_inventory"),
        ("future_leakage", "zero_leakage"),
        ("identifier_leakage", "zero_leakage"),
        ("model_tuning", "no_tuning"),
        ("threshold_tuning", "no_tuning"),
        ("feature_tuning", "no_tuning"),
        ("support_rule_tuning", "no_tuning"),
        ("watermark_tuning", "no_tuning"),
        ("episode_grouping_tuning", "no_tuning"),
        ("nondeterminism", "deterministic_tamper_evident_evidence"),
        ("tamper_miss", "deterministic_tamper_evident_evidence"),
    ),
)
def test_each_measured_evidence_failure_blocks_with_reason_code(
    qualification_request,
    failure,
    criterion,
):
    request = _real_request(qualification_request)
    measurements = _measurements_with_failure(
        request.measurements,
        failure,
    )
    blocked = evaluate_qualification(
        request.model_copy(update={"measurements": measurements}),
        trusted_signer_registry=_trusted_registry(request),
    )
    assert blocked.decision is GateDecision.BLOCKED
    assert f"CRITERION_FAILED:{criterion}" in blocked.reason_codes


def test_qualification_is_deterministic(
    qualification_request,
    trusted_signer_registry,
):
    first = evaluate_qualification(
        qualification_request,
        trusted_signer_registry=trusted_signer_registry,
    )
    second = evaluate_qualification(
        qualification_request,
        trusted_signer_registry=trusted_signer_registry,
    )
    assert first.content_hash == second.content_hash
    assert first.qualification_id == second.qualification_id


def test_ed25519_signature_detects_payload_change():
    signature = create_ed25519_signature(
        payload_sha256="a" * 64,
        private_key_bytes=PRIVATE_KEYS["domain"],
        key_id="test-key",
    )
    trusted_key = ed25519_public_key_base64(PRIVATE_KEYS["domain"])
    assert verify_signature(
        signature,
        expected_payload_sha256="a" * 64,
        trusted_public_key_base64=trusted_key,
    )
    assert not verify_signature(
        signature,
        expected_payload_sha256="b" * 64,
        trusted_public_key_base64=trusted_key,
    )


def test_evidence_bundle_seals_and_verifies(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    manifest = seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
        output_directory=tmp_path / "bundle",
        signature=synthetic_signature,
    )
    verified = verify_evidence_bundle(
        tmp_path / "bundle",
        trusted_signer_registry=trusted_signer_registry,
    )
    assert verified == manifest
    assert manifest["decision"] == "REAL_DATA_REQUIRED"


def test_real_replay_ready_bundle_requires_and_verifies_manifest_signature(
    tmp_path,
    qualification_request,
):
    request = _real_request(qualification_request)
    trusted_registry = _trusted_registry(request)

    def signer(payload_sha256):
        return create_ed25519_signature(
            payload_sha256=payload_sha256,
            private_key_bytes=PRIVATE_KEYS["qualification"],
            key_id="qualification-test-key",
        )

    manifest = seal_evidence_bundle(
        request=request,
        trusted_signer_registry=trusted_registry,
        output_directory=tmp_path / "signed",
        signature_provider=signer,
    )
    assert manifest["decision"] == "REAL_REPLAY_READY"
    assert manifest["signature"]["state"] == SignatureState.VERIFIED.value
    assert verify_evidence_bundle(
        tmp_path / "signed",
        trusted_signer_registry=trusted_registry,
    ) == manifest


def test_real_replay_ready_bundle_rejects_unsigned_manifest(
    tmp_path,
    qualification_request,
    synthetic_signature,
):
    request = _real_request(qualification_request)
    with pytest.raises(ValueError, match="verified manifest"):
        seal_evidence_bundle(
            request=request,
            trusted_signer_registry=_trusted_registry(request),
            output_directory=tmp_path / "unsigned",
            signature=synthetic_signature,
        )


def test_final_manifest_signer_must_be_distinct_from_artifact_signers(
    tmp_path,
    qualification_request,
):
    request = _real_request(qualification_request)
    registry = _trusted_registry(request)
    signers = tuple(
        signer.model_copy(
            update={
                "roles": (
                    SignerRole.DOMAIN_AUTHORITY,
                    SignerRole.QUALIFICATION_AUTHORITY,
                ),
                "deployment_pseudonym": (
                    request.deployment_pack.deployment_pseudonym
                ),
            }
        )
        if signer.key_id == "domain-test-key"
        else signer
        for signer in registry.signers
    )

    def signer(payload_sha256):
        return create_ed25519_signature(
            payload_sha256=payload_sha256,
            private_key_bytes=PRIVATE_KEYS["domain"],
            key_id="domain-test-key",
        )

    with pytest.raises(ValueError, match="distinct"):
        seal_evidence_bundle(
            request=request,
            trusted_signer_registry=registry.model_copy(
                update={"signers": signers}
            ),
            output_directory=tmp_path / "same-signer",
            signature_provider=signer,
        )


def test_evidence_bundle_detects_artifact_tampering(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    directory = tmp_path / "bundle"
    seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
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
        verify_evidence_bundle(
            directory,
            trusted_signer_registry=trusted_signer_registry,
        )


def test_bundle_verification_requires_the_out_of_band_trust_registry(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    directory = tmp_path / "bundle"
    seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
        output_directory=directory,
        signature=synthetic_signature,
    )
    different_registry = trusted_signer_registry.model_copy(
        update={"registry_id": "different-trust-root"}
    )
    with pytest.raises(ValueError, match="trusted registry"):
        verify_evidence_bundle(
            directory,
            trusted_signer_registry=different_registry,
        )


def test_evidence_bundle_survives_path_relocation(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
        output_directory=original,
        signature=synthetic_signature,
    )
    shutil.copytree(original, relocated)
    assert verify_evidence_bundle(
        relocated,
        trusted_signer_registry=trusted_signer_registry,
    )["decision"] == (
        "REAL_DATA_REQUIRED"
    )


def test_bundle_manifest_rejects_path_traversal(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    directory = tmp_path / "bundle"
    manifest = seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
        output_directory=directory,
        signature=synthetic_signature,
    )
    manifest["files"]["../outside.json"] = "0" * 64
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        verify_evidence_bundle(
            directory,
            trusted_signer_registry=trusted_signer_registry,
        )


def test_bundle_outputs_are_lf_only(
    tmp_path,
    qualification_request,
    synthetic_signature,
    trusted_signer_registry,
):
    directory = tmp_path / "bundle"
    seal_evidence_bundle(
        request=qualification_request,
        trusted_signer_registry=trusted_signer_registry,
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
    trusted_signer_registry,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        seal_evidence_bundle(
            request=qualification_request,
            trusted_signer_registry=trusted_signer_registry,
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
