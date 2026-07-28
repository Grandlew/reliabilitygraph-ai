from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.iptv_p0.contracts import (
    DeploymentPack,
    DomainPack,
    IptvP0Charter,
    LabelKind,
    OperatorTask,
    SignatureMetadata,
    SignatureState,
)


def _validate(model, **updates):
    value = model.model_dump(mode="json")
    value.update(updates)
    return type(model).model_validate(value)


def test_charter_is_deterministic_and_freezes_all_tasks(charter):
    assert charter.content_hash() == charter.content_hash()
    assert set(charter.protected_operator_tasks) == set(OperatorTask)
    assert set(charter.label_kinds) == set(LabelKind)


@pytest.mark.parametrize(
    "missing",
    (
        "source_headend",
        "core_aggregation",
        "multicast_control",
        "transport",
        "access_handoff",
    ),
)
def test_charter_rejects_missing_service_path_segment(charter, missing):
    with pytest.raises(ValidationError):
        _validate(
            charter,
            included_service_path=tuple(
                item
                for item in charter.included_service_path
                if item != missing
            ),
        )


@pytest.mark.parametrize(
    "missing",
    (
        "ott_abr",
        "drm",
        "model_training",
        "threshold_tuning",
        "gnn",
        "autonomous_remediation",
        "live_polling",
        "write_credentials",
    ),
)
def test_charter_rejects_missing_safety_exclusion(charter, missing):
    with pytest.raises(ValidationError):
        _validate(
            charter,
            explicit_exclusions=tuple(
                item for item in charter.explicit_exclusions if item != missing
            ),
        )


@pytest.mark.parametrize("missing", tuple(OperatorTask))
def test_charter_rejects_missing_protected_task(charter, missing):
    with pytest.raises(ValidationError):
        _validate(
            charter,
            protected_operator_tasks=tuple(
                item
                for item in charter.protected_operator_tasks
                if item is not missing
            ),
        )


@pytest.mark.parametrize("missing", tuple(LabelKind))
def test_charter_rejects_collapsed_outcome_labels(charter, missing):
    with pytest.raises(ValidationError):
        _validate(
            charter,
            label_kinds=tuple(
                item for item in charter.label_kinds if item is not missing
            ),
        )


@pytest.mark.parametrize(
    "update",
    (
        {"algorithm": "ed25519"},
        {"key_id": "unexpected-key"},
        {"signature_base64": "dW5leHBlY3RlZC1zaWc="},
    ),
)
def test_unsigned_synthetic_signature_cannot_name_signing_material(update):
    value = {
        "state": SignatureState.NOT_SIGNED_SYNTHETIC,
        "algorithm": "none",
        **update,
    }
    with pytest.raises(ValidationError):
        SignatureMetadata.model_validate(value)


@pytest.mark.parametrize(
    "missing",
    ("key_id", "signature_base64", "signed_payload_sha256"),
)
def test_verified_signature_requires_complete_metadata(
    verified_signature,
    missing,
):
    value = verified_signature.model_dump(mode="json")
    value[missing] = None
    with pytest.raises(ValidationError):
        SignatureMetadata.model_validate(value)


def test_domain_pack_normalizes_utc_and_has_stable_content_hash(domain_pack):
    assert domain_pack.issued_at_utc.utcoffset().total_seconds() == 0
    assert domain_pack.content_hash() == domain_pack.content_hash()


def test_domain_pack_rejects_naive_issue_time(domain_pack):
    with pytest.raises(ValidationError):
        _validate(domain_pack, issued_at_utc=datetime(2026, 1, 1))


@pytest.mark.parametrize(
    "update",
    (
        {
            "real_operator_data": True,
            "privacy_retention": {
                "data_classification": "synthetic",
                "residency_region": "test-region",
                "retention_days": 30,
                "deletion_process": "Delete the isolated synthetic fixture.",
                "direct_identifiers_prohibited": True,
                "raw_topology_identifiers_prohibited": True,
            },
        },
        {
            "real_operator_data": False,
            "privacy_retention": {
                "data_classification": "pseudonymized_operational",
                "residency_region": "test-region",
                "retention_days": 30,
                "deletion_process": "Delete the isolated approved export.",
                "direct_identifiers_prohibited": True,
                "raw_topology_identifiers_prohibited": True,
            },
        },
    ),
)
def test_deployment_pack_rejects_truth_classification_conflict(
    deployment_pack,
    update,
):
    with pytest.raises(ValidationError):
        _validate(deployment_pack, **update)


def test_deployment_pack_rejects_duplicate_sources(deployment_pack):
    source = deployment_pack.approved_source_ids[0]
    with pytest.raises(ValidationError):
        _validate(deployment_pack, approved_source_ids=(source, source))


def test_synthetic_deployment_never_completes_real_gate_identity(
    deployment_pack,
):
    assert not deployment_pack.real_gate_identity_complete


def test_contracts_reject_unknown_fields(charter):
    value = charter.model_dump(mode="json")
    value["future_contract_field"] = True
    with pytest.raises(ValidationError):
        IptvP0Charter.model_validate(value)
