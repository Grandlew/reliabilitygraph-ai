from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.shadow.bundle import generate_signing_keypair
from app.domain.nrim.shadow.evidence_anchor import (
    EvidenceAnchorService,
    IndependentFileAnchorSink,
    verify_anchor_receipt,
    verify_store_matches_checkpoint,
)
from app.domain.nrim.shadow.privacy import PrivacyViolation
from app.domain.nrim.shadow.readiness import (
    DressRehearsalEvidence,
    dress_rehearsal_gate,
    evaluate_stop_conditions,
    historical_replay_gate,
    operational_gate_summary,
)
from app.domain.nrim.shadow.sampling import (
    FrameSource,
    SamplingFrameRecord,
    SamplingPlan,
    WeightedReviewOutcome,
    inverse_probability_weighted_rate,
    sampling_frame_hash,
    sampling_seed_commitment,
    select_review_sample,
    sign_sampling_plan,
)
from app.domain.nrim.shadow.store import AppendOnlyEvidenceStore
from app.domain.nrim.shadow.storage_qualification import (
    StorageBackendKind,
    StorageQualificationEvidence,
    qualify_storage,
)


UTC = timezone.utc
NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)
DEPLOYMENT = "deployment_" + "a" * 40


def test_independently_keyed_checkpoint_receipt_detects_tampering(
    tmp_path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.sqlite")
    app_private, app_public = generate_signing_keypair()
    custodian_private, custodian_public = generate_signing_keypair()
    sink = IndependentFileAnchorSink(
        directory=tmp_path / "external-anchor-adapter",
        sink_id="independent-evidence-custodian",
        external_trust_domain="audit.example.net",
        application_trust_domain="nrim.application",
        custodian_private_key=custodian_private,
        custodian_public_key_pem=custodian_public,
    )
    service = EvidenceAnchorService(
        store=store,
        application_instance_id="shadow-instance-a",
        application_trust_domain="nrim.application",
        protocol_sha256="a" * 64,
        model_bundle_sha256="b" * 64,
        application_private_key=app_private,
        application_public_key_pem=app_public,
    )
    signed, receipt = service.checkpoint(
        sink=sink,
        sequence=1,
        previous_checkpoint_sha256="0" * 64,
        previous_evidence_count=0,
    )
    verify_anchor_receipt(
        signed_checkpoint=signed,
        receipt=receipt,
        application_public_key_pem=app_public,
        custodian_public_key_pem=custodian_public,
    )
    assert verify_store_matches_checkpoint(
        store=store,
        signed_checkpoint=signed,
    )
    with pytest.raises(ValueError, match="predecessor differs"):
        service.checkpoint(
            sink=sink,
            sequence=2,
            previous_checkpoint_sha256="f" * 64,
            previous_evidence_count=0,
        )
    second, second_receipt = service.checkpoint(
        sink=sink,
        sequence=2,
        previous_checkpoint_sha256=signed.checkpoint_sha256,
        previous_evidence_count=0,
    )
    verify_anchor_receipt(
        signed_checkpoint=second,
        receipt=second_receipt,
        application_public_key_pem=app_public,
        custodian_public_key_pem=custodian_public,
    )
    tampered = receipt.model_copy(
        update={
            "payload": receipt.payload.model_copy(
                update={"sink_id": "application-operator"}
            )
        }
    )
    with pytest.raises(ValueError, match="commitment"):
        verify_anchor_receipt(
            signed_checkpoint=signed,
            receipt=tampered,
            application_public_key_pem=app_public,
            custodian_public_key_pem=custodian_public,
        )
    with pytest.raises(ValueError, match="different trust domain"):
        IndependentFileAnchorSink(
            directory=tmp_path / "invalid-sink",
            sink_id="invalid",
            external_trust_domain="nrim.application",
            application_trust_domain="nrim.application",
            custodian_private_key=custodian_private,
            custodian_public_key_pem=custodian_public,
        )
    with pytest.raises(PrivacyViolation, match="Direct identifier"):
        store.append_review_event(
            event_id="event-with-pii",
            case_id="case-with-pii",
            event_type="unsafe_note",
            actor_pseudonym="reviewer_" + "a" * 32,
            payload={"note": "Contact operator@example.com"},
        )


def frame() -> tuple[SamplingFrameRecord, ...]:
    base = {
        "deployment_pseudonym": DEPLOYMENT,
        "hour_start_utc": NOW,
        "workload_band": "normal",
        "support_state": "in_support",
    }
    return (
        SamplingFrameRecord(
            case_id="known-incident",
            source=FrameSource.KNOWN_OPERATIONAL_INCIDENT,
            **base,
        ),
        SamplingFrameRecord(
            case_id="nrim-positive",
            source=FrameSource.NRIM_POSITIVE_EPISODE,
            **base,
        ),
        SamplingFrameRecord(
            case_id="healthy-a",
            source=FrameSource.HEALTHY_HOUR_CANDIDATE,
            **base,
        ),
        SamplingFrameRecord(
            case_id="healthy-b",
            source=FrameSource.HEALTHY_HOUR_CANDIDATE,
            **base,
        ),
    )


def test_signed_sampling_is_reproducible_and_records_probabilities() -> None:
    rows = frame()
    seed = "signed-sampling-seed-2026"
    private, public = generate_signing_keypair()
    plan = SamplingPlan(
        plan_id="review-plan-1",
        created_at_utc=NOW,
        frame_sha256=sampling_frame_hash(rows),
        seed_commitment_sha256=sampling_seed_commitment(seed),
        healthy_target_per_stratum=1,
    )
    signed = sign_sampling_plan(
        plan=plan,
        private_key=private,
        public_key_pem=public,
    )
    first = select_review_sample(
        signed_plan=signed,
        public_key_pem=public,
        seed=seed,
        frame=rows,
    )
    second = select_review_sample(
        signed_plan=signed,
        public_key_pem=public,
        seed=seed,
        frame=rows,
    )
    assert first == second
    assert first.known_incident_coverage == 1.0
    assert first.nrim_positive_coverage == 1.0
    healthy = [
        item
        for item in first.selected
        if item.source is FrameSource.HEALTHY_HOUR_CANDIDATE
    ]
    assert len(healthy) == 1
    assert healthy[0].inclusion_probability == 0.5
    with pytest.raises(ValueError, match="frame differs"):
        select_review_sample(
            signed_plan=signed,
            public_key_pem=public,
            seed=seed,
            frame=rows[:-1],
        )
    weighted = inverse_probability_weighted_rate(
        (
            WeightedReviewOutcome(
                case_id="healthy-a",
                inclusion_probability=0.5,
                exposure_hours=1.0,
                outcome=True,
            ),
            WeightedReviewOutcome(
                case_id="known-incident",
                inclusion_probability=1.0,
                exposure_hours=1.0,
                outcome=False,
            ),
        )
    )
    assert weighted["weighted_rate"] == pytest.approx(2.0 / 3.0)


def historical_metrics() -> dict:
    return {
        "real_data": True,
        "analysis_plan_preregistered": True,
        "required_feature_fraction": 1.0,
        "collector_mapping_fraction": 1.0,
        "semantic_mutation_rejection_fraction": 1.0,
        "incident_alignment_fraction": 1.0,
        "label_availability_reported": True,
        "future_leakage_count": 0,
        "replay_determinism_fraction": 1.0,
        "root_cause_mapping_fraction": 1.0,
        "topology_reconstruction_fraction": 1.0,
        "model_tuning_event_count": 0,
        "threshold_tuning_event_count": 0,
        "feature_tuning_event_count": 0,
        "support_rule_tuning_event_count": 0,
        "watermark_tuning_event_count": 0,
        "episode_grouping_tuning_event_count": 0,
        "semantic_deviation_count": 1,
        "semantic_lineage_record_count": 1,
        "data_gap_report_complete": True,
    }


def rehearsal(days: float = 7.0) -> DressRehearsalEvidence:
    return DressRehearsalEvidence(
        rehearsal_id="dress-rehearsal-1",
        started_at_utc=NOW,
        ended_at_utc=NOW + timedelta(days=days),
        real_data=True,
        predictions_hidden_from_frontline=True,
        read_only_boundary_passed=True,
        schema_compliance=0.995,
        quarantine_attribution_fraction=1.0,
        envelope_reproduction_fraction=1.0,
        evidence_envelope_completeness=1.0,
        evidence_chain_continuity=1.0,
        external_checkpoint_success_fraction=1.0,
        scheduled_cutoff_completion_fraction=0.999,
        registered_availability_floor=0.995,
        unsafe_stage2_count=0,
        restart_recovery_fraction=1.0,
        review_workflow_passed=True,
        emergency_stop_count=0,
    )


def test_historical_replay_rehearsal_and_stop_gates_are_nonpromotional() -> None:
    historical = historical_replay_gate(historical_metrics())
    assert historical["passed"]
    assert historical["promotion_evidence"] is False
    tuned = historical_metrics()
    tuned["threshold_tuning_event_count"] = 1
    assert not historical_replay_gate(tuned)["passed"]

    dress = dress_rehearsal_gate(rehearsal())
    assert dress["passed"]
    assert dress["promotion_evidence"] is False
    assert not dress_rehearsal_gate(rehearsal(6.99))["passed"]

    stop = evaluate_stop_conditions(
        {"EVIDENCE_INTEGRITY_FAILURE": True}
    )
    assert stop["pause_required"]
    assert stop["preserve_evidence"]
    assert stop["performance_tuning_authorized"] is False

    summary = operational_gate_summary(
        contracts={
            "collector_mapping_fraction": 1.0,
            "semantic_mutation_rejection_fraction": 1.0,
            "topology_reconstruction_fraction": 1.0,
            "topology_mutation_rejection_fraction": 1.0,
            "privacy_approved": False,
        },
        replay=historical,
        evidence_workflow={
            "external_anchor_verified": False,
            "storage_qualification_passed": False,
            "review_usability_passed": False,
        },
        shadow_runtime={
            "dress_rehearsal_passed": False,
            "read_only_boundary_passed": False,
        },
        pilot={
            "compiled_protocol_signed": False,
            "all_owners_approved": False,
            "deployment_count": 0,
            "topology_family_count": 0,
        },
    )
    assert not summary["all_ready"]
    assert not summary["gates"]["contracts_ready"]


def test_sqlite_cannot_be_mislabeled_as_pilot_storage() -> None:
    evidence = StorageQualificationEvidence(
        qualification_id="storage-check-1",
        tested_at_utc=NOW,
        backend_kind=StorageBackendKind.SQLITE_TEST_ADAPTER,
        infrastructure_identity="local-shadow-test-store",
        independently_administered=False,
        concurrent_process_count=2,
        concurrent_append_passed=True,
        conflicting_duplicate_test_passed=True,
        transaction_atomicity_test_passed=True,
        retention_lock_enabled=False,
        retention_policy_days=90,
        deletion_denial_test_passed=False,
        encrypted_backup_passed=True,
        restore_test_passed=True,
        restored_chain_matches=True,
        immutable_audit_export_passed=True,
        access_control_review_passed=False,
        disaster_recovery_owner="owner_" + "a" * 32,
        evidence_artifact_hashes={
            f"artifact-{index}": str(index) * 64
            for index in range(1, 6)
        },
    )
    result = qualify_storage(evidence)
    assert not result["passed"]
    assert not result["criteria"]["pilot_backend"]
