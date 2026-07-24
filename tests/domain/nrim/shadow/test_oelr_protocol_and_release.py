from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.shadow.bundle import generate_signing_keypair
from app.domain.nrim.shadow.hashing import bytes_hash
from app.domain.nrim.shadow.protocol_compiler import (
    EvidencePhase,
    OELRProtocol,
    SeparationOfDuties,
    compile_protocol,
    default_statistical_contract,
    default_stop_rules,
    register_compiled_protocol,
    verify_compiled_protocol,
)
from app.domain.nrim.shadow.release import (
    EvidenceOrigin,
    PilotContext,
    PromotionState,
    evaluate_phase_transition,
)
from app.domain.nrim.shadow.sampling import sampling_seed_commitment


UTC = timezone.utc
NOW = datetime(2026, 7, 24, tzinfo=UTC)
DEPLOYMENTS = tuple(
    f"deployment_{index}_" + "a" * 32 for index in range(5)
)


def duties(**changes) -> SeparationOfDuties:
    values = {
        "protocol_owner": "protocol_" + "a" * 32,
        "evidence_custodian": "evidence_" + "b" * 32,
        "review_lead": "review_" + "c" * 32,
        "release_authority": "release_" + "d" * 32,
        "operations_owner": "operations_" + "e" * 32,
        "privacy_owner": "privacy_" + "f" * 32,
        "security_owner": "security_" + "1" * 32,
    }
    values.update(changes)
    return SeparationOfDuties(**values)


def protocol(
    *,
    target_phase: EvidencePhase = EvidencePhase.SILENT_SHADOW,
    statistical_contract=None,
    assigned_duties=None,
) -> OELRProtocol:
    return OELRProtocol(
        protocol_id="oelr-pilot-1",
        registered_at_utc=NOW,
        target_phase=target_phase,
        planned_start_utc=NOW + timedelta(days=1),
        planned_end_utc=NOW + timedelta(weeks=9),
        deployment_pseudonyms=DEPLOYMENTS,
        topology_family_count=5,
        model_bundle_hash="a" * 64,
        bundle_public_key_sha256="b" * 64,
        feature_schema_hash="c" * 64,
        policy_hash="d" * 64,
        watermark_policy_hash="e" * 64,
        historical_replay_plan_sha256="2" * 64,
        dress_rehearsal_plan_sha256="3" * 64,
        review_workflow_plan_sha256="4" * 64,
        external_anchor_expectation_sha256="5" * 64,
        collector_contract_hashes={
            item: "f" * 64 for item in DEPLOYMENTS
        },
        topology_contract_hashes={
            item: "1" * 64 for item in DEPLOYMENTS
        },
        statistical_contract=(
            statistical_contract
            or default_statistical_contract(
                target_phase=target_phase,
                sampling_seed_commitment_sha256=(
                    sampling_seed_commitment(
                        "registered-sampling-seed"
                    )
                ),
            )
        ),
        stop_rules=default_stop_rules(),
        path_retention_rules=(
            "Retain fast only for paired severe rescue or delay value.",
            "Retain slow only for unique persistent weak-impact value.",
            "Equivalent performance selects the simpler frozen policy.",
        ),
        duties=assigned_duties or duties(),
    )


def test_protocol_compiler_is_signed_phase_correct_and_immutable(
    tmp_path,
) -> None:
    private, public = generate_signing_keypair()
    compiled = compile_protocol(protocol())
    assert (
        compiled.derived_constraints[
            "maximum_false_episodes_at_minimum_exposure"
        ]
        == 1
    )
    assert compiled.derived_constraints["human_utility_required"] is False
    registration = tmp_path / "compiled_protocol.json"
    public_path = tmp_path / "protocol_public.pem"
    public_path.write_bytes(public)
    register_compiled_protocol(
        compiled=compiled,
        registration_path=registration,
        private_key=private,
        public_key_pem=public,
    )
    assert (
        verify_compiled_protocol(
            registration_path=registration,
            public_key_path=public_path,
        )
        == compiled
    )
    with pytest.raises(FileExistsError):
        register_compiled_protocol(
            compiled=compiled,
            registration_path=registration,
            private_key=private,
            public_key_pem=public,
        )


def test_protocol_rejects_human_utility_in_hidden_phase_and_role_collision() -> None:
    advisory_contract = default_statistical_contract(
        target_phase=EvidencePhase.OPERATOR_ADVISORY,
        sampling_seed_commitment_sha256=sampling_seed_commitment(
            "registered-sampling-seed"
        ),
    )
    with pytest.raises(
        ValidationError,
        match="Human utility is unavailable",
    ):
        protocol(statistical_contract=advisory_contract)
    with pytest.raises(ValidationError, match="separately assigned"):
        duties(evidence_custodian="protocol_" + "a" * 32)


def replay_operational() -> dict:
    return {
        "real_data": True,
        "collector_mapping_fraction": 1.0,
        "semantic_mutation_rejection_fraction": 1.0,
        "topology_reconstruction_fraction": 1.0,
        "topology_mutation_rejection_fraction": 1.0,
        "privacy_design_approved": True,
        "identifier_leakage_count": 0,
        "read_only_boundary_verified": True,
        "operational_write_credentials_present": False,
        "external_trust_anchor_verified": True,
        "external_checkpoint_success_fraction": 1.0,
        "concurrent_append_test_passed": True,
        "backup_restore_test_passed": True,
        "retention_lock_test_passed": True,
        "immutable_audit_export_test_passed": True,
        "historical_replay_gate_passed": True,
        "historical_tuning_event_count": 0,
        "blinded_review_usability_passed": True,
        "consecutive_dress_rehearsal_days": 7.0,
        "dress_rehearsal_gate_passed": True,
    }


def silent_report() -> dict:
    return {
        "incident_detection": {
            "D": {
                "episode_recall": 0.95,
                "episode_recall_exact_lower_95": 0.82,
                "severe_incident_count": 14,
                "severe_recall": 1.0,
                "severe_recall_exact_lower_95": 0.807,
                "failure_families": {
                    "storage": {"incident_count": 10, "recall": 0.9}
                },
                "median_delay_hours": 0.5,
                "p90_delay_hours": 2.0,
                "mean_extra_fragments": 0.01,
            }
        },
        "alert_burden": {
            "D": {
                "episodes_per_100_healthy_hours": 0.03,
                "conservative_upper_95": 0.15,
                "heterogeneity": {
                    "deployment_intervals_complete": True,
                    "minimum_exposure_rule_applied": True,
                    "inference_status": "feasibility_only",
                    "random_effects_sensitivity": None,
                },
            }
        },
        "ranking": {
            "mrr": 0.75,
            "hits_at_1": 0.60,
            "hits_at_3": 0.90,
            "ranking_coverage": 0.95,
        },
        "support_and_data_quality": {
            "unsupported_safe_semantics": True,
            "unsafe_stage2_count": 0,
        },
        "path_attribution": {"recommendation": "retain_fast_only"},
        "adjudication": {
            "agreement_assessment_status": "interpretable"
        },
        "human_utility": {},
    }


def silent_operational() -> dict:
    return {
        "predictions_hidden_from_frontline": True,
        "schema_compliance": 1.0,
        "invalid_required_category_quarantined_fraction": 1.0,
        "feature_parity_fraction": 1.0,
        "future_leakage_count": 0,
        "prediction_reproducibility_fraction": 1.0,
        "topology_provenance_fraction": 1.0,
        "audit_field_leakage_count": 0,
        "scheduled_cutoff_completion_fraction": 1.0,
        "registered_availability_floor": 0.995,
        "evidence_durability_fraction": 1.0,
        "external_checkpoint_success_fraction": 1.0,
        "p95_latency_seconds": 1.0,
        "p95_latency_fraction_of_decision_interval": 0.01,
        "unregistered_bundle_change_count": 0,
        "state_recovery_parity_fraction": 1.0,
        "decision_interval_hours": 1.0,
        "data_quality_alerts_reported_separately": True,
        "id_false_unsupported_rate": 0.01,
        "support_dimensions_complete": True,
        "schema_shift_states_separate": True,
        "known_incident_review_fraction": 1.0,
        "nrim_incident_review_fraction": 1.0,
        "negative_sampling_probability_record_fraction": 1.0,
        "unresolved_labels_preserved": True,
        "minimum_deployments_for_cluster_claim": 10,
    }


def test_release_states_use_only_phase_available_evidence() -> None:
    replay = evaluate_phase_transition(
        target_state=PromotionState.REAL_REPLAY_READY,
        report={},
        operational=replay_operational(),
        context=PilotContext(
            evidence_origin=EvidenceOrigin.HISTORICAL_REPLAY,
            pilot_start_utc=None,
            pilot_end_utc=None,
            independent_deployment_count=5,
            topology_family_count=5,
            healthy_deployment_hours=0,
            adjudicated_incident_count=0,
            protocol_registered_before_outcomes=True,
        ),
    )
    assert replay.passed
    assert replay.state is PromotionState.REAL_REPLAY_READY
    assert not any("recall" in item for item in replay.criteria)

    silent = evaluate_phase_transition(
        target_state=PromotionState.SHADOW_VALIDATED,
        report=silent_report(),
        operational=silent_operational(),
        context=PilotContext(
            evidence_origin=EvidenceOrigin.REAL_PROSPECTIVE_SILENT,
            pilot_start_utc=NOW,
            pilot_end_utc=NOW + timedelta(weeks=8),
            independent_deployment_count=5,
            topology_family_count=5,
            healthy_deployment_hours=3000,
            adjudicated_incident_count=30,
            protocol_registered_before_outcomes=True,
            current_state=PromotionState.REAL_REPLAY_READY,
        ),
    )
    assert silent.passed
    assert silent.state is PromotionState.SHADOW_VALIDATED
    assert "engineer_top3_usefulness" not in silent.criteria


def test_synthetic_cannot_enter_real_replay_or_shadow_state() -> None:
    decision = evaluate_phase_transition(
        target_state=PromotionState.REAL_REPLAY_READY,
        report={},
        operational=replay_operational(),
        context=PilotContext(
            evidence_origin=EvidenceOrigin.SYNTHETIC,
            pilot_start_utc=None,
            pilot_end_utc=None,
            independent_deployment_count=5,
            topology_family_count=5,
            healthy_deployment_hours=0,
            adjudicated_incident_count=0,
            protocol_registered_before_outcomes=True,
        ),
    )
    assert not decision.passed
    assert decision.state is PromotionState.CANDIDATE
    assert "real_historical_evidence" in decision.blockers
