from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class EvidenceOrigin(str, Enum):
    SYNTHETIC = "synthetic"
    HISTORICAL_REPLAY = "historical_replay"
    REAL_PROSPECTIVE_SILENT = "real_prospective_silent"
    REAL_OPERATOR_ADVISORY = "real_operator_advisory"
    REAL_OPERATIONAL_ASSESSMENT = "real_operational_assessment"


class PromotionState(str, Enum):
    CANDIDATE = "candidate"
    REAL_REPLAY_READY = "real_replay_ready"
    SHADOW_VALIDATED = "shadow_validated"
    ADVISORY_VALIDATED = "advisory_validated"
    OPERATIONAL_CANDIDATE = "operational_candidate"

    # Compatibility name for callers of the original v0.7 API.
    OPERATOR_ADVISORY = "advisory_validated"


_PREDECESSOR = {
    PromotionState.REAL_REPLAY_READY: PromotionState.CANDIDATE,
    PromotionState.SHADOW_VALIDATED: PromotionState.REAL_REPLAY_READY,
    PromotionState.ADVISORY_VALIDATED: PromotionState.SHADOW_VALIDATED,
    PromotionState.OPERATIONAL_CANDIDATE: (
        PromotionState.ADVISORY_VALIDATED
    ),
}


@dataclass(frozen=True)
class PilotContext:
    evidence_origin: EvidenceOrigin
    pilot_start_utc: datetime | None
    pilot_end_utc: datetime | None
    independent_deployment_count: int
    topology_family_count: int
    healthy_deployment_hours: float
    adjudicated_incident_count: int
    protocol_registered_before_outcomes: bool
    pilot_tuning_events: int = 0
    emergency_stop_events: tuple[str, ...] = ()
    current_state: PromotionState = PromotionState.CANDIDATE

    @property
    def elapsed_weeks(self) -> float:
        if self.pilot_start_utc is None or self.pilot_end_utc is None:
            return 0.0
        return max(
            0.0,
            (self.pilot_end_utc - self.pilot_start_utc).total_seconds()
            / (7.0 * 24.0 * 3600.0),
        )


@dataclass(frozen=True)
class ReleaseDecision:
    state: PromotionState
    target_state: PromotionState
    passed: bool
    blockers: tuple[str, ...]
    criteria: Mapping[str, bool]


def _at_least(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def _equals_one(value: Any) -> bool:
    return value == 1.0


def _base_criteria(
    *,
    context: PilotContext,
    target_state: PromotionState,
) -> dict[str, bool]:
    return {
        "valid_state_transition": (
            _PREDECESSOR.get(target_state) is context.current_state
        ),
        "protocol_preregistered": (
            context.protocol_registered_before_outcomes
        ),
        "no_outcome_driven_tuning": context.pilot_tuning_events == 0,
    }


def _replay_ready_criteria(
    operational: Mapping[str, Any],
    context: PilotContext,
) -> dict[str, bool]:
    return {
        "real_historical_evidence": (
            context.evidence_origin is EvidenceOrigin.HISTORICAL_REPLAY
            and operational.get("real_data") is True
        ),
        "collector_contracts_complete": _equals_one(
            operational.get("collector_mapping_fraction")
        ),
        "semantic_mutations_fail_closed": _equals_one(
            operational.get("semantic_mutation_rejection_fraction")
        ),
        "topology_reconstruction_deterministic": _equals_one(
            operational.get("topology_reconstruction_fraction")
        ),
        "topology_mutations_fail_closed": _equals_one(
            operational.get("topology_mutation_rejection_fraction")
        ),
        "privacy_design_approved": (
            operational.get("privacy_design_approved") is True
        ),
        "identifier_leakage_zero": (
            operational.get("identifier_leakage_count") == 0
        ),
        "read_only_boundary_independently_verified": (
            operational.get("read_only_boundary_verified") is True
            and operational.get("operational_write_credentials_present")
            is False
        ),
        "external_trust_anchor_verified": (
            operational.get("external_trust_anchor_verified") is True
        ),
        "external_checkpoint_roundtrip": _equals_one(
            operational.get("external_checkpoint_success_fraction")
        ),
        "pilot_storage_qualified": (
            operational.get("concurrent_append_test_passed") is True
            and operational.get("backup_restore_test_passed") is True
            and operational.get("retention_lock_test_passed") is True
            and operational.get("immutable_audit_export_test_passed") is True
        ),
        "historical_replay_readiness": (
            operational.get("historical_replay_gate_passed") is True
        ),
        "historical_replay_without_tuning": (
            operational.get("historical_tuning_event_count") == 0
        ),
        "review_workflow_usable": (
            operational.get("blinded_review_usability_passed") is True
        ),
        "seven_day_dress_rehearsal": (
            _at_least(
                operational.get("consecutive_dress_rehearsal_days"),
                7.0,
            )
            and operational.get("dress_rehearsal_gate_passed") is True
        ),
        "minimum_5_planned_deployments": (
            context.independent_deployment_count >= 5
        ),
        "minimum_5_topology_families": (
            context.topology_family_count >= 5
        ),
    }


def _silent_criteria(
    *,
    report: Mapping[str, Any],
    operational: Mapping[str, Any],
    context: PilotContext,
) -> dict[str, bool]:
    dual = report.get("incident_detection", {}).get("D", {})
    burden = report.get("alert_burden", {}).get("D", {})
    ranking = report.get("ranking", {})
    support = report.get("support_and_data_quality", {})
    path = report.get("path_attribution", {})
    adjudication = report.get("adjudication", {})
    families = dual.get("failure_families", {})
    recall_bound_required = operational.get(
        "incident_recall_bound_required",
        True,
    )
    severe_bound_required = operational.get(
        "severe_recall_bound_required",
        True,
    )
    minimum_severe = int(
        operational.get("minimum_severe_incidents", 14)
    )
    deployment_count = context.independent_deployment_count
    real_silent = (
        context.evidence_origin
        is EvidenceOrigin.REAL_PROSPECTIVE_SILENT
    )
    return {
        "real_prospective_evidence": real_silent,
        "real_prospective_hidden_evidence": (
            real_silent
            and operational.get("predictions_hidden_from_frontline") is True
        ),
        "minimum_8_weeks": context.elapsed_weeks >= 8.0,
        "minimum_5_deployments": deployment_count >= 5,
        "minimum_5_topology_families": (
            context.topology_family_count >= 5
        ),
        "minimum_3000_healthy_hours": (
            context.healthy_deployment_hours >= 3000.0
        ),
        "minimum_30_incidents": (
            context.adjudicated_incident_count >= 30
        ),
        "schema_compliance": _at_least(
            operational.get("schema_compliance"),
            0.99,
        ),
        "invalid_categories_quarantined": _equals_one(
            operational.get(
                "invalid_required_category_quarantined_fraction"
            )
        ),
        "feature_parity": _equals_one(
            operational.get("feature_parity_fraction")
        ),
        "zero_future_leakage": (
            operational.get("future_leakage_count") == 0
        ),
        "prediction_reproducibility": _equals_one(
            operational.get("prediction_reproducibility_fraction")
        ),
        "topology_provenance": _equals_one(
            operational.get("topology_provenance_fraction")
        ),
        "zero_audit_field_leakage": (
            operational.get("audit_field_leakage_count") == 0
        ),
        "runtime_availability": _at_least(
            operational.get("scheduled_cutoff_completion_fraction"),
            float(
                operational.get(
                    "registered_availability_floor",
                    0.995,
                )
            ),
        ),
        "evidence_durability": _equals_one(
            operational.get("evidence_durability_fraction")
        ),
        "external_evidence_checkpointing": _equals_one(
            operational.get("external_checkpoint_success_fraction")
        ),
        "latency": (
            _at_most(operational.get("p95_latency_seconds"), 60.0)
            and _at_most(
                operational.get(
                    "p95_latency_fraction_of_decision_interval"
                ),
                0.10,
            )
        ),
        "bundle_integrity": (
            operational.get("unregistered_bundle_change_count") == 0
        ),
        "state_recovery": _equals_one(
            operational.get("state_recovery_parity_fraction")
        ),
        "episode_recall_point": _at_least(
            dual.get("episode_recall"),
            0.80,
        ),
        "episode_recall_bound": (
            not recall_bound_required
            or _at_least(
                dual.get("episode_recall_exact_lower_95"),
                0.80,
            )
        ),
        "severe_count": (
            dual.get("severe_incident_count", 0) >= minimum_severe
        ),
        "severe_recall_point": dual.get("severe_recall") == 1.0,
        "severe_recall_bound": (
            not severe_bound_required
            or _at_least(
                dual.get("severe_recall_exact_lower_95"),
                0.80,
            )
        ),
        "family_recall": bool(families)
        and all(
            row.get("incident_count", 0) < int(
                operational.get("family_floor_minimum_count", 5)
            )
            or _at_least(row.get("recall"), 0.70)
            for row in families.values()
        ),
        "median_delay": _at_most(
            dual.get("median_delay_hours"),
            float(operational.get("decision_interval_hours", 0.0)),
        ),
        "p90_delay": _at_most(
            dual.get("p90_delay_hours"),
            3.0
            * float(operational.get("decision_interval_hours", 0.0)),
        ),
        "fragmentation": _at_most(
            dual.get("mean_extra_fragments"),
            0.05,
        ),
        "burden_point": _at_most(
            burden.get("episodes_per_100_healthy_hours"),
            0.10,
        ),
        "burden_upper": _at_most(
            burden.get("conservative_upper_95"),
            0.20,
        ),
        "deployment_heterogeneity_reported": (
            burden.get("heterogeneity", {}).get(
                "deployment_intervals_complete"
            )
            is True
            and burden.get("heterogeneity", {}).get(
                "minimum_exposure_rule_applied"
            )
            is True
        ),
        "cluster_claim_is_appropriately_limited": (
            (
                deployment_count < int(
                    operational.get(
                        "minimum_deployments_for_cluster_claim",
                        10,
                    )
                )
                and burden.get("heterogeneity", {}).get("inference_status")
                == "feasibility_only"
            )
            or (
                deployment_count
                >= int(
                    operational.get(
                        "minimum_deployments_for_cluster_claim",
                        10,
                    )
                )
                and burden.get("heterogeneity", {}).get(
                    "random_effects_sensitivity"
                )
                is not None
            )
        ),
        "data_quality_reported_separately": (
            operational.get("data_quality_alerts_reported_separately")
            is True
        ),
        "ranking_mrr": _at_least(ranking.get("mrr"), 0.70),
        "ranking_hits1": _at_least(ranking.get("hits_at_1"), 0.55),
        "ranking_hits3": _at_least(ranking.get("hits_at_3"), 0.85),
        "ranking_coverage": _at_least(
            ranking.get("ranking_coverage"),
            0.90,
        ),
        "unsupported_safe": (
            support.get("unsupported_safe_semantics") is True
            and support.get("unsafe_stage2_count") == 0
        ),
        "id_false_unsupported_rate": _at_most(
            operational.get("id_false_unsupported_rate"),
            0.05,
        ),
        "support_dimensions_complete": (
            operational.get("support_dimensions_complete") is True
        ),
        "schema_and_shift_separate": (
            operational.get("schema_shift_states_separate") is True
        ),
        "path_decision_complete": (
            path.get("recommendation")
            not in {None, "insufficient_evidence"}
        ),
        "all_known_incidents_reviewed": _equals_one(
            operational.get("known_incident_review_fraction")
        ),
        "all_nrim_positive_episodes_reviewed": _equals_one(
            operational.get("nrim_incident_review_fraction")
        ),
        "negative_sampling_probabilities_complete": _equals_one(
            operational.get(
                "negative_sampling_probability_record_fraction"
            )
        ),
        "unresolved_labels_preserved": (
            operational.get("unresolved_labels_preserved") is True
        ),
        "agreement_assessment_phase_appropriate": (
            adjudication.get("agreement_assessment_status")
            in {"interpretable", "not_reportable_preregistered_reason"}
        ),
        "no_human_utility_claim": (
            not report.get("human_utility")
            or report.get("human_utility", {}).get("case_count", 0) == 0
        ),
    }


def _advisory_criteria(
    *,
    report: Mapping[str, Any],
    operational: Mapping[str, Any],
    context: PilotContext,
) -> dict[str, bool]:
    ranking = report.get("ranking", {})
    utility = report.get("human_utility", {})
    return {
        "real_operator_advisory_evidence": (
            context.evidence_origin
            is EvidenceOrigin.REAL_OPERATOR_ADVISORY
        ),
        "workflow_comparison_preregistered": (
            operational.get("workflow_comparison_preregistered") is True
        ),
        "paired_or_rollout_aware_analysis": (
            utility.get("analysis_design")
            in {"paired", "randomized", "stepped_rollout"}
        ),
        "engineer_top3_usefulness": _at_least(
            ranking.get("engineer_top3_usefulness"),
            0.70,
        ),
        "time_to_hypothesis_improvement": _at_least(
            utility.get("time_to_hypothesis_median_improvement"),
            0.20,
        ),
        "components_investigated_reduction": _at_least(
            utility.get("components_investigated_median_reduction"),
            0.20,
        ),
        "override_reason_completeness": _equals_one(
            utility.get("override_reason_completeness")
        ),
        "automation_bias_assessed": (
            utility.get("automation_bias_assessed") is True
        ),
        "no_automatic_remediation": (
            operational.get("automatic_remediation_enabled") is False
        ),
    }


def _operational_criteria(
    operational: Mapping[str, Any],
    context: PilotContext,
) -> dict[str, bool]:
    return {
        "real_operational_assessment": (
            context.evidence_origin
            is EvidenceOrigin.REAL_OPERATIONAL_ASSESSMENT
        ),
        "security_assessment_approved": (
            operational.get("security_assessment_approved") is True
        ),
        "resilience_and_recovery_approved": (
            operational.get("resilience_recovery_approved") is True
        ),
        "change_management_approved": (
            operational.get("change_management_approved") is True
        ),
        "deployment_owner_approved": (
            operational.get("deployment_owner_approved") is True
        ),
        "privacy_owner_approved": (
            operational.get("privacy_owner_approved") is True
        ),
        "independent_reviewer_approved": (
            operational.get("independent_reviewer_approved") is True
        ),
        "no_autonomous_remediation": (
            operational.get("automatic_remediation_enabled") is False
        ),
    }


def evaluate_phase_transition(
    *,
    target_state: PromotionState,
    report: Mapping[str, Any],
    operational: Mapping[str, Any],
    context: PilotContext,
) -> ReleaseDecision:
    if target_state is PromotionState.CANDIDATE:
        raise ValueError("CANDIDATE is an initial state, not a promotion gate")
    criteria = _base_criteria(
        context=context,
        target_state=target_state,
    )
    if target_state is PromotionState.REAL_REPLAY_READY:
        criteria.update(_replay_ready_criteria(operational, context))
    elif target_state is PromotionState.SHADOW_VALIDATED:
        criteria.update(
            _silent_criteria(
                report=report,
                operational=operational,
                context=context,
            )
        )
    elif target_state is PromotionState.ADVISORY_VALIDATED:
        criteria.update(
            _advisory_criteria(
                report=report,
                operational=operational,
                context=context,
            )
        )
    elif target_state is PromotionState.OPERATIONAL_CANDIDATE:
        criteria.update(_operational_criteria(operational, context))
    else:
        raise ValueError(f"Unsupported target state: {target_state}")

    blockers = [name for name, passed in criteria.items() if not passed]
    blockers.extend(
        f"emergency_stop:{item}"
        for item in context.emergency_stop_events
    )
    passed = not blockers
    return ReleaseDecision(
        state=target_state if passed else context.current_state,
        target_state=target_state,
        passed=passed,
        blockers=tuple(blockers),
        criteria=criteria,
    )


def evaluate_release(
    *,
    report: Mapping[str, Any],
    operational: Mapping[str, Any],
    context: PilotContext,
    advisory_phase: bool = False,
) -> ReleaseDecision:
    """Compatibility wrapper around the phase-specific release evaluator."""

    target = (
        PromotionState.ADVISORY_VALIDATED
        if advisory_phase
        else PromotionState.SHADOW_VALIDATED
    )
    return evaluate_phase_transition(
        target_state=target,
        report=report,
        operational=operational,
        context=context,
    )
