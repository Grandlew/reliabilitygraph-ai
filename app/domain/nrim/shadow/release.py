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


class PromotionState(str, Enum):
    CANDIDATE = "candidate"
    SHADOW_VALIDATED = "shadow_validated"
    OPERATOR_ADVISORY = "operator_advisory"


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
    passed: bool
    blockers: tuple[str, ...]
    criteria: Mapping[str, bool]


def _at_least(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def evaluate_release(
    *,
    report: Mapping[str, Any],
    operational: Mapping[str, Any],
    context: PilotContext,
    advisory_phase: bool = False,
) -> ReleaseDecision:
    """Apply the preregistered v0.7 gates without synthetic promotion."""

    dual = report.get("incident_detection", {}).get("D", {})
    burden = report.get("alert_burden", {}).get("D", {})
    ranking = report.get("ranking", {})
    support = report.get("support_and_data_quality", {})
    path = report.get("path_attribution", {})
    adjudication = report.get("adjudication", {})
    utility = report.get("human_utility", {})
    families = dual.get("failure_families", {})

    criteria = {
        "real_prospective_evidence": (
            context.evidence_origin
            in {
                EvidenceOrigin.REAL_PROSPECTIVE_SILENT,
                EvidenceOrigin.REAL_OPERATOR_ADVISORY,
            }
        ),
        "protocol_preregistered": (
            context.protocol_registered_before_outcomes
        ),
        "no_pilot_tuning": context.pilot_tuning_events == 0,
        "minimum_8_weeks": context.elapsed_weeks >= 8.0,
        "minimum_5_deployments": (
            context.independent_deployment_count >= 5
        ),
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
        "invalid_categories_quarantined": (
            operational.get(
                "invalid_required_category_quarantined_fraction"
            )
            == 1.0
        ),
        "feature_parity": (
            operational.get("feature_parity_fraction") == 1.0
        ),
        "zero_future_leakage": (
            operational.get("future_leakage_count") == 0
        ),
        "prediction_reproducibility": (
            operational.get("prediction_reproducibility_fraction")
            == 1.0
        ),
        "topology_provenance": (
            operational.get("topology_provenance_fraction") == 1.0
        ),
        "zero_audit_field_leakage": (
            operational.get("audit_field_leakage_count") == 0
        ),
        "runtime_availability": _at_least(
            operational.get("scheduled_cutoff_completion_fraction"),
            0.995,
        ),
        "evidence_durability": (
            operational.get("evidence_durability_fraction") == 1.0
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
        "state_recovery": (
            operational.get("state_recovery_parity_fraction") == 1.0
        ),
        "episode_recall": _at_least(
            dual.get("episode_recall"),
            0.80,
        ),
        "severe_recall": (
            dual.get("severe_incident_count", 0) > 0
            and dual.get("severe_recall") == 1.0
        ),
        "family_recall": bool(families)
        and all(
            row.get("incident_count", 0) < 5
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
        "burden_concentration": _at_most(
            burden.get("maximum_deployment_concentration_ratio"),
            2.0,
        ),
        "data_quality_reported_separately": (
            operational.get("data_quality_alerts_reported_separately")
            is True
        ),
        "ranking_mrr": _at_least(ranking.get("mrr"), 0.70),
        "ranking_hits1": _at_least(
            ranking.get("hits_at_1"),
            0.55,
        ),
        "ranking_hits3": _at_least(
            ranking.get("hits_at_3"),
            0.85,
        ),
        "ranking_coverage": _at_least(
            ranking.get("ranking_coverage"),
            0.90,
        ),
        "static_prior_incremental_value": (
            operational.get("static_prior_incremental_value") is True
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
        "adjudication_agreement": (
            adjudication.get("impact_kappa") is not None
            and adjudication.get("impact_kappa") >= 0.70
            and adjudication.get("cause_family_kappa") is not None
            and adjudication.get("cause_family_kappa") >= 0.70
        ),
        "confirmed_incident_adjudication_complete": (
            operational.get(
                "confirmed_incident_adjudication_fraction"
            )
            == 1.0
        ),
        "nrim_incident_review_complete": (
            operational.get("nrim_incident_review_fraction") == 1.0
        ),
    }
    if advisory_phase:
        criteria.update(
            {
                "engineer_top3_usefulness": _at_least(
                    ranking.get("engineer_top3_usefulness"),
                    0.70,
                ),
                "time_to_hypothesis_improvement": _at_least(
                    utility.get(
                        "time_to_hypothesis_median_improvement"
                    ),
                    0.20,
                ),
                "components_investigated_reduction": _at_least(
                    utility.get(
                        "components_investigated_median_reduction"
                    ),
                    0.20,
                ),
                "override_reason_completeness": (
                    utility.get("override_reason_completeness") == 1.0
                ),
            }
        )
    blockers = [
        name for name, passed in criteria.items() if not passed
    ]
    blockers.extend(
        f"emergency_stop:{item}"
        for item in context.emergency_stop_events
    )
    passed = not blockers
    if passed and advisory_phase:
        state = PromotionState.OPERATOR_ADVISORY
    elif passed:
        state = PromotionState.SHADOW_VALIDATED
    else:
        state = PromotionState.CANDIDATE
    return ReleaseDecision(
        state=state,
        passed=passed,
        blockers=tuple(blockers),
        criteria=criteria,
    )
