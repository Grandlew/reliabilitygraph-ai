from __future__ import annotations

from .reasoning import (
    FailureHypothesis,
    HypothesisStatus,
    InterventionRisk,
    RecommendedIntervention,
)


def create_interventions(
    hypothesis: FailureHypothesis,
) -> list[RecommendedIntervention]:
    if hypothesis.status not in {
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.LEADING,
        HypothesisStatus.CONFIRMED,
    }:
        return []

    if hypothesis.failure_type == "storage_capacity_saturation":
        return [
            RecommendedIntervention(
                intervention_id="intervention_storage_observe",
                name="Inspect and validate CatchUP storage capacity",
                hypothesis_id=hypothesis.hypothesis_id,
                target_node_ids=hypothesis.target_node_ids,
                rationale=(
                    "The current evidence suggests storage capacity "
                    "may be contributing to recording failures."
                ),
                expected_effect=(
                    "Determine whether storage capacity is genuinely "
                    "constraining CatchUP recording."
                ),
                operational_risk=InterventionRisk.OBSERVE_ONLY,
                reversible=True,
                rollback_plan="No system state is changed.",
                requires_engineer_approval=False,
                verification_test=(
                    "Confirm free capacity, growth rate, cleanup state, "
                    "and allocation limits."
                ),
            ),
            RecommendedIntervention(
                intervention_id="intervention_storage_capacity_change",
                name="Adjust retention or expand storage after approval",
                hypothesis_id=hypothesis.hypothesis_id,
                target_node_ids=hypothesis.target_node_ids,
                rationale=(
                    "If capacity saturation is confirmed, storage demand "
                    "must be reduced or capacity increased."
                ),
                expected_effect=(
                    "Restore stable CatchUP recording and prevent "
                    "capacity exhaustion."
                ),
                operational_risk=InterventionRisk.MEDIUM,
                reversible=True,
                rollback_plan=(
                    "Restore the previous retention policy or remove "
                    "the newly introduced storage configuration if the "
                    "change causes degradation."
                ),
                requires_engineer_approval=True,
                verification_test=(
                    "Compare recording success, storage utilization, "
                    "and write latency before and after the intervention."
                ),
            ),
        ]

    if hypothesis.failure_type == "storage_io_degradation":
        return [
            RecommendedIntervention(
                intervention_id="intervention_storage_io_inspection",
                name="Inspect storage I/O and hardware health",
                hypothesis_id=hypothesis.hypothesis_id,
                target_node_ids=hypothesis.target_node_ids,
                rationale=(
                    "Elevated I/O latency and recording failures may "
                    "indicate storage-path degradation."
                ),
                expected_effect=(
                    "Identify whether latency comes from saturation, "
                    "filesystem errors, hardware problems, or queueing."
                ),
                operational_risk=InterventionRisk.OBSERVE_ONLY,
                reversible=True,
                rollback_plan="No system state is changed.",
                requires_engineer_approval=False,
                verification_test=(
                    "Inspect disk health, filesystem errors, queue depth, "
                    "and write throughput."
                ),
            )
        ]

    if hypothesis.failure_type == "catchup_application_failure":
        return [
            RecommendedIntervention(
                intervention_id="intervention_worker_inspection",
                name="Inspect CatchUP recording-worker state",
                hypothesis_id=hypothesis.hypothesis_id,
                target_node_ids=hypothesis.target_node_ids,
                rationale=(
                    "The failure may be isolated to the application or "
                    "recording workers."
                ),
                expected_effect=(
                    "Identify worker crashes, blocked jobs, exceptions, "
                    "or configuration failures."
                ),
                operational_risk=InterventionRisk.OBSERVE_ONLY,
                reversible=True,
                rollback_plan="No system state is changed.",
                requires_engineer_approval=False,
                verification_test=(
                    "Inspect worker state, restart count, application "
                    "exceptions, and per-channel recording results."
                ),
            )
        ]

    return []
