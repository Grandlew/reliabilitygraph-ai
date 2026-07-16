from __future__ import annotations

from .reasoning import DiagnosticOption, InterventionRisk


def calculate_diagnostic_utility(
    diagnostic: DiagnosticOption,
) -> float:
    score = (
        0.40 * diagnostic.information_gain
        + 0.20 * diagnostic.hypothesis_separation
        + 0.15 * diagnostic.safety
        + 0.10 * diagnostic.reversibility
        + 0.10 * diagnostic.speed
        + 0.05 * diagnostic.low_cost
    )

    if diagnostic.operational_risk in {
        InterventionRisk.HIGH,
        InterventionRisk.PROHIBITED_AUTONOMOUSLY,
    }:
        score *= 0.25

    return round(score, 4)


def create_catchup_diagnostic_options() -> list[DiagnosticOption]:
    options = [
        DiagnosticOption(
            diagnostic_id="diag_storage_capacity",
            name="Inspect CatchUP storage capacity and growth",
            question=(
                "What are current free capacity, recent growth rate, "
                "retention cleanup status, and allocation limits?"
            ),
            hypothesis_ids=[
                "hyp_storage_capacity",
                "hyp_storage_io",
            ],
            information_gain=0.92,
            hypothesis_separation=0.80,
            safety=1.00,
            reversibility=1.00,
            speed=0.90,
            low_cost=0.95,
            operational_risk=InterventionRisk.OBSERVE_ONLY,
            requires_engineer=False,
            expected_evidence=[
                "Free storage capacity",
                "Storage growth rate",
                "Cleanup-job status",
                "Storage allocation limits",
            ],
        ),
        DiagnosticOption(
            diagnostic_id="diag_storage_io",
            name="Inspect storage I/O health",
            question=(
                "Are write latency, I/O queue depth, filesystem errors, "
                "or disk errors abnormal?"
            ),
            hypothesis_ids=[
                "hyp_storage_io",
                "hyp_storage_capacity",
            ],
            information_gain=0.90,
            hypothesis_separation=0.85,
            safety=1.00,
            reversibility=1.00,
            speed=0.85,
            low_cost=0.90,
            operational_risk=InterventionRisk.OBSERVE_ONLY,
            requires_engineer=False,
            expected_evidence=[
                "Write latency",
                "I/O queue depth",
                "Filesystem errors",
                "Disk health",
            ],
        ),
        DiagnosticOption(
            diagnostic_id="diag_recording_worker",
            name="Inspect CatchUP recording-worker health",
            question=(
                "Are recording workers running, restarting, blocked, "
                "or reporting exceptions?"
            ),
            hypothesis_ids=[
                "hyp_catchup_application",
                "hyp_storage_io",
            ],
            information_gain=0.82,
            hypothesis_separation=0.90,
            safety=1.00,
            reversibility=1.00,
            speed=0.80,
            low_cost=0.90,
            operational_risk=InterventionRisk.OBSERVE_ONLY,
            requires_engineer=False,
            expected_evidence=[
                "Worker process state",
                "Restart count",
                "Application exceptions",
            ],
        ),
        DiagnosticOption(
            diagnostic_id="diag_reduce_retention",
            name="Temporarily reduce CatchUP retention",
            question=(
                "Does reducing retention restore reliable recording?"
            ),
            hypothesis_ids=[
                "hyp_storage_capacity",
            ],
            information_gain=0.95,
            hypothesis_separation=0.95,
            safety=0.40,
            reversibility=0.85,
            speed=0.55,
            low_cost=0.80,
            operational_risk=InterventionRisk.MEDIUM,
            requires_engineer=True,
            expected_evidence=[
                "Recording success after retention change",
                "Storage utilization after cleanup",
            ],
        ),
    ]

    return [
        option.model_copy(
            update={
                "utility_score": calculate_diagnostic_utility(option),
            }
        )
        for option in options
    ]


def rank_diagnostics(
    diagnostics: list[DiagnosticOption],
    *,
    leading_hypothesis_ids: set[str],
) -> list[DiagnosticOption]:
    relevant = [
        diagnostic
        for diagnostic in diagnostics
        if set(diagnostic.hypothesis_ids)
        & leading_hypothesis_ids
    ]

    return sorted(
        relevant,
        key=lambda item: item.utility_score,
        reverse=True,
    )
