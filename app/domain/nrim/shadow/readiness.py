from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class TaskState(str, Enum):
    COMPLETE = "complete"
    IMPLEMENTED_NOT_OPERATIONALLY_VALIDATED = (
        "implemented_not_operationally_validated"
    )
    EXTERNAL_EVIDENCE_REQUIRED = "external_evidence_required"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BacklogTaskStatus:
    task_number: int
    name: str
    state: TaskState
    evidence: tuple[str, ...]
    blocker: str | None = None


def historical_replay_gate(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "required_features_reconstructable": (
            metrics.get("required_feature_fraction", 0.0) >= 0.95
        ),
        "incident_alignment": (
            metrics.get("incident_alignment_fraction", 0.0) >= 0.90
        ),
        "future_leakage_zero": (
            metrics.get("future_leakage_count") == 0
        ),
        "replay_determinism": (
            metrics.get("replay_determinism_fraction") == 1.0
        ),
        "root_cause_mapping": (
            metrics.get("root_cause_mapping_fraction", 0.0) >= 0.95
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "truth_note": (
            "Historical replay is a readiness check, not prospective "
            "operational efficacy evidence."
        ),
    }


def gate_progress(
    tasks: tuple[BacklogTaskStatus, ...],
) -> dict[str, Any]:
    by_number = {task.task_number: task for task in tasks}

    def complete(numbers: range) -> bool:
        return all(
            by_number.get(number) is not None
            and by_number[number].state is TaskState.COMPLETE
            for number in numbers
        )

    return {
        "gate_A_contracts_ready": complete(range(1, 6)),
        "gate_B_replay_ready": complete(range(6, 11)),
        "gate_C_shadow_runtime_ready": complete(range(11, 16)),
        "gate_D_evidence_workflow_ready": complete(range(16, 19)),
        "gate_E_pilot_ready": complete(range(19, 21)),
    }
