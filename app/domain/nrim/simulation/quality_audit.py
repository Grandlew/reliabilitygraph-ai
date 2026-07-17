from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetAudit:
    scenario_count: int = 0
    failure_type_counts: Counter[str] = field(
        default_factory=Counter
    )
    topology_counts: Counter[str] = field(
        default_factory=Counter
    )

    leakage_warnings: list[str] = field(
        default_factory=list
    )
    integrity_errors: list[str] = field(
        default_factory=list
    )


FORBIDDEN_OBSERVABLE_KEYS = {
    "hidden_state",
    "fault_parameters",
    "caused_by_fault_id",
    "propagation_records",
    "latent_capacity_factor",
    "latent_latency_factor",
    "latent_error_factor",
}


def recursively_find_keys(
    value: Any,
    *,
    path: str = "root",
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            found.append((child_path, key))
            found.extend(
                recursively_find_keys(
                    child,
                    path=child_path,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                recursively_find_keys(
                    child,
                    path=f"{path}[{index}]",
                )
            )

    return found


def audit_observable_scenario(
    scenario: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []

    for path, key in recursively_find_keys(scenario):
        if key in FORBIDDEN_OBSERVABLE_KEYS:
            warnings.append(
                f"Forbidden hidden key found at {path}."
            )

    scenario_id = str(
        scenario.get("scenario_id", "")
    ).lower()

    forbidden_id_tokens = {
        "storage_failure",
        "io_degradation",
        "cleanup_failure",
        "worker_failure",
        "healthy",
    }

    for token in forbidden_id_tokens:
        if token in scenario_id:
            warnings.append(
                "Scenario ID appears to encode the class: "
                f"{scenario_id}"
            )

    return warnings


def audit_root_cause_labels(
    scenario: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    labels = (
        scenario.get("labels", {})
        .get("root_cause_node", {})
    )

    positive_count = sum(
        int(value)
        for value in labels.values()
    )

    failure_type = (
        scenario.get("labels", {})
        .get("failure_type")
    )

    expected = 0 if failure_type == "healthy" else 1

    if positive_count != expected:
        errors.append(
            f"Expected {expected} root-cause positives; "
            f"found {positive_count}."
        )

    return errors
