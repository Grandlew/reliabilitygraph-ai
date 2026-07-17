from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .models import FailureType


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


def recursively_find_strings(
    value: Any,
    *,
    path: str = "root",
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(
                recursively_find_strings(
                    child,
                    path=f"{path}.{key}",
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                recursively_find_strings(
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

    failure_names = {
        variant.casefold()
        for failure_type in FailureType
        if failure_type != FailureType.HEALTHY
        for variant in (
            failure_type.value,
            failure_type.value.replace("_", " "),
            failure_type.value.replace(
                "_",
                " ",
            ).replace("storage io", "storage i/o"),
            failure_type.name,
        )
    }
    for path, value in recursively_find_strings(
        scenario.get("telemetry", []),
        path="root.telemetry",
    ):
        normalized = value.casefold()
        if any(name in normalized for name in failure_names):
            warnings.append(
                "Observable telemetry text contains a failure class "
                f"name at {path}."
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

    raw_failure_type = (
        scenario.get("labels", {})
        .get("failure_type")
    )

    try:
        failure_type = FailureType(raw_failure_type)
    except (TypeError, ValueError):
        errors.append(
            f"Unknown failure type label: {raw_failure_type!r}."
        )
        return errors

    expected = 0 if failure_type == FailureType.HEALTHY else 1

    if positive_count != expected:
        errors.append(
            f"Expected {expected} root-cause positives; "
            f"found {positive_count}."
        )

    return errors
