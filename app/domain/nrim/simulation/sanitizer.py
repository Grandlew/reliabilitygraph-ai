from __future__ import annotations

from copy import deepcopy
from typing import Any


FORBIDDEN_TOP_LEVEL_KEYS = {
    "simulation_metadata",
    "environment_id",
    "pair_id",
    "scenario_plan",
    "missingness_report",
}


FORBIDDEN_RECURSIVE_KEYS = {
    "confounder_type",
    "model_visible",
    "simulated_transient",
    "topology_seed",
    "workload_seed",
    "observation_seed",
    "fault_seed",
    "fault_id",
    "caused_by_fault_id",
    "hidden_state",
    "latent_capacity_factor",
    "latent_latency_factor",
    "latent_error_factor",
    "propagation_records",
    "root_cause_node_id",
    "incident_onset_time",
}


FORBIDDEN_ATTRIBUTE_KEYS = {
    "simulated_transient",
    "confounder_type",
    "fault_type",
    "root_cause",
}


def recursively_remove_keys(
    value: Any,
    forbidden_keys: set[str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: recursively_remove_keys(
                child,
                forbidden_keys,
            )
            for key, child in value.items()
            if key not in forbidden_keys
        }

    if isinstance(value, list):
        return [
            recursively_remove_keys(
                child,
                forbidden_keys,
            )
            for child in value
        ]

    return value


def sanitize_attributes(
    attributes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []

    for attribute in attributes:
        key = str(attribute.get("key", ""))

        if key in FORBIDDEN_ATTRIBUTE_KEYS:
            continue

        clean_attribute = recursively_remove_keys(
            attribute,
            FORBIDDEN_RECURSIVE_KEYS,
        )

        sanitized.append(clean_attribute)

    return sanitized


def sanitize_telemetry_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    sanitized = recursively_remove_keys(
        deepcopy(event),
        FORBIDDEN_RECURSIVE_KEYS,
    )

    attributes = sanitized.get("attributes", [])

    if isinstance(attributes, list):
        sanitized["attributes"] = sanitize_attributes(
            attributes
        )

    return sanitized


def sanitize_context_event(
    event: dict[str, Any],
) -> dict[str, Any] | None:
    sanitized = recursively_remove_keys(
        deepcopy(event),
        FORBIDDEN_RECURSIVE_KEYS,
    )

    allowed = {
        "event_id",
        "signal_name",
        "observed_at",
        "statement",
        "value",
        "unit",
        "component_node_id",
    }

    sanitized = {
        key: value
        for key, value in sanitized.items()
        if key in allowed
    }

    required = {
        "signal_name",
        "observed_at",
    }

    if not required.issubset(sanitized):
        return None

    return sanitized


def sanitize_observable_scenario(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    sanitized = deepcopy(scenario)

    for key in FORBIDDEN_TOP_LEVEL_KEYS:
        sanitized.pop(key, None)

    sanitized = recursively_remove_keys(
        sanitized,
        FORBIDDEN_RECURSIVE_KEYS,
    )

    telemetry = sanitized.get("telemetry", [])

    sanitized["telemetry"] = [
        sanitize_telemetry_event(event)
        for event in telemetry
        if isinstance(event, dict)
    ]

    context_events = sanitized.get(
        "context_events",
        [],
    )

    sanitized_context = []

    for event in context_events:
        if not isinstance(event, dict):
            continue

        sanitized_event = sanitize_context_event(event)

        if sanitized_event is not None:
            sanitized_context.append(
                sanitized_event
            )

    sanitized["context_events"] = sanitized_context

    return sanitized
