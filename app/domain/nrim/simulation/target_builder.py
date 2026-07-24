from __future__ import annotations

from datetime import datetime
from typing import Any

from .temporal_windowing import (
    TemporalWindowBoundary,
    parse_timestamp,
)


def extract_ground_truth(
    hidden_scenario: dict[str, Any],
) -> dict[str, Any]:
    ground_truth = hidden_scenario.get(
        "ground_truth"
    )

    if not isinstance(ground_truth, dict):
        raise ValueError(
            "Hidden scenario has no ground truth."
        )

    return ground_truth


def build_window_targets(
    *,
    node_ids: list[str],
    hidden_scenario: dict[str, Any],
    boundary: TemporalWindowBoundary,
) -> dict[str, Any]:
    ground_truth = extract_ground_truth(
        hidden_scenario
    )

    root_cause_node_id = ground_truth.get(
        "root_cause_node_id"
    )

    incident_onset_raw = ground_truth.get(
        "incident_onset_time"
    )

    incident_onset = (
        parse_timestamp(
            str(incident_onset_raw)
        )
        if incident_onset_raw
        else None
    )

    impact_times = sorted(
        parse_timestamp(str(value))
        for value in ground_truth.get(
            "observable_impact_times",
            [],
        )
    )
    recovery_raw = ground_truth.get(
        "recovery_time"
    )
    recovery_time = (
        parse_timestamp(str(recovery_raw))
        if recovery_raw
        else None
    )
    # Backward compatibility for older generated scenarios.
    if not impact_times and incident_onset is not None:
        impact_times = [incident_onset]

    current_incident = any(
        boundary.observation_start
        <= impact_time
        <= boundary.observation_cutoff
        for impact_time in impact_times
    )

    future_incident = any(
        boundary.observation_cutoff
        < impact_time
        <= boundary.prediction_end
        for impact_time in impact_times
    )

    if incident_onset is None:
        time_to_incident_hours = None
        time_to_incident_mask = 0.0
    elif current_incident:
        time_to_incident_hours = 0.0
        time_to_incident_mask = 1.0
    else:
        next_impacts = [
            impact_time
            for impact_time in impact_times
            if impact_time > boundary.observation_cutoff
        ]
        time_to_incident_hours = (
            (
                min(next_impacts)
                - boundary.observation_cutoff
            ).total_seconds()
            / 3600.0
            if next_impacts
            else None
        )

        time_to_incident_mask = float(
            bool(next_impacts)
            and min(next_impacts)
            <= boundary.prediction_end
        )

    affected_service_ids = set(
        ground_truth.get(
            "affected_service_node_ids",
            [],
        )
    )

    root_cause_labels = [
        int(node_id == root_cause_node_id)
        for node_id in node_ids
    ]

    affected_service_labels = [
        int(node_id in affected_service_ids)
        for node_id in node_ids
    ]

    if root_cause_node_id is None:
        if sum(root_cause_labels) != 0:
            raise ValueError(
                "Healthy target has positive root cause."
            )
    else:
        if sum(root_cause_labels) != 1:
            raise ValueError(
                "Fault target must contain one root cause."
            )

    return {
        "root_cause_node": root_cause_labels,
        "current_incident": int(
            current_incident
        ),
        "fault_present": int(
            ground_truth.get("fault_id") is not None
            and ground_truth.get("injection_time") is not None
            and parse_timestamp(
                str(ground_truth["injection_time"])
            )
            <= boundary.observation_cutoff
            and (
                recovery_time is None
                or boundary.observation_cutoff
                < recovery_time
            )
        ),
        "future_incident": int(
            future_incident
        ),
        "failure_type": ground_truth[
            "failure_type"
        ],
        "affected_service_node": (
            affected_service_labels
        ),
        "time_to_incident_hours": (
            time_to_incident_hours
        ),
        "time_to_incident_mask": (
            time_to_incident_mask
        ),
    }
