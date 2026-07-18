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

    current_incident = bool(
        incident_onset is not None
        and incident_onset
        <= boundary.observation_cutoff
    )

    future_incident = bool(
        incident_onset is not None
        and boundary.observation_cutoff
        < incident_onset
        <= boundary.prediction_end
    )

    if incident_onset is None:
        time_to_incident_hours = None
        time_to_incident_mask = 0.0
    elif current_incident:
        time_to_incident_hours = 0.0
        time_to_incident_mask = 1.0
    else:
        time_to_incident_hours = (
            incident_onset
            - boundary.observation_cutoff
        ).total_seconds() / 3600.0

        time_to_incident_mask = float(
            incident_onset
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
