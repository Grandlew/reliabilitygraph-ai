from __future__ import annotations

from datetime import datetime

from .models import (
    FailureType,
    FaultSpecification,
    PropagationRecord,
    ScenarioGroundTruth,
    ScenarioKind,
)


def build_healthy_ground_truth(
    *,
    scenario_id: str,
    random_seed: int,
    simulator_version: str = "0.1.0",
    labeling_version: str = "0.1.0",
) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        scenario_id=scenario_id,
        scenario_kind=ScenarioKind.HEALTHY_CONTROL,
        failure_type=FailureType.HEALTHY,
        root_cause_node_id=None,
        fault_id=None,
        injection_time=None,
        incident_onset_time=None,
        affected_service_node_ids=[],
        propagation_edge_ids=[],
        simulator_version=simulator_version,
        labeling_version=labeling_version,
        random_seed=random_seed,
    )


def build_fault_ground_truth(
    *,
    scenario_id: str,
    fault: FaultSpecification,
    propagation_records: list[PropagationRecord],
    affected_service_node_ids: list[str],
    incident_onset_time: datetime | None,
    random_seed: int,
    simulator_version: str = "0.1.0",
    labeling_version: str = "0.1.0",
) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        scenario_id=scenario_id,
        scenario_kind=ScenarioKind.SINGLE_FAULT,
        failure_type=fault.failure_type,
        root_cause_node_id=fault.target_node_id,
        fault_id=fault.fault_id,
        injection_time=fault.injection_time,
        incident_onset_time=incident_onset_time,
        affected_service_node_ids=sorted(
            set(affected_service_node_ids)
        ),
        propagation_edge_ids=sorted(
            {
                record.edge_id
                for record in propagation_records
            }
        ),
        simulator_version=simulator_version,
        labeling_version=labeling_version,
        random_seed=random_seed,
    )


def root_cause_node_labels(
    *,
    node_ids: list[str],
    ground_truth: ScenarioGroundTruth,
) -> dict[str, int]:
    labels = {
        node_id: int(
            node_id == ground_truth.root_cause_node_id
        )
        for node_id in node_ids
    }

    expected_positive_count = (
        0
        if ground_truth.scenario_kind
        == ScenarioKind.HEALTHY_CONTROL
        else 1
    )

    actual_positive_count = sum(labels.values())

    if actual_positive_count != expected_positive_count:
        raise ValueError(
            "Root-cause label count does not match scenario kind."
        )

    return labels
