from datetime import datetime, timezone

from app.domain.nrim.simulation.labeler import (
    build_fault_ground_truth,
    build_healthy_ground_truth,
    root_cause_node_labels,
)
from app.domain.nrim.simulation.models import (
    FailureType,
    FaultSpecification,
)


def test_healthy_scenario_has_no_positive_root_cause() -> None:
    ground_truth = build_healthy_ground_truth(
        scenario_id="scenario_a",
        random_seed=42,
    )

    labels = root_cause_node_labels(
        node_ids=["node_1", "node_2"],
        ground_truth=ground_truth,
    )

    assert sum(labels.values()) == 0


def test_fault_scenario_has_one_positive_root_cause() -> None:
    now = datetime.now(timezone.utc)

    fault = FaultSpecification(
        failure_type=(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        target_node_id="storage_1",
        injection_time=now,
        severity=0.8,
    )

    ground_truth = build_fault_ground_truth(
        scenario_id="scenario_b",
        fault=fault,
        propagation_records=[],
        affected_service_node_ids=["catchup_1"],
        incident_onset_time=now,
        random_seed=42,
    )

    labels = root_cause_node_labels(
        node_ids=[
            "storage_1",
            "catchup_1",
            "middleware_1",
        ],
        ground_truth=ground_truth,
    )

    assert labels["storage_1"] == 1
    assert sum(labels.values()) == 1
