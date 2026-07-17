from __future__ import annotations

from copy import deepcopy

from .fault_catalog import FAULT_CATALOG
from .models import (
    DeploymentTopology,
    FaultSpecification,
    HiddenNodeState,
    HealthState,
)


def validate_fault_target(
    *,
    topology: DeploymentTopology,
    fault: FaultSpecification,
) -> None:
    node_by_id = {
        node.node_id: node
        for node in topology.nodes
    }

    target = node_by_id.get(fault.target_node_id)

    if target is None:
        raise ValueError(
            f"Fault target does not exist: "
            f"{fault.target_node_id}"
        )

    definition = FAULT_CATALOG.get(fault.failure_type)

    if definition is None:
        raise ValueError(
            f"No fault definition exists for "
            f"{fault.failure_type.value}."
        )

    if target.node_type not in definition.valid_target_types:
        raise ValueError(
            f"Fault {fault.failure_type.value} cannot target "
            f"node type {target.node_type.value}."
        )


def inject_fault(
    *,
    topology: DeploymentTopology,
    fault: FaultSpecification,
    states: dict[str, HiddenNodeState],
) -> dict[str, HiddenNodeState]:
    validate_fault_target(
        topology=topology,
        fault=fault,
    )

    updated = deepcopy(states)
    target_state = updated[fault.target_node_id]

    severity = fault.severity

    if fault.failure_type.value == (
        "storage_capacity_saturation"
    ):
        target_state.latent_capacity_factor = max(
            0.0,
            1.0 - 0.70 * severity,
        )
        target_state.health_state = HealthState.DEGRADED

    elif fault.failure_type.value == (
        "storage_io_degradation"
    ):
        target_state.latent_latency_factor = (
            1.0 + 5.0 * severity
        )
        target_state.latent_error_factor = (
            1.0 + 8.0 * severity
        )
        target_state.health_state = HealthState.DEGRADED

    elif fault.failure_type.value == (
        "cleanup_job_failure"
    ):
        target_state.latent_error_factor = (
            1.0 + 3.0 * severity
        )
        target_state.health_state = HealthState.DEGRADED

    elif fault.failure_type.value == (
        "catchup_worker_failure"
    ):
        target_state.latent_error_factor = (
            1.0 + 10.0 * severity
        )
        target_state.health_state = HealthState.CRITICAL

    else:
        raise ValueError(
            f"Unsupported failure type: "
            f"{fault.failure_type.value}"
        )

    target_state.caused_by_fault_id = fault.fault_id

    return updated
