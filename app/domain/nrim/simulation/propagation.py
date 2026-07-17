from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta

from .models import (
    DeploymentTopology,
    FaultSpecification,
    HealthState,
    HiddenNodeState,
    PropagationRecord,
    SimulationEdgeType,
)


ALLOWED_PROPAGATION_EDGES = {
    SimulationEdgeType.DEPENDS_ON,
    SimulationEdgeType.STORES_ON,
    SimulationEdgeType.SERVES,
    SimulationEdgeType.CONNECTED_TO,
}


def propagate_fault(
    *,
    topology: DeploymentTopology,
    fault: FaultSpecification,
    states: dict[str, HiddenNodeState],
    timestamp: datetime,
    minimum_strength: float = 0.20,
) -> tuple[
    dict[str, HiddenNodeState],
    list[PropagationRecord],
]:
    updated = deepcopy(states)
    records: list[PropagationRecord] = []

    outgoing: dict[str, list] = {}

    for edge in topology.edges:
        if edge.edge_type not in ALLOWED_PROPAGATION_EDGES:
            continue

        outgoing.setdefault(
            edge.target_node_id,
            [],
        ).append(edge)

    queue: deque[tuple[str, float, datetime]] = deque(
        [
            (
                fault.target_node_id,
                fault.severity,
                timestamp,
            )
        ]
    )

    visited_strength: dict[str, float] = {
        fault.target_node_id: fault.severity
    }

    while queue:
        current_node_id, current_strength, current_time = (
            queue.popleft()
        )

        for edge in outgoing.get(current_node_id, []):
            propagated_strength = (
                current_strength
                * edge.propagation_strength
                * 0.75
            )

            if propagated_strength < minimum_strength:
                continue

            previous_strength = visited_strength.get(
                edge.source_node_id,
                0.0,
            )

            if propagated_strength <= previous_strength:
                continue

            propagation_time = current_time + timedelta(
                minutes=edge.propagation_delay_minutes
            )

            target_state = updated[edge.source_node_id]
            target_state.latent_error_factor *= (
                1.0 + propagated_strength
            )

            if propagated_strength >= 0.75:
                target_state.health_state = HealthState.CRITICAL
            else:
                target_state.health_state = HealthState.DEGRADED

            target_state.caused_by_fault_id = fault.fault_id

            records.append(
                PropagationRecord(
                    source_node_id=current_node_id,
                    target_node_id=edge.source_node_id,
                    edge_id=edge.edge_id,
                    fault_id=fault.fault_id,
                    propagation_started_at=propagation_time,
                    effect_strength=propagated_strength,
                )
            )

            visited_strength[edge.source_node_id] = (
                propagated_strength
            )

            queue.append(
                (
                    edge.source_node_id,
                    propagated_strength,
                    propagation_time,
                )
            )

    return updated, records
