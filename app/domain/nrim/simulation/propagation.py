from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from heapq import heappop, heappush

from .models import (
    DeploymentTopology,
    FaultSpecification,
    HealthState,
    HiddenNodeState,
    PropagationRecord,
    SimulationEdge,
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

    outgoing: dict[str, list[SimulationEdge]] = {}

    for edge in topology.edges:
        if edge.edge_type not in ALLOWED_PROPAGATION_EDGES:
            continue

        outgoing.setdefault(
            edge.target_node_id,
            [],
        ).append(edge)

    queue: list[tuple[float, int, str, datetime, SimulationEdge | None]] = []
    sequence = 0
    heappush(
        queue,
        (
            -fault.severity,
            sequence,
            fault.target_node_id,
            timestamp,
            None,
        ),
    )

    visited_nodes: set[str] = set()
    recorded_edges: set[str] = set()

    while queue:
        (
            negative_strength,
            _,
            current_node_id,
            current_time,
            incoming_edge,
        ) = heappop(queue)
        current_strength = -negative_strength

        if current_node_id in visited_nodes:
            continue

        visited_nodes.add(current_node_id)

        if incoming_edge is not None:
            target_state = updated[current_node_id]
            target_state.latent_error_factor *= (
                1.0 + current_strength
            )

            if current_strength >= 0.75:
                target_state.health_state = HealthState.CRITICAL
            else:
                target_state.health_state = HealthState.DEGRADED

            target_state.caused_by_fault_id = fault.fault_id

            if incoming_edge.edge_id not in recorded_edges:
                records.append(
                    PropagationRecord(
                        source_node_id=incoming_edge.target_node_id,
                        target_node_id=current_node_id,
                        edge_id=incoming_edge.edge_id,
                        fault_id=fault.fault_id,
                        propagation_started_at=current_time,
                        effect_strength=current_strength,
                    )
                )
                recorded_edges.add(incoming_edge.edge_id)

        for edge in outgoing.get(current_node_id, []):
            if edge.source_node_id in visited_nodes:
                continue

            propagated_strength = (
                current_strength
                * edge.propagation_strength
                * 0.75
            )

            if propagated_strength < minimum_strength:
                continue

            propagation_time = current_time + timedelta(
                minutes=edge.propagation_delay_minutes
            )
            sequence += 1
            heappush(
                queue,
                (
                    -propagated_strength,
                    sequence,
                    edge.source_node_id,
                    propagation_time,
                    edge,
                )
            )

    return updated, records
