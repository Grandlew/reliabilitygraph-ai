from datetime import datetime, timezone

from app.domain.nrim.simulation.dataset_builder import (
    initialize_states,
)
from app.domain.nrim.simulation.models import (
    DeploymentTopology,
    FailureType,
    FaultSpecification,
    SimulationEdge,
    SimulationEdgeType,
    SimulationNode,
    SimulationNodeType,
)
from app.domain.nrim.simulation.propagation import (
    propagate_fault,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def test_storage_fault_propagates_to_catchup_service() -> None:
    topology = generate_hotel_topology(
        room_count=100,
        floor_count=4,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )

    now = datetime.now(timezone.utc)

    states = initialize_states(
        topology=topology,
        timestamp=now,
    )

    fault = FaultSpecification(
        failure_type=(
            FailureType.STORAGE_IO_DEGRADATION
        ),
        target_node_id="catchup_storage_1",
        injection_time=now,
        severity=0.9,
    )

    updated, records = propagate_fault(
        topology=topology,
        fault=fault,
        states=states,
        timestamp=now,
    )

    assert records
    assert any(
        record.target_node_id
        == "catchup_service_1"
        for record in records
    )

    assert (
        updated["catchup_service_1"]
        .latent_error_factor
        > 1.0
    )


def test_propagation_ignores_cycles_and_parallel_paths() -> None:
    service = SimulationNode(
        node_id="service",
        node_type=SimulationNodeType.CATCHUP_SERVICE,
        name="CatchUP service",
    )
    storage = SimulationNode(
        node_id="storage",
        node_type=SimulationNodeType.CATCHUP_STORAGE,
        name="Storage",
    )
    topology = DeploymentTopology(
        deployment_family="test",
        room_count=10,
        floor_count=1,
        nodes=[service, storage],
        edges=[
            SimulationEdge(
                source_node_id="service",
                target_node_id="storage",
                edge_type=SimulationEdgeType.STORES_ON,
            ),
            SimulationEdge(
                source_node_id="service",
                target_node_id="storage",
                edge_type=SimulationEdgeType.CONNECTED_TO,
                propagation_strength=0.8,
            ),
            SimulationEdge(
                source_node_id="storage",
                target_node_id="service",
                edge_type=SimulationEdgeType.CONNECTED_TO,
            ),
        ],
    )
    now = datetime.now(timezone.utc)
    states = initialize_states(topology=topology, timestamp=now)
    fault = FaultSpecification(
        failure_type=FailureType.STORAGE_IO_DEGRADATION,
        target_node_id="storage",
        injection_time=now,
        severity=1.0,
    )

    _, records = propagate_fault(
        topology=topology,
        fault=fault,
        states=states,
        timestamp=now,
    )

    assert len(records) == 1
    assert records[0].target_node_id == "service"
    assert len({record.edge_id for record in records}) == len(records)
