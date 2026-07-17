import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.models import (
    DeploymentTopology,
    SimulationEdge,
    SimulationEdgeType,
    SimulationNode,
    SimulationNodeType,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def make_required_nodes() -> list[SimulationNode]:
    return [
        SimulationNode(
            node_id="catchup_service",
            node_type=(
                SimulationNodeType.CATCHUP_SERVICE
            ),
            name="CatchUP",
        ),
        SimulationNode(
            node_id="storage",
            node_type=(
                SimulationNodeType.CATCHUP_STORAGE
            ),
            name="Storage",
        ),
    ]


def test_topology_rejects_missing_endpoint() -> None:
    with pytest.raises(ValidationError):
        DeploymentTopology(
            deployment_family="test",
            room_count=10,
            floor_count=1,
            nodes=make_required_nodes(),
            edges=[
                SimulationEdge(
                    source_node_id="catchup_service",
                    target_node_id="missing",
                    edge_type=(
                        SimulationEdgeType.STORES_ON
                    ),
                )
            ],
        )


def test_topology_requires_storage_relationship() -> None:
    with pytest.raises(ValidationError):
        DeploymentTopology(
            deployment_family="test",
            room_count=10,
            floor_count=1,
            nodes=make_required_nodes(),
            edges=[
                SimulationEdge(
                    source_node_id="catchup_service",
                    target_node_id="storage",
                    edge_type=(
                        SimulationEdgeType.DEPENDS_ON
                    ),
                )
            ],
        )


def test_non_shared_storage_uses_dedicated_service_pairs() -> None:
    topology = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=True,
        shared_storage=False,
        seed=42,
    )

    services = {
        node.node_id
        for node in topology.nodes
        if node.node_type == SimulationNodeType.CATCHUP_SERVICE
    }
    storage_nodes = [
        node
        for node in topology.nodes
        if node.node_type == SimulationNodeType.CATCHUP_STORAGE
    ]
    storage_edges = [
        edge
        for edge in topology.edges
        if edge.edge_type == SimulationEdgeType.STORES_ON
    ]

    assert topology.shared_storage is False
    assert len(services) == 2
    assert len(storage_nodes) == 2
    assert len(storage_edges) == 2
    assert len({edge.target_node_id for edge in storage_edges}) == 2
    assert sum(
        node.capacity["usable_capacity_tb"]
        for node in storage_nodes
    ) == 16.0
