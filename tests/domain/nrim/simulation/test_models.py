import pytest
from pydantic import ValidationError

from app.domain.nrim.simulation.models import (
    DeploymentTopology,
    SimulationEdge,
    SimulationEdgeType,
    SimulationNode,
    SimulationNodeType,
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
