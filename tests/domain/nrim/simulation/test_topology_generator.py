from app.domain.nrim.simulation.models import (
    SimulationNodeType,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def test_generator_creates_one_switch_per_floor() -> None:
    topology = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )

    distribution_switches = [
        node
        for node in topology.nodes
        if node.node_type
        == SimulationNodeType.DISTRIBUTION_SWITCH
    ]

    assert len(distribution_switches) == 6


def test_generator_is_reproducible() -> None:
    first = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )

    second = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )

    first_nodes = [
        node.model_dump()
        for node in first.nodes
    ]
    second_nodes = [
        node.model_dump()
        for node in second.nodes
    ]

    assert first_nodes == second_nodes


def test_redundancy_adds_second_middleware() -> None:
    topology = generate_hotel_topology(
        room_count=180,
        floor_count=6,
        redundant_middleware=True,
        shared_storage=True,
        seed=42,
    )

    middleware_nodes = [
        node
        for node in topology.nodes
        if node.node_type
        == SimulationNodeType.MIDDLEWARE
    ]

    assert len(middleware_nodes) == 2
