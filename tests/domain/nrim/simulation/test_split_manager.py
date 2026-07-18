from app.domain.nrim.simulation.scenario_design import (
    DatasetSplit,
)
from app.domain.nrim.simulation.split_manager import (
    deterministic_group_split,
    topology_fingerprint,
    validate_group_isolation,
)


def make_topology() -> dict:
    return {
        "deployment_family": "small",
        "room_count": 50,
        "floor_count": 2,
        "nodes": [
            {
                "node_id": "a",
                "node_type": "catchup_service",
                "capacity": {},
                "configuration": {},
                "metadata": {},
            },
            {
                "node_id": "b",
                "node_type": "catchup_storage",
                "capacity": {},
                "configuration": {},
                "metadata": {},
            },
        ],
        "edges": [
            {
                "edge_id": "random_edge_id",
                "source_node_id": "a",
                "target_node_id": "b",
                "edge_type": "stores_on",
                "propagation_delay_minutes": 5,
                "propagation_strength": 1.0,
            }
        ],
    }


def test_fingerprint_ignores_edge_identifier() -> None:
    first = make_topology()
    second = make_topology()

    second["edges"][0]["edge_id"] = (
        "different_random_id"
    )

    assert (
        topology_fingerprint(first)
        == topology_fingerprint(second)
    )


def test_group_split_is_deterministic() -> None:
    first = deterministic_group_split(
        group_id="topology_abc"
    )

    second = deterministic_group_split(
        group_id="topology_abc"
    )

    assert first == second


def test_ood_always_uses_ood_split() -> None:
    split = deterministic_group_split(
        group_id="topology_abc",
        ood=True,
    )

    assert split == DatasetSplit.OOD_TEST


def test_pair_split_leakage_is_detected() -> None:
    records = [
        {
            "pair_id": "pair_1",
            "topology_fingerprint": "topology_1",
            "split": "train",
        },
        {
            "pair_id": "pair_1",
            "topology_fingerprint": "topology_1",
            "split": "test",
        },
    ]

    errors = validate_group_isolation(records)

    assert errors
