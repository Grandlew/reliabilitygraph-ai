from app.domain.nrim.baselines.topology_scoring import (
    build_propagation_adjacency,
    propagate_scores,
)


def test_stores_on_propagates_from_service_to_storage() -> None:
    adjacency = build_propagation_adjacency(
        node_ids=[
            "service",
            "storage",
        ],
        edge_index=[[0, 1]],
        edge_features=[
            [
                1.0,
                0.0,
                1.0,
            ]
        ],
        edge_feature_names=[
            "edge_type__stores_on",
            "propagation_delay_minutes",
            "propagation_strength",
        ],
    )

    assert "service" in adjacency
    assert adjacency["service"][0][0] == "storage"


def test_propagation_increases_dependency_score() -> None:
    result = propagate_scores(
        initial_scores={
            "service": 1.0,
            "storage": 0.2,
        },
        adjacency={
            "service": [
                ("storage", 1.0)
            ]
        },
        propagation_decay=0.5,
        maximum_hops=1,
    )

    assert result["storage"] > 0.2


def test_zero_hops_preserves_initial_scores() -> None:
    initial = {
        "a": 1.0,
        "b": 0.0,
    }

    result = propagate_scores(
        initial_scores=initial,
        adjacency={
            "a": [("b", 1.0)]
        },
        maximum_hops=0,
    )

    assert result == initial


def test_cycle_does_not_explode() -> None:
    result = propagate_scores(
        initial_scores={
            "a": 1.0,
            "b": 0.0,
        },
        adjacency={
            "a": [("b", 1.0)],
            "b": [("a", 1.0)],
        },
        propagation_decay=0.8,
        maximum_hops=10,
    )

    assert result["a"] < 10.0
    assert result["b"] < 10.0
