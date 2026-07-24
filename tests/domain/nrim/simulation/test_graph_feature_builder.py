from datetime import datetime, timezone

from app.domain.nrim.simulation.feature_schema import (
    build_feature_schema,
)
from app.domain.nrim.simulation.graph_feature_builder import (
    build_edge_feature_matrix,
    build_node_feature_matrix,
)


def make_topology() -> dict:
    return {
        "nodes": [
            {
                "node_id": "catchup_service_1",
                "node_type": "catchup_service",
            },
            {
                "node_id": "catchup_storage_1",
                "node_type": "catchup_storage",
            },
        ],
        "edges": [
            {
                "source_node_id": "catchup_service_1",
                "target_node_id": "catchup_storage_1",
                "edge_type": "stores_on",
                "propagation_delay_minutes": 5,
                "propagation_strength": 1.0,
            }
        ],
    }


def test_node_feature_rows_match_nodes() -> None:
    cutoff = datetime.now(timezone.utc)

    node_ids, matrix = build_node_feature_matrix(
        topology=make_topology(),
        telemetry=[],
        context_events=[],
        cutoff=cutoff,
        schema=build_feature_schema(),
    )

    assert len(node_ids) == 2
    assert len(matrix) == 2


def test_missing_signal_has_missing_indicator() -> None:
    cutoff = datetime.now(timezone.utc)
    schema = build_feature_schema()

    node_ids, matrix = build_node_feature_matrix(
        topology=make_topology(),
        telemetry=[],
        context_events=[],
        cutoff=cutoff,
        schema=schema,
    )

    feature_names = [
        feature.name
        for feature in schema.node_features
    ]

    missing_index = feature_names.index(
        "system__disk__utilization__missing"
    )

    storage_index = node_ids.index(
        "catchup_storage_1"
    )
    service_index = node_ids.index(
        "catchup_service_1"
    )

    assert matrix[storage_index][missing_index] == 1.0
    assert matrix[service_index][missing_index] == 0.0


def test_edge_rows_match_edges() -> None:
    schema = build_feature_schema()

    edge_index, edge_features = (
        build_edge_feature_matrix(
            topology=make_topology(),
            node_ids=[
                "catchup_service_1",
                "catchup_storage_1",
            ],
            schema=schema,
        )
    )

    assert len(edge_index) == 1
    assert len(edge_features) == 1
