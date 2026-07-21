from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .models import NodeScore


REVERSE_PROPAGATION_EDGE_TYPES = {
    "depends_on",
    "stores_on",
}


FORWARD_PROPAGATION_EDGE_TYPES = {
    "serves",
    "connected_to",
    "sends_to",
}


def edge_type_from_row(
    *,
    edge_row: list[float],
    edge_feature_names: list[str],
) -> str | None:
    active_types = []

    for index, name in enumerate(
        edge_feature_names
    ):
        if not name.startswith("edge_type__"):
            continue

        if float(edge_row[index]) > 0.5:
            active_types.append(
                name.removeprefix("edge_type__")
            )

    if len(active_types) > 1:
        raise ValueError(
            "Edge has multiple active edge types."
        )

    return (
        active_types[0]
        if active_types
        else None
    )


def build_propagation_adjacency(
    *,
    node_ids: list[str],
    edge_index: list[list[int]],
    edge_features: list[list[float]],
    edge_feature_names: list[str],
) -> dict[str, list[tuple[str, float]]]:
    if len(edge_index) != len(edge_features):
        raise ValueError(
            "Edge index and edge features differ."
        )

    adjacency: dict[
        str,
        list[tuple[str, float]],
    ] = defaultdict(list)

    delay_index = (
        edge_feature_names.index(
            "propagation_delay_minutes"
        )
        if "propagation_delay_minutes"
        in edge_feature_names
        else None
    )

    strength_index = (
        edge_feature_names.index(
            "propagation_strength"
        )
        if "propagation_strength"
        in edge_feature_names
        else None
    )

    for indices, feature_row in zip(
        edge_index,
        edge_features,
        strict=True,
    ):
        source_index, target_index = indices

        source_id = node_ids[source_index]
        target_id = node_ids[target_index]

        edge_type = edge_type_from_row(
            edge_row=feature_row,
            edge_feature_names=edge_feature_names,
        )

        strength = (
            float(feature_row[strength_index])
            if strength_index is not None
            else 1.0
        )

        delay = (
            float(feature_row[delay_index])
            if delay_index is not None
            else 0.0
        )

        delay_discount = 1.0 / (
            1.0 + max(0.0, delay) / 60.0
        )

        effective_strength = max(
            0.0,
            min(
                1.0,
                strength * delay_discount,
            ),
        )

        if edge_type in REVERSE_PROPAGATION_EDGE_TYPES:
            adjacency[source_id].append(
                (
                    target_id,
                    effective_strength,
                )
            )

        elif edge_type in FORWARD_PROPAGATION_EDGE_TYPES:
            adjacency[target_id].append(
                (
                    source_id,
                    effective_strength,
                )
            )

    return dict(adjacency)


def propagate_scores(
    *,
    initial_scores: dict[str, float],
    adjacency: dict[
        str,
        list[tuple[str, float]],
    ],
    propagation_decay: float = 0.65,
    maximum_hops: int = 3,
) -> dict[str, float]:

    final_scores = initial_scores.copy()
    if maximum_hops <= 0:
        return final_scores

    # Queue entries: (source_node_id, current_node_id, current_score, hops_left)
    queue = deque()
    for node_id, score in initial_scores.items():
        if score > 0:
            queue.append((node_id, node_id, score, maximum_hops))

    while queue:
        source, current, score, hops = queue.popleft()

        if hops <= 0:
            continue

        neighbors = adjacency.get(current, [])
        for neighbor_id, weight in neighbors:
            # Attenuate
            propagated_value = score * weight * propagation_decay

            if propagated_value > final_scores.get(neighbor_id, 0.0):
                final_scores[neighbor_id] = propagated_value
                # Continue propagation from this neighbor
                queue.append((source, neighbor_id, propagated_value, hops - 1))

    return final_scores


def topology_propagation_baseline(
    *,
    window: dict[str, Any],
    local_scores: list[NodeScore],
    propagation_decay: float = 0.65,
    maximum_hops: int = 3,
) -> list[NodeScore]:
    node_ids = [
        str(node_id)
        for node_id in window["node_ids"]
    ]

    adjacency = build_propagation_adjacency(
        node_ids=node_ids,
        edge_index=list(window["edge_index"]),
        edge_features=list(
            window["edge_features"]
        ),
        edge_feature_names=list(
            window["edge_feature_names"]
        ),
    )

    initial = {
        item.node_id: item.score
        for item in local_scores
    }

    propagated = propagate_scores(
        initial_scores=initial,
        adjacency=adjacency,
        propagation_decay=propagation_decay,
        maximum_hops=maximum_hops,
    )

    local_by_id = {
        item.node_id: item
        for item in local_scores
    }

    return [
        NodeScore(
            node_id=node_id,
            score=propagated.get(
                node_id,
                initial.get(node_id, 0.0),
            ),
            evidence={
                **local_by_id[node_id].evidence,
                "topology_increment": max(
                    0.0,
                    propagated.get(node_id, 0.0)
                    - initial.get(node_id, 0.0),
                ),
            },
        )
        for node_id in node_ids
    ]
