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

EDGE_TYPE_COMPATIBILITY = {
    "depends_on": 1.00,
    "stores_on": 1.00,
    "serves": 0.80,
    "sends_to": 0.70,
    "connected_to": 0.45,
}

DIAGNOSTIC_REVERSE_COMPATIBILITY = {
    "depends_on": 0.35,
    "stores_on": 0.55,
    "serves": 0.30,
    "sends_to": 0.25,
    "connected_to": 0.35,
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
) -> dict[str, list[tuple[str, float, str]]]:
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
                strength
                * delay_discount
                * EDGE_TYPE_COMPATIBILITY.get(
                    edge_type or "",
                    0.0,
                ),
            ),
        )

        if edge_type in REVERSE_PROPAGATION_EDGE_TYPES:
            adjacency[source_id].append(
                (
                    target_id,
                    effective_strength,
                    edge_type,
                )
            )
            adjacency[target_id].append(
                (
                    source_id,
                    effective_strength
                    * DIAGNOSTIC_REVERSE_COMPATIBILITY[
                        edge_type
                    ],
                    f"{edge_type}__diagnostic_reverse",
                )
            )

        elif edge_type in FORWARD_PROPAGATION_EDGE_TYPES:
            adjacency[target_id].append(
                (
                    source_id,
                    effective_strength,
                    edge_type,
                )
            )
            adjacency[source_id].append(
                (
                    target_id,
                    effective_strength
                    * DIAGNOSTIC_REVERSE_COMPATIBILITY[
                        edge_type
                    ],
                    f"{edge_type}__diagnostic_reverse",
                )
            )

    return dict(adjacency)


def propagate_scores(
    *,
    initial_scores: dict[str, float],
    adjacency: dict[
        str,
        list[tuple[str, float] | tuple[str, float, str]],
    ],
    propagation_decay: float = 0.65,
    maximum_hops: int = 3,
    aggregation_weight: float = 0.50,
) -> dict[str, float]:
    if maximum_hops <= 0:
        return initial_scores.copy()
    if not 0.0 <= propagation_decay <= 1.0:
        raise ValueError("propagation_decay must be between zero and one")
    if not 0.0 <= aggregation_weight <= 1.0:
        raise ValueError("aggregation_weight must be between zero and one")

    strongest_path: dict[tuple[str, str], float] = {}

    for source, source_score in initial_scores.items():
        if source_score <= 0.0:
            continue

        queue = deque(
            [
                (
                    source,
                    float(source_score),
                    maximum_hops,
                    frozenset({source}),
                )
            ]
        )

        while queue:
            current, score, hops, path = queue.popleft()
            if hops <= 0:
                continue

            for edge in adjacency.get(current, []):
                neighbor_id = edge[0]
                edge_weight = edge[1]
                if neighbor_id in path:
                    continue

                # Propagate only evidence that is not already explained by
                # the destination's local score. This makes topology a
                # residual signal instead of a monotonic copy of anomaly.
                destination_local = float(
                    initial_scores.get(neighbor_id, 0.0)
                )
                propagated = (
                    max(0.0, score - destination_local)
                    * max(0.0, min(1.0, edge_weight))
                    * propagation_decay
                )
                key = (source, neighbor_id)
                if propagated <= strongest_path.get(key, 0.0):
                    continue

                strongest_path[key] = propagated
                queue.append(
                    (
                        neighbor_id,
                        propagated,
                        hops - 1,
                        path | {neighbor_id},
                    )
                )

    incoming: dict[str, list[float]] = defaultdict(list)
    for (source, target), value in strongest_path.items():
        if source != target:
            incoming[target].append(value)

    return {
        node_id: min(
            1.0,
            max(0.0, float(local_score))
            + aggregation_weight
            * (
                sum(incoming[node_id])
                / len(incoming[node_id])
                if incoming[node_id]
                else 0.0
            ),
        )
        for node_id, local_score in initial_scores.items()
    }


def topology_propagation_baseline(
    *,
    window: dict[str, Any],
    local_scores: list[NodeScore],
    propagation_decay: float = 0.65,
    maximum_hops: int = 3,
    aggregation_weight: float = 0.50,
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
        aggregation_weight=aggregation_weight,
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
