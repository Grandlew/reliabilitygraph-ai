from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

from .feature_schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SIGNAL_APPLICABLE_NODE_TYPES,
    SIGNAL_NAMES,
    FeatureSchema,
)
from .temporal_windowing import parse_timestamp


LOW_QUALITY_VALUES = {
    "low",
    "quarantined",
}


CRITICAL_NODE_TYPES = {
    "middleware",
    "database",
    "catchup_service",
    "catchup_storage",
    "core_switch",
}


def safe_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric = float(value)

        if math.isfinite(numeric):
            return numeric

    return None


def linear_slope(
    timestamps: list[datetime],
    values: list[float],
) -> float:
    if len(values) < 2:
        return 0.0

    start = timestamps[0]

    x = [
        (timestamp - start).total_seconds()
        / 3600.0
        for timestamp in timestamps
    ]

    x_mean = mean(x)
    y_mean = mean(values)

    denominator = sum(
        (item - x_mean) ** 2
        for item in x
    )

    if denominator == 0:
        return 0.0

    numerator = sum(
        (x_value - x_mean)
        * (y_value - y_mean)
        for x_value, y_value
        in zip(x, values, strict=True)
    )

    return numerator / denominator


def build_signal_statistics(
    *,
    events: list[dict[str, Any]],
    cutoff: datetime,
) -> dict[str, float]:
    numeric_events = []

    for event in events:
        numeric_value = safe_numeric(
            event.get("value")
        )

        if numeric_value is None:
            continue

        timestamp = parse_timestamp(
            str(event["observed_at"])
        )

        numeric_events.append(
            (
                timestamp,
                numeric_value,
                str(event.get("quality", "medium")),
            )
        )

    numeric_events.sort(
        key=lambda item: item[0]
    )

    if not numeric_events:
        return {
            "latest": 0.0,
            "mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "standard_deviation": 0.0,
            "slope": 0.0,
            "count": 0.0,
            "low_quality_fraction": 0.0,
            "hours_since_latest": 0.0,
            "missing": 1.0,
        }

    timestamps = [
        item[0] for item in numeric_events
    ]
    values = [
        item[1] for item in numeric_events
    ]
    qualities = [
        item[2] for item in numeric_events
    ]

    low_quality_count = sum(
        quality in LOW_QUALITY_VALUES
        for quality in qualities
    )

    hours_since_latest = (
        cutoff - timestamps[-1]
    ).total_seconds() / 3600.0

    return {
        "latest": values[-1],
        "mean": mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": (
            pstdev(values)
            if len(values) > 1
            else 0.0
        ),
        "slope": linear_slope(
            timestamps,
            values,
        ),
        "count": float(len(values)),
        "low_quality_fraction": (
            low_quality_count / len(values)
        ),
        "hours_since_latest": max(
            0.0,
            hours_since_latest,
        ),
        "missing": 0.0,
    }


def build_node_feature_matrix(
    *,
    topology: dict[str, Any],
    telemetry: list[dict[str, Any]],
    context_events: list[dict[str, Any]],
    cutoff: datetime,
    schema: FeatureSchema,
) -> tuple[
    list[str],
    list[list[float]],
]:
    node_features = [
        definition.name
        for definition in schema.node_features
    ]

    default_by_name = {
        definition.name: definition.default_value
        for definition in schema.node_features
    }

    nodes = topology.get("nodes", [])
    edges = topology.get("edges", [])

    node_ids = [
        str(node["node_id"])
        for node in nodes
    ]

    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)

    for edge in edges:
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])

        outgoing[source] += 1
        incoming[target] += 1

    telemetry_by_node_signal: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for event in telemetry:
        node_id = str(
            event.get("component_node_id", "")
        )
        signal_name = str(
            event.get("signal_name", "")
        )

        telemetry_by_node_signal[
            (node_id, signal_name)
        ].append(event)

    changes_by_node: dict[str, int] = defaultdict(int)
    global_change_count = 0

    for event in context_events:
        global_change_count += 1
        node_id = event.get("component_node_id")

        if node_id:
            changes_by_node[str(node_id)] += 1

    matrix: list[list[float]] = []

    for node in nodes:
        node_id = str(node["node_id"])
        node_type = str(node["node_type"])

        row = dict(default_by_name)

        for known_node_type in NODE_TYPES:
            row[
                f"node_type__{known_node_type}"
            ] = float(
                node_type == known_node_type
            )

        row["in_degree"] = float(
            incoming[node_id]
        )
        row["out_degree"] = float(
            outgoing[node_id]
        )
        row["total_degree"] = float(
            incoming[node_id]
            + outgoing[node_id]
        )
        row["critical_service"] = float(
            node_type in CRITICAL_NODE_TYPES
        )
        row["recent_change_event_count"] = float(
            changes_by_node[node_id]
            + global_change_count
        )
        for signal_name in SIGNAL_NAMES:
            applicable = (
                node_type
                in SIGNAL_APPLICABLE_NODE_TYPES[
                    signal_name
                ]
            )
            safe_signal = signal_name.replace(
                ".",
                "__",
            )
            row[f"{safe_signal}__applicable"] = float(
                applicable
            )

            if not applicable:
                row[f"{safe_signal}__missing"] = 0.0
                continue

            statistics = build_signal_statistics(
                events=telemetry_by_node_signal[
                    (node_id, signal_name)
                ],
                cutoff=cutoff,
            )

            for statistic, value in (
                statistics.items()
            ):
                row[
                    f"{safe_signal}__{statistic}"
                ] = float(value)

        matrix.append(
            [
                float(row[name])
                for name in node_features
            ]
        )

    return node_ids, matrix


def build_edge_feature_matrix(
    *,
    topology: dict[str, Any],
    node_ids: list[str],
    schema: FeatureSchema,
) -> tuple[
    list[list[int]],
    list[list[float]],
]:
    node_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    feature_names = [
        definition.name
        for definition in schema.edge_features
    ]

    default_by_name = {
        definition.name: definition.default_value
        for definition in schema.edge_features
    }

    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []

    for edge in topology.get("edges", []):
        source_id = str(
            edge["source_node_id"]
        )
        target_id = str(
            edge["target_node_id"]
        )

        if (
            source_id not in node_index
            or target_id not in node_index
        ):
            raise ValueError(
                "Edge references unknown node."
            )

        edge_type = str(edge["edge_type"])

        row = dict(default_by_name)

        for known_edge_type in EDGE_TYPES:
            row[
                f"edge_type__{known_edge_type}"
            ] = float(
                edge_type == known_edge_type
            )

        row["propagation_delay_minutes"] = float(
            edge.get(
                "propagation_delay_minutes",
                0.0,
            )
        )
        row["propagation_strength"] = float(
            edge.get(
                "propagation_strength",
                1.0,
            )
        )
        row["reverse_dependency_direction"] = float(
            edge_type
            in {
                "depends_on",
                "stores_on",
            }
        )

        edge_index.append(
            [
                node_index[source_id],
                node_index[target_id],
            ]
        )

        edge_features.append(
            [
                float(row[name])
                for name in feature_names
            ]
        )

    return edge_index, edge_features
