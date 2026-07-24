from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .feature_access import FeatureAccessor, node_rows_by_id
from .models import NodeScore
from .root_cause_baselines import (
    anomaly_baseline,
    error_evidence_baseline,
    infer_node_type,
    score_map,
    static_criticality_baseline,
)
from .topology_scoring import topology_propagation_baseline


STAGE2_FROZEN_SIGNAL_PREFIXES = (
    "system__disk__utilization",
    "system__disk__io_latency",
    "system__disk__io_errors",
    "iptv__catchup__recording_failures",
    "system__process__restart_count",
    "iptv__session__active_count",
)
STAGE2_EXCLUDED_SIGNAL_PREFIXES = (
    "iptv__catchup__service_availability",
)


def _frozen_stage2_window(
    window: dict[str, Any],
) -> dict[str, Any]:
    """Hide post-freeze Stage 1 signals from every Stage 2 feature path."""

    names = list(window["node_feature_names"])
    excluded = [
        index
        for index, name in enumerate(names)
        if any(
            prefix in name
            for prefix
            in STAGE2_EXCLUDED_SIGNAL_PREFIXES
        )
    ]
    if not excluded:
        return window
    frozen = dict(window)
    frozen["node_features"] = [
        [
            (
                0.0
                if index in excluded
                else float(value)
            )
            for index, value in enumerate(row)
        ]
        for row in window["node_features"]
    ]
    return frozen


@dataclass(frozen=True)
class RootCauseFusionModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    use_topology: bool = True


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def node_fusion_features(
    window: dict[str, Any],
    *,
    use_topology: bool = True,
) -> dict[str, dict[str, float]]:
    stage2_window = _frozen_stage2_window(
        window
    )
    anomaly = anomaly_baseline(
        window=stage2_window
    )
    errors = error_evidence_baseline(
        window=stage2_window
    )
    criticality = static_criticality_baseline(
        window=stage2_window
    )
    topology = topology_propagation_baseline(
        window=stage2_window,
        local_scores=anomaly,
    )
    anomaly_map = score_map(anomaly)
    error_map = score_map(errors)
    criticality_map = score_map(criticality)
    topology_map = score_map(topology)
    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = node_rows_by_id(window)

    output: dict[str, dict[str, float]] = {}
    for node_id, row in rows.items():
        slopes = [
            accessor.get(row, name)
            for name in accessor.feature_names
            if name.endswith("__slope")
            and name.removesuffix(
                "__slope"
            )
            in STAGE2_FROZEN_SIGNAL_PREFIXES
        ]
        temporal_deltas = []
        for name in accessor.feature_names:
            if not name.endswith("__latest"):
                continue
            if (
                name.removesuffix("__latest")
                not in STAGE2_FROZEN_SIGNAL_PREFIXES
            ):
                continue
            mean_name = (
                name.removesuffix("__latest")
                + "__mean"
            )
            temporal_deltas.append(
                accessor.get(row, name)
                - accessor.get(row, mean_name)
            )
        node_type = infer_node_type(
            row=row,
            accessor=accessor,
        )
        topology_increment = float(
            topology_map[node_id].evidence.get(
                "topology_increment",
                0.0,
            )
        )
        output[node_id] = {
            "local_anomaly": anomaly_map[node_id].score,
            "error_evidence": error_map[node_id].score,
            "static_criticality": criticality_map[node_id].score,
            "topology_residual": (
                topology_increment if use_topology else 0.0
            ),
            "temporal_positive_slope": max(
                [0.0, *slopes]
            ),
            "temporal_delta": max(
                [0.0, *temporal_deltas]
            ),
            "in_degree": accessor.get(row, "in_degree"),
            "out_degree": accessor.get(row, "out_degree"),
            "is_storage": float(
                node_type == "catchup_storage"
            ),
            "is_service": float(
                node_type == "catchup_service"
            ),
        }
    return output


def fit_root_cause_fusion(
    *,
    training_windows: list[dict[str, Any]],
    use_topology: bool = True,
    learning_rate: float = 0.04,
    iterations: int = 400,
    l2_penalty: float = 0.02,
    random_seed: int = 42,
) -> RootCauseFusionModel:
    matrix_rows: list[dict[str, float]] = []
    labels: list[int] = []
    for window in training_windows:
        targets = list(
            window["targets"]["root_cause_node"]
        )
        if not any(int(value) for value in targets):
            continue
        features = node_fusion_features(
            window,
            use_topology=use_topology,
        )
        for index, node_id in enumerate(
            window["node_ids"]
        ):
            matrix_rows.append(features[str(node_id)])
            labels.append(int(targets[index]))
    if not matrix_rows or len(set(labels)) != 2:
        raise ValueError(
            "Root-cause fusion requires positive and negative nodes"
        )

    feature_names = tuple(sorted(matrix_rows[0]))
    matrix = [
        [row[name] for name in feature_names]
        for row in matrix_rows
    ]
    centers = tuple(
        mean(row[column] for row in matrix)
        for column in range(len(feature_names))
    )
    scales = tuple(
        max(
            1e-6,
            math.sqrt(
                mean(
                    (row[column] - centers[column]) ** 2
                    for row in matrix
                )
            ),
        )
        for column in range(len(feature_names))
    )
    standardized = [
        [
            (value - center) / scale
            for value, center, scale in zip(
                row,
                centers,
                scales,
                strict=True,
            )
        ]
        for row in matrix
    ]
    positives = sum(labels)
    negatives = len(labels) - positives
    class_weights = (
        len(labels) / (2.0 * negatives),
        len(labels) / (2.0 * positives),
    )
    rng = random.Random(random_seed)
    weights = [
        rng.uniform(-0.01, 0.01)
        for _ in feature_names
    ]
    bias = math.log(
        (positives + 0.5) / (negatives + 0.5)
    )
    for _ in range(iterations):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, label in zip(
            standardized,
            labels,
            strict=True,
        ):
            prediction = _sigmoid(
                bias
                + sum(
                    weight * value
                    for weight, value in zip(
                        weights,
                        row,
                        strict=True,
                    )
                )
            )
            error = (
                prediction - label
            ) * class_weights[label]
            bias_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        count = len(labels)
        bias -= learning_rate * bias_gradient / count
        for index in range(len(weights)):
            weights[index] -= learning_rate * (
                gradients[index] / count
                + l2_penalty * weights[index]
            )

    return RootCauseFusionModel(
        feature_names=feature_names,
        means=centers,
        scales=scales,
        weights=tuple(weights),
        bias=bias,
        use_topology=use_topology,
    )


def learned_fusion_scores(
    *,
    window: dict[str, Any],
    model: RootCauseFusionModel,
) -> list[NodeScore]:
    features = node_fusion_features(
        window,
        use_topology=model.use_topology,
    )
    results = []
    for node_id in window["node_ids"]:
        node_id = str(node_id)
        values = features[node_id]
        standardized = [
            (values[name] - center) / scale
            for name, center, scale in zip(
                model.feature_names,
                model.means,
                model.scales,
                strict=True,
            )
        ]
        score = _sigmoid(
            model.bias
            + sum(
                weight * value
                for weight, value in zip(
                    model.weights,
                    standardized,
                    strict=True,
                )
            )
        )
        results.append(
            NodeScore(
                node_id=node_id,
                score=score,
                evidence={
                    name: values[name]
                    for name in model.feature_names
                },
            )
        )
    return results
