from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Sequence

from .feature_access import FeatureAccessor
from .incident_detector import SIGNAL_PREFIXES, incident_features
from .temporal_episode_gate import (
    EpisodeRecord,
    SupportAssessment,
    SupportStatus,
)


@dataclass(frozen=True)
class FeatureSupport:
    name: str
    axis: str
    lower: float
    upper: float
    scale: float
    hard: bool = False


@dataclass(frozen=True)
class DomainDiscriminator:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float


@dataclass(frozen=True)
class ShiftSentinelModel:
    supports: tuple[FeatureSupport, ...]
    domain_discriminator: DomainDiscriminator
    support_threshold: float


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _longest_path_depth(window: dict[str, Any]) -> float:
    node_count = len(window["node_ids"])
    incoming: dict[int, list[int]] = {index: [] for index in range(node_count)}
    for source, target in window["edge_index"]:
        incoming[int(target)].append(int(source))
    memo: dict[int, int] = {}
    visiting: set[int] = set()

    def depth(node: int) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return 0
        visiting.add(node)
        value = 0
        for parent in incoming[node]:
            value = max(value, 1 + depth(parent))
        visiting.remove(node)
        memo[node] = value
        return value

    return float(max((depth(node) for node in range(node_count)), default=0))


def sentinel_features(
    *,
    window: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, tuple[str, float]]:
    """Observable support features grouped by the registered OOD axes."""

    node_count = max(1, len(window["node_ids"]))
    accessor = FeatureAccessor(list(window["node_feature_names"]))
    rows = [[float(value) for value in row] for row in window["node_features"]]
    features: dict[str, tuple[str, float]] = {
        "topology__log_node_count": ("topology", math.log1p(node_count)),
        "topology__edge_density": (
            "topology",
            len(window["edge_index"]) / node_count,
        ),
        "topology__depth": ("topology", _longest_path_depth(window)),
        "workload__log_room_count": (
            "workload",
            math.log1p(float(metadata["room_count"])),
        ),
        "workload__floor_count": (
            "workload",
            float(metadata["floor_count"]),
        ),
    }
    optional_workload = {
        "retention_days": (
            lambda value: math.log1p(float(value))
        ),
        "base_occupancy_fraction": float,
        "catchup_recording_channels": (
            lambda value: math.log1p(float(value))
        ),
        "average_bitrate_mbps": float,
    }
    for name, transform in optional_workload.items():
        value = metadata.get(name)
        if value is not None:
            features[f"workload__{name}"] = (
                "workload",
                transform(value),
            )
    for name in accessor.feature_names:
        if not name.startswith("node_type__"):
            continue
        features[f"topology__{name}"] = (
            "topology",
            sum(accessor.get(row, name) for row in rows) / node_count,
        )

    incident = incident_features(window)
    features["telemetry__missing_fraction"] = (
        "telemetry",
        float(incident["missing_fraction"]),
    )
    features["telemetry__scenario_missing_fraction"] = (
        "telemetry",
        float(metadata.get("missing_fraction", 0.0)),
    )
    features["telemetry__anomaly_mean"] = (
        "telemetry",
        float(incident["anomaly_mean"]),
    )
    features["telemetry__error_mean"] = (
        "telemetry",
        float(incident["error_mean"]),
    )
    session_key = "iptv_session_active_count_mean_mean"
    features["workload__active_sessions"] = (
        "workload",
        math.log1p(max(0.0, float(incident.get(session_key, 0.0)))),
    )
    for prefix in SIGNAL_PREFIXES:
        applicable_name = f"{prefix}__applicable"
        applicable = [
            accessor.get(row, applicable_name, default=0.0) for row in rows
        ]
        features[
            "applicability__" + prefix.replace("__", "_")
        ] = (
            "applicability",
            mean(applicable) if applicable else 0.0,
        )
    return features


def _scenario_features(
    *,
    record: EpisodeRecord,
    metadata: dict[str, Any],
) -> dict[str, tuple[str, float]]:
    rows = [
        sentinel_features(window=window, metadata=metadata)
        for window in record.windows
    ]
    return {
        name: (
            rows[0][name][0],
            median(row[name][1] for row in rows),
        )
        for name in rows[0]
    }


def _fit_domain_discriminator(
    matrix: list[list[float]],
    feature_names: tuple[str, ...],
) -> DomainDiscriminator:
    centers = tuple(mean(row[index] for row in matrix) for index in range(len(feature_names)))
    scales = tuple(
        max(
            1e-6,
            math.sqrt(
                mean((row[index] - centers[index]) ** 2 for row in matrix)
            ),
        )
        for index in range(len(feature_names))
    )
    normal = [
        [(value - center) / scale for value, center, scale in zip(row, centers, scales, strict=True)]
        for row in matrix
    ]
    # Synthetic shifts are calibration probes only. They never contribute
    # incident labels and never use validation/test/OOD data.
    shifted = []
    for row in normal:
        shifted.append(
            [
                value + (3.0 if index % 2 == 0 else -3.0)
                for index, value in enumerate(row)
            ]
        )
    training = normal + shifted
    labels = [0] * len(normal) + [1] * len(shifted)
    weights = [0.0] * len(feature_names)
    bias = 0.0
    for _ in range(600):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, label in zip(training, labels, strict=True):
            prediction = _sigmoid(
                bias
                + sum(
                    weight * value
                    for weight, value in zip(weights, row, strict=True)
                )
            )
            error = prediction - label
            bias_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        bias -= 0.05 * bias_gradient / len(training)
        for index in range(len(weights)):
            weights[index] -= 0.05 * (
                gradients[index] / len(training) + 0.02 * weights[index]
            )
    train_scores = [
        _sigmoid(
            bias
            + sum(
                weight * value
                for weight, value in zip(weights, row, strict=True)
            )
        )
        for row in normal
    ]
    return DomainDiscriminator(
        feature_names=feature_names,
        means=centers,
        scales=scales,
        weights=tuple(weights),
        bias=bias,
        threshold=max(0.80, _quantile(train_scores, 0.99)),
    )


def fit_shift_sentinels(
    *,
    training_records: Sequence[EpisodeRecord],
    scenario_metadata: dict[str, dict[str, Any]],
    registered_operational_bounds: (
        dict[str, tuple[float, float]] | None
    ) = None,
) -> ShiftSentinelModel:
    if not training_records:
        raise ValueError("Shift sentinel fitting requires training scenarios")
    rows = [
        _scenario_features(
            record=record,
            metadata=scenario_metadata[record.scenario_id],
        )
        for record in training_records
    ]
    feature_names = tuple(sorted(rows[0]))
    supports = []
    for name in feature_names:
        axis = rows[0][name][0]
        values = [row[name][1] for row in rows]
        registered_bounds = (
            registered_operational_bounds or {}
        ).get(name)
        hard = registered_bounds is not None or name in {
            "workload__log_room_count",
            "workload__retention_days",
        }
        lower = (
            registered_bounds[0]
            if registered_bounds is not None
            else min(values)
            if hard
            else _quantile(values, 0.01)
        )
        upper = (
            registered_bounds[1]
            if registered_bounds is not None
            else max(values)
            if hard
            else _quantile(values, 0.99)
        )
        iqr = _quantile(values, 0.75) - _quantile(values, 0.25)
        supports.append(
            FeatureSupport(
                name=name,
                axis=axis,
                lower=lower,
                upper=upper,
                scale=max(1e-6, iqr, (upper - lower) / 2.0),
                hard=hard,
            )
        )
    matrix = [[row[name][1] for name in feature_names] for row in rows]
    provisional = ShiftSentinelModel(
        supports=tuple(supports),
        domain_discriminator=_fit_domain_discriminator(matrix, feature_names),
        support_threshold=1.0,
    )
    scenario_max_scores = [
        max(
            (
                assess_support(
                    model=provisional,
                    window=window,
                    metadata=scenario_metadata[record.scenario_id],
                ).support_score
                for window in record.windows
            ),
            default=0.0,
        )
        for record in training_records
    ]
    # Calibrate on whole scenario maxima. A per-window percentile would
    # compound across overlapping windows and produce excessive scenario-level
    # false OOD alarms.
    return ShiftSentinelModel(
        supports=provisional.supports,
        domain_discriminator=provisional.domain_discriminator,
        support_threshold=max(1.0, _quantile(scenario_max_scores, 0.99)),
    )


def assess_support(
    *,
    model: ShiftSentinelModel,
    window: dict[str, Any],
    metadata: dict[str, Any],
) -> SupportAssessment:
    features = sentinel_features(window=window, metadata=metadata)
    axis_scores: dict[str, float] = {}
    values = {}
    hard_violation = False
    for support in model.supports:
        value = float(features[support.name][1])
        values[support.name] = value
        distance = max(
            0.0,
            support.lower - value,
            value - support.upper,
        ) / support.scale
        if support.hard and distance > 0.0:
            hard_violation = True
            distance = max(
                distance,
                model.support_threshold + 1.0,
            )
        axis_scores[support.axis] = max(
            axis_scores.get(support.axis, 0.0), distance
        )
    discriminator = model.domain_discriminator
    standardized = [
        (values[name] - center) / scale
        for name, center, scale in zip(
            discriminator.feature_names,
            discriminator.means,
            discriminator.scales,
            strict=True,
        )
    ]
    domain_score = _sigmoid(
        discriminator.bias
        + sum(
            weight * value
            for weight, value in zip(
                discriminator.weights, standardized, strict=True
            )
        )
    )
    axis_scores["domain"] = domain_score / discriminator.threshold
    support_score = max(axis_scores.values(), default=0.0)
    return SupportAssessment(
        status=(
            SupportStatus.UNSUPPORTED
            if hard_violation
            or support_score > model.support_threshold
            else SupportStatus.IN_DISTRIBUTION
        ),
        support_score=support_score,
        axis_scores=tuple(sorted(axis_scores.items())),
    )
