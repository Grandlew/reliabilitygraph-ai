from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .abstention import threshold_candidates
from .feature_access import FeatureAccessor
from .root_cause_baselines import (
    anomaly_baseline,
    error_evidence_baseline,
)


@dataclass(frozen=True)
class IncidentThresholdSelection:
    threshold: float
    feasible: bool
    false_selection_rate: float
    coverage: float
    maximum_cohort_false_selection_rate: float = 0.0
    cohort_false_selection_rates: tuple[
        tuple[str, float],
        ...,
    ] = ()


@dataclass(frozen=True)
class IncidentDetectorModel:
    """Deterministic standard-library logistic incident detector."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    ood_distance_threshold: float

    def feature_vector(
        self,
        window: dict[str, Any],
    ) -> tuple[float, ...]:
        features = incident_features(window)
        return tuple(
            float(features.get(name, 0.0))
            for name in self.feature_names
        )

    def standardized_vector(
        self,
        window: dict[str, Any],
    ) -> tuple[float, ...]:
        return tuple(
            (value - center) / scale
            for value, center, scale in zip(
                self.feature_vector(window),
                self.means,
                self.scales,
                strict=True,
            )
        )

    def predict_score(
        self,
        window: dict[str, Any],
    ) -> float:
        linear = self.bias + sum(
            weight * value
            for weight, value in zip(
                self.weights,
                self.standardized_vector(window),
                strict=True,
            )
        )
        return _sigmoid(linear)

    def ood_distance(
        self,
        window: dict[str, Any],
    ) -> float:
        standardized = self.standardized_vector(window)
        if not standardized:
            return 0.0
        # RMS standardized distance is topology-size invariant.
        return math.sqrt(
            sum(value * value for value in standardized)
            / len(standardized)
        )

    def is_out_of_distribution(
        self,
        window: dict[str, Any],
    ) -> bool:
        return (
            self.ood_distance(window)
            > self.ood_distance_threshold
        )


SIGNAL_PREFIXES = (
    "system__disk__utilization",
    "system__disk__io_latency",
    "system__disk__io_errors",
    "iptv__catchup__recording_failures",
    "system__process__restart_count",
    "iptv__session__active_count",
    "iptv__catchup__service_availability",
)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _quantile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def _applicable_values(
    *,
    rows: list[list[float]],
    accessor: FeatureAccessor,
    prefix: str,
    statistic: str,
) -> list[float]:
    value_name = f"{prefix}__{statistic}"
    applicable_name = f"{prefix}__applicable"
    missing_name = f"{prefix}__missing"
    return [
        accessor.get(row, value_name)
        for row in rows
        if accessor.get(row, applicable_name, default=1.0) > 0.5
        and accessor.get(row, missing_name, default=0.0) < 0.5
    ]


def incident_features(
    window: dict[str, Any],
) -> dict[str, float]:
    """Create topology-normalized aggregate and temporal features."""
    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = [
        [float(value) for value in row]
        for row in window["node_features"]
    ]
    anomaly_scores = [
        item.score
        for item in anomaly_baseline(window=window)
    ]
    error_scores = [
        item.score
        for item in error_evidence_baseline(window=window)
    ]
    features: dict[str, float] = {
        "anomaly_max": max(anomaly_scores, default=0.0),
        "anomaly_mean": mean(anomaly_scores) if anomaly_scores else 0.0,
        "anomaly_q75": _quantile(anomaly_scores, 0.75),
        "anomaly_fraction_positive": (
            sum(value > 0.05 for value in anomaly_scores)
            / len(anomaly_scores)
            if anomaly_scores
            else 0.0
        ),
        "error_max": max(error_scores, default=0.0),
        "error_mean": mean(error_scores) if error_scores else 0.0,
        "error_q75": _quantile(error_scores, 0.75),
        "error_fraction_positive": (
            sum(value > 0.02 for value in error_scores)
            / len(error_scores)
            if error_scores
            else 0.0
        ),
        "change_event_fraction": (
            sum(
                accessor.get(row, "recent_change_event_count")
                for row in rows
            )
            / max(1, len(rows))
        ),
    }
    missing_total = 0
    applicable_total = 0
    for prefix in SIGNAL_PREFIXES:
        applicable_name = f"{prefix}__applicable"
        missing_name = f"{prefix}__missing"
        for row in rows:
            if accessor.get(
                row,
                applicable_name,
                default=1.0,
            ) > 0.5:
                applicable_total += 1
                missing_total += int(
                    accessor.get(
                        row,
                        missing_name,
                        default=0.0,
                    )
                    > 0.5
                )

        for statistic in (
            "latest",
            "mean",
            "maximum",
            "slope",
            "standard_deviation",
        ):
            values = _applicable_values(
                rows=rows,
                accessor=accessor,
                prefix=prefix,
                statistic=statistic,
            )
            key = prefix.replace("__", "_")
            features[f"{key}_{statistic}_max"] = max(
                values,
                default=0.0,
            )
            features[f"{key}_{statistic}_mean"] = (
                mean(values) if values else 0.0
            )

        latest = _applicable_values(
            rows=rows,
            accessor=accessor,
            prefix=prefix,
            statistic="latest",
        )
        historical = _applicable_values(
            rows=rows,
            accessor=accessor,
            prefix=prefix,
            statistic="mean",
        )
        key = prefix.replace("__", "_")
        features[f"{key}_temporal_delta_max"] = max(
            (
                current - baseline
                for current, baseline in zip(
                    latest,
                    historical,
                    strict=False,
                )
            ),
            default=0.0,
        )
        history_z = [
            accessor.get(
                row,
                f"history__{prefix}__robust_z",
            )
            for row in rows
            if accessor.get(
                row,
                f"{prefix}__applicable",
                default=1.0,
            )
            > 0.5
        ]
        history_delta = [
            accessor.get(
                row,
                f"history__{prefix}__delta",
            )
            for row in rows
            if accessor.get(
                row,
                f"{prefix}__applicable",
                default=1.0,
            )
            > 0.5
        ]
        history_persistence = [
            accessor.get(
                row,
                f"history__{prefix}__persistence_count",
            )
            for row in rows
            if accessor.get(
                row,
                f"{prefix}__applicable",
                default=1.0,
            )
            > 0.5
        ]
        features[f"{key}_history_z_max"] = max(
            history_z,
            default=0.0,
        )
        features[f"{key}_history_z_mean"] = (
            mean(history_z) if history_z else 0.0
        )
        features[f"{key}_history_delta_max"] = max(
            history_delta,
            default=0.0,
        )
        features[f"{key}_persistence_max"] = max(
            history_persistence,
            default=0.0,
        )
        features[f"{key}_affected_fraction"] = (
            sum(value >= 1.5 for value in history_z)
            / len(history_z)
            if history_z
            else 0.0
        )

    features["missing_fraction"] = (
        missing_total / applicable_total
        if applicable_total
        else 0.0
    )
    return features


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _percentile(
    values: list[float],
    fraction: float,
) -> float:
    return _quantile(values, fraction)


def fit_incident_detector(
    *,
    training_windows: list[dict[str, Any]],
    learning_rate: float = 0.03,
    iterations: int = 1_500,
    l2_penalty: float = 0.01,
    scenario_metadata: dict[str, dict[str, Any]] | None = None,
    use_counterfactual_pairs: bool = True,
    cohort_balancing: bool = True,
    feature_mode: str = "full",
    pairwise_weight: float = 0.50,
    hard_negative_weight: float = 1.25,
) -> IncidentDetectorModel:
    if not training_windows:
        raise ValueError("Incident detector training requires windows")

    rows = [
        incident_features(window)
        for window in training_windows
    ]
    labels = [
        int(window["targets"]["current_incident"])
        for window in training_windows
    ]
    if len(set(labels)) != 2:
        raise ValueError(
            "Incident detector training requires both target classes"
        )

    if feature_mode not in {"raw", "full"}:
        raise ValueError("feature_mode must be 'raw' or 'full'")
    feature_names = tuple(
        sorted(
            name
            for name in rows[0]
            if (
                feature_mode == "full"
                or not any(
                    marker in name
                    for marker in (
                        "_history_",
                        "_persistence_",
                        "_affected_fraction",
                    )
                )
            )
        )
    )
    matrix = [
        [float(row[name]) for name in feature_names]
        for row in rows
    ]
    means = tuple(
        mean(row[column] for row in matrix)
        for column in range(len(feature_names))
    )
    scales = tuple(
        max(
            1e-6,
            math.sqrt(
                mean(
                    (
                        row[column]
                        - means[column]
                    )
                    ** 2
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
                means,
                scales,
                strict=True,
            )
        ]
        for row in matrix
    ]

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    class_weights = (
        len(labels) / (2.0 * negative_count),
        len(labels) / (2.0 * positive_count),
    )
    sample_weights = [1.0] * len(labels)
    if scenario_metadata and cohort_balancing:
        cohort_keys = []
        for window, label in zip(
            training_windows,
            labels,
            strict=True,
        ):
            metadata = scenario_metadata[
                str(window["source_scenario_id"])
            ]
            cohort_keys.append(
                (
                    label,
                    str(metadata["topology_fingerprint"]),
                )
            )
        cohort_counts: dict[tuple[Any, ...], int] = {}
        for key in cohort_keys:
            cohort_counts[key] = (
                cohort_counts.get(key, 0) + 1
            )
        raw_weights = [
            1.0 / math.sqrt(cohort_counts[key])
            for key in cohort_keys
        ]
        normalization = mean(raw_weights)
        sample_weights = [
            value / normalization
            for value in raw_weights
        ]
        # Confounder-family labels are evaluation-only. Weighting is grouped
        # by independent topology and target class, never by simulator
        # annotations unavailable at inference time.

    pair_indices: list[tuple[int, int]] = []
    if (
        scenario_metadata
        and use_counterfactual_pairs
        and pairwise_weight > 0.0
    ):
        buckets: dict[
            tuple[str, str],
            list[int],
        ] = {}
        for index, window in enumerate(training_windows):
            metadata = scenario_metadata[
                str(window["source_scenario_id"])
            ]
            key = (
                str(metadata["pair_id"]),
                str(window["observation_cutoff"]),
            )
            buckets.setdefault(key, []).append(index)
        for indices in buckets.values():
            positives = [
                index for index in indices if labels[index] == 1
            ]
            healthy_controls = [
                index
                for index in indices
                if bool(
                    scenario_metadata[
                        str(
                            training_windows[index][
                                "source_scenario_id"
                            ]
                        )
                    ]["healthy"]
                )
            ]
            for positive in positives:
                for healthy in healthy_controls:
                    pair_indices.append((positive, healthy))
    weights = [0.0] * len(feature_names)
    bias = math.log(
        (positive_count + 0.5)
        / (negative_count + 0.5)
    )

    for _ in range(iterations):
        weight_gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, label, cohort_weight in zip(
            standardized,
            labels,
            sample_weights,
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
            sample_weight = (
                class_weights[label] * cohort_weight
            )
            error = (prediction - label) * sample_weight
            bias_gradient += error
            for index, value in enumerate(row):
                weight_gradients[index] += error * value

        for positive_index, healthy_index in pair_indices:
            difference = [
                positive - healthy
                for positive, healthy in zip(
                    standardized[positive_index],
                    standardized[healthy_index],
                    strict=True,
                )
            ]
            margin = sum(
                weight * value
                for weight, value in zip(
                    weights,
                    difference,
                    strict=True,
                )
            )
            pair_error = (
                _sigmoid(margin) - 1.0
            ) * pairwise_weight
            for index, value in enumerate(difference):
                weight_gradients[index] += pair_error * value

        count = len(labels) + pairwise_weight * len(
            pair_indices
        )
        bias -= learning_rate * bias_gradient / count
        for index in range(len(weights)):
            weights[index] -= learning_rate * (
                weight_gradients[index] / count
                + l2_penalty * weights[index]
            )

    provisional = IncidentDetectorModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        weights=tuple(weights),
        bias=bias,
        ood_distance_threshold=float("inf"),
    )
    training_distances = [
        provisional.ood_distance(window)
        for window in training_windows
    ]
    return IncidentDetectorModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        weights=tuple(weights),
        bias=bias,
        ood_distance_threshold=_percentile(
            training_distances,
            0.99,
        ),
    )


def incident_detection_score(
    window: dict[str, Any],
    *,
    model: IncidentDetectorModel | None = None,
) -> float:
    """Score a window with the fitted detector or legacy heuristic."""
    if model is not None:
        return model.predict_score(window)

    anomaly = anomaly_baseline(window=window)
    errors = error_evidence_baseline(window=window)
    strongest_anomaly = max(
        (item.score for item in anomaly),
        default=0.0,
    )
    strongest_error = max(
        (item.score for item in errors),
        default=0.0,
    )
    return min(
        1.0,
        strongest_anomaly + 0.5 * strongest_error,
    )


def choose_incident_threshold(
    *,
    validation_windows: list[dict[str, Any]],
    maximum_false_selection_rate: float = 0.10,
    minimum_incident_coverage: float = 0.70,
    model: IncidentDetectorModel | None = None,
    scenario_metadata: dict[str, dict[str, Any]] | None = None,
    maximum_cohort_false_selection_rate: float = 0.15,
) -> IncidentThresholdSelection:
    if not validation_windows:
        raise ValueError("Incident threshold selection requires windows")

    rows = [
        (
            incident_detection_score(window, model=model),
            int(window["targets"]["current_incident"]),
            (
                "+".join(
                    sorted(
                        scenario_metadata[
                            str(window["source_scenario_id"])
                        ]["confounders"]
                    )
                )
                or "none"
                if scenario_metadata
                else "all"
            ),
        )
        for window in validation_windows
    ]
    negative_count = sum(
        target == 0 for _, target, _ in rows
    )
    positive_count = sum(
        target == 1 for _, target, _ in rows
    )
    if not negative_count or not positive_count:
        raise ValueError(
            "Incident threshold selection requires both target classes"
        )

    candidates = threshold_candidates(
        [score for score, _, _ in rows]
    )
    ranked_candidates = []
    for threshold in candidates:
        false_rate = (
            sum(
                score >= threshold and target == 0
                for score, target, _ in rows
            )
            / negative_count
        )
        coverage = (
            sum(
                score >= threshold and target == 1
                for score, target, _ in rows
            )
            / positive_count
        )
        cohort_rates = {}
        for cohort in {
            cohort
            for _, target, cohort in rows
            if target == 0
        }:
            cohort_rows = [
                (score, target)
                for score, target, row_cohort in rows
                if row_cohort == cohort and target == 0
            ]
            cohort_rates[cohort] = _safe_rate(
                sum(
                    score >= threshold
                    for score, _ in cohort_rows
                ),
                len(cohort_rows),
            )
        maximum_cohort_rate = max(
            cohort_rates.values(),
            default=0.0,
        )
        feasible = (
            false_rate <= maximum_false_selection_rate
            and coverage >= minimum_incident_coverage
            and maximum_cohort_rate
            <= maximum_cohort_false_selection_rate
        )
        violation = (
            max(0.0, false_rate - maximum_false_selection_rate)
            + max(0.0, minimum_incident_coverage - coverage)
            + max(
                0.0,
                maximum_cohort_rate
                - maximum_cohort_false_selection_rate,
            )
        )
        ranked_candidates.append(
            (
                feasible,
                -violation,
                coverage - false_rate,
                coverage,
                -threshold,
                threshold,
                false_rate,
                maximum_cohort_rate,
                tuple(sorted(cohort_rates.items())),
            )
        )

    ranked_candidates.sort(reverse=True)
    best = ranked_candidates[0]
    return IncidentThresholdSelection(
        threshold=best[5],
        feasible=best[0],
        false_selection_rate=best[6],
        coverage=best[3],
        maximum_cohort_false_selection_rate=best[7],
        cohort_false_selection_rates=best[8],
    )


def incident_operating_curve(
    *,
    windows: list[dict[str, Any]],
    model: IncidentDetectorModel,
) -> list[dict[str, float]]:
    """Return every distinct validation operating point."""
    rows = [
        (
            model.predict_score(window),
            int(window["targets"]["current_incident"]),
        )
        for window in windows
    ]
    negatives = sum(not target for _, target in rows)
    positives = sum(target for _, target in rows)
    if not negatives or not positives:
        return []
    return [
        {
            "threshold": threshold,
            "healthy_false_selection_rate": (
                sum(
                    score >= threshold and not target
                    for score, target in rows
                )
                / negatives
            ),
            "faulty_coverage": (
                sum(
                    score >= threshold and bool(target)
                    for score, target in rows
                )
                / positives
            ),
        }
        for threshold in threshold_candidates(
            [score for score, _ in rows]
        )
    ]
