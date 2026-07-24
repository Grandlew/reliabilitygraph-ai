from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from statistics import mean, median
from typing import Any, Sequence

from .feature_access import FeatureAccessor
from .temporal_episode_gate import (
    EpisodeRecord,
    GateState,
    ScenarioEpisodeResult,
    SupportAssessment,
    SupportStatus,
    TemporalDecision,
    _logit,
    _parse_datetime,
    summarize_episode_results,
)


OPERATIONAL_METADATA_FIELDS = (
    "room_count",
    "floor_count",
    "retention_days",
    "base_occupancy_fraction",
    "catchup_recording_channels",
    "average_bitrate_mbps",
    "shared_storage",
    "redundant_middleware",
)


@dataclass(frozen=True)
class HealthyResidualModel:
    """Expected healthy Stage 1 log-odds under observable conditions."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    residual_center: float
    residual_scale: float

    def expected_logit(
        self,
        *,
        window: dict[str, Any],
        metadata: dict[str, Any],
        prior_logits: Sequence[float] = (),
    ) -> float:
        features = residual_context_features(
            window=window,
            metadata=metadata,
            prior_logits=prior_logits,
        )
        standardized = (
            (
                float(features[name]) - center
            )
            / scale
            for name, center, scale in zip(
                self.feature_names,
                self.means,
                self.scales,
                strict=True,
            )
        )
        return self.bias + sum(
            weight * value
            for weight, value in zip(
                self.weights,
                standardized,
                strict=True,
            )
        )

    def standardized_residual(
        self,
        *,
        window: dict[str, Any],
        probability: float,
        metadata: dict[str, Any],
        prior_logits: Sequence[float] = (),
    ) -> float:
        raw = (
            _logit(probability)
            - self.expected_logit(
                window=window,
                metadata=metadata,
                prior_logits=prior_logits,
            )
            - self.residual_center
        )
        return raw / self.residual_scale


@dataclass(frozen=True)
class ImpactConfirmation:
    confirmed: bool
    score: float
    family_count: int
    affected_node_fraction: float
    causal_consistent: bool
    families: tuple[str, ...]


@dataclass(frozen=True)
class DualPathConfig:
    """Risk-controlled fast/slow temporal episode policy."""

    fast_probability_threshold: float = 0.85
    fast_residual_threshold: float = 2.0
    impact_score_threshold: float = 0.55
    impact_minimum_families: int = 2
    impact_minimum_node_fraction: float = 0.08
    require_causal_consistency: bool = True
    slow_reference_kappa: float = 0.25
    slow_threshold: float = 3.0
    slow_input_cap: float = 5.0
    raw_probability_center: float = 0.50
    suspect_probability_threshold: float = 0.65
    suspect_residual_threshold: float = 1.0
    maximum_suspect_windows: int = 2
    recovery_residual_threshold: float = 0.25
    minimum_recovery_windows: int = 2
    cooldown_windows: int = 1
    unsupported_escalation_probability: float = 0.80
    use_residual: bool = True
    enable_fast_path: bool = True
    require_impact_confirmation: bool = True
    enable_slow_path: bool = True
    enable_support_sentinel: bool = True

    def __post_init__(self) -> None:
        probabilities = (
            self.fast_probability_threshold,
            self.raw_probability_center,
            self.suspect_probability_threshold,
            self.unsupported_escalation_probability,
        )
        if any(
            not 0.0 < value < 1.0
            for value in probabilities
        ):
            raise ValueError(
                "Dual-path probabilities must be strictly between zero "
                "and one"
            )
        if self.slow_threshold <= 0.0:
            raise ValueError(
                "slow_threshold must be positive"
            )
        if self.slow_input_cap <= 0.0:
            raise ValueError(
                "slow_input_cap must be positive"
            )
        if self.impact_minimum_families < 1:
            raise ValueError(
                "impact_minimum_families must be positive"
            )
        if not 0.0 <= self.impact_minimum_node_fraction <= 1.0:
            raise ValueError(
                "impact_minimum_node_fraction must be in [0, 1]"
            )
        if self.minimum_recovery_windows < 1:
            raise ValueError(
                "minimum_recovery_windows must be positive"
            )


@dataclass(frozen=True)
class DualPathCalibrationResult:
    config: DualPathConfig
    feasible: bool
    objective: float
    metrics: dict[str, Any]
    evaluated_configurations: int
    feasibility_failures: tuple[str, ...]
    candidate_summaries: tuple[
        dict[str, Any],
        ...,
    ] = ()


def _quantile(
    values: Sequence[float],
    fraction: float,
) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def _safe_numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return 0.0


def residual_context_features(
    *,
    window: dict[str, Any],
    metadata: dict[str, Any],
    prior_logits: Sequence[float] = (),
) -> dict[str, float]:
    """Build label-free healthy-context features with causal history only."""

    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = [
        [float(value) for value in row]
        for row in window["node_features"]
    ]
    node_count = max(1, len(rows))
    features: dict[str, float] = {
        "topology__log_node_count": math.log1p(node_count),
        "topology__edges_per_node": (
            len(window["edge_index"]) / node_count
        ),
        "observable__change_events_per_node": (
            sum(
                accessor.get(
                    row,
                    "recent_change_event_count",
                )
                for row in rows
            )
            / node_count
        ),
        "history__prior_logit_mean": (
            mean(prior_logits[-3:])
            if prior_logits
            else 0.0
        ),
        "history__available": float(bool(prior_logits)),
    }
    cutoff = _parse_datetime(
        window["observation_cutoff"]
    )
    hour_angle = 2.0 * math.pi * cutoff.hour / 24.0
    features["time__hour_sin"] = math.sin(hour_angle)
    features["time__hour_cos"] = math.cos(hour_angle)
    features["time__weekend"] = float(
        cutoff.weekday() >= 5
    )

    for field in OPERATIONAL_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None:
            features[f"workload__{field}"] = 0.0
            features[
                f"workload__{field}__available"
            ] = 0.0
            continue
        numeric = _safe_numeric(value)
        if field in {
            "room_count",
            "floor_count",
            "retention_days",
            "catchup_recording_channels",
        }:
            numeric = math.log1p(max(0.0, numeric))
        features[f"workload__{field}"] = numeric
        features[
            f"workload__{field}__available"
        ] = 1.0

    for name in accessor.feature_names:
        if name.startswith("node_type__"):
            features[
                f"topology__fraction__{name}"
            ] = (
                sum(
                    accessor.get(row, name)
                    for row in rows
                )
                / node_count
            )
        elif name.endswith("__applicable"):
            features[
                f"applicability__{name}"
            ] = (
                sum(
                    accessor.get(row, name)
                    for row in rows
                )
                / node_count
            )
        elif name.endswith("__missing"):
            features[
                f"missingness__{name}"
            ] = (
                sum(
                    accessor.get(row, name)
                    for row in rows
                )
                / node_count
            )
    return features


def _solve_linear_system(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    """Solve a small dense system using pivoted Gauss-Jordan elimination."""

    size = len(vector)
    augmented = [
        [*matrix[row], vector[row]]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(
                augmented[row][column]
            ),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [
            value / divisor
            for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return [
        augmented[row][-1]
        for row in range(size)
    ]


def _weighted_ridge(
    *,
    matrix: Sequence[Sequence[float]],
    targets: Sequence[float],
    sample_weights: Sequence[float],
    penalty: float,
) -> list[float]:
    dimension = len(matrix[0]) + 1
    normal = [
        [0.0] * dimension
        for _ in range(dimension)
    ]
    right = [0.0] * dimension
    for row, target, weight in zip(
        matrix,
        targets,
        sample_weights,
        strict=True,
    ):
        design = [1.0, *row]
        for left in range(dimension):
            right[left] += (
                weight * design[left] * target
            )
            for column in range(left, dimension):
                contribution = (
                    weight
                    * design[left]
                    * design[column]
                )
                normal[left][column] += contribution
                if left != column:
                    normal[column][left] += contribution
    for index in range(1, dimension):
        normal[index][index] += penalty
    normal[0][0] += 1e-9
    return _solve_linear_system(normal, right)


def fit_healthy_residual_model(
    *,
    training_records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    scenario_metadata: dict[str, dict[str, Any]],
    ridge_penalty: float = 0.25,
    huber_iterations: int = 4,
) -> HealthyResidualModel:
    """Fit only on healthy training scenarios, equal-weighting topologies."""

    rows: list[dict[str, float]] = []
    targets: list[float] = []
    topology_groups: list[str] = []
    for record in training_records:
        if not record.healthy_control:
            continue
        sequence_probabilities = probabilities[
            record.scenario_id
        ]
        if len(sequence_probabilities) != len(
            record.windows
        ):
            raise ValueError(
                "Residual probabilities must align with windows"
            )
        prior_logits: list[float] = []
        metadata = scenario_metadata[
            record.scenario_id
        ]
        for window, probability in zip(
            record.windows,
            sequence_probabilities,
            strict=True,
        ):
            rows.append(
                residual_context_features(
                    window=window,
                    metadata=metadata,
                    prior_logits=prior_logits,
                )
            )
            current_logit = _logit(
                float(probability)
            )
            targets.append(current_logit)
            topology_groups.append(
                record.topology_group
            )
            prior_logits.append(current_logit)
    if not rows:
        raise ValueError(
            "Healthy residual fitting requires healthy training windows"
        )

    feature_names = tuple(sorted(rows[0]))
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
    group_counts: dict[str, int] = defaultdict(int)
    for group in topology_groups:
        group_counts[group] += 1
    base_weights = [
        1.0 / group_counts[group]
        for group in topology_groups
    ]
    normalization = mean(base_weights)
    base_weights = [
        value / normalization
        for value in base_weights
    ]
    robust_weights = list(base_weights)
    coefficients = [0.0] * (
        len(feature_names) + 1
    )
    for _ in range(max(1, huber_iterations)):
        coefficients = _weighted_ridge(
            matrix=standardized,
            targets=targets,
            sample_weights=robust_weights,
            penalty=ridge_penalty,
        )
        residuals = [
            target
            - coefficients[0]
            - sum(
                weight * value
                for weight, value in zip(
                    coefficients[1:],
                    row,
                    strict=True,
                )
            )
            for row, target in zip(
                standardized,
                targets,
                strict=True,
            )
        ]
        center = median(residuals)
        mad = median(
            abs(value - center)
            for value in residuals
        )
        scale = max(1e-6, 1.4826 * mad)
        cutoff = 1.5 * scale
        robust_weights = [
            base
            * min(
                1.0,
                cutoff
                / max(
                    1e-12,
                    abs(residual - center),
                ),
            )
            for base, residual in zip(
                base_weights,
                residuals,
                strict=True,
            )
        ]

    final_residuals = [
        target
        - coefficients[0]
        - sum(
            weight * value
            for weight, value in zip(
                coefficients[1:],
                row,
                strict=True,
            )
        )
        for row, target in zip(
            standardized,
            targets,
            strict=True,
        )
    ]
    residual_center = median(final_residuals)
    residual_mad = median(
        abs(value - residual_center)
        for value in final_residuals
    )
    residual_scale = max(
        0.10,
        1.4826 * residual_mad,
    )
    return HealthyResidualModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        weights=tuple(coefficients[1:]),
        bias=coefficients[0],
        residual_center=residual_center,
        residual_scale=residual_scale,
    )


def _feature_values(
    *,
    rows: Sequence[Sequence[float]],
    accessor: FeatureAccessor,
    name: str,
    applicable_name: str,
) -> list[tuple[int, float]]:
    return [
        (index, accessor.get(list(row), name))
        for index, row in enumerate(rows)
        if accessor.get(
            list(row),
            applicable_name,
            default=0.0,
        )
        > 0.5
        and accessor.get(
            list(row),
            applicable_name.replace(
                "__applicable",
                "__missing",
            ),
            default=0.0,
        )
        < 0.5
    ]


def confirm_observable_impact(
    *,
    window: dict[str, Any],
    score_threshold: float = 0.55,
    minimum_families: int = 2,
    minimum_node_fraction: float = 0.08,
    require_causal_consistency: bool = True,
) -> ImpactConfirmation:
    """Confirm impact without classifier scores or simulator annotations."""

    accessor = FeatureAccessor(
        list(window["node_feature_names"])
    )
    rows = [
        [float(value) for value in row]
        for row in window["node_features"]
    ]
    node_count = max(1, len(rows))
    family_nodes: dict[str, set[int]] = defaultdict(set)
    strengths: dict[str, float] = {}
    persistent_families: set[str] = set()

    def positive_family(
        family: str,
        prefix: str,
        *,
        absolute_threshold: float | None = None,
    ) -> None:
        applicable = f"{prefix}__applicable"
        z_values = _feature_values(
            rows=rows,
            accessor=accessor,
            name=f"history__{prefix}__robust_z",
            applicable_name=applicable,
        )
        latest_values = _feature_values(
            rows=rows,
            accessor=accessor,
            name=f"{prefix}__latest",
            applicable_name=applicable,
        )
        persistence_values = _feature_values(
            rows=rows,
            accessor=accessor,
            name=(
                f"history__{prefix}"
                "__persistence_count"
            ),
            applicable_name=applicable,
        )
        z_strength = max(
            (
                max(0.0, value) / 3.0
                for _, value in z_values
            ),
            default=0.0,
        )
        absolute_strength = 0.0
        if absolute_threshold is not None:
            absolute_strength = max(
                (
                    max(
                        0.0,
                        value - absolute_threshold,
                    )
                    / max(1.0, absolute_threshold)
                    for _, value in latest_values
                ),
                default=0.0,
            )
        strength = min(
            1.0,
            max(z_strength, absolute_strength),
        )
        strengths[family] = max(
            strengths.get(family, 0.0),
            strength,
        )
        for index, value in z_values:
            if value >= 1.5:
                family_nodes[family].add(index)
        if absolute_threshold is not None:
            for index, value in latest_values:
                if value > absolute_threshold:
                    family_nodes[family].add(index)
        if any(
            value >= 2.0
            for _, value in persistence_values
        ):
            persistent_families.add(family)

    session_prefix = "iptv__session__active_count"
    session_applicable = (
        f"{session_prefix}__applicable"
    )
    session_z = _feature_values(
        rows=rows,
        accessor=accessor,
        name=(
            f"history__{session_prefix}__robust_z"
        ),
        applicable_name=session_applicable,
    )
    session_delta = _feature_values(
        rows=rows,
        accessor=accessor,
        name=f"history__{session_prefix}__delta",
        applicable_name=session_applicable,
    )
    session_strength = min(
        1.0,
        max(
            [
                0.0,
                *(
                    max(0.0, -value) / 3.0
                    for _, value in session_z
                ),
                *(
                    max(0.0, -value) / 5.0
                    for _, value in session_delta
                ),
            ]
        ),
    )
    strengths["service_traffic_decline"] = (
        session_strength
    )
    for index, value in session_z:
        if value <= -1.5:
            family_nodes[
                "service_traffic_decline"
            ].add(index)

    availability_prefix = (
        "iptv__catchup__service_availability"
    )
    availability_applicable = (
        f"{availability_prefix}__applicable"
    )
    availability_z = _feature_values(
        rows=rows,
        accessor=accessor,
        name=(
            f"history__{availability_prefix}"
            "__robust_z"
        ),
        applicable_name=availability_applicable,
    )
    availability_latest = _feature_values(
        rows=rows,
        accessor=accessor,
        name=f"{availability_prefix}__latest",
        applicable_name=availability_applicable,
    )
    availability_strength = min(
        1.0,
        max(
            [
                0.0,
                *(
                    max(0.0, -value) / 3.0
                    for _, value in availability_z
                ),
                *(
                    max(0.0, 98.0 - value) / 20.0
                    for _, value in availability_latest
                ),
            ]
        ),
    )
    strengths["service_availability_decline"] = (
        availability_strength
    )
    for index, value in availability_z:
        if value <= -1.5:
            family_nodes[
                "service_availability_decline"
            ].add(index)
    for index, value in availability_latest:
        if value <= 91.0:
            family_nodes[
                "service_availability_decline"
            ].add(index)

    positive_family(
        "recording_errors",
        "iptv__catchup__recording_failures",
        absolute_threshold=0.0,
    )
    positive_family(
        "storage_latency",
        "system__disk__io_latency",
        absolute_threshold=20.0,
    )
    positive_family(
        "storage_errors",
        "system__disk__io_errors",
        absolute_threshold=0.0,
    )
    positive_family(
        "storage_saturation",
        "system__disk__utilization",
        absolute_threshold=80.0,
    )
    positive_family(
        "process_restarts",
        "system__process__restart_count",
        absolute_threshold=0.0,
    )

    selected_families = tuple(
        sorted(
            family
            for family, strength in strengths.items()
            if strength >= 0.35
        )
    )
    affected_nodes = set().union(
        *(
            family_nodes[family]
            for family in selected_families
        )
    ) if selected_families else set()
    breadth = len(affected_nodes) / node_count

    causal_consistent = False
    if len(selected_families) >= 2:
        for left_position, left in enumerate(
            selected_families
        ):
            for right in selected_families[
                left_position + 1 :
            ]:
                if (
                    family_nodes[left]
                    & family_nodes[right]
                ):
                    causal_consistent = True
                    break
            if causal_consistent:
                break
    if not causal_consistent and affected_nodes:
        for source, target in window["edge_index"]:
            if (
                int(source) in affected_nodes
                and int(target) in affected_nodes
            ):
                causal_consistent = True
                break
    ordered_strengths = sorted(
        strengths.values(),
        reverse=True,
    )
    strongest = (
        ordered_strengths[0]
        if ordered_strengths
        else 0.0
    )
    second = (
        ordered_strengths[1]
        if len(ordered_strengths) > 1
        else 0.0
    )
    score = min(
        1.0,
        0.45 * strongest
        + 0.35 * second
        + 0.10 * min(1.0, breadth / 0.10)
        + 0.10 * float(causal_consistent),
    )
    breadth_confirmed = (
        len(selected_families) >= minimum_families
        or breadth >= minimum_node_fraction
    )
    service_impact_confirmed = (
        "service_availability_decline"
        in selected_families
        or (
            "service_traffic_decline"
            in selected_families
            and (
                "recording_errors"
                in persistent_families
            )
        )
    )
    confirmed = (
        score >= score_threshold
        and breadth_confirmed
        and service_impact_confirmed
        and (
            causal_consistent
            or not require_causal_consistency
        )
    )
    return ImpactConfirmation(
        confirmed=confirmed,
        score=score,
        family_count=len(selected_families),
        affected_node_fraction=breadth,
        causal_consistent=causal_consistent,
        families=selected_families,
    )


class DualPathEpisodeGate:
    """Support-first gate with impact fast path and residual CUSUM."""

    def __init__(
        self,
        *,
        config: DualPathConfig,
        residual_model: HealthyResidualModel | None,
    ) -> None:
        if config.use_residual and residual_model is None:
            raise ValueError(
                "Residual mode requires a healthy residual model"
            )
        self.config = config
        self.residual_model = residual_model

    def run(
        self,
        *,
        record: EpisodeRecord,
        probabilities: Sequence[float],
        metadata: dict[str, Any],
        metadata_sequence: Sequence[dict[str, Any]] | None = None,
        support: Sequence[SupportAssessment] | None = None,
    ) -> ScenarioEpisodeResult:
        count = len(record.windows)
        if len(probabilities) != count:
            raise ValueError(
                "A probability is required for every window"
            )
        support = support or (
            [SupportAssessment()] * count
        )
        if len(support) != count:
            raise ValueError(
                "Support assessments must align with windows"
            )
        metadata_rows = (
            tuple(metadata_sequence)
            if metadata_sequence is not None
            else (metadata,) * count
        )
        if len(metadata_rows) != count:
            raise ValueError(
                "Metadata rows must align with windows"
            )

        state = GateState.HEALTHY
        slow_statistic = 0.0
        suspect_age = 0
        recovery_age = 0
        cooldown = 0
        prior_logits: list[float] = []
        decisions: list[TemporalDecision] = []

        for (
            window,
            raw_probability,
            assessment,
            window_metadata,
        ) in zip(
            record.windows,
            probabilities,
            support,
            metadata_rows,
            strict=True,
        ):
            probability = min(
                1.0,
                max(0.0, float(raw_probability)),
            )
            if self.config.use_residual:
                assert self.residual_model is not None
                residual = (
                    self.residual_model.standardized_residual(
                        window=window,
                        probability=probability,
                        metadata=window_metadata,
                        prior_logits=prior_logits,
                    )
                )
            else:
                residual = (
                    _logit(probability)
                    - _logit(
                        self.config.raw_probability_center
                    )
                )
            prior_logits.append(_logit(probability))
            impact = confirm_observable_impact(
                window=window,
                score_threshold=(
                    self.config.impact_score_threshold
                ),
                minimum_families=(
                    self.config.impact_minimum_families
                ),
                minimum_node_fraction=(
                    self.config.impact_minimum_node_fraction
                ),
                require_causal_consistency=(
                    self.config.require_causal_consistency
                ),
            )
            cooldown = max(0, cooldown - 1)

            if (
                self.config.enable_support_sentinel
                and assessment.status
                is SupportStatus.UNSUPPORTED
            ):
                visible_state = (
                    GateState.ESCALATE
                    if probability
                    >= self.config.unsupported_escalation_probability
                    else GateState.UNKNOWN
                )
                decisions.append(
                    TemporalDecision(
                        window_id=str(
                            window["window_id"]
                        ),
                        timestamp=_parse_datetime(
                            window["observation_cutoff"]
                        ),
                        probability=probability,
                        evidence=residual,
                        state=visible_state,
                        support_status=assessment.status,
                        support_score=(
                            assessment.support_score
                        ),
                        residual=residual,
                        impact_score=impact.score,
                        slow_statistic=slow_statistic,
                        activation_path="unsupported",
                    )
                )
                continue

            slow_input = max(
                -self.config.slow_input_cap,
                min(
                    self.config.slow_input_cap,
                    residual,
                ),
            )
            slow_statistic = max(
                0.0,
                slow_statistic
                + slow_input
                - self.config.slow_reference_kappa,
            )
            slow_triggered = (
                self.config.enable_slow_path
                and slow_statistic
                >= self.config.slow_threshold
                and cooldown == 0
            )
            fast_evidence = (
                probability
                >= self.config.fast_probability_threshold
                or residual
                >= self.config.fast_residual_threshold
            )
            fast_confirmation = (
                impact.confirmed
                if self.config.require_impact_confirmation
                else True
            )
            fast_triggered = (
                self.config.enable_fast_path
                and fast_evidence
                and fast_confirmation
                and (
                    cooldown == 0
                    or impact.confirmed
                )
            )
            if fast_triggered and slow_triggered:
                activation_path = "both"
            elif fast_triggered:
                activation_path = "fast"
            elif slow_triggered:
                activation_path = "slow"
            else:
                activation_path = "none"

            model_suspect = (
                probability
                >= self.config.suspect_probability_threshold
                or residual
                >= self.config.suspect_residual_threshold
            )
            clear_evidence = (
                residual
                <= self.config.recovery_residual_threshold
                and not impact.confirmed
                and probability
                < self.config.suspect_probability_threshold
            )

            if state is GateState.HEALTHY:
                if fast_triggered or slow_triggered:
                    state = GateState.INCIDENT
                    recovery_age = 0
                elif model_suspect:
                    state = GateState.SUSPECT
                    suspect_age = 1
            elif state is GateState.SUSPECT:
                suspect_age += 1
                if fast_triggered or slow_triggered:
                    state = GateState.INCIDENT
                    recovery_age = 0
                elif (
                    not model_suspect
                    or suspect_age
                    >= self.config.maximum_suspect_windows
                ):
                    state = GateState.HEALTHY
                    suspect_age = 0
            elif state is GateState.INCIDENT:
                if clear_evidence:
                    recovery_age += 1
                    if (
                        recovery_age
                        >= self.config.minimum_recovery_windows
                    ):
                        state = GateState.RECOVERY
                        slow_statistic = 0.0
                else:
                    recovery_age = 0
            elif state is GateState.RECOVERY:
                if fast_triggered or slow_triggered:
                    state = GateState.INCIDENT
                    recovery_age = 0
                elif clear_evidence:
                    state = GateState.HEALTHY
                    cooldown = self.config.cooldown_windows
                    recovery_age = 0
                else:
                    state = GateState.SUSPECT
                    suspect_age = 1

            decisions.append(
                TemporalDecision(
                    window_id=str(window["window_id"]),
                    timestamp=_parse_datetime(
                        window["observation_cutoff"]
                    ),
                    probability=probability,
                    evidence=residual,
                    state=state,
                    support_status=assessment.status,
                    support_score=assessment.support_score,
                    residual=residual,
                    impact_score=impact.score,
                    slow_statistic=slow_statistic,
                    fast_triggered=fast_triggered,
                    slow_triggered=slow_triggered,
                    activation_path=activation_path,
                )
            )

        return ScenarioEpisodeResult(
            scenario_id=record.scenario_id,
            split=record.split,
            topology_group=record.topology_group,
            confounder_family=(
                record.confounder_family
            ),
            healthy_control=record.healthy_control,
            failure_family=record.failure_family,
            decisions=tuple(decisions),
            truth=tuple(
                bool(
                    int(
                        window["targets"][
                            "current_incident"
                        ]
                    )
                )
                for window in record.windows
            ),
            fault_duration_hours=(
                record.fault_duration_hours
            ),
        )


def stage2_eligible_decision(
    decision: TemporalDecision,
) -> bool:
    """Only supported, confirmed incidents may enter root-cause ranking."""

    return (
        decision.state is GateState.INCIDENT
        and decision.support_status
        is SupportStatus.IN_DISTRIBUTION
    )


def default_dual_path_config_grid() -> tuple[
    DualPathConfig,
    ...,
]:
    """Preregistered validation-only policy grid."""

    slow_settings = (
        (0.50, 6.0),
        (0.75, 10.0),
        (1.00, 14.0),
        (1.50, 20.0),
    )
    return tuple(
        DualPathConfig(
            fast_probability_threshold=fast_probability,
            fast_residual_threshold=fast_residual,
            impact_score_threshold=impact_threshold,
            impact_minimum_families=minimum_families,
            slow_reference_kappa=kappa,
            slow_threshold=slow_threshold,
            minimum_recovery_windows=recovery_windows,
        )
        for fast_probability in (0.90, 0.97)
        for fast_residual in (3.0, 5.0)
        for impact_threshold in (0.65, 0.80)
        for minimum_families in (2, 3)
        for kappa, slow_threshold in slow_settings
        for recovery_windows in (2, 3)
    )


def registered_dual_path_ablation_configs(
    calibrated: DualPathConfig,
) -> dict[str, DualPathConfig | None]:
    """A–H protocol; A and B use their frozen external references."""

    return {
        "A_independent_plus_sentinel": None,
        "B_persistence_plus_sentinel": None,
        "C_sequential_raw_probability": replace(
            calibrated,
            use_residual=False,
            enable_fast_path=False,
        ),
        "D_sequential_residual": replace(
            calibrated,
            use_residual=True,
            enable_fast_path=False,
        ),
        "E_probability_fast_override": replace(
            calibrated,
            use_residual=True,
            enable_fast_path=True,
            require_impact_confirmation=False,
        ),
        "F_impact_fast_only": replace(
            calibrated,
            use_residual=True,
            enable_fast_path=True,
            require_impact_confirmation=True,
            enable_slow_path=False,
        ),
        "G_full_dual_path": calibrated,
        "H_full_without_sentinel": replace(
            calibrated,
            enable_support_sentinel=False,
        ),
    }


def _validation_failures(
    metrics: dict[str, Any],
) -> list[str]:
    failures = []
    if (
        metrics[
            "healthy_scenario_false_selection_rate"
        ]
        > 0.10
    ):
        failures.append("scenario_false_selection")
    if (
        metrics[
            "healthy_scenario_false_selection_ucb"
        ]
        > 0.10
    ):
        failures.append("global_risk_ucb")
    if metrics["worst_confounder_point_risk"] > 0.10:
        failures.append("confounder_point_risk")
    if metrics["worst_confounder_ucb"] > 0.15:
        failures.append("confounder_risk_ucb")
    if metrics["incident_episode_recall"] < 0.80:
        failures.append("episode_recall")
    if any(
        float(row["episode_recall"]) < 0.70
        for row in metrics[
            "failure_family_metrics"
        ].values()
    ):
        failures.append("failure_family_recall")
    median_delay = metrics[
        "median_detection_delay_hours"
    ]
    if median_delay is None or median_delay > 2.0:
        failures.append("median_detection_delay")
    p90_delay = metrics[
        "p90_detection_delay_hours"
    ]
    if p90_delay is None or p90_delay > 4.0:
        failures.append("p90_detection_delay")
    capacity = metrics[
        "failure_family_metrics"
    ].get(
        "storage_capacity_saturation",
        {},
    ).get("median_detection_delay_hours")
    if capacity is None or capacity > 4.0:
        failures.append("capacity_detection_delay")
    if metrics["mean_fragmentation"] > 0.05:
        failures.append("fragmentation")
    if (
        metrics[
            "unsupported_safe_semantics_rate"
        ]
        < 1.0
    ):
        failures.append("unsupported_safe_semantics")
    return failures


def calibrate_dual_path_gate(
    *,
    validation_records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    scenario_metadata: dict[str, dict[str, Any]],
    residual_model: HealthyResidualModel,
    candidate_configs: Sequence[DualPathConfig],
    support: (
        dict[str, Sequence[SupportAssessment]]
        | None
    ) = None,
) -> DualPathCalibrationResult:
    """Select exclusively on validation topology clusters."""

    if not validation_records or not candidate_configs:
        raise ValueError(
            "Calibration requires validation records and configs"
        )
    evaluated = []
    candidate_summaries = []
    for config in candidate_configs:
        gate = DualPathEpisodeGate(
            config=config,
            residual_model=(
                residual_model
                if config.use_residual
                else None
            ),
        )
        results = [
            gate.run(
                record=record,
                probabilities=probabilities[
                    record.scenario_id
                ],
                metadata=scenario_metadata[
                    record.scenario_id
                ],
                support=(
                    support[record.scenario_id]
                    if support is not None
                    else None
                ),
            )
            for record in validation_records
        ]
        metrics = summarize_episode_results(results)
        failures = _validation_failures(metrics)
        delay = metrics[
            "median_detection_delay_hours"
        ]
        objective = (
            float(metrics["incident_episode_recall"])
            - 2.0
            * float(
                metrics[
                    "healthy_scenario_false_selection_rate"
                ]
            )
            - 0.02 * float(delay or 8.0)
            - 0.10
            * float(metrics["mean_fragmentation"])
        )
        evaluated.append(
            (
                not failures,
                objective,
                config,
                metrics,
                failures,
            )
        )
        candidate_summaries.append(
            {
                "config": asdict(config),
                "feasible": not failures,
                "objective": objective,
                "feasibility_failures": (
                    tuple(failures)
                ),
                "healthy_scenario_risk": metrics[
                    "healthy_scenario_false_selection_rate"
                ],
                "healthy_topology_risk": metrics[
                    "healthy_topology_false_selection_rate"
                ],
                "healthy_topology_risk_ucb": metrics[
                    "healthy_scenario_false_selection_ucb"
                ],
                "worst_confounder_point_risk": metrics[
                    "worst_confounder_point_risk"
                ],
                "worst_confounder_ucb": metrics[
                    "worst_confounder_ucb"
                ],
                "episode_recall": metrics[
                    "incident_episode_recall"
                ],
                "median_delay_hours": metrics[
                    "median_detection_delay_hours"
                ],
                "p90_delay_hours": metrics[
                    "p90_detection_delay_hours"
                ],
                "mean_fragmentation": metrics[
                    "mean_fragmentation"
                ],
            }
        )
    feasible = [
        candidate
        for candidate in evaluated
        if candidate[0]
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda candidate: candidate[1],
        )
    else:
        def normalized_violation(
            candidate: tuple[
                bool,
                float,
                DualPathConfig,
                dict[str, Any],
                list[str],
            ],
        ) -> float:
            metrics = candidate[3]
            family_recall = min(
                (
                    float(row["episode_recall"])
                    for row in metrics[
                        "failure_family_metrics"
                    ].values()
                ),
                default=0.0,
            )
            delay = metrics[
                "median_detection_delay_hours"
            ]
            p90 = metrics[
                "p90_detection_delay_hours"
            ]
            capacity = metrics[
                "failure_family_metrics"
            ].get(
                "storage_capacity_saturation",
                {},
            ).get(
                "median_detection_delay_hours"
            )
            return sum(
                (
                    max(
                        0.0,
                        metrics[
                            "healthy_scenario_false_selection_rate"
                        ]
                        / 0.10
                        - 1.0,
                    ),
                    max(
                        0.0,
                        metrics[
                            "healthy_scenario_false_selection_ucb"
                        ]
                        / 0.10
                        - 1.0,
                    ),
                    max(
                        0.0,
                        metrics[
                            "worst_confounder_point_risk"
                        ]
                        / 0.10
                        - 1.0,
                    ),
                    max(
                        0.0,
                        metrics[
                            "worst_confounder_ucb"
                        ]
                        / 0.15
                        - 1.0,
                    ),
                    max(
                        0.0,
                        0.80
                        - metrics[
                            "incident_episode_recall"
                        ],
                    )
                    / 0.20,
                    max(
                        0.0,
                        0.70 - family_recall,
                    )
                    / 0.30,
                    max(
                        0.0,
                        float(delay or 10.0) / 2.0
                        - 1.0,
                    ),
                    max(
                        0.0,
                        float(p90 or 10.0) / 4.0
                        - 1.0,
                    ),
                    max(
                        0.0,
                        float(capacity or 10.0) / 4.0
                        - 1.0,
                    ),
                    max(
                        0.0,
                        metrics["mean_fragmentation"]
                        / 0.05
                        - 1.0,
                    ),
                )
            )

        selected = min(
            evaluated,
            key=lambda candidate: (
                normalized_violation(candidate),
                -candidate[1],
            ),
        )
    return DualPathCalibrationResult(
        config=selected[2],
        feasible=selected[0],
        objective=selected[1],
        metrics=selected[3],
        evaluated_configurations=len(evaluated),
        feasibility_failures=tuple(selected[4]),
        candidate_summaries=tuple(
            sorted(
                candidate_summaries,
                key=lambda row: (
                    not row["feasible"],
                    -float(row["objective"]),
                ),
            )
        ),
    )
