from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from app.domain.nrim.baselines.temporal_episode_gate import (
    exact_binomial_upper_bound,
)

from .contracts import (
    AdjudicationConfidence,
    DecisionState,
    IncidentAdjudication,
    IncidentSeverity,
    PredictionEnvelope,
    ReviewAnswer,
)


POLICIES = ("F", "S", "D")


@dataclass(frozen=True)
class HealthyExposure:
    deployment_pseudonym: str
    topology_family: str
    healthy_hours: float
    nonactionable_episodes: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.healthy_hours <= 0.0:
            raise ValueError("Healthy exposure must be positive")
        if set(self.nonactionable_episodes) != set(POLICIES):
            raise ValueError("Healthy exposure requires F/S/D episode counts")
        if any(value < 0 for value in self.nonactionable_episodes.values()):
            raise ValueError("Episode counts cannot be negative")


@dataclass(frozen=True)
class IncidentCase:
    incident_id: str
    deployment_pseudonym: str
    failure_family: str
    severity: IncidentSeverity
    onset_utc: datetime
    recovery_utc: datetime | None
    in_support: bool
    data_quality_blocked: bool
    detections: Mapping[str, datetime | None]
    fragments: Mapping[str, int]
    confirmed_root_component: str | None
    stage2_top_k: tuple[str, ...]
    ranking_available: bool
    engineer_top3_useful: bool | None

    def __post_init__(self) -> None:
        if set(self.detections) != set(POLICIES):
            raise ValueError("Incident case requires F/S/D detection results")
        if set(self.fragments) != set(POLICIES):
            raise ValueError("Incident case requires F/S/D fragmentation")


@dataclass(frozen=True)
class HumanUtilityCase:
    investigation_id: str
    baseline_minutes_to_hypothesis: float
    advisory_minutes_to_hypothesis: float
    baseline_components_investigated: int
    advisory_components_investigated: int
    override_reason_recorded: bool

    def __post_init__(self) -> None:
        if min(
            self.baseline_minutes_to_hypothesis,
            self.advisory_minutes_to_hypothesis,
        ) < 0.0:
            raise ValueError("Investigation durations cannot be negative")
        if min(
            self.baseline_components_investigated,
            self.advisory_components_investigated,
        ) < 0:
            raise ValueError("Investigation component counts cannot be negative")


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _runs(values: Sequence[bool]) -> int:
    count = 0
    active = False
    for value in values:
        if value and not active:
            count += 1
        active = value
    return count


def _poisson_cdf(events: int, rate: float) -> float:
    term = math.exp(-rate)
    total = term
    for index in range(1, events + 1):
        term *= rate / index
        total += term
    return min(1.0, total)


def exact_poisson_rate_upper(
    *,
    events: int,
    exposure: float,
    confidence: float = 0.95,
) -> float:
    """One-sided exact Poisson upper rate under the stated Poisson model."""

    if events < 0 or exposure <= 0.0:
        raise ValueError("Events must be nonnegative and exposure positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between zero and one")
    alpha = 1.0 - confidence
    low = float(events)
    high = max(1.0, float(events + 1))
    while _poisson_cdf(events, high) > alpha:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _poisson_cdf(events, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high / exposure


def exact_poisson_rate_lower(
    *,
    events: int,
    exposure: float,
    confidence: float = 0.95,
) -> float:
    """One-sided exact Poisson lower rate under the Poisson model."""

    if events < 0 or exposure <= 0.0:
        raise ValueError("Events must be nonnegative and exposure positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between zero and one")
    if events == 0:
        return 0.0
    target = confidence
    low = 0.0
    high = float(events)
    while _poisson_cdf(events - 1, high) > target:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _poisson_cdf(events - 1, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return high / exposure


def exact_binomial_recall_lower(
    *,
    detected: int,
    trials: int,
    confidence: float = 0.95,
) -> float | None:
    if trials == 0:
        return None
    misses = trials - detected
    return 1.0 - exact_binomial_upper_bound(
        events=misses,
        trials=trials,
        confidence=confidence,
    ).upper_confidence_bound


def _random_effects_rate_sensitivity(
    rows: Sequence[tuple[int, float]],
) -> dict[str, float] | None:
    """Approximate random-effects sensitivity on log Poisson rates.

    A 0.5 continuity correction is explicit because zero-event deployments
    otherwise have undefined log rates. This is a heterogeneity sensitivity,
    never a substitute for the exact deployment intervals.
    """

    if len(rows) < 10:
        return None
    log_rates = [
        math.log((events + 0.5) / exposure)
        for events, exposure in rows
    ]
    variances = [1.0 / (events + 0.5) for events, _ in rows]
    fixed_weights = [1.0 / value for value in variances]
    total_weight = sum(fixed_weights)
    fixed_mean = sum(
        weight * value
        for weight, value in zip(
            fixed_weights,
            log_rates,
            strict=True,
        )
    ) / total_weight
    q = sum(
        weight * (value - fixed_mean) ** 2
        for weight, value in zip(
            fixed_weights,
            log_rates,
            strict=True,
        )
    )
    denominator = total_weight - (
        sum(weight**2 for weight in fixed_weights) / total_weight
    )
    tau_squared = max(
        0.0,
        (q - (len(rows) - 1)) / denominator,
    )
    random_weights = [
        1.0 / (variance + tau_squared)
        for variance in variances
    ]
    random_total = sum(random_weights)
    pooled_log_rate = sum(
        weight * value
        for weight, value in zip(
            random_weights,
            log_rates,
            strict=True,
        )
    ) / random_total
    standard_error = math.sqrt(1.0 / random_total)
    i_squared = (
        max(0.0, (q - (len(rows) - 1)) / q)
        if q > 0.0
        else 0.0
    )
    return {
        "deployment_count": float(len(rows)),
        "pooled_rate_per_100_hours": 100.0
        * math.exp(pooled_log_rate),
        "lower_95_per_100_hours": 100.0
        * math.exp(pooled_log_rate - 1.959963984540054 * standard_error),
        "upper_95_per_100_hours": 100.0
        * math.exp(pooled_log_rate + 1.959963984540054 * standard_error),
        "tau_squared_log_rate": tau_squared,
        "i_squared": i_squared,
        "continuity_correction_events": 0.5,
    }


def _cluster_bootstrap(
    *,
    rows_by_deployment: Mapping[str, Sequence[Any]],
    statistic,
    seed: int,
    iterations: int = 2000,
) -> dict[str, float | int | None]:
    deployments = sorted(rows_by_deployment)
    if not deployments:
        return {
            "deployment_count": 0,
            "iterations": 0,
            "lower_95": None,
            "upper_95": None,
        }
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [
            deployments[rng.randrange(len(deployments))]
            for _ in deployments
        ]
        rows = [
            row
            for deployment in sample
            for row in rows_by_deployment[deployment]
        ]
        values.append(float(statistic(rows)))
    return {
        "deployment_count": len(deployments),
        "iterations": iterations,
        "lower_95": _quantile(values, 0.025),
        "upper_95": _quantile(values, 0.975),
    }


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    categories = sorted(
        {value for pair in pairs for value in pair}
    )
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_rates = {
        category: sum(left == category for left, _ in pairs) / len(pairs)
        for category in categories
    }
    right_rates = {
        category: sum(right == category for _, right in pairs) / len(pairs)
        for category in categories
    }
    expected = sum(
        left_rates[category] * right_rates[category]
        for category in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_assessment(
    pairs: Sequence[tuple[str, str]],
    *,
    minimum_pairs: int = 20,
) -> dict[str, Any]:
    if len(pairs) < minimum_pairs:
        return {
            "status": "not_reportable_preregistered_reason",
            "reason": "insufficient_double_review_sample",
            "pair_count": len(pairs),
            "kappa": None,
        }
    left_categories = {left for left, _ in pairs}
    right_categories = {right for _, right in pairs}
    if len(left_categories) < 2 or len(right_categories) < 2:
        return {
            "status": "not_reportable_preregistered_reason",
            "reason": "degenerate_category_prevalence",
            "pair_count": len(pairs),
            "kappa": None,
        }
    return {
        "status": "interpretable",
        "reason": None,
        "pair_count": len(pairs),
        "kappa": cohen_kappa(pairs),
    }


def _latest_adjudications(
    adjudications: Iterable[IncidentAdjudication],
) -> list[IncidentAdjudication]:
    latest = {}
    for item in adjudications:
        key = item.adjudication_id
        if (
            key not in latest
            or latest[key].adjudication_version
            < item.adjudication_version
        ):
            latest[key] = item
    return list(latest.values())


def incident_cases_from_evidence(
    *,
    predictions: Sequence[PredictionEnvelope],
    adjudications: Sequence[IncidentAdjudication],
) -> list[IncidentCase]:
    """Join prospective evidence to final labels without changing predictions."""

    by_deployment: dict[str, list[PredictionEnvelope]] = defaultdict(list)
    for prediction in predictions:
        by_deployment[prediction.deployment_pseudonym].append(prediction)
    cases = []
    for label in _latest_adjudications(adjudications):
        if (
            label.customer_impact_present is not ReviewAnswer.YES
            or label.impact_onset_utc is None
        ):
            continue
        rows = sorted(
            (
                item
                for item in by_deployment[label.deployment_pseudonym]
                if label.review_window_start_utc
                <= item.decision_cutoff_utc
                <= label.review_window_end_utc
            ),
            key=lambda item: item.decision_cutoff_utc,
        )
        detections: dict[str, datetime | None] = {}
        fragments = {}
        for policy in POLICIES:
            selected = [
                item.counterfactual_policy_states.get(policy)
                is DecisionState.INCIDENT
                for item in rows
            ]
            detections[policy] = next(
                (
                    item.decision_cutoff_utc
                    for item, selected_item in zip(
                        rows,
                        selected,
                        strict=True,
                    )
                    if selected_item
                ),
                None,
            )
            fragments[policy] = max(0, _runs(selected) - 1)
        supported_rows = [
            item
            for item in rows
            if item.final_decision
            not in {
                DecisionState.UNKNOWN,
                DecisionState.ESCALATE,
                DecisionState.DATA_QUALITY_ESCALATION,
            }
        ]
        ranking_row = next(
            (
                item
                for item in rows
                if item.final_decision is DecisionState.INCIDENT
                and item.stage2_top_k
            ),
            None,
        )
        cases.append(
            IncidentCase(
                incident_id=label.adjudication_id,
                deployment_pseudonym=label.deployment_pseudonym,
                failure_family=label.root_cause_family,
                severity=label.severity,
                onset_utc=label.impact_onset_utc,
                recovery_utc=label.recovery_utc,
                in_support=bool(supported_rows),
                data_quality_blocked=any(
                    item.final_decision
                    is DecisionState.DATA_QUALITY_ESCALATION
                    for item in rows
                ),
                detections=detections,
                fragments=fragments,
                confirmed_root_component=label.confirmed_root_component,
                stage2_top_k=(
                    tuple(
                        item.component_pseudonym
                        for item in ranking_row.stage2_top_k
                    )
                    if ranking_row is not None
                    else ()
                ),
                ranking_available=ranking_row is not None,
                engineer_top3_useful=label.nrim_top3_useful,
            )
        )
    return cases


def _relative_improvement(baseline: Sequence[float], new: Sequence[float]) -> (
    float | None
):
    if not baseline or not new or len(baseline) != len(new):
        return None
    baseline_median = median(baseline)
    if baseline_median <= 0.0:
        return None
    return (baseline_median - median(new)) / baseline_median


class ProspectiveMetricsEngine:
    def __init__(
        self,
        *,
        decision_interval_hours: float,
        bootstrap_seed: int = 707,
    ) -> None:
        if decision_interval_hours <= 0.0:
            raise ValueError("Decision interval must be positive")
        self.decision_interval_hours = decision_interval_hours
        self.bootstrap_seed = bootstrap_seed

    def _detection(
        self,
        incidents: Sequence[IncidentCase],
        policy: str,
    ) -> dict[str, Any]:
        included = list(incidents)
        conditionally_evaluable = [
            item
            for item in included
            if item.in_support and not item.data_quality_blocked
        ]
        detected = [
            item for item in included if item.detections[policy] is not None
        ]
        conditional_detected = [
            item
            for item in conditionally_evaluable
            if item.detections[policy] is not None
        ]
        misses = len(included) - len(detected)
        recall = len(detected) / len(included) if included else None
        miss_bound = exact_binomial_upper_bound(
            events=misses,
            trials=len(included),
        )
        delays = [
            (
                item.detections[policy] - item.onset_utc
            ).total_seconds()
            / 3600.0
            for item in detected
            if item.detections[policy] is not None
        ]
        severe = [
            item
            for item in included
            if item.severity
            in {IncidentSeverity.HIGH, IncidentSeverity.CRITICAL}
        ]
        severe_detected = sum(
            item.detections[policy] is not None for item in severe
        )
        by_family = {}
        for family in sorted({item.failure_family for item in included}):
            rows = [
                item for item in included if item.failure_family == family
            ]
            by_family[family] = {
                "incident_count": len(rows),
                "recall": (
                    sum(
                        item.detections[policy] is not None
                        for item in rows
                    )
                    / len(rows)
                ),
            }
        grouped: dict[str, list[IncidentCase]] = defaultdict(list)
        for item in included:
            grouped[item.deployment_pseudonym].append(item)

        def recall_stat(rows: Sequence[IncidentCase]) -> float:
            return (
                sum(item.detections[policy] is not None for item in rows)
                / len(rows)
                if rows
                else 0.0
            )

        return {
            "included_incident_count": len(included),
            "in_support_incident_count": len(conditionally_evaluable),
            "unsupported_or_data_blocked_incident_count": (
                len(included) - len(conditionally_evaluable)
            ),
            "detected_incident_count": len(detected),
            "episode_recall": recall,
            "episode_recall_denominator": (
                "all included adjudicated incident episodes"
            ),
            "episode_recall_exact_lower_95": (
                1.0 - miss_bound.upper_confidence_bound
                if included
                else None
            ),
            "conditional_in_support_recall": (
                len(conditional_detected) / len(conditionally_evaluable)
                if conditionally_evaluable
                else None
            ),
            "severe_incident_count": len(severe),
            "severe_recall": (
                severe_detected / len(severe) if severe else None
            ),
            "severe_recall_exact_lower_95": (
                exact_binomial_recall_lower(
                    detected=severe_detected,
                    trials=len(severe),
                )
                if severe
                else None
            ),
            "median_delay_hours": (
                median(delays) if delays else None
            ),
            "p90_delay_hours": _quantile(delays, 0.90),
            "mean_extra_fragments": (
                mean(item.fragments[policy] for item in included)
                if included
                else None
            ),
            "failure_families": by_family,
            "deployment_cluster_bootstrap": _cluster_bootstrap(
                rows_by_deployment=grouped,
                statistic=recall_stat,
                seed=self.bootstrap_seed,
            ),
        }

    def _burden(
        self,
        exposures: Sequence[HealthyExposure],
        policy: str,
    ) -> dict[str, Any]:
        hours = sum(item.healthy_hours for item in exposures)
        events = sum(
            int(item.nonactionable_episodes[policy])
            for item in exposures
        )
        rate = 100.0 * events / hours if hours else None
        poisson_upper = (
            100.0
            * exact_poisson_rate_upper(
                events=events,
                exposure=hours,
            )
            if hours
            else None
        )
        grouped: dict[str, list[HealthyExposure]] = defaultdict(list)
        for item in exposures:
            grouped[item.deployment_pseudonym].append(item)

        def rate_stat(rows: Sequence[HealthyExposure]) -> float:
            total_hours = sum(item.healthy_hours for item in rows)
            total_events = sum(
                item.nonactionable_episodes[policy]
                for item in rows
            )
            return (
                100.0 * total_events / total_hours
                if total_hours
                else 0.0
            )

        bootstrap = _cluster_bootstrap(
            rows_by_deployment=grouped,
            statistic=rate_stat,
            seed=self.bootstrap_seed + 1,
        )
        deployment_rows = {}
        for deployment, rows in sorted(grouped.items()):
            deployment_hours = sum(item.healthy_hours for item in rows)
            deployment_events = sum(
                item.nonactionable_episodes[policy] for item in rows
            )
            deployment_rows[deployment] = {
                "healthy_hours": deployment_hours,
                "nonactionable_episode_count": deployment_events,
                "episodes_per_100_healthy_hours": (
                    100.0 * deployment_events / deployment_hours
                ),
                "exact_lower_95_per_100_hours": (
                    100.0
                    * exact_poisson_rate_lower(
                        events=deployment_events,
                        exposure=deployment_hours,
                        confidence=0.975,
                    )
                ),
                "exact_upper_95_per_100_hours": (
                    100.0
                    * exact_poisson_rate_upper(
                        events=deployment_events,
                        exposure=deployment_hours,
                        confidence=0.975,
                    )
                ),
            }
        bootstrap_upper = bootstrap["upper_95"]
        conservative_upper = (
            max(
                float(poisson_upper),
                float(bootstrap_upper),
            )
            if poisson_upper is not None
            and bootstrap_upper is not None
            else poisson_upper
        )
        return {
            "healthy_deployment_hours": hours,
            "nonactionable_episode_count": events,
            "episodes_per_100_healthy_hours": rate,
            "poisson_model_upper_95": poisson_upper,
            "deployment_cluster_bootstrap": bootstrap,
            "conservative_upper_95": conservative_upper,
            "by_deployment": deployment_rows,
            "heterogeneity": {
                "deployment_intervals_complete": all(
                    row["healthy_hours"] > 0.0
                    and row["exact_upper_95_per_100_hours"] is not None
                    for row in deployment_rows.values()
                ),
                "minimum_exposure_hours_for_rate_ratio": 100.0,
                "minimum_exposure_rule_applied": True,
                "maximum_rate_ratio_above_minimum_exposure": (
                    max(
                        row["episodes_per_100_healthy_hours"] / rate
                        for row in deployment_rows.values()
                        if row["healthy_hours"] >= 100.0
                    )
                    if rate
                    and any(
                        row["healthy_hours"] >= 100.0
                        for row in deployment_rows.values()
                    )
                    else None
                ),
                "inference_status": (
                    "random_effects_sensitivity"
                    if len(deployment_rows) >= 10
                    else "feasibility_only"
                ),
                "random_effects_sensitivity": (
                    _random_effects_rate_sensitivity(
                        [
                            (
                                int(row["nonactionable_episode_count"]),
                                float(row["healthy_hours"]),
                            )
                            for row in deployment_rows.values()
                        ]
                    )
                ),
                "assumption_note": (
                    "Exact deployment intervals condition on Poisson counts. "
                    "The random-effects result uses a documented 0.5-event "
                    "continuity correction and is sensitivity analysis only."
                ),
            },
        }

    def _ranking(self, incidents: Sequence[IncidentCase]) -> dict[str, Any]:
        eligible = [
            item
            for item in incidents
            if item.in_support
            and not item.data_quality_blocked
            and item.confirmed_root_component is not None
        ]
        ranked = [item for item in eligible if item.ranking_available]
        reciprocal = []
        hits1 = 0
        hits3 = 0
        for item in ranked:
            try:
                rank = (
                    item.stage2_top_k.index(
                        item.confirmed_root_component
                    )
                    + 1
                )
            except ValueError:
                rank = None
            reciprocal.append(1.0 / rank if rank else 0.0)
            hits1 += int(rank == 1)
            hits3 += int(rank is not None and rank <= 3)
        useful = [
            item.engineer_top3_useful
            for item in eligible
            if item.engineer_top3_useful is not None
        ]
        grouped: dict[str, list[IncidentCase]] = defaultdict(list)
        for item in eligible:
            grouped[item.deployment_pseudonym].append(item)

        def ranking_stat(
            rows: Sequence[IncidentCase],
            metric: str,
        ) -> float:
            available = [item for item in rows if item.ranking_available]
            if metric == "coverage":
                return len(available) / len(rows) if rows else 0.0
            values = []
            for item in available:
                try:
                    rank = (
                        item.stage2_top_k.index(
                            item.confirmed_root_component
                        )
                        + 1
                    )
                except ValueError:
                    rank = None
                if metric == "mrr":
                    values.append(1.0 / rank if rank else 0.0)
                elif metric == "hits1":
                    values.append(float(rank == 1))
                else:
                    values.append(
                        float(rank is not None and rank <= 3)
                    )
            return mean(values) if values else 0.0

        return {
            "eligible_confirmed_incidents": len(eligible),
            "ranked_incidents": len(ranked),
            "ranking_coverage": (
                len(ranked) / len(eligible) if eligible else None
            ),
            "mrr": mean(reciprocal) if reciprocal else None,
            "hits_at_1": hits1 / len(ranked) if ranked else None,
            "hits_at_3": hits3 / len(ranked) if ranked else None,
            "engineer_top3_usefulness": (
                sum(bool(value) for value in useful) / len(useful)
                if useful
                else None
            ),
            "deployment_stratified_uncertainty": {
                metric: _cluster_bootstrap(
                    rows_by_deployment=grouped,
                    statistic=(
                        lambda rows, selected=metric: ranking_stat(
                            rows,
                            selected,
                        )
                    ),
                    seed=self.bootstrap_seed + offset,
                )
                for offset, metric in enumerate(
                    ("mrr", "hits1", "hits3", "coverage"),
                    start=10,
                )
            },
        }

    @staticmethod
    def _support(predictions: Sequence[PredictionEnvelope]) -> dict[str, Any]:
        unsupported = [
            item
            for item in predictions
            if item.final_decision
            in {DecisionState.UNKNOWN, DecisionState.ESCALATE}
        ]
        blocked = [
            item
            for item in predictions
            if item.final_decision
            is DecisionState.DATA_QUALITY_ESCALATION
        ]
        unsafe = [
            item
            for item in unsupported + blocked
            if item.stage2_top_k
        ]
        by_deployment: dict[str, list[PredictionEnvelope]] = defaultdict(list)
        for item in predictions:
            by_deployment[item.deployment_pseudonym].append(item)
        return {
            "prediction_count": len(predictions),
            "unsupported_count": len(unsupported),
            "unsupported_rate": (
                len(unsupported) / len(predictions) if predictions else None
            ),
            "data_quality_blocked_count": len(blocked),
            "unsafe_stage2_count": len(unsafe),
            "unsupported_safe_semantics": not unsafe,
            "by_deployment": {
                deployment: {
                    "prediction_count": len(rows),
                    "unsupported_rate": (
                        sum(
                            item.final_decision
                            in {
                                DecisionState.UNKNOWN,
                                DecisionState.ESCALATE,
                            }
                            for item in rows
                        )
                        / len(rows)
                    ),
                    "data_quality_blocked_rate": (
                        sum(
                            item.final_decision
                            is DecisionState.DATA_QUALITY_ESCALATION
                            for item in rows
                        )
                        / len(rows)
                    ),
                }
                for deployment, rows in sorted(by_deployment.items())
            },
        }

    def _path_attribution(
        self,
        *,
        incidents: Sequence[IncidentCase],
        burden: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        fast = self._detection(incidents, "F")
        slow = self._detection(incidents, "S")
        dual = self._detection(incidents, "D")
        rows = [
            item
            for item in incidents
            if item.in_support and not item.data_quality_blocked
        ]
        fast_unique = [
            item
            for item in rows
            if item.detections["F"] is not None
            and item.detections["S"] is None
        ]
        slow_unique = [
            item
            for item in rows
            if item.detections["S"] is not None
            and item.detections["F"] is None
        ]
        fast_excess = (
            burden["F"]["episodes_per_100_healthy_hours"]
            - burden["S"]["episodes_per_100_healthy_hours"]
            if burden["F"]["episodes_per_100_healthy_hours"] is not None
            and burden["S"]["episodes_per_100_healthy_hours"] is not None
            else None
        )
        fast_delays = [
            (
                item.detections["F"] - item.onset_utc
            ).total_seconds()
            / 3600.0
            for item in rows
            if item.detections["F"] is not None
        ]
        slow_delays = [
            (
                item.detections["S"] - item.onset_utc
            ).total_seconds()
            / 3600.0
            for item in rows
            if item.detections["S"] is not None
        ]
        fast_delay_gain = (
            median(slow_delays) - median(fast_delays)
            if slow_delays and fast_delays
            else None
        )
        severe_fast_rescue = any(
            item.severity
            in {IncidentSeverity.HIGH, IncidentSeverity.CRITICAL}
            for item in fast_unique
        )
        short_rows = [
            item
            for item in rows
            if item.recovery_utc is not None
            and (
                item.recovery_utc - item.onset_utc
            ).total_seconds()
            / 3600.0
            <= 2.0 * self.decision_interval_hours
        ]

        def short_recall(policy: str) -> float | None:
            return (
                sum(
                    item.detections[policy] is not None
                    for item in short_rows
                )
                / len(short_rows)
                if short_rows
                else None
            )

        fast_short_increment = (
            short_recall("F") - short_recall("S")
            if short_recall("F") is not None
            and short_recall("S") is not None
            else None
        )
        fast_value = (
            (
                fast_delay_gain is not None
                and fast_delay_gain >= self.decision_interval_hours
            )
            or severe_fast_rescue
            or (
                fast_short_increment is not None
                and fast_short_increment >= 0.05
            )
        )
        slow_increment = (
            len(slow_unique) / len(rows) if rows else None
        )
        slow_value = (
            slow_increment is not None and slow_increment >= 0.05
        )
        fast_burden_ok = fast_excess is not None and fast_excess <= 0.02
        if fast_value and fast_burden_ok and slow_value:
            recommendation = "retain_dual"
        elif fast_value and fast_burden_ok:
            recommendation = "retain_fast_only"
        elif slow_value:
            recommendation = "retain_slow_only"
        elif rows:
            recommendation = "simplify_to_lower_burden_single_path"
        else:
            recommendation = "insufficient_evidence"
        return {
            "F": fast,
            "S": slow,
            "D": dual,
            "fast_unique_incidents": len(fast_unique),
            "slow_unique_incidents": len(slow_unique),
            "fast_delay_gain_hours": fast_delay_gain,
            "fast_short_recall_increment": fast_short_increment,
            "fast_severe_rescue": severe_fast_rescue,
            "fast_excess_burden_per_100_hours": fast_excess,
            "fast_retention_condition": fast_value and fast_burden_ok,
            "slow_incremental_incident_fraction": slow_increment,
            "slow_retention_condition": slow_value,
            "recommendation": recommendation,
        }

    @staticmethod
    def _human_utility(
        cases: Sequence[HumanUtilityCase],
    ) -> dict[str, Any]:
        return {
            "investigation_count": len(cases),
            "time_to_hypothesis_median_improvement": _relative_improvement(
                [
                    item.baseline_minutes_to_hypothesis
                    for item in cases
                ],
                [
                    item.advisory_minutes_to_hypothesis
                    for item in cases
                ],
            ),
            "components_investigated_median_reduction": _relative_improvement(
                [
                    float(item.baseline_components_investigated)
                    for item in cases
                ],
                [
                    float(item.advisory_components_investigated)
                    for item in cases
                ],
            ),
            "override_reason_completeness": (
                sum(item.override_reason_recorded for item in cases)
                / len(cases)
                if cases
                else None
            ),
        }

    def evaluate(
        self,
        *,
        predictions: Sequence[PredictionEnvelope],
        adjudications: Sequence[IncidentAdjudication],
        healthy_exposures: Sequence[HealthyExposure],
        human_utility_cases: Sequence[HumanUtilityCase] = (),
        human_utility_analysis_design: str | None = None,
        automation_bias_assessed: bool = False,
        agreement_pairs_impact: Sequence[tuple[str, str]] = (),
        agreement_pairs_cause: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        incidents = incident_cases_from_evidence(
            predictions=predictions,
            adjudications=adjudications,
        )
        burden = {
            policy: self._burden(healthy_exposures, policy)
            for policy in POLICIES
        }
        return {
            "metric_protocol_version": "0.7.1-oelr",
            "independent_unit": (
                "deployment_cluster_and_adjudicated_incident_episode"
            ),
            "incident_detection": {
                policy: self._detection(incidents, policy)
                for policy in POLICIES
            },
            "alert_burden": burden,
            "ranking": self._ranking(incidents),
            "support_and_data_quality": self._support(predictions),
            "path_attribution": self._path_attribution(
                incidents=incidents,
                burden=burden,
            ),
            "adjudication": {
                "final_incident_count": len(incidents),
                "impact_agreement": agreement_assessment(
                    agreement_pairs_impact
                ),
                "cause_family_agreement": agreement_assessment(
                    agreement_pairs_cause
                ),
                "agreement_assessment_status": (
                    "interpretable"
                    if agreement_assessment(
                        agreement_pairs_impact
                    )["status"]
                    == "interpretable"
                    and agreement_assessment(
                        agreement_pairs_cause
                    )["status"]
                    == "interpretable"
                    else "not_reportable_preregistered_reason"
                ),
                "confirmed_fraction": (
                    sum(
                        item.confidence
                        is AdjudicationConfidence.CONFIRMED
                        for item in _latest_adjudications(adjudications)
                    )
                    / len(_latest_adjudications(adjudications))
                    if adjudications
                    else None
                ),
            },
            "human_utility": {
                **self._human_utility(human_utility_cases),
                "analysis_design": human_utility_analysis_design,
                "automation_bias_assessed": automation_bias_assessed,
            },
        }
