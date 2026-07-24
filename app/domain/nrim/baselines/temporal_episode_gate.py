from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from statistics import mean, median
from typing import Any, Iterable, Sequence


class GateState(str, Enum):
    """Operational states exposed by the temporal incident gate."""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    INCIDENT = "incident"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"
    ESCALATE = "escalate"


class SupportStatus(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EpisodeRecord:
    """Scenario-clustered sequence contract.

    Identifiers and evaluation-only labels live on the record, outside the
    feature tensors consumed by either learned model.
    """

    scenario_id: str
    split: str
    topology_group: str
    ordered_window_ids: tuple[str, ...]
    windows: tuple[dict[str, Any], ...]
    impact_onset: datetime | None
    recovery_timestamp: datetime | None
    confounder_family: str
    healthy_control: bool = False
    failure_family: str = "unknown"
    ood_axis_labels: tuple[str, ...] = ()
    fault_duration_hours: float | None = None


@dataclass(frozen=True)
class SequenceAssemblyAudit:
    scenario_count: int
    window_count: int
    overlapping_window_pairs: int
    cross_split_scenario_count: int
    duplicate_window_count: int
    non_monotonic_sequence_count: int
    leakage_free: bool


@dataclass(frozen=True)
class SupportAssessment:
    status: SupportStatus = SupportStatus.IN_DISTRIBUTION
    support_score: float = 0.0
    axis_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TemporalGateConfig:
    """Deterministic evidence and explicit-duration state policy."""

    window_threshold: float = 0.5
    evidence_center: float = 0.5
    decay: float = 0.75
    entry_threshold: float = 1.0
    exit_threshold: float = 0.25
    impact_weight: float = 0.5
    recovery_weight: float = 0.5
    minimum_suspect_windows: int = 2
    minimum_incident_windows: int = 1
    minimum_recovery_windows: int = 2
    maximum_suspect_windows: int = 4
    cooldown_windows: int = 1
    severity_override_probability: float = 0.95
    unsupported_escalation_probability: float = 0.80

    def __post_init__(self) -> None:
        probabilities = (
            self.window_threshold,
            self.evidence_center,
            self.severity_override_probability,
            self.unsupported_escalation_probability,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("Gate probabilities must be strictly between 0 and 1")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between zero and one")
        if self.exit_threshold > self.entry_threshold:
            raise ValueError("exit_threshold cannot exceed entry_threshold")
        durations = (
            self.minimum_suspect_windows,
            self.minimum_incident_windows,
            self.minimum_recovery_windows,
            self.maximum_suspect_windows,
            self.cooldown_windows,
        )
        if any(value < 0 for value in durations):
            raise ValueError("State durations cannot be negative")
        if self.maximum_suspect_windows < self.minimum_suspect_windows:
            raise ValueError(
                "maximum_suspect_windows cannot be below minimum_suspect_windows"
            )


@dataclass(frozen=True)
class TemporalDecision:
    window_id: str
    timestamp: datetime
    probability: float
    evidence: float
    state: GateState
    support_status: SupportStatus
    support_score: float
    residual: float = 0.0
    impact_score: float = 0.0
    slow_statistic: float = 0.0
    fast_triggered: bool = False
    slow_triggered: bool = False
    activation_path: str = "none"


@dataclass(frozen=True)
class ScenarioEpisodeResult:
    scenario_id: str
    split: str
    topology_group: str
    confounder_family: str
    healthy_control: bool
    failure_family: str
    decisions: tuple[TemporalDecision, ...]
    truth: tuple[bool, ...]
    fault_duration_hours: float | None = None


@dataclass(frozen=True)
class RiskBound:
    events: int
    trials: int
    point_estimate: float
    upper_confidence_bound: float
    confidence: float


@dataclass(frozen=True)
class TemporalCalibrationResult:
    config: TemporalGateConfig
    feasible: bool
    objective: float
    metrics: dict[str, Any]
    evaluated_configurations: int
    feasibility_failures: tuple[str, ...]


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _window_interval(window: dict[str, Any]) -> tuple[datetime, datetime]:
    return (
        _parse_datetime(window["observation_start"]),
        _parse_datetime(window["observation_cutoff"]),
    )


def _derive_ood_axes(
    *,
    metadata: dict[str, Any],
    split: str,
    train_room_range: tuple[int, int],
    train_floor_range: tuple[int, int],
    train_missingness_modes: set[str],
    train_missing_range: tuple[float, float],
) -> tuple[str, ...]:
    if split != "ood_test":
        return ()
    axes = []
    rooms = int(metadata["room_count"])
    floors = int(metadata["floor_count"])
    missing_mode = str(metadata["missingness_mode"])
    missing_fraction = float(metadata["missing_fraction"])
    # The registered OOD generator changes workload scale. Exact topology IDs
    # are never compared: split isolation intentionally makes every
    # fingerprint unseen, while the generated structural range can overlap.
    axes.append("workload")
    if not train_room_range[0] <= rooms <= train_room_range[1]:
        axes.append("workload")
    if not train_floor_range[0] <= floors <= train_floor_range[1]:
        axes.append("workload")
    if missing_mode not in train_missingness_modes:
        axes.append("applicability")
    if not train_missing_range[0] <= missing_fraction <= train_missing_range[1]:
        axes.append("telemetry")
    return tuple(sorted(set(axes)))


def assemble_episode_records(
    *,
    windows: Iterable[dict[str, Any]],
    scenario_metadata: dict[str, dict[str, Any]],
) -> tuple[list[EpisodeRecord], SequenceAssemblyAudit]:
    """Build chronological scenario sequences and reject split leakage."""

    windows = list(windows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_windows: set[str] = set()
    duplicate_window_count = 0
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        window_id = str(window["window_id"])
        if window_id in seen_windows:
            duplicate_window_count += 1
        seen_windows.add(window_id)
        scenario_id = str(window["source_scenario_id"])
        grouped[scenario_id].append(window)
        scenario_splits[scenario_id].add(str(window["split"]))

    cross_split = sum(len(splits) > 1 for splits in scenario_splits.values())
    non_monotonic = 0
    overlap_count = 0
    records: list[EpisodeRecord] = []

    train_metadata = [
        record
        for record in scenario_metadata.values()
        if str(record["split"]) == "train"
    ]
    if not train_metadata:
        raise ValueError("Sequence assembly requires training metadata")
    train_rooms = [int(record["room_count"]) for record in train_metadata]
    train_floors = [int(record["floor_count"]) for record in train_metadata]
    train_missing = [float(record["missing_fraction"]) for record in train_metadata]
    train_modes = {str(record["missingness_mode"]) for record in train_metadata}

    for scenario_id, scenario_windows in grouped.items():
        if scenario_id not in scenario_metadata:
            raise ValueError(f"Missing scenario metadata for {scenario_id}")
        ordered = sorted(
            scenario_windows,
            key=lambda window: _parse_datetime(window["observation_cutoff"]),
        )
        cutoffs = [
            _parse_datetime(window["observation_cutoff"]) for window in ordered
        ]
        if len(cutoffs) != len(set(cutoffs)):
            non_monotonic += 1
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_start, previous_end = _window_interval(previous)
            current_start, current_end = _window_interval(current)
            if current_start < previous_end and previous_start < current_end:
                overlap_count += 1

        truth = [
            bool(int(window["targets"]["current_incident"])) for window in ordered
        ]
        onset_index = next(
            (index for index, target in enumerate(truth) if target),
            None,
        )
        recovery_index = None
        if onset_index is not None:
            recovery_index = next(
                (
                    index
                    for index in range(onset_index + 1, len(truth))
                    if not truth[index] and any(truth[onset_index:index])
                ),
                None,
            )
        metadata = scenario_metadata[scenario_id]
        confounder_family = "+".join(sorted(metadata["confounders"])) or "none"
        axes = _derive_ood_axes(
            metadata=metadata,
            split=str(ordered[0]["split"]),
            train_room_range=(min(train_rooms), max(train_rooms)),
            train_floor_range=(min(train_floors), max(train_floors)),
            train_missingness_modes=train_modes,
            train_missing_range=(min(train_missing), max(train_missing)),
        )
        records.append(
            EpisodeRecord(
                scenario_id=scenario_id,
                split=str(ordered[0]["split"]),
                topology_group=str(metadata["topology_fingerprint"]),
                ordered_window_ids=tuple(
                    str(window["window_id"]) for window in ordered
                ),
                windows=tuple(ordered),
                impact_onset=(
                    cutoffs[onset_index] if onset_index is not None else None
                ),
                recovery_timestamp=(
                    cutoffs[recovery_index]
                    if recovery_index is not None
                    else None
                ),
                confounder_family=confounder_family,
                healthy_control=bool(metadata["healthy"]),
                failure_family=str(metadata["failure_type"]),
                ood_axis_labels=axes,
                fault_duration_hours=(
                    float(
                        metadata[
                            "fault_duration_hours"
                        ]
                    )
                    if metadata.get(
                        "fault_duration_hours"
                    )
                    is not None
                    else None
                ),
            )
        )

    leakage_free = not (
        duplicate_window_count or cross_split or non_monotonic
    )
    audit = SequenceAssemblyAudit(
        scenario_count=len(records),
        window_count=len(windows),
        overlapping_window_pairs=overlap_count,
        cross_split_scenario_count=cross_split,
        duplicate_window_count=duplicate_window_count,
        non_monotonic_sequence_count=non_monotonic,
        leakage_free=leakage_free,
    )
    if not leakage_free:
        raise ValueError(
            "Sequence assembly leakage audit failed: "
            f"cross_split={cross_split}, duplicates={duplicate_window_count}, "
            f"non_monotonic={non_monotonic}"
        )
    return sorted(records, key=lambda record: record.scenario_id), audit


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(clipped / (1.0 - clipped))


class TemporalEpisodeGate:
    """Causal evidence accumulator plus explicit-duration controller."""

    def __init__(self, config: TemporalGateConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        record: EpisodeRecord,
        probabilities: Sequence[float],
        support: Sequence[SupportAssessment] | None = None,
        impact_signals: Sequence[float] | None = None,
        recovery_signals: Sequence[float] | None = None,
    ) -> ScenarioEpisodeResult:
        count = len(record.windows)
        if len(probabilities) != count:
            raise ValueError("A probability is required for every window")
        support = support or [SupportAssessment()] * count
        impact_signals = impact_signals or [0.0] * count
        recovery_signals = recovery_signals or [0.0] * count
        if any(
            len(values) != count
            for values in (support, impact_signals, recovery_signals)
        ):
            raise ValueError("All temporal inputs must align with the sequence")

        evidence = 0.0
        state = GateState.HEALTHY
        suspect_age = 0
        incident_age = 0
        recovery_age = 0
        cooldown = 0
        decisions: list[TemporalDecision] = []
        center_logit = _logit(self.config.evidence_center)

        for window, probability, assessment, impact, recovery in zip(
            record.windows,
            probabilities,
            support,
            impact_signals,
            recovery_signals,
            strict=True,
        ):
            probability = min(1.0, max(0.0, float(probability)))
            evidence = max(
                0.0,
                self.config.decay * evidence
                + _logit(probability)
                - center_logit
                + self.config.impact_weight * max(0.0, float(impact))
                - self.config.recovery_weight * max(0.0, float(recovery)),
            )
            cooldown = max(0, cooldown - 1)

            if assessment.status is SupportStatus.UNSUPPORTED:
                visible_state = (
                    GateState.ESCALATE
                    if probability
                    >= self.config.unsupported_escalation_probability
                    else GateState.UNKNOWN
                )
            else:
                severe = probability >= self.config.severity_override_probability
                above_entry = (
                    evidence >= self.config.entry_threshold
                    and probability >= self.config.window_threshold
                )
                below_exit = (
                    evidence <= self.config.exit_threshold
                    and probability < self.config.window_threshold
                )

                if state is GateState.HEALTHY:
                    if severe:
                        state = GateState.INCIDENT
                        incident_age = 1
                    elif above_entry and cooldown == 0:
                        if self.config.minimum_suspect_windows == 0:
                            state = GateState.INCIDENT
                            incident_age = 1
                        else:
                            state = GateState.SUSPECT
                            suspect_age = 1
                elif state is GateState.SUSPECT:
                    suspect_age += 1
                    if severe or (
                        above_entry
                        and suspect_age >= self.config.minimum_suspect_windows
                    ):
                        state = GateState.INCIDENT
                        incident_age = 1
                    elif (
                        not above_entry
                        or suspect_age >= self.config.maximum_suspect_windows
                    ):
                        state = GateState.HEALTHY
                        suspect_age = 0
                        cooldown = self.config.cooldown_windows
                elif state is GateState.INCIDENT:
                    incident_age += 1
                    if (
                        below_exit
                        and incident_age >= self.config.minimum_incident_windows
                    ):
                        state = GateState.RECOVERY
                        recovery_age = 1
                elif state is GateState.RECOVERY:
                    recovery_age += 1
                    if severe or above_entry:
                        state = GateState.INCIDENT
                        incident_age = 1
                    elif (
                        below_exit
                        and recovery_age >= self.config.minimum_recovery_windows
                    ):
                        state = GateState.HEALTHY
                        recovery_age = 0
                        cooldown = self.config.cooldown_windows
                visible_state = state

            decisions.append(
                TemporalDecision(
                    window_id=str(window["window_id"]),
                    timestamp=_parse_datetime(window["observation_cutoff"]),
                    probability=probability,
                    evidence=evidence,
                    state=visible_state,
                    support_status=assessment.status,
                    support_score=assessment.support_score,
                )
            )

        return ScenarioEpisodeResult(
            scenario_id=record.scenario_id,
            split=record.split,
            topology_group=record.topology_group,
            confounder_family=record.confounder_family,
            healthy_control=record.healthy_control,
            failure_family=record.failure_family,
            decisions=tuple(decisions),
            truth=tuple(
                bool(int(window["targets"]["current_incident"]))
                for window in record.windows
            ),
            fault_duration_hours=(
                record.fault_duration_hours
            ),
        )


def run_independent_window_policy(
    *,
    record: EpisodeRecord,
    probabilities: Sequence[float],
    threshold: float,
    support: Sequence[SupportAssessment] | None = None,
) -> ScenarioEpisodeResult:
    """Frozen reference: each window is selected independently."""

    support = support or [SupportAssessment()] * len(record.windows)
    decisions = []
    for window, probability, assessment in zip(
        record.windows, probabilities, support, strict=True
    ):
        if assessment.status is SupportStatus.UNSUPPORTED:
            state = (
                GateState.ESCALATE
                if probability >= threshold
                else GateState.UNKNOWN
            )
        else:
            state = (
                GateState.INCIDENT
                if probability >= threshold
                else GateState.HEALTHY
            )
        decisions.append(
            TemporalDecision(
                window_id=str(window["window_id"]),
                timestamp=_parse_datetime(window["observation_cutoff"]),
                probability=float(probability),
                evidence=0.0,
                state=state,
                support_status=assessment.status,
                support_score=assessment.support_score,
            )
        )
    return ScenarioEpisodeResult(
        scenario_id=record.scenario_id,
        split=record.split,
        topology_group=record.topology_group,
        confounder_family=record.confounder_family,
        healthy_control=record.healthy_control,
        failure_family=record.failure_family,
        decisions=tuple(decisions),
        truth=tuple(
            bool(int(window["targets"]["current_incident"]))
            for window in record.windows
        ),
        fault_duration_hours=(
            record.fault_duration_hours
        ),
    )


def _runs(values: Sequence[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _selected(decision: TemporalDecision) -> bool:
    return decision.state in {GateState.INCIDENT, GateState.ESCALATE}


def exact_binomial_upper_bound(
    *,
    events: int,
    trials: int,
    confidence: float = 0.95,
) -> RiskBound:
    """One-sided exact Clopper-Pearson binomial upper bound."""

    if trials < 0 or not 0 <= events <= trials:
        raise ValueError("events must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if trials == 0:
        return RiskBound(events, trials, 0.0, 1.0, confidence)
    point = events / trials
    if events == trials:
        upper = 1.0
    else:
        alpha = 1.0 - confidence

        def cdf(probability: float) -> float:
            return sum(
                math.comb(trials, index)
                * probability**index
                * (1.0 - probability) ** (trials - index)
                for index in range(events + 1)
            )

        low = point
        high = 1.0
        for _ in range(80):
            midpoint = (low + high) / 2.0
            if cdf(midpoint) > alpha:
                low = midpoint
            else:
                high = midpoint
        upper = high
    return RiskBound(events, trials, point, upper, confidence)


def _hours_between(
    decisions: Sequence[TemporalDecision], left: int, right: int
) -> float:
    return max(
        0.0,
        (
            decisions[right].timestamp - decisions[left].timestamp
        ).total_seconds()
        / 3600.0,
    )


def _quantile(
    values: Sequence[float],
    fraction: float,
) -> float | None:
    if not values:
        return None
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


def summarize_episode_results(
    results: Sequence[ScenarioEpisodeResult],
    *,
    risk_confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute deployment-facing metrics at scenario and episode level."""

    # The truth vector distinguishes impact episodes, while the immutable
    # EpisodeRecord contract distinguishes deliberately fault-free controls.
    # ScenarioEpisodeResult carries that distinction for risk accounting.
    healthy = [
        result
        for result in results
        if getattr(result, "healthy_control", False)
    ]
    faulty = [result for result in results if any(result.truth)]
    healthy_false = [
        result for result in healthy if any(map(_selected, result.decisions))
    ]
    healthy_by_topology: dict[str, list[ScenarioEpisodeResult]] = defaultdict(
        list
    )
    for result in healthy:
        healthy_by_topology[result.topology_group].append(result)
    topology_false_events = sum(
        any(
            any(map(_selected, result.decisions))
            for result in topology_results
        )
        for topology_results in healthy_by_topology.values()
    )
    risk = exact_binomial_upper_bound(
        events=topology_false_events,
        trials=len(healthy_by_topology),
        confidence=risk_confidence,
    )

    total_healthy_hours = 0.0
    false_episode_count = 0
    for result in healthy:
        selected = [_selected(decision) for decision in result.decisions]
        false_episode_count += len(_runs(selected))
        if len(result.decisions) > 1:
            spacing = median(
                _hours_between(result.decisions, index, index + 1)
                for index in range(len(result.decisions) - 1)
            )
            total_healthy_hours += (
                _hours_between(
                    result.decisions, 0, len(result.decisions) - 1
                )
                + spacing
            )
        elif result.decisions:
            total_healthy_hours += 1.0

    true_episode_count = 0
    detected_episode_count = 0
    detection_delays = []
    recovery_delays = []
    fragments = []
    duration_buckets: dict[str, list[bool]] = defaultdict(list)
    for result in faulty:
        selected = [_selected(decision) for decision in result.decisions]
        predicted_runs = _runs(selected)
        for truth_start, truth_end in _runs(result.truth):
            true_episode_count += 1
            detections = [
                index
                for index in range(truth_start, truth_end + 1)
                if selected[index]
            ]
            if detections:
                detected_episode_count += 1
                detection_delays.append(
                    _hours_between(result.decisions, truth_start, detections[0])
                )
            registered_duration = (
                result.fault_duration_hours
            )
            if registered_duration is not None:
                if len(result.decisions) > 1:
                    decision_spacing = median(
                        _hours_between(
                            result.decisions,
                            index,
                            index + 1,
                        )
                        for index in range(
                            len(result.decisions) - 1
                        )
                    )
                else:
                    decision_spacing = 2.0
                duration_windows = max(
                    1,
                    math.ceil(
                        registered_duration
                        / max(
                            1e-6,
                            decision_spacing,
                        )
                    ),
                )
            else:
                duration_windows = (
                    truth_end - truth_start + 1
                )
            if duration_windows <= 2:
                duration_bucket = (
                    "1_to_2_decision_windows"
                )
            elif duration_windows <= 4:
                duration_bucket = (
                    "3_to_4_decision_windows"
                )
            else:
                duration_bucket = (
                    "longer_than_4_decision_windows"
                )
            duration_buckets[duration_bucket].append(
                bool(detections)
            )
            overlapping = sum(
                not (end < truth_start or start > truth_end)
                for start, end in predicted_runs
            )
            fragments.append(max(0, overlapping - 1))
            if truth_end + 1 < len(selected):
                first_clear = next(
                    (
                        index
                        for index in range(truth_end + 1, len(selected))
                        if not selected[index]
                    ),
                    None,
                )
                if first_clear is not None:
                    recovery_delays.append(
                        _hours_between(
                            result.decisions, truth_end, first_clear
                        )
                    )

    cohort_rows: dict[str, list[ScenarioEpisodeResult]] = defaultdict(list)
    for result in healthy:
        cohort_rows[result.confounder_family].append(result)
    cohort_risk = {}
    for cohort, rows in cohort_rows.items():
        rows_by_topology: dict[
            str, list[ScenarioEpisodeResult]
        ] = defaultdict(list)
        for result in rows:
            rows_by_topology[result.topology_group].append(result)
        events = sum(
            any(
                any(map(_selected, result.decisions))
                for result in topology_rows
            )
            for topology_rows in rows_by_topology.values()
        )
        bound = exact_binomial_upper_bound(
            events=events,
            trials=len(rows_by_topology),
            confidence=risk_confidence,
        )
        cohort_risk[cohort] = {
            "events": events,
            "trials": len(rows_by_topology),
            "scenario_count": len(rows),
            "point_estimate": (
                sum(
                    any(map(_selected, result.decisions))
                    for result in rows
                )
                / len(rows)
            ),
            "cluster_point_estimate": bound.point_estimate,
            "upper_confidence_bound": bound.upper_confidence_bound,
        }

    unsupported_decisions = [
        decision
        for result in results
        for decision in result.decisions
        if decision.support_status is SupportStatus.UNSUPPORTED
    ]
    unsupported_safe = sum(
        decision.state in {GateState.UNKNOWN, GateState.ESCALATE}
        for decision in unsupported_decisions
    )
    state_counts = {
        state.value: sum(
            decision.state is state
            for result in results
            for decision in result.decisions
        )
        for state in GateState
    }
    activation_path_counts = {
        path: sum(
            decision.activation_path == path
            for result in results
            for decision in result.decisions
        )
        for path in (
            "none",
            "fast",
            "slow",
            "both",
            "unsupported",
        )
    }
    failure_family_metrics = {}
    for family in sorted({result.failure_family for result in faulty}):
        family_results = [
            result for result in faulty if result.failure_family == family
        ]
        family_true_episodes = 0
        family_detected_episodes = 0
        family_delays = []
        for result in family_results:
            selected = [_selected(decision) for decision in result.decisions]
            for truth_start, truth_end in _runs(result.truth):
                family_true_episodes += 1
                detections = [
                    index
                    for index in range(truth_start, truth_end + 1)
                    if selected[index]
                ]
                if detections:
                    family_detected_episodes += 1
                    family_delays.append(
                        _hours_between(
                            result.decisions,
                            truth_start,
                            detections[0],
                        )
                    )
        failure_family_metrics[family] = {
            "scenario_count": len(family_results),
            "true_episode_count": family_true_episodes,
            "detected_episode_count": family_detected_episodes,
            "episode_recall": (
                family_detected_episodes / family_true_episodes
                if family_true_episodes
                else 0.0
            ),
            "median_detection_delay_hours": (
                median(family_delays) if family_delays else None
            ),
            "p90_detection_delay_hours": _quantile(
                family_delays,
                0.90,
            ),
        }

    topology_metrics = {}
    for topology in sorted({result.topology_group for result in results}):
        topology_results = [
            result for result in results if result.topology_group == topology
        ]
        topology_healthy = [
            result for result in topology_results if result.healthy_control
        ]
        topology_impact = [
            result for result in topology_results if any(result.truth)
        ]
        topology_metrics[topology] = {
            "scenario_count": len(topology_results),
            "healthy_false_selection_rate": (
                sum(
                    any(map(_selected, result.decisions))
                    for result in topology_healthy
                )
                / len(topology_healthy)
                if topology_healthy
                else None
            ),
            "impact_scenario_detection_rate": (
                sum(
                    any(
                        _selected(decision) and truth
                        for decision, truth in zip(
                            result.decisions,
                            result.truth,
                            strict=True,
                        )
                    )
                    for result in topology_impact
                )
                / len(topology_impact)
                if topology_impact
                else None
            ),
        }
    return {
        "scenario_count": len(results),
        "healthy_scenario_count": len(healthy),
        "healthy_independent_topology_count": len(healthy_by_topology),
        "impact_scenario_count": len(faulty),
        "healthy_scenario_false_selection_rate": (
            len(healthy_false) / len(healthy) if healthy else 0.0
        ),
        "healthy_topology_false_selection_rate": risk.point_estimate,
        "healthy_scenario_false_selection_ucb": risk.upper_confidence_bound,
        "risk_confidence": risk.confidence,
        "false_episodes_per_100_healthy_scenario_hours": (
            100.0 * false_episode_count / total_healthy_hours
            if total_healthy_hours
            else 0.0
        ),
        "true_episode_count": true_episode_count,
        "detected_episode_count": detected_episode_count,
        "incident_episode_recall": (
            detected_episode_count / true_episode_count
            if true_episode_count
            else 0.0
        ),
        "median_detection_delay_hours": (
            median(detection_delays) if detection_delays else None
        ),
        "mean_detection_delay_hours": (
            mean(detection_delays) if detection_delays else None
        ),
        "p90_detection_delay_hours": _quantile(
            detection_delays,
            0.90,
        ),
        "mean_fragmentation": mean(fragments) if fragments else 0.0,
        "median_recovery_delay_hours": (
            median(recovery_delays) if recovery_delays else None
        ),
        "confounder_risk": cohort_risk,
        "failure_family_metrics": failure_family_metrics,
        "episode_duration_metrics": {
            bucket: {
                "episode_count": len(detected),
                "episode_recall": (
                    sum(detected) / len(detected)
                    if detected
                    else 0.0
                ),
            }
            for bucket, detected in sorted(
                duration_buckets.items()
            )
        },
        "topology_group_metrics": topology_metrics,
        "worst_confounder_point_risk": max(
            (
                float(row["point_estimate"])
                for row in cohort_risk.values()
            ),
            default=0.0,
        ),
        "worst_confounder_ucb": max(
            (
                float(row["upper_confidence_bound"])
                for row in cohort_risk.values()
            ),
            default=1.0,
        ),
        "unsupported_decision_count": len(unsupported_decisions),
        "state_counts": state_counts,
        "activation_path_counts": activation_path_counts,
        "unsupported_safe_semantics_rate": (
            unsupported_safe / len(unsupported_decisions)
            if unsupported_decisions
            else 1.0
        ),
        "unsupported_unknown_rate": (
            sum(
                decision.state is GateState.UNKNOWN
                for decision in unsupported_decisions
            )
            / len(unsupported_decisions)
            if unsupported_decisions
            else 0.0
        ),
        "unsupported_escalation_rate": (
            sum(
                decision.state is GateState.ESCALATE
                for decision in unsupported_decisions
            )
            / len(unsupported_decisions)
            if unsupported_decisions
            else 0.0
        ),
    }


def scenario_cluster_bootstrap(
    *,
    values_by_scenario: dict[str, float],
    seed: int = 42,
    samples: int = 2_000,
) -> dict[str, float]:
    """Bootstrap scenarios as clusters; never resample overlapping windows."""

    if not values_by_scenario:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    scenario_ids = sorted(values_by_scenario)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [
            scenario_ids[generator.randrange(len(scenario_ids))]
            for _ in scenario_ids
        ]
        estimates.append(mean(values_by_scenario[item] for item in selected))
    estimates.sort()
    return {
        "mean": mean(values_by_scenario.values()),
        "ci_lower": estimates[int(0.025 * (samples - 1))],
        "ci_upper": estimates[int(0.975 * (samples - 1))],
    }


def calibrate_temporal_gate(
    *,
    validation_records: Sequence[EpisodeRecord],
    probabilities: dict[str, Sequence[float]],
    candidate_configs: Sequence[TemporalGateConfig],
    impact_signals: dict[str, Sequence[float]] | None = None,
    recovery_signals: dict[str, Sequence[float]] | None = None,
    support: dict[str, Sequence[SupportAssessment]] | None = None,
    maximum_global_risk_ucb: float = 0.10,
    maximum_confounder_risk_ucb: float = 0.15,
    minimum_episode_recall: float = 0.70,
    maximum_median_delay_hours: float = 3.0,
    risk_confidence: float = 0.95,
) -> TemporalCalibrationResult:
    """Select on validation scenario clusters under registered constraints."""

    if not validation_records or not candidate_configs:
        raise ValueError("Calibration requires records and candidate configs")
    candidates = []
    for config in candidate_configs:
        gate = TemporalEpisodeGate(config)
        results = [
            gate.run(
                record=record,
                probabilities=probabilities[record.scenario_id],
                support=(
                    support[record.scenario_id]
                    if support is not None
                    else None
                ),
                impact_signals=(
                    impact_signals[record.scenario_id]
                    if impact_signals is not None
                    else None
                ),
                recovery_signals=(
                    recovery_signals[record.scenario_id]
                    if recovery_signals is not None
                    else None
                ),
            )
            for record in validation_records
        ]
        metrics = summarize_episode_results(
            results, risk_confidence=risk_confidence
        )
        failures = []
        if metrics["healthy_scenario_false_selection_ucb"] > maximum_global_risk_ucb:
            failures.append("global_risk_ucb")
        if metrics["worst_confounder_ucb"] > maximum_confounder_risk_ucb:
            failures.append("confounder_risk_ucb")
        if metrics["incident_episode_recall"] < minimum_episode_recall:
            failures.append("episode_recall")
        delay = metrics["median_detection_delay_hours"]
        if delay is None or delay > maximum_median_delay_hours:
            failures.append("detection_delay")
        objective = (
            float(metrics["incident_episode_recall"])
            - float(metrics["healthy_scenario_false_selection_rate"])
            - 0.01 * float(delay or maximum_median_delay_hours * 2.0)
            - 0.05 * float(metrics["mean_fragmentation"])
        )
        candidates.append((not failures, objective, config, metrics, failures))

    feasible = [candidate for candidate in candidates if candidate[0]]
    if feasible:
        selected = max(feasible, key=lambda item: item[1])
    else:
        selected = min(
            candidates,
            key=lambda item: (
                len(item[4]),
                max(
                    0.0,
                    item[3]["healthy_scenario_false_selection_ucb"]
                    - maximum_global_risk_ucb,
                )
                + max(
                    0.0,
                    item[3]["worst_confounder_ucb"]
                    - maximum_confounder_risk_ucb,
                )
                + max(
                    0.0,
                    minimum_episode_recall
                    - item[3]["incident_episode_recall"],
                ),
                -item[1],
            ),
        )
    return TemporalCalibrationResult(
        config=selected[2],
        feasible=selected[0],
        objective=selected[1],
        metrics=selected[3],
        evaluated_configurations=len(candidates),
        feasibility_failures=tuple(selected[4]),
    )


def default_temporal_config_grid(
    *, window_threshold: float
) -> tuple[TemporalGateConfig, ...]:
    """Small preregistered grid; all choices are selected on validation only."""

    configs = []
    for decay in (0.50, 0.75, 0.90):
        for entry in (0.5, 1.0, 1.5, 2.0):
            for suspect in (1, 2, 3):
                configs.append(
                    TemporalGateConfig(
                        window_threshold=window_threshold,
                        evidence_center=max(0.05, window_threshold * 0.8),
                        decay=decay,
                        entry_threshold=entry,
                        exit_threshold=entry * 0.20,
                        minimum_suspect_windows=suspect,
                        maximum_suspect_windows=max(3, suspect + 1),
                        minimum_recovery_windows=2,
                        severity_override_probability=max(
                            0.95, window_threshold
                        ),
                    )
                )
    return tuple(configs)


def ablation_configs(
    *,
    calibrated: TemporalGateConfig,
) -> dict[str, TemporalGateConfig | None]:
    """Registered A-G temporal ablations from the recommendation."""

    return {
        "A_independent_window": None,
        "B_hysteresis": replace(
            calibrated,
            decay=0.0,
            entry_threshold=0.0,
            exit_threshold=0.0,
            minimum_suspect_windows=1,
            minimum_recovery_windows=1,
        ),
        "C_evidence_accumulator": replace(
            calibrated,
            minimum_suspect_windows=0,
            minimum_recovery_windows=1,
        ),
        "D_duration_states": calibrated,
        "E_scenario_risk_control": calibrated,
        "F_shift_sentinel": None,
        "G_full_rtieg": calibrated,
    }
