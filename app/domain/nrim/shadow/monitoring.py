from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from .contracts import (
    DataQualityState,
    DecisionState,
    PredictionEnvelope,
)


@dataclass(frozen=True)
class MonitoringDimensions:
    deployment_pseudonym: str
    topology_version: str
    software_version: str
    workload_band: str
    collector_family: str


@dataclass(frozen=True)
class MonitoringAlert:
    code: str
    severity: str
    dimensions: MonitoringDimensions | None
    details: Mapping[str, Any]


class RuntimeHealthMonitor:
    """Operational telemetry for the NRIM service itself."""

    def __init__(self, *, decision_interval_seconds: float) -> None:
        if decision_interval_seconds <= 0.0:
            raise ValueError("Decision interval must be positive")
        self.decision_interval_seconds = decision_interval_seconds
        self.scheduled_cutoffs = 0
        self.completed_or_blocked_cutoffs = 0
        self.evidence_write_attempts = 0
        self.evidence_write_successes = 0
        self.latencies_seconds: list[float] = []
        self.backlog_depths: list[int] = []
        self.unregistered_bundle_changes = 0
        self.state_recovery_checks = 0
        self.state_recovery_matches = 0

    def record_cutoff(
        self,
        *,
        envelope: PredictionEnvelope | None,
        evidence_write_succeeded: bool,
        backlog_depth: int = 0,
    ) -> None:
        self.scheduled_cutoffs += 1
        self.evidence_write_attempts += 1
        if evidence_write_succeeded:
            self.evidence_write_successes += 1
        if envelope is not None:
            self.completed_or_blocked_cutoffs += 1
            self.latencies_seconds.append(
                envelope.inference_latency_ms / 1000.0
            )
        self.backlog_depths.append(max(0, int(backlog_depth)))

    def record_bundle_mismatch(self) -> None:
        self.unregistered_bundle_changes += 1

    def record_state_recovery(self, *, matches: bool) -> None:
        self.state_recovery_checks += 1
        self.state_recovery_matches += int(matches)

    @staticmethod
    def _quantile(values: Sequence[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, math.ceil(fraction * len(ordered)) - 1),
        )
        return float(ordered[index])

    def summary(self) -> dict[str, Any]:
        p95 = self._quantile(self.latencies_seconds, 0.95)
        return {
            "scheduled_cutoff_completion_fraction": (
                self.completed_or_blocked_cutoffs / self.scheduled_cutoffs
                if self.scheduled_cutoffs
                else None
            ),
            "evidence_durability_fraction": (
                self.evidence_write_successes / self.evidence_write_attempts
                if self.evidence_write_attempts
                else None
            ),
            "p95_latency_seconds": p95,
            "p95_latency_fraction_of_decision_interval": (
                p95 / self.decision_interval_seconds
                if p95 is not None
                else None
            ),
            "maximum_backlog_depth": max(self.backlog_depths, default=0),
            "unregistered_bundle_change_count": (
                self.unregistered_bundle_changes
            ),
            "state_recovery_parity_fraction": (
                self.state_recovery_matches / self.state_recovery_checks
                if self.state_recovery_checks
                else None
            ),
        }


class DataQualityMonitor:
    def __init__(self) -> None:
        self.expected_records = 0
        self.accepted_records = 0
        self.quarantined_records = 0
        self.blocking_category_failures = 0
        self.blocking_category_quarantined = 0
        self.future_leakage_count = 0
        self.audit_field_leakage_count = 0
        self.topology_provenance_count = 0
        self.prediction_count = 0
        self.reproduced_prediction_count = 0
        self.parity_checks = 0
        self.parity_matches = 0

    def record_contract(
        self,
        *,
        accepted: bool,
        blocking_category_failure: bool = False,
    ) -> None:
        self.expected_records += 1
        self.accepted_records += int(accepted)
        self.quarantined_records += int(not accepted)
        self.blocking_category_failures += int(blocking_category_failure)
        self.blocking_category_quarantined += int(
            blocking_category_failure and not accepted
        )

    def record_prediction_audit(
        self,
        *,
        topology_linked: bool,
        reproducible: bool,
        future_leakage: bool = False,
        audit_field_leakage: bool = False,
    ) -> None:
        self.prediction_count += 1
        self.topology_provenance_count += int(topology_linked)
        self.reproduced_prediction_count += int(reproducible)
        self.future_leakage_count += int(future_leakage)
        self.audit_field_leakage_count += int(audit_field_leakage)

    def record_parity(self, *, matches: bool) -> None:
        self.parity_checks += 1
        self.parity_matches += int(matches)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_compliance": (
                self.accepted_records / self.expected_records
                if self.expected_records
                else None
            ),
            "quarantine_rate": (
                self.quarantined_records / self.expected_records
                if self.expected_records
                else None
            ),
            "invalid_required_category_quarantined_fraction": (
                self.blocking_category_quarantined
                / self.blocking_category_failures
                if self.blocking_category_failures
                else None
            ),
            "future_leakage_count": self.future_leakage_count,
            "audit_field_leakage_count": self.audit_field_leakage_count,
            "topology_provenance_fraction": (
                self.topology_provenance_count / self.prediction_count
                if self.prediction_count
                else None
            ),
            "prediction_reproducibility_fraction": (
                self.reproduced_prediction_count / self.prediction_count
                if self.prediction_count
                else None
            ),
            "feature_parity_fraction": (
                self.parity_matches / self.parity_checks
                if self.parity_checks
                else None
            ),
        }


class SupportRateMonitor:
    """Detect abrupt support-routing changes without changing inference."""

    def __init__(
        self,
        *,
        baseline_size: int = 100,
        recent_size: int = 20,
        absolute_change_threshold: float = 0.20,
    ) -> None:
        if baseline_size < 1 or recent_size < 1:
            raise ValueError("Support monitor windows must be positive")
        self.baseline_size = baseline_size
        self.recent_size = recent_size
        self.absolute_change_threshold = absolute_change_threshold
        self._history: dict[
            tuple[str, str, str, str, str],
            deque[bool],
        ] = defaultdict(
            lambda: deque(maxlen=baseline_size + recent_size)
        )

    @staticmethod
    def _key(
        dimensions: MonitoringDimensions,
    ) -> tuple[str, str, str, str, str]:
        return (
            dimensions.deployment_pseudonym,
            dimensions.topology_version,
            dimensions.software_version,
            dimensions.workload_band,
            dimensions.collector_family,
        )

    def record(
        self,
        *,
        envelope: PredictionEnvelope,
        dimensions: MonitoringDimensions,
    ) -> MonitoringAlert | None:
        unsupported = envelope.final_decision in {
            DecisionState.UNKNOWN,
            DecisionState.ESCALATE,
        }
        history = self._history[self._key(dimensions)]
        history.append(unsupported)
        if len(history) < self.baseline_size + self.recent_size:
            return None
        rows = list(history)
        baseline = rows[: self.baseline_size]
        recent = rows[-self.recent_size :]
        baseline_rate = sum(baseline) / len(baseline)
        recent_rate = sum(recent) / len(recent)
        delta = recent_rate - baseline_rate
        if abs(delta) < self.absolute_change_threshold:
            return None
        return MonitoringAlert(
            code="ABRUPT_SUPPORT_RATE_CHANGE",
            severity="warning",
            dimensions=dimensions,
            details={
                "baseline_rate": baseline_rate,
                "recent_rate": recent_rate,
                "absolute_delta": delta,
            },
        )

    def rates(self) -> dict[str, Any]:
        return {
            "|".join(key): {
                "sample_count": len(rows),
                "unsupported_rate": (
                    sum(rows) / len(rows) if rows else None
                ),
            }
            for key, rows in sorted(self._history.items())
        }


class FeatureDriftMonitor:
    """Monitoring-only robust range/median drift against registered baselines."""

    def __init__(
        self,
        *,
        registered_ranges: Mapping[str, tuple[float, float]],
        registered_medians: Mapping[str, float],
        minimum_batch: int = 20,
        median_shift_fraction: float = 0.25,
    ) -> None:
        if set(registered_ranges) != set(registered_medians):
            raise ValueError("Drift range and median feature sets differ")
        self.ranges = dict(registered_ranges)
        self.medians = dict(registered_medians)
        self.minimum_batch = minimum_batch
        self.median_shift_fraction = median_shift_fraction

    def evaluate(
        self,
        rows: Sequence[Mapping[str, float]],
    ) -> dict[str, Any]:
        if len(rows) < self.minimum_batch:
            return {
                "status": "insufficient_evidence",
                "sample_count": len(rows),
                "features": {},
            }
        findings = {}
        for feature, (lower, upper) in self.ranges.items():
            values = [float(row[feature]) for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Drift observations must be finite")
            span = max(1e-9, upper - lower)
            current_median = median(values)
            out_of_range = sum(
                value < lower or value > upper for value in values
            ) / len(values)
            shift = abs(
                current_median - self.medians[feature]
            ) / span
            findings[feature] = {
                "median": current_median,
                "registered_median": self.medians[feature],
                "normalized_median_shift": shift,
                "out_of_range_fraction": out_of_range,
                "investigate": (
                    shift >= self.median_shift_fraction
                    or out_of_range >= 0.05
                ),
            }
        return {
            "status": "investigate"
            if any(item["investigate"] for item in findings.values())
            else "stable",
            "sample_count": len(rows),
            "features": findings,
        }


def structured_runtime_event(
    *,
    event_name: str,
    severity: str,
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    """OpenTelemetry-compatible structured event payload."""

    return {
        "event.name": event_name,
        "severity_text": severity.upper(),
        "attributes": {
            f"nrim.shadow.{key}": value
            for key, value in sorted(attributes.items())
        },
    }
