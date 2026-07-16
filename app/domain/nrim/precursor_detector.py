from __future__ import annotations

from dataclasses import dataclass

from .temporal import (
    EarlyWarning,
    ForecastStatus,
    PrecursorEvidence,
    ThresholdForecast,
    WarningTier,
)
from .telemetry import (
    BaselineStatus,
    CanonicalTelemetryEvent,
    TelemetryQuality,
)


@dataclass(frozen=True)
class PrecursorPolicy:
    early_warning_horizon_hours: float = 72.0
    high_risk_horizon_hours: float = 24.0
    minimum_independent_signals: int = 2
    minimum_evidence_strength: float = 0.50


def _latest_event(
    events: list[CanonicalTelemetryEvent],
    signal_name: str,
) -> CanonicalTelemetryEvent | None:
    matches = [
        event
        for event in events
        if event.signal_name == signal_name
        and event.quality != TelemetryQuality.QUARANTINED
    ]

    if not matches:
        return None

    return max(matches, key=lambda item: item.observed_at)


def detect_catchup_storage_precursor(
    *,
    case_id: str,
    forecast: ThresholdForecast,
    events: list[CanonicalTelemetryEvent],
    policy: PrecursorPolicy | None = None,
) -> EarlyWarning:
    policy = policy or PrecursorPolicy()
    evidence: list[PrecursorEvidence] = []

    storage_latency = _latest_event(
        events,
        "system.disk.io_latency",
    )
    recording_failures = _latest_event(
        events,
        "iptv.catchup.recording_failures",
    )
    io_errors = _latest_event(
        events,
        "system.disk.io_errors",
    )
    cleanup_failures = _latest_event(
        events,
        "iptv.catchup.cleanup_failures",
    )

    if (
        forecast.status == ForecastStatus.AVAILABLE
        and forecast.central_hours_to_threshold is not None
    ):
        if (
            forecast.central_hours_to_threshold
            <= policy.early_warning_horizon_hours
        ):
            strength = (
                0.90
                if forecast.central_hours_to_threshold
                <= policy.high_risk_horizon_hours
                else 0.70
            )

            evidence.append(
                PrecursorEvidence(
                    evidence_id=f"precursor:{forecast.forecast_id}",
                    signal_name=forecast.signal_name,
                    event_ids=forecast.supporting_event_ids,
                    statement=(
                        "Storage threshold crossing is forecast within "
                        f"{forecast.central_hours_to_threshold:.1f} hours."
                    ),
                    strength=strength,
                    quality=TelemetryQuality.HIGH,
                )
            )

    if storage_latency is not None:
        latency_abnormal = (
            storage_latency.baseline_status
            in {
                BaselineStatus.WARNING,
                BaselineStatus.ANOMALOUS,
                BaselineStatus.CRITICAL,
            }
            or (
                isinstance(storage_latency.value, (int, float))
                and float(storage_latency.value) >= 30.0
            )
        )

        if latency_abnormal:
            evidence.append(
                PrecursorEvidence(
                    evidence_id=(
                        f"precursor:{storage_latency.event_id}"
                    ),
                    signal_name=storage_latency.signal_name,
                    event_ids=[storage_latency.event_id],
                    statement=(
                        "Storage I/O latency is elevated and may "
                        "indicate storage-path degradation."
                    ),
                    strength=0.65,
                    quality=storage_latency.quality,
                )
            )

    if recording_failures is not None:
        if (
            isinstance(recording_failures.value, (int, float))
            and float(recording_failures.value) > 0
        ):
            evidence.append(
                PrecursorEvidence(
                    evidence_id=(
                        f"precursor:{recording_failures.event_id}"
                    ),
                    signal_name=recording_failures.signal_name,
                    event_ids=[recording_failures.event_id],
                    statement=(
                        "CatchUP recording failures are currently "
                        "present."
                    ),
                    strength=0.80,
                    quality=recording_failures.quality,
                )
            )

    if io_errors is not None:
        if (
            isinstance(io_errors.value, (int, float))
            and float(io_errors.value) > 0
        ):
            evidence.append(
                PrecursorEvidence(
                    evidence_id=f"precursor:{io_errors.event_id}",
                    signal_name=io_errors.signal_name,
                    event_ids=[io_errors.event_id],
                    statement="Storage I/O errors are present.",
                    strength=0.90,
                    quality=io_errors.quality,
                )
            )

    if cleanup_failures is not None:
        if (
            isinstance(cleanup_failures.value, (int, float))
            and float(cleanup_failures.value) > 0
        ):
            evidence.append(
                PrecursorEvidence(
                    evidence_id=(
                        f"precursor:{cleanup_failures.event_id}"
                    ),
                    signal_name=cleanup_failures.signal_name,
                    event_ids=[cleanup_failures.event_id],
                    statement=(
                        "CatchUP cleanup failures may accelerate "
                        "storage consumption."
                    ),
                    strength=0.70,
                    quality=cleanup_failures.quality,
                )
            )

    usable_evidence = [
        item
        for item in evidence
        if item.strength >= policy.minimum_evidence_strength
        and item.quality != TelemetryQuality.QUARANTINED
    ]

    independent_signals = {
        item.signal_name
        for item in usable_evidence
    }

    recording_failure_present = any(
        item.signal_name
        == "iptv.catchup.recording_failures"
        for item in usable_evidence
    )

    forecast_near = (
        forecast.status == ForecastStatus.AVAILABLE
        and forecast.central_hours_to_threshold is not None
        and forecast.central_hours_to_threshold
        <= policy.high_risk_horizon_hours
    )

    threshold_crossed = (
        forecast.status
        == ForecastStatus.THRESHOLD_ALREADY_CROSSED
    )

    if threshold_crossed or (
        recording_failure_present
        and len(independent_signals) >= 2
    ):
        tier = WarningTier.ACTIVE_INCIDENT
        conclusion = (
            "CatchUP storage-path degradation may already be "
            "affecting service. Immediate engineer validation is "
            "required."
        )
    elif (
        forecast_near
        and len(independent_signals)
        >= policy.minimum_independent_signals
    ):
        tier = WarningTier.HIGH_RISK
        conclusion = (
            "CatchUP storage-path failure risk is high. Capacity "
            "crossing is near and independent deterioration signals "
            "are present."
        )
    elif (
        forecast.status == ForecastStatus.AVAILABLE
        and forecast.central_hours_to_threshold is not None
        and forecast.central_hours_to_threshold
        <= policy.early_warning_horizon_hours
    ):
        tier = WarningTier.EARLY_WARNING
        conclusion = (
            "CatchUP storage exhaustion is forecast within the "
            "configured early-warning horizon."
        )
    elif usable_evidence:
        tier = WarningTier.WATCH
        conclusion = (
            "One or more storage-path deterioration indicators "
            "require continued observation."
        )
    else:
        tier = WarningTier.NONE
        conclusion = (
            "No sufficiently supported CatchUP storage precursor "
            "is currently detected."
        )

    return EarlyWarning(
        case_id=case_id,
        deployment_id=forecast.deployment_id,
        component_node_id=forecast.component_node_id,
        failure_signature_id=(
            "FS-CATCHUP-STORAGE-PRECURSOR-001"
        ),
        tier=tier,
        title="CatchUP storage-path early warning",
        conclusion=conclusion,
        evidence=usable_evidence,
        forecast_id=forecast.forecast_id,
        recommended_next_test=(
            "Inspect free capacity, storage growth, cleanup-job "
            "status, write latency, and recording-worker health."
            if tier != WarningTier.NONE
            else None
        ),
        requires_engineer_review=(
            tier
            in {
                WarningTier.HIGH_RISK,
                WarningTier.ACTIVE_INCIDENT,
            }
        ),
        limitations=[
            "Synthetic development thresholds are used.",
            "No NetUP-certified operating limits are available.",
            "The forecast assumes near-term trend continuity.",
            "Root cause is not confirmed by this warning.",
        ],
    )
