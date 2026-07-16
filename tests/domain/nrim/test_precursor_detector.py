from datetime import datetime, timezone

from app.domain.nrim.precursor_detector import (
    detect_catchup_storage_precursor,
)
from app.domain.nrim.telemetry import (
    CanonicalTelemetryEvent,
    GoldenSignal,
    TelemetryQuality,
    TelemetrySignalType,
)
from app.domain.nrim.temporal import (
    ForecastStatus,
    ThresholdForecast,
    WarningTier,
)


def make_forecast(
    hours: float,
) -> ThresholdForecast:
    now = datetime.now(timezone.utc)

    return ThresholdForecast(
        deployment_id="deployment_1",
        component_node_id="storage_1",
        signal_name="system.disk.utilization",
        unit="percent",
        latest_observation_at=now,
        latest_value=88.0,
        threshold=95.0,
        status=ForecastStatus.AVAILABLE,
        slope_per_hour=0.5,
        central_hours_to_threshold=hours,
        earliest_hours_to_threshold=max(0.0, hours - 3),
        latest_hours_to_threshold=hours + 5,
        trend_quality=0.9,
        supporting_event_ids=["storage_event"],
    )


def make_metric(
    *,
    event_id: str,
    signal_name: str,
    value: float,
    unit: str,
) -> CanonicalTelemetryEvent:
    now = datetime.now(timezone.utc)

    return CanonicalTelemetryEvent(
        event_id=event_id,
        deployment_id="deployment_1",
        component_node_id="storage_1",
        service_node_ids=["service_catchup"],
        observed_at=now,
        ingested_at=now,
        signal_type=TelemetrySignalType.METRIC,
        signal_name=signal_name,
        value=value,
        unit=unit,
        collection_source="test",
        quality=TelemetryQuality.HIGH,
        golden_signal=GoldenSignal.ERRORS,
    )


def test_near_forecast_and_latency_create_high_risk() -> None:
    warning = detect_catchup_storage_precursor(
        case_id="case_1",
        forecast=make_forecast(12.0),
        events=[
            make_metric(
                event_id="latency_1",
                signal_name="system.disk.io_latency",
                value=45.0,
                unit="milliseconds",
            )
        ],
    )

    assert warning.tier == WarningTier.HIGH_RISK


def test_far_forecast_creates_watch_or_none() -> None:
    warning = detect_catchup_storage_precursor(
        case_id="case_1",
        forecast=make_forecast(120.0),
        events=[],
    )

    assert warning.tier in {
        WarningTier.NONE,
        WarningTier.WATCH,
    }


def test_recording_failures_with_storage_evidence_is_incident() -> None:
    warning = detect_catchup_storage_precursor(
        case_id="case_1",
        forecast=make_forecast(12.0),
        events=[
            make_metric(
                event_id="failure_1",
                signal_name=(
                    "iptv.catchup.recording_failures"
                ),
                value=5.0,
                unit="failures_per_5_minutes",
            )
        ],
    )

    assert warning.tier == WarningTier.ACTIVE_INCIDENT
